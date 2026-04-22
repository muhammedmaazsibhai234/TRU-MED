
from collections import OrderedDict  
import torch                        
import torch.nn as nn                
import torch.nn.functional as F      
try:                                  
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:                    
    checkpoint_wrapper = lambda x: x

from timm.models import register_model            
from timm.models.layers import DropPath, trunc_normal_ 
from timm.models.vision_transformer import _cfg    

from .dssa import DSSA                              
from ._common import AttentionLePE, DWConv          

def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not supported!')

class EvidentialHead(nn.Module):
    """
    Evidential (Dirichlet) head that operates on per-token features.

    Input:  z  [B, H, W, C]
    Output:
        alpha_map  [B, H, W, num_classes]
        p_mean_map [B, H, W, num_classes]
        sigma_map  [B, H, W]   (scalar uncertainty per token)
    """
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
            nn.Softplus()  # evidence ≥ 0
        )
        self.num_classes = num_classes
        self.in_dim = in_dim

    def forward(self, z: torch.Tensor):
        # z: [B, H, W, C]
        assert z.dim() == 4, f"EvidentialHead expects [B,H,W,C], got {z.shape}"
        B, H, W, C = z.shape
        assert C == self.in_dim, f"Expected in_dim={self.in_dim}, got C={C}"

        # Flatten spatial dims → per-token features
        z_flat = z.reshape(B * H * W, C)  # [B*H*W, C]

        e_flat = self.net(z_flat)         # [B*H*W, num_classes]
        alpha_flat = e_flat + 1.0         # Dirichlet α = e + 1
        S_flat = alpha_flat.sum(-1, keepdim=True)       # [B*H*W, 1]
        p_mean_flat = alpha_flat / S_flat               # [B*H*W, num_classes]
        sigma_flat = (self.num_classes / S_flat).clamp(max=1.0).squeeze(-1)  # [B*H*W]

        # Reshape back to spatial maps
        alpha_map = alpha_flat.view(B, H, W, self.num_classes)
        p_mean_map = p_mean_flat.view(B, H, W, self.num_classes)
        sigma_map = sigma_flat.view(B, H, W)  # [B, H, W]

        return alpha_map, p_mean_map, sigma_map



class PrototypeHead(nn.Module):
    """Prototype-augmented classifier head (keeps DSSA unchanged).

    Classifies by comparing spatial tokens to learned prototypes ("this looks like that"),
    then aggregating evidence across tokens and prototypes per class.

    Optional auxiliary losses (returned in aux dict) improve reliability:
      - cluster_loss: encourages each prototype to be close to at least one token in the batch
      - diversity_loss: discourages prototype collapse (encourages diverse prototypes)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_prototypes_per_class: int = 8,
        temp: float = 10.0,
        agg: str = "logsumexp",
        use_cosine: bool = True,
        eps: float = 1e-6,
        diversity_mode: str = "all",   # "all" or "within_class"
        diversity_power: float = 2.0,
    ):
        super().__init__()
        assert agg in ("logsumexp", "max", "mean")
        assert diversity_mode in ("all", "within_class")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.agg = agg
        self.use_cosine = use_cosine
        self.eps = eps
        self.diversity_mode = diversity_mode
        self.diversity_power = diversity_power

        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, in_dim))
        trunc_normal_(self.prototypes, std=0.02)

        proto_classes = []
        for c in range(num_classes):
            proto_classes += [c] * num_prototypes_per_class
        self.register_buffer("proto_class", torch.tensor(proto_classes, dtype=torch.long), persistent=False)

        # learnable temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temp))))

    def _compute_sim(self, tokens: torch.Tensor):
        """tokens: [B,T,C] -> sim: [B,T,K]"""
        protos = self.prototypes
        if self.use_cosine:
            tokens_n = F.normalize(tokens, dim=-1, eps=self.eps)
            protos_n = F.normalize(protos, dim=-1, eps=self.eps)
        else:
            tokens_n = tokens
            protos_n = protos

        temp = torch.exp(self.log_temp).clamp(1e-4, 100.0)
        sim = torch.matmul(tokens_n, protos_n.t()) * temp
        return sim, protos_n

    def _proto_aggregate_tokens(self, sim: torch.Tensor):
        """Aggregate token->prototype similarity across tokens.
        sim: [B,T,K] -> proto_evidence: [B,K]
        """
        if self.agg == "logsumexp":
            return torch.logsumexp(sim, dim=1)
        if self.agg == "max":
            return sim.max(dim=1).values
        return sim.mean(dim=1)

    def _proto_aggregate_classes(self, proto_evidence: torch.Tensor):
        """Aggregate prototype evidence to class logits.
        proto_evidence: [B,K] -> logits: [B,num_classes]
        """
        B = proto_evidence.shape[0]
        logits = proto_evidence.new_full((B, self.num_classes), -1e9)
        for c in range(self.num_classes):
            m = (self.proto_class == c)
            if self.agg == "logsumexp":
                logits[:, c] = torch.logsumexp(proto_evidence[:, m], dim=-1)
            elif self.agg == "max":
                logits[:, c] = proto_evidence[:, m].max(dim=-1).values
            else:
                logits[:, c] = proto_evidence[:, m].mean(dim=-1)
        return logits

    def compute_aux_losses(self, sim: torch.Tensor, protos_n: torch.Tensor):
        """Compute prototype auxiliary losses.

        sim: [B,T,K] similarities (after optional cosine + temperature)
        protos_n: [K,C] normalized prototypes if cosine else raw prototypes

        Returns:
          cluster_loss (scalar), diversity_loss (scalar)
        """
        # --- Cluster loss: each prototype should match at least one token in the batch.
        # For each prototype k, find max similarity over tokens (per sample), then mean over batch+prototypes.
        # We minimize negative similarity (maximize match).
        # sim_max: [B,K]
        sim_max = sim.max(dim=1).values
        cluster_loss = -sim_max.mean()

        # --- Diversity loss: discourage prototype collapse.
        # Use cosine similarity matrix between prototypes (if not cosine, we still normalize for diversity).
        p = F.normalize(self.prototypes, dim=-1, eps=self.eps)
        cos = p @ p.t()  # [K,K]
        K = cos.shape[0]
        eye = torch.eye(K, device=cos.device, dtype=cos.dtype)
        cos_off = cos * (1.0 - eye)

        if self.diversity_mode == "within_class":
            # penalize similarities within same class (encourage diverse modes per class)
            same = (self.proto_class.view(-1, 1) == self.proto_class.view(1, -1)).to(cos.dtype)
            same_off = same * (1.0 - eye)
            vals = cos_off * same_off
        else:
            # penalize all off-diagonal similarities
            vals = cos_off

        diversity_loss = (vals.abs() ** self.diversity_power).mean()
        return cluster_loss, diversity_loss

    def forward(self, feats: torch.Tensor, return_aux: bool = False, compute_proto_loss: bool = False):
        """feats: [B,C,H,W]"""
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # [B,T,C]

        sim, protos_n = self._compute_sim(tokens)            # [B,T,K]
        proto_evidence = self._proto_aggregate_tokens(sim)   # [B,K]
        logits = self._proto_aggregate_classes(proto_evidence)

        if return_aux:
            aux = {"proto_evidence": proto_evidence.detach()}
            if compute_proto_loss:
                cl, dl = self.compute_aux_losses(sim, protos_n)
                aux["proto_cluster_loss"] = cl
                aux["proto_diversity_loss"] = dl
            return logits, aux

        return logits


class BlockU(nn.Module):
    

    def __init__(self,
                 dim: int,
                 mlp_ratio: float,
                 stage,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = -1.,
                 num_heads: int = 8,
                 n_win: int = 7,
                 qk_scale=None,
                 topk: int = 4,
                 mlp_dwconv: bool = False,
                 side_dwconv: int = 5,
                 before_attn_dwconv: int = 3,
                 pre_norm: bool = True,
                 auto_pad: bool = False,
                 # uncertainty control
                 num_classes: int = 2,
                 beta_gate: float = 0.5,
                 mlp_dropout_p: float = 0.2,
                 # DSSA sparsity / routing knobs
                 token_keep_ratio: float = 0.25,
                 route_init_scale: float = 10.0,
                 stop_grad_routing: bool = False):
        super().__init__()

        # Positional DWConv
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(
                dim, dim,
                kernel_size=before_attn_dwconv,
                padding=before_attn_dwconv // 2,
                groups=dim
            )
        else:
            self.pos_embed = lambda x: 0

        # LayerNorm before attention / MLP (applied in NHWC)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # Attention: DSSA / LePE / Identity
        if topk > 0:
            self.attn = DSSA(
                dim=dim,
                num_heads=num_heads,
                n_win=n_win,
                qk_scale=qk_scale,
                topk=topk,
                side_dwconv=side_dwconv,
                auto_pad=auto_pad,
                attn_backend='torch'
            ,
                token_keep_ratio=token_keep_ratio,
                route_init_scale=route_init_scale,
                stop_grad_routing=stop_grad_routing)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        else:
            self.attn = nn.Identity()

        # MLP with optional DWConv and dropout
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Dropout(mlp_dropout_p),
            nn.Linear(int(mlp_ratio * dim), dim)
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # LayerScale
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
        else:
            self.use_layer_scale = False

        self.pre_norm = pre_norm
        self.stage = stage  # e.g. [0], [1], [2], [3]

        # Evidential per-token uncertainty head
        self.unc_head = EvidentialHead(in_dim=dim, hidden=256, num_classes=num_classes)

        # Uncertainty gate strength (per stage, passed from outer class)
        self.beta_gate = beta_gate

        # UGTR: router + refinement
        router_hidden = max(dim // 4, 16)
        self.router = nn.Sequential(
            nn.Conv2d(dim, router_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(router_hidden, 1, kernel_size=1),  # logits
        )

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

        # Learnable strength for refine branch
        self.lambda_refine = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Caches for debugging / analysis
        self.last_alpha = None
        self.last_pmean = None
        self.last_sigma_map = None
        self.last_sigma_global = None

        self.last_mask_raw = None
        self.last_mask = None
        self.last_evidence_sum = None
    def _compute_sigma(self, x_nhwc: torch.Tensor):
        
        alpha_map, p_mean_map, sigma_map = self.unc_head(x_nhwc)  # [B,H,W,C/num_classes/H,W]

        # Cache raw outputs
        self.last_alpha = alpha_map.detach()
        self.last_pmean = p_mean_map.detach()

        # Normalize sigma_map per image to [0,1]
        # Use sigma_map as-is (already derived from Dirichlet strength), no per-image min-max.
        # sigma_map: [B,H,W] in [0,1] (clamped in EvidentialHead)
        sigma_clamped = sigma_map.clamp(0.0, 1.0)

        self.last_sigma_map = sigma_clamped.detach()

        # Global scalar = mean uncertainty over tokens
        sigma_global = sigma_clamped.mean(dim=(1, 2))  # [B]
        self.last_sigma_global = sigma_global.detach()

        return sigma_global, sigma_clamped

    def forward(self, x: torch.Tensor, tissue_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        use_ugtr = (self.stage[0] >= 1)

        x = x + self.pos_embed(x)  # [B,C,H,W]
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]

        if self.pre_norm:
            x1 = self.norm1(x)  # [B,H,W,C]

            if use_ugtr:
                sigma, sigma_map = self._compute_sigma(x1)  # [B], [B,H,W]
            else:
                sigma = x1.new_zeros(x1.shape[0])          # [B]
                sigma_map = None

            attn_out = self.attn(x1)  # [B,H,W,C]

            if use_ugtr:
                x1_nchw = x1.permute(0, 3, 1, 2)          # [B,C,H,W]
                mask_logits = self.router(x1_nchw)        # [B,1,H,W] (logits)
                mask = torch.sigmoid(mask_logits)         # for routing only

                # store raw router output (logits, before sigmoid/masking)
                self.last_mask_raw = mask_logits  # keep grad for routing loss

                # Option B: explicitly suppress background (tissue_mask: 1=tissue, 0=background)
                if tissue_mask is not None:
                    # tissue_mask expected shape [B,1,H,W]
                    mask = mask * tissue_mask

                # suppress high-uncertainty tokens (avoid refining uncertain/background-like regions)
                mask = mask * (1.0 - sigma_map.unsqueeze(1))      # [B,1,H,W]

                self.last_mask = mask.detach()
                # evidence_sum per token: sum(evidence) = sum(alpha-1)
                if self.last_alpha is not None:
                    self.last_evidence_sum = (self.last_alpha - 1.0).sum(dim=-1).detach()  # [B,H,W]
                refine_out = self.refine(x1_nchw)         # [B,C,H,W]
                refine_out = refine_out.permute(0, 2, 3, 1)  # [B,H,W,C]

                mask_nhwc = mask.permute(0, 2, 3, 1)      # [B,H,W,1]

                delta = mask_nhwc * (self.lambda_refine * (refine_out - attn_out))
                # Apply uncertainty gating ONLY to the refinement delta (preserve baseline attention)
                scale = (1.0 - self.beta_gate * sigma).view(-1, 1, 1, 1)  # [B,1,1,1]
                delta = delta * scale
                attn_routed = attn_out + delta
            else:
                attn_routed = attn_out

            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * attn_routed)
            else:
                x = x + self.drop_path(attn_routed)

            x2 = self.norm2(x)
            mlp_out = self.mlp(x2)
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma2 * mlp_out)
            else:
                x = x + self.drop_path(mlp_out)
        else:
            attn_out = self.attn(x)
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * attn_out))
            else:
                x = self.norm1(x + self.drop_path(attn_out))

            mlp_out = self.mlp(x)
            if self.use_layer_scale:
                x = self.norm2(x + self.drop_path(self.gamma2 * mlp_out))
            else:
                x = self.norm2(x + self.drop_path(mlp_out))

        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]
        return x

class MedFormerUncertaintyDSSA(nn.Module):
    """
    MedFormer backbone with DSSA, per-token evidential uncertainty, and UGTR.

    Changes vs earlier:
      - Stage-dependent beta_gate (weaker in shallow, stronger in deep stages)
      - Final norm is LayerNorm over channels (on NHWC), not BatchNorm2d
    """

    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 num_classes=2,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 qk_scale=None,
                 representation_size=None,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 use_checkpoint_stages=[],
                 # DSSA / attention hyper-params
                 n_win=7,
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[64, 128, 320, 512],
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 mlp_ratios=[4, 4, 4, 4],
                 mlp_dwconv=False,
                 # uncertainty knobs
                 beta_gate=0.5,
                 mlp_dropout_p=0.2,
                 # DSSA sparsity / routing knobs
                 token_keep_ratio: float = 0.25,
                 route_init_scale: float = 10.0,
                 # prototype head (improves decision by prototype evidence)
                 use_proto_head: bool = False,
                 num_prototypes_per_class: int = 8,
                 proto_temp: float = 10.0,
                 proto_agg: str = 'logsumexp',
                 proto_use_cosine: bool = True,
                 # prototype regularizers (aux losses)
                 proto_cluster_weight: float = 0.05,
                 proto_diversity_weight: float = 0.01,
                 proto_diversity_mode: str = "within_class",
                 stop_grad_routing: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # ---- Downsample / stem ----
        self.downsample_layers = nn.ModuleList()

        # Stem: two convs
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.add_module('pe', get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if 0 in use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        # Stages 1–3 downsampling
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and (i + 1) in pe_stages:
                downsample_layer.add_module('pe', get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if (i + 1) in use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)

        # ---- Transformer stages ----
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        # Stage-dependent beta_gate (scale global beta_gate across depths)
        # e.g. if beta_gate=0.5 → [0.125, 0.25, 0.375, 0.5]
        beta_scales = [0.25, 0.5, 0.75, 1.0]
        beta_stages = [beta_gate * s for s in beta_scales]

        for i in range(4):
            stage_blocks = []
            for j in range(depth[i]):
                stage_blocks.append(
                    BlockU(
                        dim=embed_dim[i],
                        mlp_ratio=mlp_ratios[i],
                        stage=[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_scale=qk_scale,
                        topk=topks[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        num_classes=num_classes,
                        beta_gate=beta_stages[i],
                        mlp_dropout_p=mlp_dropout_p
                    ,
                        token_keep_ratio=token_keep_ratio,
                        route_init_scale=route_init_scale,
                        stop_grad_routing=stop_grad_routing)
                )
            stage = nn.Sequential(*stage_blocks)
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        # Final norm before head: LayerNorm over channels (on NHWC)
        self.norm = nn.LayerNorm(embed_dim[-1])

        # Feature dimension after pooling (optionally project with pre_logits)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim[-1], representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.num_features = embed_dim[-1]
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Prototype-augmented classifier head 
        self.use_proto_head = use_proto_head
        self.proto_cluster_weight = float(proto_cluster_weight)
        self.proto_diversity_weight = float(proto_diversity_weight)
        self.proto_diversity_mode = proto_diversity_mode
        if use_proto_head:
            self.proto_head = PrototypeHead(
                in_dim=self.num_features,
                num_classes=num_classes,
                num_prototypes_per_class=num_prototypes_per_class,
                temp=proto_temp,
                agg=proto_agg,
                use_cosine=proto_use_cosine,
                diversity_mode=proto_diversity_mode,
            )
        else:
            self.proto_head = None


        # Initialize weights
        self.apply(self._init_weights)

    # ----- Weight init -----
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    # ----- Classifier helpers -----
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def get_block(self, stage_idx: int, block_idx: int) -> nn.Module:
        return self.stages[stage_idx][block_idx]

    def forward_features(self, x: torch.Tensor, tissue_mask: torch.Tensor = None, return_aux: bool = False):
        # Conv stem + transformer stages
        aux = None
        if return_aux:
            aux = {"masks": [], "masks_raw": [], "sigmas": [], "evidence_sum": [], "routing_loss": None}
        for i in range(4):
            x = self.downsample_layers[i](x)
            # iterate blocks so we can collect UGTR signals
            for blk in self.stages[i]:
                if tissue_mask is not None:
                    tmask = F.interpolate(tissue_mask.float(), size=x.shape[2:], mode="nearest")
                else:
                    tmask = None
                x = blk(x, tissue_mask=tmask)
                if return_aux and hasattr(blk, "last_mask") and blk.last_mask is not None:
                    aux["masks"].append(blk.last_mask)
                    aux["masks_raw"].append(blk.last_mask_raw if blk.last_mask_raw is not None else blk.last_mask)
                    aux["sigmas"].append(blk.last_sigma_map if getattr(blk, "last_sigma_map", None) is not None else None)
                    if getattr(blk, "last_evidence_sum", None) is not None:
                        aux["evidence_sum"].append(blk.last_evidence_sum)

                    # Supervise router with tissue mask (if provided) using raw router output.
                    if tissue_mask is not None and getattr(blk, "last_mask_raw", None) is not None:
                        # Ensure shapes: [B,1,H,W]
                        tmask_sup = tmask.float()
                        logits = blk.last_mask_raw  # raw logits
                        target = tmask_sup.float()
                        if logits.shape != target.shape:
                            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
                        bce = F.binary_cross_entropy_with_logits(logits, target)
                        aux["routing_loss"] = bce if aux["routing_loss"] is None else (aux["routing_loss"] + bce)


        # x: [B, C, H, W] → NHWC for LayerNorm
        x = x.permute(0, 2, 3, 1)   # [B,H,W,C]
        x = self.norm(x)            # LN over C
        x = x.permute(0, 3, 1, 2)   # [B,C,H,W]

        if return_aux:
            return x, aux
        return x

    def forward(self, x: torch.Tensor, tissue_mask: torch.Tensor = None, return_aux: bool = False):
        if return_aux:
            feats, aux = self.forward_features(x, tissue_mask=tissue_mask, return_aux=True)
        else:
            feats = self.forward_features(x, tissue_mask=tissue_mask, return_aux=False)
            aux = None

        # feats already LayerNorm'ed in forward_features
        # Decision head
        if self.use_proto_head and (self.proto_head is not None):
            if return_aux:
                logits, proto_aux = self.proto_head(feats, return_aux=True, compute_proto_loss=True)
                if aux is not None:
                    aux['proto'] = proto_aux
                    # Weighted prototype regularizer (add to CE in your training loop)
                    if 'proto_cluster_loss' in proto_aux and 'proto_diversity_loss' in proto_aux:
                        aux['proto_reg_loss'] = (self.proto_cluster_weight * proto_aux['proto_cluster_loss']
                                                + self.proto_diversity_weight * proto_aux['proto_diversity_loss'])
            else:
                logits = self.proto_head(feats, return_aux=False)
        else:
            pooled = feats.flatten(2).mean(-1)               # [B,C] (original global avg pooling)
            pooled = self.pre_logits(pooled)                 # [B,num_features]
            logits = self.head(pooled)

        if return_aux:
            return logits, aux
        return logits

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T).clamp(min=1e-4, max=100.0)
        return logits / T

@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, mc_runs: int = 20):
    model.eval()
    def _enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(_enable_dropout)

    logits_T = []
    for _ in range(mc_runs):
        logits_T.append(model(x))  
    logits_T = torch.stack(logits_T, dim=0)  
    probs_T = logits_T.softmax(-1)           

    p_mean = probs_T.mean(0)                 
    H_total = -(p_mean * (p_mean.clamp_min(1e-12)).log()).sum(-1)         
    H_each  = -(probs_T * (probs_T.clamp_min(1e-12)).log()).sum(-1)       
    H_alea  = H_each.mean(0)                                             
    MI = H_total - H_alea                                                
    return p_mean, H_total, H_alea, MI, logits_T

@register_model
def tru_med(pretrained=False, pretrained_cfg=None,
                               pretrained_cfg_overlay=None, **kwargs):
    model = MedFormerUncertaintyDSSA(
        depth=[2, 2, 8, 2],
        embed_dim=[32, 64, 128, 256], mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[32, 64, 128, 256],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def tru_med(pretrained=False, pretrained_cfg=None,
                                pretrained_cfg_overlay=None, **kwargs):
    model = MedFormerUncertaintyDSSA(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def tru_med(pretrained=False, pretrained_cfg=None,
                               pretrained_cfg_overlay=None, **kwargs):
    model = MedFormerUncertaintyDSSA(
        depth=[2, 2, 8, 2],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        pre_norm=True,
        pe=None,
        **kwargs)
    model.default_cfg = _cfg()
    return model