from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention  # type: ignore
from diffusers.utils import deprecate


class CAOFSASFProcessor:
    """
    Drop-in replacement for diffusers' AttnProcessor that implements:
      - SASF: Self-Attention Style Fusion (AdaIN / DSIN)
      - CAOF: Cross-Attention Optimal Transport-based feature blending

    Mirrors the inlined behavior but lives outside diffusers so the library files remain unchanged.
    """

    def __init__(
        self,
        enable_CAOF: bool = True,
        enable_SASF: bool = True,
        sasf_mode: str = "DSIN",  # "AdaIN" or "DSIN"
        use_ot: bool = True,
    ):
        self.enable_CAOF = enable_CAOF
        self.enable_SASF = enable_SASF
        self.sasf_mode = sasf_mode
        self.use_ot = use_ot

        self.config = {
            "w0": 0.7,
            "source_percentile": 60,
            "dest_percentile": 60,
            "gamma": 0.1,
            "epsilon": 0.1,
            "lambda_feature": 1,
            "lambda_spatial": 0,
        }

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecate(
                "scale",
                "1.0.0",
                (
                    "The `scale` argument is deprecated and will be ignored. "
                    "Please remove it; pass scale via `cross_attention_kwargs` on the pipeline instead."
                ),
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            height = width = int(hidden_states.shape[1] ** 0.5)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # SASF pre (self-attention)
        if self.enable_SASF and encoder_hidden_states is None and hidden_states.shape[0] != 1:
            hidden_states = self.SASF(hidden_states)

        # QKV
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # SASF on K/V when shapes match (self-like)
        if (
            self.enable_SASF
            and encoder_hidden_states is not None
            and encoder_hidden_states.shape == hidden_states.shape
            and hidden_states.shape[0] != 1
        ):
            key, value = self.SASF(hidden_states, key, value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # CAOF (cross-attn-like case)
        if (
            self.enable_CAOF
            and encoder_hidden_states is not None
            and encoder_hidden_states.shape != hidden_states.shape
            and encoder_hidden_states.shape[0] != 1
        ):
            hidden_states = self.CAOF(attn, hidden_states, attention_probs, height, width)

        # out proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    # ---------------- SASF ----------------
    def SASF(
        self,
        hidden_states: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Self-Attention Style Fusion.

        If key,value are None => apply style normalization (AdaIN / DSIN) to hidden_states.
        If key,value exist => do the K/V region swap.
        """
        eps = 1e-5

        def adain(cnt_feat: torch.Tensor, sty_feat: torch.Tensor) -> torch.Tensor:
            cnt_mean = cnt_feat.mean(dim=0, keepdim=True)
            cnt_std = cnt_feat.std(dim=0, keepdim=True) + eps
            sty_mean = sty_feat.mean(dim=0, keepdim=True)
            sty_std = sty_feat.std(dim=0, keepdim=True) + eps
            return (cnt_feat - cnt_mean) / cnt_std * sty_std + sty_mean

        def local_smoothing_1d(feat: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
            half_k = (kernel_size - 1) // 2
            x = torch.arange(-half_k, half_k + 1, device=feat.device, dtype=feat.dtype)
            gauss = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss = gauss / gauss.sum()
            kernel_1d = gauss.view(1, 1, -1)

            feat = feat.unsqueeze(0).transpose(1, 2)  # [1, C, T]
            channels = feat.shape[1]
            kernel_1d = kernel_1d.expand(channels, 1, kernel_size)
            padding = (kernel_size - 1) // 2
            smoothed = F.conv1d(feat, kernel_1d, stride=1, padding=padding, groups=channels)
            smoothed = smoothed.transpose(1, 2).squeeze(0)  # [T, C]
            return smoothed

        def inject_details(adain_out, cnt_feat, sty_feat, alpha: float):
            cnt_smooth = local_smoothing_1d(cnt_feat, kernel_size=5, sigma=2.5)
            sty_smooth = local_smoothing_1d(sty_feat, kernel_size=5, sigma=2.5)
            contentHF = cnt_feat - cnt_smooth
            styleHF = sty_feat - sty_smooth
            return adain_out + alpha * (styleHF - contentHF)

        # Style normalization path (no K/V provided)
        if key is None and value is None:
            # slot meanings from your example:
            # 0: uncond, 1: init, 2: target, 3: blend, 4: style
            cnt_feat = hidden_states[2, :, :]
            sty_feat = hidden_states[4, :, :]

            if self.sasf_mode.lower() == "adain":
                hidden_states[2, :, :] = adain(cnt_feat, sty_feat)
            else:  # "DSIN"
                adain_out = adain(cnt_feat, sty_feat)
                final_out = inject_details(adain_out, cnt_feat, sty_feat, alpha=0.5)
                hidden_states[2, :, :] = final_out

            return hidden_states

        # K/V swap region (as in your snippet)
        key[20:30, :, :], value[20:30, :, :] = (key[40:50, :, :], value[40:50, :, :])
        return key, value

    # ---------------- CAOF ----------------
    def CAOF(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_probs: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        w0 = self.config["w0"]
        source_percentile = self.config["source_percentile"]
        dest_percentile = self.config["dest_percentile"]
        gamma = self.config["gamma"]
        epsilon = self.config["epsilon"]
        lambda_feature = self.config["lambda_feature"]
        lambda_spatial = self.config["lambda_spatial"]

        device = hidden_states.device
        indices = {"uncond": 0, "init": 1, "target": 2, "blend": 3, "style": 4}

        attention_probs_target = attention_probs[
            indices["target"] * attn.heads : (indices["target"] + 1) * attn.heads, :, :
        ]
        attention_probs_blend = attention_probs[
            indices["blend"] * attn.heads : (indices["blend"] + 1) * attn.heads, :, :
        ]

        target_token_index = 1
        blend_token_index = 1

        attention_target_to_target = attention_probs_target[:, :, target_token_index].mean(dim=0)
        attention_blend_to_blend = attention_probs_blend[:, :, blend_token_index].mean(dim=0)

        source_threshold = torch.quantile(attention_blend_to_blend, source_percentile / 100.0)
        dest_threshold = torch.quantile(attention_target_to_target, dest_percentile / 100.0)

        source_mask = attention_blend_to_blend >= source_threshold
        dest_mask = attention_target_to_target >= dest_threshold

        source_indices = source_mask.nonzero(as_tuple=False).squeeze()
        dest_indices = dest_mask.nonzero(as_tuple=False).squeeze()

        x_s = source_indices % width
        y_s = source_indices // width
        x_d = dest_indices % width
        y_d = dest_indices // width

        hidden_states_target = hidden_states[indices["target"], :, :]
        hidden_states_blend = hidden_states[indices["blend"], :, :]

        source_embeddings = hidden_states_blend[source_indices, :]
        dest_embeddings = hidden_states_target[dest_indices, :]

        # simple index-wise alpha blend (no OT)
        if not self.use_ot:
            m = source_embeddings.size(0)
            n = dest_embeddings.size(0)
            k = min(m, n)
            blended = dest_embeddings.clone()
            blended[:k] = (1.0 - w0) * dest_embeddings[:k] + w0 * source_embeddings[:k]
            updated_target = hidden_states_target.clone()
            updated_target[dest_indices, :] = blended
            hidden_states[indices["target"], :, :] = updated_target
            return hidden_states

        # OT path
        m = source_embeddings.shape[0]
        n = dest_embeddings.shape[0]

        dest_norm = F.normalize(dest_embeddings, p=2, dim=1)
        source_norm = F.normalize(source_embeddings, p=2, dim=1)
        feature_distance = 1 - torch.mm(dest_norm, source_norm.t())

        dest_positions = torch.stack([x_d, y_d], dim=1).float()
        source_positions = torch.stack([x_s, y_s], dim=1).float()
        spatial_distance = torch.cdist(dest_positions, source_positions, p=2)

        cost_matrix = lambda_feature * feature_distance + lambda_spatial * spatial_distance
        if cost_matrix.numel() > 0:
            cost_matrix = cost_matrix / cost_matrix.max().clamp(min=1e-8)

        supply = torch.full((m,), 1.0 / max(m, 1), device=device)
        demand = torch.full((n,), 1.0 / max(n, 1), device=device)

        transport_plan = self.sinkhorn(supply, demand, cost_matrix, gamma, max_iter=1000, tol=1e-6)

        if m > 0 and n > 0:
            min_supply = 1.0 / m
            source_total_mass = transport_plan.sum(dim=0)
            unused_sources = (source_total_mass < min_supply).nonzero(as_tuple=False).squeeze()
            if unused_sources.numel() > 0:
                adjustment = (min_supply - source_total_mass[unused_sources]) / n
                transport_plan[:, unused_sources] += adjustment.unsqueeze(0)
                transport_plan = transport_plan / transport_plan.sum(dim=1, keepdim=True)

            assignment_counts = (transport_plan > 0).sum(dim=0).float()
            mu_A = assignment_counts.mean()
            sigma_A = assignment_counts.std()
            CV = sigma_A / mu_A.clamp_min(1e-8)

            if CV > epsilon:
                gamma = gamma * 1.5
                transport_plan = self.sinkhorn(supply, demand, cost_matrix, gamma, max_iter=1000, tol=1e-6)

        blended_embeddings = (1.0 - w0) * dest_embeddings + w0 * torch.mm(transport_plan, source_embeddings)
        updated_hidden_states_target = hidden_states_target.clone()
        updated_hidden_states_target[dest_indices, :] = blended_embeddings

        hidden_states[indices["target"], :, :] = updated_hidden_states_target
        return hidden_states

    @staticmethod
    def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, gamma: float, max_iter: int = 1000, tol: float = 1e-6):
        if C.numel() == 0:
            return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)

        K = torch.exp(-C / max(gamma, 1e-8))
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(max_iter):
            u_prev = u.clone()
            Ku = K.t() @ v
            Kv = K @ u
            Ku = Ku + 1e-12
            Kv = Kv + 1e-12
            u = a / Ku
            v = b / Kv
            if torch.max(torch.abs(u - u_prev)) < tol:
                break

        T = torch.diag(v) @ K @ torch.diag(u)
        return T
