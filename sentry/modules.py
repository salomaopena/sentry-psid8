"""SENTRY core modules (paper Sec. III-C).

Implements Eqs. (1) and (2) exactly:
  (1) ConvGRU per pyramid level (gates z, r; candidate h~; update h_t)
  (2) Motion-gated fusion:
      M_t = sigma(W_m * |P_t - P_{t-1}|)
      F_t = h_t (1 + alpha * M_t) + P_t     <- additive skip guarantees the
                                               frame-level floor
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """2D ConvGRU cell (Ballas et al., 2016). Preserves spatial structure."""

    def __init__(self, in_ch: int, hidden_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.hidden_ch = hidden_ch
        # z and r gates computed jointly for efficiency
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, k, padding=p)
        self.cand = nn.Conv2d(in_ch + hidden_ch, hidden_ch, k, padding=p)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3))
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde                     # Eq. (1)


class MotionGate(nn.Module):
    """M_t = sigma(W_m * |P_t - P_{t-1}|); learned scalar alpha per level."""

    def __init__(self, in_ch: int, alpha_init: float = 0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, feat: torch.Tensor, prev_feat: torch.Tensor | None) -> torch.Tensor:
        if prev_feat is None:
            return torch.zeros_like(feat[:, :1])
        return torch.sigmoid(self.conv((feat - prev_feat).abs()))


class TFMLevel(nn.Module):
    """Single-level TFM: ConvGRU + motion gate + Eq. (2) fusion.

    Projects the hidden state back to the pyramid width when C_h != C_p,
    so the detection head stays untouched.
    """

    def __init__(self, pyr_ch: int, hidden_ch: int, k: int = 3, alpha_init: float = 0.5):
        super().__init__()
        self.gru = ConvGRUCell(pyr_ch, hidden_ch, k)
        self.gate = MotionGate(pyr_ch, alpha_init)
        self.proj = (nn.Identity() if hidden_ch == pyr_ch
                     else nn.Conv2d(hidden_ch, pyr_ch, 1))

    def forward(self, feat, state):
        h_prev, feat_prev = state
        h = self.gru(feat, h_prev)
        m = self.gate(feat, feat_prev)
        alpha = self.gate.alpha
        fused = self.proj(h) * (1 + alpha * m) + feat        # Eq. (2)
        return fused, (h.detach(), feat.detach()), {"z_mean": None, "m_mean": m.mean()}


class TemporalFeatureMemory(nn.Module):
    """Full TFM: one TFMLevel per pyramid level {P3, P4, P5}.

    Streaming usage:
        tfm.reset()                      # at the start of each clip/stream
        fused, ev = tfm(list_of_feats)   # every frame
    `ev` carries the evidence terms (mean motion-gate activation) that feed
    the structured alert record (Sec. III-E).
    """

    def __init__(self, pyr_chs=(256, 512, 1024), hidden_ch=128, k=3, alpha_init=0.5):
        super().__init__()
        self.levels = nn.ModuleList(
            TFMLevel(c, hidden_ch, k, alpha_init) for c in pyr_chs
        )
        self._states = None

    def reset(self):
        self._states = [(None, None) for _ in self.levels]

    def forward(self, feats: list[torch.Tensor]):
        if self._states is None or len(self._states) != len(feats):
            self.reset()
        fused, evid = [], []
        for i, (lvl, f) in enumerate(zip(self.levels, feats)):
            out, new_state, ev = lvl(f, self._states[i])
            self._states[i] = new_state
            fused.append(out)
            evid.append(ev)
        m_mean = torch.stack([e["m_mean"] for e in evid]).mean()
        return fused, {"motion_gate_mean": m_mean}

    def extra_flops_per_frame(self, spatial_sizes, pyr_chs=(256, 512, 1024),
                              hidden_ch=128, k=3):
        """Analytical estimate from Sec. III-C: O(k^2 (C_p + C_h) C_h H W) per
        level. `spatial_sizes` = [(H3,W3),(H4,W4),(H5,W5)]. Returns extra GFLOPs."""
        total = 0
        for (H, W), cp in zip(spatial_sizes, pyr_chs):
            per_gate = k * k * (cp + hidden_ch) * hidden_ch * H * W
            total += per_gate * 3            # z, r, candidate
            total += 9 * cp * H * W          # 3x3 conv -> 1-channel motion gate
        return 2 * total / 1e9               # MACs -> FLOPs
