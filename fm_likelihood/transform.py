"""
fm_likelihood.transform
=========================

Precipitation transform: log1p, affine to [-1, 1], exact log-Jacobian,
uniform dequantisation. Ported unchanged from fm_likelihood_extremes_v2.py's
PrecipTransform (research programme Section 2, "What is correct": "log|du/dp|
... is correct ... self-test 4 verifies it by finite differences. Correct.").
Nothing in this module's math changes in the refactor, only where the
randomness for dequantisation comes from (an explicit generator argument
here, never a shared stateful one, see fm_likelihood.seeds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PrecipConfig:
    """Must match the trained model's preprocessing exactly."""
    log1p_max: float = 6.7859          # log1p of the training-set maximum
    wet_threshold_mm: float = 1.0      # wet-day threshold (mm/day)
    dequant_bin_mm: float = 0.0625     # EMPIRICALLY VERIFIED 2026-08-16 (Q7), not
                                        # assumed. MSWX_Pcp_Daily_Ind_HR.nc's real
                                        # precision is a uniform 1/16 mm, confirmed
                                        # flat across every magnitude bin from 0.25
                                        # to 512 mm/day (ruled out float16-style
                                        # magnitude-dependent scaling directly, see
                                        # check_mswx_quantization_by_magnitude.py).
                                        # Was 0.1 (assumed, wrong) through all of
                                        # E0a-c and the original 1461-day E1 run.
                                        # Correcting this is NOT a constant shift to
                                        # BPD: verify_dequant_bin_correction.py showed
                                        # the actual shift is field-dependent, driven
                                        # by each field's dry-pixel fraction (log1p's
                                        # sensitivity near zero to the dequant noise
                                        # width, compounded over ~122,500 mostly-dry
                                        # pixels), ranging 0.08 (driest test field) to
                                        # 0.44 (wettest) bits/dim, nowhere near the
                                        # naive closed-form estimate of 0.68. The full
                                        # 1461-day BPD table must be regenerated under
                                        # this corrected value; it cannot be patched
                                        # after the fact by adding a constant.
    clip_min_mm: float = 0.0


class PrecipTransform:
    r"""
    p (mm/day, >= 0)  --dequantise-->  p~  --to_model-->  u in [-1, 1]

        s = log1p(p)
        u = 2 s / L - 1,           L = log1p_max
        p = expm1((u + 1) L / 2)   (inverse)

    log|du/dp| = log(2 / L) - s, summed over pixels gives the change of
    variables term for converting a u-space density to a p-space density:

        log p_phys(p) = log p_model(u) + sum_pixels log|du/dp|
    """

    def __init__(self, cfg: PrecipConfig):
        self.cfg = cfg
        self.L = float(cfg.log1p_max)

    def to_model(self, p_mm: Tensor) -> Tensor:
        s = torch.log1p(p_mm.clamp_min(self.cfg.clip_min_mm))
        return 2.0 * s / self.L - 1.0

    def to_physical(self, u: Tensor) -> Tensor:
        s = (u + 1.0) * self.L / 2.0
        return torch.expm1(s).clamp_min(self.cfg.clip_min_mm)

    def log_dudp_sum(self, p_mm: Tensor) -> Tensor:
        """sum_pixels log|du/dp| per sample; shape (B,)."""
        s = torch.log1p(p_mm.clamp_min(self.cfg.clip_min_mm))
        per_elem = math.log(2.0 / self.L) - s
        return per_elem.flatten(1).sum(dim=1)

    def dequantize(self, p_mm: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """Uniform dequantisation noise in [0, delta). `generator` should
        come from fm_likelihood.seeds.torch_generator(field_id, condition,
        replicate, purpose="dequant") for anything that will be reported.
        If omitted, torch's global RNG state is used (fine for scratch
        exploration, never for a result that goes in a figure or table)."""
        delta = self.cfg.dequant_bin_mm
        gen_device = generator.device if generator is not None else p_mm.device
        noise = torch.rand(p_mm.shape, generator=generator, device=gen_device,
                            dtype=p_mm.dtype).to(p_mm.device) * delta
        return (p_mm + noise).clamp_min(self.cfg.clip_min_mm)
