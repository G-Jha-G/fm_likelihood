"""
fm_likelihood.bpd
===================

Physical-space bits per dimension. Ported from fm_likelihood_extremes_v2.py's
bits_per_dim, math unchanged:

    p (mm) --dequant--> p~ --to_model--> u --CNF--> log p_model(u)
    log p_phys(p~) = log p_model(u) + sum_pixels log|du/dp|
    bpd = -(log p_phys(p~) + D log(delta)) / (D log 2)

Two additions per the brief's Section 7.2 API contract and the padding
resolution (Section 6, E0):
  - the full accounting dict, not just bits_per_dim
  - bits_per_interior_dim, reported alongside bits_per_dim (normalised by
    D_total) once D_interior is known. Until mask.py lands (blocked on
    Section 13 Q1), pass d_interior=None and it is simply omitted rather
    than silently wrong.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
from torch import Tensor

from fm_likelihood.cnf import ConditionalCNFLikelihood
from fm_likelihood.transform import PrecipTransform

LOG2 = math.log(2.0)


def bits_per_dim(engine: ConditionalCNFLikelihood, transform: PrecipTransform,
                  p_hr_mm: Tensor, cond_u: Tensor,
                  dequant_generator: Optional[torch.Generator] = None,
                  hutchinson_generator: Optional[torch.Generator] = None,
                  d_interior: Optional[int] = None) -> dict:
    """
    dequant_generator, hutchinson_generator: pass fm_likelihood.seeds
    generators keyed by (field_id, condition, replicate, purpose). Two
    calls that are meant to be a common-random-numbers pair (E2, E3) must
    receive generators built from the SAME condition string; two calls
    meant to be independent replicates must vary it.

    d_interior: total interior (unpadded) pixel count, from mask.py. If
    None, bits_per_interior_dim is not computed (see N2: the padded region
    is a degenerate manifold and D_total overstates the true dimension).
    """
    cfg = transform.cfg
    D_total = p_hr_mm[0].numel()

    p_deq = transform.dequantize(p_hr_mm, dequant_generator)
    u = transform.to_model(p_deq)
    out = engine.log_prob(u, cond_u, hutchinson_generator=hutchinson_generator)

    lp_model = out["log_prob"].to(torch.float64)
    log_jac = transform.log_dudp_sum(p_deq).to(torch.float64)
    lp_phys = lp_model + log_jac

    bpd_total = -(lp_phys + D_total * math.log(cfg.dequant_bin_mm)) / (D_total * LOG2)

    result = {
        "log_prob_model": lp_model,          # nats, u space
        "log_jac": log_jac,                  # nats, u -> p
        "log_prob_phys": lp_phys,            # nats, physical space
        "bpd_total": bpd_total,              # normalised by D_total (padded)
        "bits_per_interior_dim": None,
        "d_total": D_total,
        "d_interior": d_interior,
        "n_function_evals": out["n_function_evals"],
        "solver": out["solver"],
        "n_steps": out["n_steps"],
        "probes": out["probes"],
    }

    if d_interior is not None:
        result["bits_per_interior_dim"] = -(
            lp_phys + d_interior * math.log(cfg.dequant_bin_mm)
        ) / (d_interior * LOG2)
    else:
        warnings.warn(
            "d_interior not supplied; bits_per_interior_dim is None. This is "
            "expected until mask.py lands (Section 13 Q1). Do not substitute "
            "bpd_total for it in any paired-difference reporting once mask.py "
            "exists, N2 in the research programme is exactly this mistake.",
            stacklevel=2,
        )

    return result
