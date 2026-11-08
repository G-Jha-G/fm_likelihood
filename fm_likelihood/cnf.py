"""
fm_likelihood.cnf
===================

Exact conditional log-density via the instantaneous change-of-variables
identity, integrated along the probability-flow ODE. Ported from
fm_likelihood_extremes_v2.py's ConditionalCNFLikelihood.

WHAT CHANGED FROM THE PILOT (deliberately, everything else is identical)
--------------------------------------------------------------------------
1. No internal stateful generator. The pilot's `self._gen`, seeded once in
   __init__ and advanced on every log_prob() call, is exactly finding N4/N5
   in the research programme: it made every saved array call-order
   dependent, and it is the reason lve_bpd, typ_real, rvr_bpd, cmm_matched
   turned out to be four different noisy draws of the same quantity rather
   than four identical numbers. log_prob() now takes an explicit
   `hutchinson_generator`; callers get one from fm_likelihood.seeds so two
   calls with the same (field_id, condition, replicate, purpose) always
   draw the same probes, and common-random-number pairing (E2, E3) is
   possible by construction rather than by accident of shared state.

2. Function-evaluation counting. Every log_prob() call now returns
   n_function_evals, needed for the compute-budget arithmetic in the brief
   (Section 1.3) and for reporting cost alongside every numerical
   validation result in E0.

MATH, UNCHANGED AND ALREADY VERIFIED CORRECT
----------------------------------------------
Augmented system  d/dt [x; a] = [v; -div(v)], integrated from t=1 (data) to
t=0 (base), a(1)=0. Then a(0) = log p_base(x0) - log p_data(x1), so

    log p_theta(x1 | cond) = log N(x0; 0, I) - a(0).

Research programme Section 2: "The augmented system is ... Line 493
implements exactly this ... No sign error." Preserve this exactly; do not
"clean up" the sign convention, it is correct as written.
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from fm_likelihood.velocity import VelocityField

try:
    from torchdiffeq import odeint as _odeint
    _HAS_TORCHDIFFEQ = True
except Exception:
    _HAS_TORCHDIFFEQ = False

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class LikelihoodConfig:
    method: str = "dopri5"             # "dopri5" (adaptive) | "rk4" (fixed-step)
    rtol: float = 1e-5
    atol: float = 1e-5
    n_steps_fixed: int = 100
    hutchinson_probes: int = 8
    exact_trace: bool = False          # exact divergence, only for tiny D (tests)
    use_float64: bool = True
    t0: float = 1.0                    # data time
    t1: float = 0.0                    # base time


def _rademacher_like(x: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    gen_device = generator.device if generator is not None else x.device
    r = torch.randint(0, 2, x.shape, generator=generator, device=gen_device,
                       dtype=x.dtype)
    return (r.to(x.device) * 2.0 - 1.0)


def divergence_hutchinson(vel: VelocityField, x: Tensor, t: Tensor, cond: Tensor,
                           eps_list: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    """Stochastic trace: tr(J) ~ mean_k eps_k^T J eps_k. eps_list is the
    SAME list reused across the whole trajectory (FFJORD convention), so
    the per-trajectory estimate is unbiased."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = vel(x, t, cond)
        div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if not v.requires_grad:
            return v.detach(), div
        for eps in eps_list:
            (vjp,) = torch.autograd.grad(v, x, grad_outputs=eps, retain_graph=True,
                                          create_graph=False, allow_unused=True)
            if vjp is None:
                continue
            div = div + (vjp * eps).flatten(1).sum(dim=1)
        div = div / float(len(eps_list))
    return v.detach(), div.detach()


def divergence_exact(vel: VelocityField, x: Tensor, t: Tensor, cond: Tensor
                      ) -> Tuple[Tensor, Tensor]:
    """Exact divergence, O(D) backward passes. Validation/toy use only."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = vel(x, t, cond)
        B = x.shape[0]
        D = x[0].numel()
        div = torch.zeros(B, device=x.device, dtype=x.dtype)
        if not v.requires_grad:
            return v.detach(), div
        v_flat = v.flatten(1)
        for i in range(D):
            (grad_i,) = torch.autograd.grad(v_flat[:, i].sum(), x,
                                             retain_graph=True, create_graph=False,
                                             allow_unused=True)
            if grad_i is None:
                continue
            div = div + grad_i.flatten(1)[:, i]
    return v.detach(), div.detach()


class ConditionalCNFLikelihood:
    """Exact conditional log-density log p_theta(x1 | cond) for a velocity
    field whose generative ODE is the probability-flow ODE."""

    def __init__(self, velocity: VelocityField, cfg: LikelihoodConfig):
        self.velocity = velocity
        self.cfg = cfg

    @staticmethod
    def _base_logprob(x0: Tensor) -> Tensor:
        D = x0[0].numel()
        sq = (x0 ** 2).flatten(1).sum(dim=1)
        return -0.5 * sq - 0.5 * D * LOG_2PI

    def _make_dynamics(self, cond: Tensor, eps_list, nfe_counter: list):
        cfg = self.cfg

        def dynamics(t: Tensor, state):
            x, _a = state
            if cfg.exact_trace:
                v, div = divergence_exact(self.velocity, x, t, cond)
                # 1 forward pass plus D backward passes (one per dimension)
                nfe_counter[0] += 1 + x[0].numel()
            else:
                v, div = divergence_hutchinson(self.velocity, x, t, cond, eps_list)
                # 1 forward pass plus one VJP per Hutchinson probe, matching
                # the brief's Section 1.3 compute arithmetic exactly
                nfe_counter[0] += 1 + len(eps_list)
            return v, -div

        return dynamics

    @staticmethod
    def _rk4(dynamics, state0, t0, t1, n_steps):
        x, a = state0
        ts = torch.linspace(t0, t1, n_steps + 1, device=x.device, dtype=x.dtype)
        for i in range(n_steps):
            t = ts[i]
            h = ts[i + 1] - ts[i]

            def f(tt, st):
                return dynamics(tt, st)

            k1x, k1a = f(t, (x, a))
            k2x, k2a = f(t + 0.5 * h, (x + 0.5 * h * k1x, a + 0.5 * h * k1a))
            k3x, k3a = f(t + 0.5 * h, (x + 0.5 * h * k2x, a + 0.5 * h * k2a))
            k4x, k4a = f(t + h, (x + h * k3x, a + h * k3a))
            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            a = a + (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        return x, a

    @torch.no_grad()
    def log_prob(self, x1: Tensor, cond: Tensor,
                  hutchinson_generator: Optional[torch.Generator] = None) -> dict:
        """
        hutchinson_generator: pass fm_likelihood.seeds.torch_generator(
            field_id, condition, replicate, purpose="hutchinson") for
        anything that will be reported. Omitting it falls back to the
        global torch RNG state, which is fine for a quick interactive
        check and never fine for a saved result.
        """
        if hutchinson_generator is None:
            warnings.warn(
                "log_prob() called with no hutchinson_generator; using the global "
                "RNG state. Fine for a scratch check, not for anything you intend "
                "to save, plot, or report. Pass fm_likelihood.seeds.torch_generator(...).",
                stacklevel=2,
            )

        cfg = self.cfg
        dtype = torch.float64 if cfg.use_float64 else x1.dtype
        x1 = x1.to(dtype)
        cond = cond.to(dtype)

        eps_list = [
            _rademacher_like(x1, hutchinson_generator).to(x1.device)
            for _ in range(cfg.hutchinson_probes)
        ]
        a0 = torch.zeros(x1.shape[0], device=x1.device, dtype=dtype)
        nfe_counter = [0]
        dynamics = self._make_dynamics(cond, eps_list, nfe_counter)

        if cfg.method == "dopri5" and _HAS_TORCHDIFFEQ:
            t_span = torch.tensor([cfg.t0, cfg.t1], device=x1.device, dtype=dtype)
            sol = _odeint(lambda t, st: dynamics(t, st), (x1, a0), t_span,
                           rtol=cfg.rtol, atol=cfg.atol, method="dopri5")
            xT, aT = sol[0][-1], sol[1][-1]
            solver_used = "dopri5"
        else:
            if cfg.method == "dopri5" and not _HAS_TORCHDIFFEQ:
                print("[warn] torchdiffeq unavailable; falling back to fixed-step RK4.",
                      file=sys.stderr)
            xT, aT = self._rk4(dynamics, (x1, a0), cfg.t0, cfg.t1, cfg.n_steps_fixed)
            solver_used = "rk4"

        base_lp = self._base_logprob(xT)
        log_prob = base_lp - aT
        return {
            "log_prob": log_prob,
            "base_log_prob": base_lp,
            "delta_log_det": -aT,
            "x0": xT,
            "n_function_evals": nfe_counter[0],
            "solver": solver_used,
            "n_steps": cfg.n_steps_fixed if solver_used == "rk4" else None,
            "probes": cfg.hutchinson_probes if not cfg.exact_trace else None,
        }
