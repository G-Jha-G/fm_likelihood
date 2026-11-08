"""
fm_likelihood.velocity
========================

Adapter wrapping the trained Flow-Matching network so it satisfies the
VelocityField protocol used by fm_likelihood.cnf. Ported unchanged from
fm_likelihood_extremes_v2.py's FMVelocity: same dtype casting (ODE state
travels in float64, the network runs in its native float32), same time
scaling, same channel-concatenation convention.

DDPMProbabilityFlowVelocity (T2.2 in the brief) is deliberately NOT ported
yet. It needs N10 fixed first (the discrete `t_in = round(s*(T-1))` time
staircase is inconsistent with continuous-time ODE integration) and isn't
on the critical path for Phase 0 to 3. Port it only when T2.2 is actually
scheduled, so a half-fixed version doesn't sit around getting used by
accident.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

import torch
from torch import Tensor


class VelocityField(Protocol):
    """A conditional velocity field v_theta(x_t, t, cond) -> dx/dt, in
    MODEL space (i.e. after PrecipTransform.to_model)."""
    def __call__(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor: ...


class FMVelocity:
    r"""
    Expected underlying network signature:

        net(x_in, t_scaled) -> velocity

    where x_in = concat([x_t, cond], dim=1) and t_scaled = t * time_scale.
    Velocity has the same shape as x_t (HR-state channels only).
    """

    def __init__(self, net: torch.nn.Module, time_scale: float = 1000.0,
                 concat_cond: bool = True,
                 call_fn: Optional[Callable[[torch.nn.Module, Tensor, Tensor], Tensor]] = None):
        self.net = net
        self.time_scale = float(time_scale)
        self.concat_cond = concat_cond
        self.call_fn = call_fn

    def __call__(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        orig_dtype = x_t.dtype
        net_dtype = next(self.net.parameters()).dtype

        if t.ndim == 0:
            t = t.expand(x_t.shape[0])

        t_scaled = (t * self.time_scale).to(net_dtype)
        x_in = torch.cat([x_t, cond], dim=1) if self.concat_cond else x_t
        x_in = x_in.to(net_dtype)

        if self.call_fn is not None:
            v = self.call_fn(self.net, x_in, t_scaled)
        else:
            v = self.net(x_in, t_scaled)

        return v.to(orig_dtype)
