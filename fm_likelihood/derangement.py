"""
fm_likelihood.derangement
============================

Fixes finding N9: the pilot's _derangement() repairs fixed points by
swapping each fixed index with its neighbour. That works in the common
case but is not provably correct: if indices i and i+1 are BOTH fixed
points after the initial random permutation, swapping i with i+1 fixes
both, but if i, i+1, i+2 are all fixed and the repair processes them in
order, swapping i with i+1 can leave i+1 newly fixed (since perm[i+1] was
i+1's own repaired value, not necessarily different from i+1), or leave
i+2 fixed if it wasn't reached by a swap. The pilot's own comment even
flags this as a "low probability" risk rather than a proven guarantee.

This version uses reject-and-resample instead: draw a random permutation,
check for zero fixed points, and if any exist, draw again. This is simple
and provably correct, the function cannot return a value with a fixed
point, by construction, not by repair. It is also cheap: the probability
a random permutation of n elements has zero fixed points converges to
1/e ~ 0.368 as n grows (already close at n=20), so the expected number of
draws needed is ~e ~ 2.718, a handful of resamples at n=147, not a
performance concern.
"""
from __future__ import annotations

from typing import Optional

import torch


def derangement(n: int, generator: Optional[torch.Generator] = None,
                 max_tries: int = 10_000) -> torch.Tensor:
    """A permutation of 0..n-1 with provably zero fixed points.

    generator: pass fm_likelihood.seeds.torch_generator(field_id=-1,
    condition="derangement", replicate=0, purpose="mismatch") or similar,
    for reproducibility. field_id=-1 (or any fixed sentinel) is
    appropriate here since a derangement is drawn once per experiment
    run, not once per field.

    Raises RuntimeError if no derangement is found in max_tries draws;
    at n=147 this should never happen (expected ~2.7 tries), a failure
    here would indicate n is very small (n<2) or the generator is
    somehow degenerate, not bad luck.
    """
    if n < 2:
        raise ValueError(f"no derangement exists for n={n} (need n >= 2)")
    identity = torch.arange(n)
    for _ in range(max_tries):
        perm = torch.randperm(n, generator=generator)
        if not bool((perm == identity).any()):
            return perm
    raise RuntimeError(
        f"failed to find a derangement of size {n} in {max_tries} tries; "
        "this should not happen for n >= 2 with a working generator"
    )
