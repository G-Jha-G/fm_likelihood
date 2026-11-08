"""
fm_likelihood.seeds
====================

Deterministic key derivation, replacing the pilot's shared stateful
generators (engine._gen for Hutchinson probes, dq for dequantisation
noise), which made every result call-order dependent (finding N5 /
N4 in the research programme, PROJECT_BRIEF Section 3.2).

Every stochastic draw anywhere in this package (Hutchinson probes,
dequantisation noise, surrogate construction, the conditioning-mismatch
derangement) must derive its seed from rng_key(), never from a shared
generator advancing across calls. Two calls with identical arguments
always return identical output; that is what makes results reproducible
independent of what order experiments happen to run in.

COMMON RANDOM NUMBERS FOR PAIRED COMPARISONS (brief Section 7.2, and the
E2/E3 designs): both members of a paired comparison must receive the
IDENTICAL dequant key and the IDENTICAL hutchinson key. Concretely, for
a field/condition pair (A, B) being compared, call

    rng_key(field_id, condition="A", replicate=0, purpose="dequant")
    rng_key(field_id, condition="B", replicate=0, purpose="dequant")

with the SAME field_id and SAME replicate, varying only `condition`, if
you want the two draws to be independent (e.g. two genuinely different
perturbations). To share the identical draw between A and B instead
(cancelling numerical noise from the difference, which is what E2 and E3
require), pass the SAME condition string to both calls, for example
condition="shared", and only vary the semantic meaning downstream.
"""
from __future__ import annotations

import hashlib

KNOWN_PURPOSES = ("dequant", "hutchinson", "surrogate", "mismatch")


def rng_key(field_id: int, condition: str, replicate: int, purpose: str) -> int:
    """Deterministic 31-bit key from (field_id, condition, replicate, purpose).

    purpose is typically one of KNOWN_PURPOSES, but new experiments (block
    permutation scale, conditioning shift distance, etc.) may introduce new
    purpose strings; nothing here enforces the set, it is documentation of
    current usage, not a closed enum.
    """
    h = hashlib.blake2b(
        f"{field_id}|{condition}|{replicate}|{purpose}".encode(), digest_size=8
    )
    return int.from_bytes(h.digest(), "little") % (2**31 - 1)


def torch_generator(field_id: int, condition: str, replicate: int, purpose: str,
                     device: str = "cpu"):
    """Convenience wrapper: an already-seeded torch.Generator, since every
    stochastic call site in the pilot (Rademacher probes, dequant noise)
    consumes a torch.Generator, not a bare int."""
    import torch
    key = rng_key(field_id, condition, replicate, purpose)
    gen = torch.Generator(device=device)
    gen.manual_seed(key)
    return gen


def numpy_rng(field_id: int, condition: str, replicate: int, purpose: str):
    """Same idea, for call sites that want a numpy Generator instead
    (surrogate construction, block bootstrap resampling)."""
    import numpy as np
    key = rng_key(field_id, condition, replicate, purpose)
    return np.random.default_rng(key)
