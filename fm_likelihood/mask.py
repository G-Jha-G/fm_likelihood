"""
fm_likelihood.mask
=====================

Padding and interior-mask bookkeeping. Answers Section 13 Q1: MSWX India
domain is 350x350 unpadded, reflect-padded to 352x352 for the UNet (div=16
padding needs ph=pw=2, split 1 pixel each side). D_interior = 122500,
D_total = 123904.

WHY THIS MODULE EXISTS (finding N2)
--------------------------------------
Verified directly (not assumed): torch's F.pad(mode='reflect') at pad
width 1 does not duplicate the edge row, it copies the row ONE STEP IN
(row index 1, not row 0) into the new border. This means the padded
352x352 field contains two exact duplicate rows and two exact duplicate
columns, real interior data appearing twice at the same field, not
independent information. The joint density is being evaluated on a
degenerate manifold in the padded region: the true density there is a
delta function (values are deterministic copies of already-observed
values), and whatever the network assigns to it is not meaningful.

This does not invalidate the pilot's relative comparisons directly (the
pad fraction is the same for every field and every condition, so paired
differences in TOTAL NATS are unaffected), but it does mean:
  - Absolute bpd_total (normalised by D_total) is a biased estimate of
    the true per-pixel information content, since D_total counts cells
    that carry zero new information.
  - extremity_index and any field-summary statistic must be computed on
    the interior only, not the padded field, or they inherit the same
    duplication.
  - Reporting must distinguish bpd_total (padded, comparable within this
    study) from bits_per_interior_dim (unpadded, the honest per-pixel
    number), per the brief's Section 6 padding resolution rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PaddingInfo:
    h: int              # unpadded height
    w: int               # unpadded width
    div: int              # padding multiple required by the UNet (16)
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @property
    def h_padded(self) -> int:
        return self.h + self.pad_top + self.pad_bottom

    @property
    def w_padded(self) -> int:
        return self.w + self.pad_left + self.pad_right

    @property
    def d_interior(self) -> int:
        return self.h * self.w

    @property
    def d_total(self) -> int:
        return self.h_padded * self.w_padded

    @property
    def pad_fraction(self) -> float:
        return 1.0 - (self.d_interior / self.d_total)


def compute_padding(h: int, w: int, div: int = 16) -> PaddingInfo:
    """Reproduces the pilot's exact padding arithmetic (fm_likelihood_
    extremes_v2.py load_test_fields, lines ~833-837), so mask.py and the
    original data-loading path always agree on where the interior sits."""
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    pad_top, pad_left = ph // 2, pw // 2
    pad_bottom, pad_right = ph - pad_top, pw - pad_left
    return PaddingInfo(h=h, w=w, div=div,
                        pad_top=pad_top, pad_bottom=pad_bottom,
                        pad_left=pad_left, pad_right=pad_right)


def interior_slices(info: PaddingInfo):
    """Row and column slices into the PADDED array that recover exactly
    the original unpadded interior. Usage:
        row_sl, col_sl = interior_slices(info)
        interior = padded_field[..., row_sl, col_sl]
    """
    row_sl = slice(info.pad_top, info.pad_top + info.h)
    col_sl = slice(info.pad_left, info.pad_left + info.w)
    return row_sl, col_sl


def interior_mask(info: PaddingInfo, device=None, dtype=torch.bool) -> Tensor:
    """Boolean mask, shape (h_padded, w_padded), True in the interior,
    False in the padded border. Assert no pad cells with
    `assert not mask[~mask].any()`-style checks in tests, not by eye."""
    m = torch.zeros(info.h_padded, info.w_padded, dtype=dtype, device=device)
    row_sl, col_sl = interior_slices(info)
    m[row_sl, col_sl] = True
    return m


def to_interior(field: Tensor, info: PaddingInfo) -> Tensor:
    """Crop the last two dimensions of a padded field down to the
    interior. field shape (..., H_padded, W_padded) -> (..., H, W)."""
    row_sl, col_sl = interior_slices(info)
    return field[..., row_sl, col_sl]
