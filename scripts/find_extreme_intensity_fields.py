#!/usr/bin/env python3
"""
find_extreme_intensity_fields.py
===================================

E0b's field selection sorts by WET FRACTION (fraction of interior pixels
above the wet threshold), which is a reasonable reading of the brief's
"spanning the wetness range" but is not the same axis as field intensity
(99.9th percentile mm/day). A field can have a low wet fraction (mostly
dry) and still contain one small pocket of very intense rain, giving it
high 99.9th-percentile intensity, exactly the kind of field this paper's
extreme-event focus cares about most, and exactly the kind of field the
current wet-fraction-based selection could have missed entirely.

This reads the pilot's already-saved likelihood_extremes.npz (no GPU, no
model, no new inference needed) and reports the field indices with the
highest lve_extremity, so those specific fields can be added to the E0b
probe-convergence check via --field-ids.

USAGE
-----
    python scripts/find_extreme_intensity_fields.py --n-fields 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, default=Path("likelihood_extremes.npz"))
    p.add_argument("--n-fields", type=int, default=3,
                   help="how many of the most extreme-intensity fields to report")
    args = p.parse_args()

    if not args.npz.exists():
        print(f"error: {args.npz} not found")
        return 1

    data = np.load(args.npz)
    extremity = np.asarray(data["lve_extremity"], dtype=np.float64)
    n = len(extremity)

    order = np.argsort(extremity)[::-1]  # descending, most extreme first
    top = order[: args.n_fields]

    print(f"{n} fields total. Top {args.n_fields} by 99.9th-percentile intensity "
          f"(lve_extremity):\n")
    print(f"{'field_id':>8} {'extremity_mm':>14}")
    for idx in top:
        print(f"{idx:>8} {extremity[idx]:>14.2f}")

    field_id_str = ",".join(str(i) for i in top.tolist())
    print(f"\nFor E0b, run with:")
    print(f"  --field-ids \"{field_id_str}\"")

    # sanity check: how much do these overlap with the wet-fraction-based
    # selection already tested (fields 2, 48, 134)? If there's no overlap,
    # that's itself informative about how different the two selection
    # criteria really are for this dataset.
    already_tested = {2, 48, 134}
    new_fields = set(top.tolist()) - already_tested
    overlap = set(top.tolist()) & already_tested
    print(f"\nOverlap with already-tested fields {sorted(already_tested)}: "
          f"{sorted(overlap) if overlap else 'none'}")
    print(f"Genuinely new fields to test: {sorted(new_fields)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
