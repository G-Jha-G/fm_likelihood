"""
tests.test_defaults
===================

Guards the preprocessing defaults that the published numbers depend on.

The paper states that dequantization uses MSWX's quantization step of
0.0625 mm (1/16 mm), verified from the raw data. Every production script
builds its transform with PrecipTransform(PrecipConfig()), so the paper and
the code agree only if the default below is 0.0625. If this test fails, the
absolute bits-per-dimension values from a rerun will not match the paper
(paired differences are much less affected, but do not rely on that).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm_likelihood.transform import PrecipConfig


def test_dequant_bin_matches_paper():
    assert PrecipConfig().dequant_bin_mm == 0.0625


def test_wet_threshold_matches_paper():
    assert PrecipConfig().wet_threshold_mm == 1.0
