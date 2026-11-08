"""
tests.test_cnf_closed_form
============================

Ports run_selftest's tests 1-3 from fm_likelihood_extremes_v2.py into a
real test suite, unchanged targets and tolerances, validating the CNF
engine against closed-form Gaussians independent of any real trained
network. Adds two tests for what actually changed in the refactor: NFE
accounting, and the statelessness that fixes N4/N5.

This must pass before the CNF core (fm_likelihood.cnf) is trusted for
anything. Per PROJECT_BRIEF Section 12, Phase 0 go/no-go: "Go when the
closed form CNF test and the finite difference Jacobian test pass."
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fm_likelihood.cnf import ConditionalCNFLikelihood, LikelihoodConfig
from fm_likelihood.seeds import torch_generator

LOG_2PI = math.log(2.0 * math.pi)


class _LinearVelocity:
    """v(x,t) = lam * x. Pushforward of N(0,I) at t=0 to t=1 is
    N(0, e^{2 lam} I), an analytic target."""
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, x, t, cond):
        return self.lam * x


class _ShiftVelocity:
    """v(x,t) = c, divergence 0. Pushforward N(0,I) -> N(c, I)."""
    def __init__(self, c: float):
        self.c = c

    def __call__(self, x, t, cond):
        return torch.full_like(x, self.c)


def _gaussian_logprob(x, mean: float, var: float):
    D = x[0].numel()
    sq = ((x - mean) ** 2).flatten(1).sum(dim=1)
    return -0.5 * sq / var - 0.5 * D * math.log(var) - 0.5 * D * LOG_2PI


def test_linear_velocity_matches_analytic_gaussian_exact_trace():
    torch.manual_seed(0)
    D, B = 6, 64
    dummy_cond = torch.zeros(B, 1)
    for lam in (-0.6, 0.4):
        x1 = torch.randn(B, D, dtype=torch.float64) * math.exp(lam)
        analytic = _gaussian_logprob(x1, 0.0, math.exp(2 * lam))
        eng = ConditionalCNFLikelihood(
            _LinearVelocity(lam),
            LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                              rtol=1e-7, atol=1e-7))
        lp = eng.log_prob(x1, dummy_cond)["log_prob"]
        err = (lp - analytic).abs().max().item()
        assert err < 1e-3 * max(1.0, analytic.abs().mean().item()), \
            f"lam={lam}: max abs error {err:.3e} exceeds tolerance"


def test_linear_velocity_hutchinson_matches_exact_trace_in_mean():
    torch.manual_seed(0)
    D, B = 6, 64
    dummy_cond = torch.zeros(B, 1)
    for lam in (-0.6, 0.4):
        x1 = torch.randn(B, D, dtype=torch.float64) * math.exp(lam)
        analytic = _gaussian_logprob(x1, 0.0, math.exp(2 * lam))
        gen = torch_generator(field_id=0, condition="selftest", replicate=0,
                               purpose="hutchinson")
        eng_hutch = ConditionalCNFLikelihood(
            _LinearVelocity(lam),
            LikelihoodConfig(method="dopri5", exact_trace=False, hutchinson_probes=8,
                              use_float64=True, rtol=1e-6, atol=1e-6))
        lp_hutch = eng_hutch.log_prob(x1, dummy_cond, hutchinson_generator=gen)["log_prob"]
        err_mean = (lp_hutch.mean() - analytic.mean()).abs().item()
        assert err_mean < 0.5, f"lam={lam}: mean Hutchinson error {err_mean:.3e} too large"


def test_shift_velocity_matches_analytic_gaussian():
    torch.manual_seed(0)
    D, B = 6, 64
    dummy_cond = torch.zeros(B, 1)
    c = 0.7
    x1 = torch.randn(B, D, dtype=torch.float64) + c
    analytic = _gaussian_logprob(x1, c, 1.0)
    eng = ConditionalCNFLikelihood(
        _ShiftVelocity(c),
        LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                          rtol=1e-7, atol=1e-7))
    lp = eng.log_prob(x1, dummy_cond)["log_prob"]
    err = (lp - analytic).abs().max().item()
    assert err < 1e-4 * max(1.0, analytic.abs().mean().item())


def test_rk4_agrees_with_dopri5_on_linear_velocity():
    torch.manual_seed(0)
    D, B = 6, 64
    dummy_cond = torch.zeros(B, 1)
    lam = 0.4
    x1 = torch.randn(B, D, dtype=torch.float64) * math.exp(lam)
    eng_rk4 = ConditionalCNFLikelihood(
        _LinearVelocity(lam),
        LikelihoodConfig(method="rk4", exact_trace=True, n_steps_fixed=200,
                          use_float64=True))
    eng_dp = ConditionalCNFLikelihood(
        _LinearVelocity(lam),
        LikelihoodConfig(method="dopri5", exact_trace=True, use_float64=True,
                          rtol=1e-7, atol=1e-7))
    d = (eng_rk4.log_prob(x1, dummy_cond)["log_prob"]
         - eng_dp.log_prob(x1, dummy_cond)["log_prob"]).abs().max().item()
    assert d < 1e-3 * max(1.0, x1.abs().mean().item()) + 1e-3


def test_nfe_matches_brief_compute_arithmetic():
    """Section 1.3: 'n_steps x RK4 stages x [1 forward + K VJPs]'. With
    exact_trace, D backward passes stand in for K VJPs."""
    torch.manual_seed(0)
    D, B = 6, 8
    dummy_cond = torch.zeros(B, 1)
    x1 = torch.randn(B, D, dtype=torch.float64)
    eng = ConditionalCNFLikelihood(
        _LinearVelocity(0.1),
        LikelihoodConfig(method="rk4", exact_trace=True, n_steps_fixed=25,
                          use_float64=True))
    out = eng.log_prob(x1, dummy_cond)
    expected = 25 * 4 * (1 + D)  # n_steps * rk4_stages * (1 forward + D backward)
    assert out["n_function_evals"] == expected
    assert out["solver"] == "rk4"


def test_two_calls_same_generator_key_give_identical_result():
    """The point of the refactor: statelessness. Two log_prob calls seeded
    from the same (field_id, condition, replicate, purpose) key must return
    bit-identical results, unlike the pilot's shared generator, which
    advanced between calls and is exactly why lve_bpd, typ_real, rvr_bpd,
    cmm_matched turned out to be four different noisy draws (N4/N5)."""
    torch.manual_seed(0)
    D, B = 6, 8
    dummy_cond = torch.zeros(B, 1)
    x1 = torch.randn(B, D, dtype=torch.float64)
    eng = ConditionalCNFLikelihood(
        _LinearVelocity(0.2),
        LikelihoodConfig(method="rk4", exact_trace=False, hutchinson_probes=4,
                          n_steps_fixed=25, use_float64=True))
    gen1 = torch_generator(field_id=3, condition="matched", replicate=0, purpose="hutchinson")
    gen2 = torch_generator(field_id=3, condition="matched", replicate=0, purpose="hutchinson")
    lp1 = eng.log_prob(x1, dummy_cond, hutchinson_generator=gen1)["log_prob"]
    lp2 = eng.log_prob(x1, dummy_cond, hutchinson_generator=gen2)["log_prob"]
    assert torch.equal(lp1, lp2)


class _CoupledVelocity:
    """v(x,t) = x + coupling * flip(x). Off-diagonal Jacobian (couples
    dimension i with dimension D-1-i). Needed specifically for the test
    below: a purely diagonal Jacobian (e.g. _LinearVelocity) gives a
    Hutchinson estimator with ZERO variance regardless of which Rademacher
    probes are drawn, since eps_i^2 = 1 identically, so eps^T diag(d) eps
    = sum(d) exactly for any eps. Only off-diagonal structure makes the
    estimate sensitive to which probes were drawn, which is what this
    test needs to exercise."""
    def __init__(self, coupling: float = 0.15):
        self.coupling = coupling

    def __call__(self, x, t, cond):
        return x + self.coupling * x.flip(dims=[-1])


def test_different_condition_string_gives_different_result():
    """Varying only `condition` in the key must change the Hutchinson
    draw, or the common-random-numbers pairing in E2/E3 would silently
    degenerate into always-identical, hiding real effects rather than
    cancelling noise. Uses _CoupledVelocity, not _LinearVelocity, see
    above for why a diagonal Jacobian can't distinguish this."""
    torch.manual_seed(0)
    D, B = 6, 8
    dummy_cond = torch.zeros(B, 1)
    x1 = torch.randn(B, D, dtype=torch.float64)
    eng = ConditionalCNFLikelihood(
        _CoupledVelocity(0.15),
        LikelihoodConfig(method="rk4", exact_trace=False, hutchinson_probes=4,
                          n_steps_fixed=25, use_float64=True))
    gen1 = torch_generator(field_id=3, condition="A", replicate=0, purpose="hutchinson")
    gen2 = torch_generator(field_id=3, condition="B", replicate=0, purpose="hutchinson")
    lp1 = eng.log_prob(x1, dummy_cond, hutchinson_generator=gen1)["log_prob"]
    lp2 = eng.log_prob(x1, dummy_cond, hutchinson_generator=gen2)["log_prob"]
    assert not torch.equal(lp1, lp2)


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod, inspect.isfunction)
             if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
