"""Error-controlled stepping via TSADAPTBASIC and the embedded pair."""

import pytest
from conftest import ARK_SSP_G5 as ARK_SSP
from conftest import EXACT_DECAY, scalar_problem
from firedrake import *
from firedrake.exceptions import ConvergenceError

import firedrake_ts

EXACT = EXACT_DECAY


def _decay(extra, dt=1e-2):
    u, u_t, v = scalar_problem()
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    params = dict(ARK_SSP, ts_time_step=dt, ts_exact_final_time="stepover")
    params.update(extra)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=params,
        options_prefix="",
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


def test_evaluatestep_gives_the_embedded_solution():
    """Order p and order p-1 completions must differ, and both be sane."""
    solver, _ = _decay({"ts_adapt_type": "none"}, dt=0.1)
    ctx = solver.ts.getPythonContext()
    full = solver.ts.getSolution().duplicate()
    embedded = solver.ts.getSolution().duplicate()
    ctx.evaluatestep(solver.ts, ctx._tab.order, full)
    ctx.evaluatestep(solver.ts, ctx._tab.order - 1, embedded)
    assert full.norm() > 0.0
    assert embedded.norm() > 0.0
    diff = full.copy()
    diff.axpy(-1.0, embedded)
    assert 0.0 < diff.norm() < full.norm()


def test_adapt_basic_selects_steps():
    # matchstep, not the module default stepover: TSAdaptChoose's own
    # generic wrapper (tsadapt.c, ahead of the type-specific ...Choose_Basic
    # this calls) clips next_h to land exactly on max_time whenever adaptivity
    # is active, so the comparison against EXACT below is against t=1 exactly
    # rather than wherever an unclipped last step happened to overshoot to.
    solver, value = _decay(
        {
            "ts_adapt_type": "basic",
            "ts_adapt_dt_min": 1e-6,
            "ts_adapt_dt_max": 0.2,
            "ts_rtol": 1e-6,
            "ts_atol": 1e-8,
            "ts_exact_final_time": "matchstep",
        }
    )
    assert solver.ts.getStepNumber() > 0
    assert abs(value - EXACT) < 1e-4
    # The controller must actually have changed dt away from the initial guess.
    assert abs(solver.ts.getTimeStep() - 1e-2) > 1e-9


def test_exhausted_rejections_raises_convergence_error():
    """ts_max_reject=0 forbids even a single retry, so a step deliberately
    sized to fail its only attempt (an oversized initial guess against a
    tight tolerance) must surface as a clean ConvergenceError rather than
    silently continuing or hanging.
    """
    from firedrake.petsc import PETSc

    try:
        with pytest.raises(ConvergenceError, match="DIVERGED_STEP_REJECTED"):
            _decay(
                {
                    "ts_adapt_type": "basic",
                    "ts_adapt_dt_min": 1e-8,
                    "ts_adapt_dt_max": 1.0,
                    "ts_rtol": 1e-10,
                    "ts_atol": 1e-12,
                    "ts_max_reject": 0,
                    "ts_error_if_step_fails": False,
                },
                dt=0.9,
            )
    finally:
        opts = PETSc.Options()
        if opts.hasName("ts_max_step_rejections"):
            opts.delValue("ts_max_step_rejections")


def test_design_order_two_without_the_limiter():
    """Order 2, unlimited: halving dt should cut error by roughly 2**2 = 4."""
    _, coarse = _decay({"ts_adapt_type": "none"}, dt=4e-2)
    _, fine = _decay({"ts_adapt_type": "none"}, dt=2e-2)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"


def test_adapt_choose_is_told_whether_the_last_attempt_failed():
    """``accept`` is an IN/OUT argument to TSAdaptChoose, not pure output.

    ``TSAdaptChoose_Basic`` reads it on the rejection branch::

        if (enorm > 1) {
          if (!*accept) safety *= adapt->reject_safety;
    """
    from firedrake_ts import ark_ssp

    seen = []
    ratios = []
    real_choose = ark_ssp.ts_adapt_choose

    def spy(adapt, ts, h, last_accepted=True):
        seen.append(last_accepted)
        result = real_choose(adapt, ts, h, last_accepted)
        _, accept = result
        if not accept:  # this attempt is being rejected
            h_as_failed, _ = real_choose(adapt, ts, h, False)
            h_as_accepted, _ = real_choose(adapt, ts, h, True)
            ratios.append(h_as_failed / h_as_accepted)
        return result

    ark_ssp.ts_adapt_choose = spy
    try:
        # A big first step against a tight-but-reachable tolerance forces a
        # rejection immediately; ts_max_steps bounds the run so a tolerance
        # that cannot be met does not turn this into an endless shrink.
        _decay(
            {
                "ts_adapt_type": "basic",
                "ts_rtol": 1e-9,
                "ts_atol": 1e-11,
                "ts_max_steps": 4,
                # See the docstring: without this the default clip floor
                # absorbs the factor being measured.
                "ts_adapt_clip": "1e-8,10",
            },
            dt=0.5,
        )
    finally:
        ark_ssp.ts_adapt_choose = real_choose

    assert seen, "TSAdaptChoose was never called"
    assert seen[0] is True, (
        "the first attempt of the first step must report the previous attempt "
        f"as accepted, got {seen[0]}"
    )
    assert any(x is False for x in seen), (
        "no retry ever reported a failed previous attempt, so the rejection "
        "path was not exercised; tighten the tolerance"
    )
    assert ratios, "no step was rejected, so the seeding could not be measured"
    for ratio in ratios:
        assert ratio == pytest.approx(0.5, rel=1e-9), (
            f"seeding accept=False must shrink next_h by reject_safety=0.5; "
            f"observed ratios {ratios} -- if this is 1.0 the argument is "
            "being ignored"
        )
