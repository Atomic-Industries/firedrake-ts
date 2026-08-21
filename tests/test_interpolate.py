"""Dense output: X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i."""

import numpy as np
import pytest
from conftest import ARK_SSP_G5 as ARK_SSP
from conftest import scalar_problem
from firedrake import *

import firedrake_ts


def _solver(dt=0.1, tmax=1.0, tableau="esdirk_gamma5"):
    u, u_t, v = scalar_problem()
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, tmax), G=-inner(u, v) * dx
    )
    return firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type=tableau,
            ts_adapt_type="none",
            ts_time_step=dt,
            ts_exact_final_time="interpolate",
        ),
        options_prefix="",
    ), u


def test_endpoint_matches_the_accepted_step_exactly():
    """X(1) must equal y_{n+1}; a mismatch means the wrong weights."""
    solver, _ = _solver()
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    end = ts.getSolution().duplicate()
    ctx.interpolate(ts, ts.getTime(), end)
    diff = end.copy()
    diff.axpy(-1.0, ts.getSolution())
    assert diff.norm() < 1e-12, f"|X(1) - y_n+1| = {diff.norm()}"


def test_interpolation_is_at_least_first_order():
    """Midpoint interpolation must beat piecewise-constant."""
    solver, _ = _solver(dt=0.2, tmax=1.0)
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    mid = ts.getSolution().duplicate()
    t_mid = ts.getTime() - 0.5 * ts.getTimeStep()
    ctx.interpolate(ts, t_mid, mid)
    exact = np.exp(-t_mid)
    assert abs(float(mid.getArray()[0]) - exact) < 1e-2


def test_exact_final_time_interpolate_lands_on_tmax():
    """Exercises TSInterpolate through the driver, not just directly."""
    solver, u = _solver(dt=0.03, tmax=1.0)
    solver.solve()
    assert solver.ts.getSolveTime() == pytest.approx(1.0, abs=1e-12)
    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-3)


def test_dense_output_stays_bounded_on_a_stiff_step():
    """d.g = 0 must actually keep X(theta) bounded through interpolate."""
    u, u_t, v = scalar_problem()
    lam = 1e6
    h = 1.0
    F = inner(u_t, v) * dx + lam * inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, h), G=None)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_adapt_type="none",
            ts_time_step=h,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    end = ts.getSolution().duplicate()
    values = []
    for theta in np.linspace(0.001, 0.999, 999):
        t = (ts.getTime() - h) + theta * h
        ctx.interpolate(ts, t, end)
        values.append(float(end.getArray()[0]))
    values = np.array(values)
    assert values.max() < 1.1, (
        f"X(theta) reached {values.max()} on a stiff step (lambda={lam}); "
        "expected it to stay near [y_n+1, y_n] = [~0, 1] rather than "
        "overshoot the way a naive d ignoring At's null direction would"
    )
    assert values.min() > -0.1, (
        f"X(theta) reached {values.min()} on a stiff step (lambda={lam}); "
        "expected it to stay near [y_n+1, y_n] = [~0, 1] rather than "
        "diverge below the endpoints"
    )


def test_interpolate_ignores_ydot_for_a_purely_explicit_tableau():
    """A tableau with no implicit part must contribute no Ydot term."""
    solver, _ = _solver(dt=0.2, tmax=1.0, tableau="ssprk2")
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    for vec in ctx._Ydot:
        vec.set(np.nan)
    mid = ts.getSolution().duplicate()
    t_mid = ts.getTime() - 0.5 * ts.getTimeStep()
    ctx.interpolate(ts, t_mid, mid)
    values = mid.getArray()
    assert np.all(np.isfinite(values)), (
        f"interpolate read _Ydot for a tableau with no implicit part; got {values}"
    )
    # And the interpolant is still right, not merely finite.
    assert abs(values[0] - np.exp(-t_mid)) < 0.05 * np.exp(-t_mid)
