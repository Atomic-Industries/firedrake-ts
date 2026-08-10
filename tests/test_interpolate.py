"""Dense output: X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts

ARK_SSP = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


def _solver(dt=0.1, tmax=1.0):
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, tmax), G=-inner(u, v) * dx
    )
    return firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
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
    """Exercises TSInterpolate through the driver, not just directly.

    dt=0.03 does not divide tmax=1.0 evenly, so the last accepted step
    overshoots; ``ts_exact_final_time: interpolate`` is what is supposed to
    pull the reported solution back to tmax via this stepper's
    ``interpolate``. Check that against ``getSolveTime``, not ``getTime``:
    PETSc's own TSGetTime documents "this time may not correspond to the
    final time set with TSSetMaxTime(), use TSGetSolveTime()" -- ts->ptime
    is left at the overshot step-completion time (here ~1.02) even though
    TSSolve interpolated the *solution* back to tmax; ts->solvetime is the
    one clamped to tmax.
    """
    solver, u = _solver(dt=0.03, tmax=1.0)
    solver.solve()
    assert solver.ts.getSolveTime() == pytest.approx(1.0, abs=1e-12)
    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-3)
