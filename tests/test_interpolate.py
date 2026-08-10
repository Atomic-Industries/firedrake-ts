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


def test_dense_output_stays_bounded_on_a_stiff_step():
    """d.g = 0 must actually keep X(theta) bounded through interpolate.

    ``test_dense_output_coefficients_are_stiff_safe`` in test_tableaux.py
    checks d . g = 0 algebraically, on the coefficients alone; nothing
    drives that property through the real ``interpolate`` call on an
    actual stiff step. This closes that gap: F = u_t + lambda*u with
    lambda = 1e6 puts z = lambda*h at -1e6, deep in the stiff regime, with
    an L-stable, stiffly accurate tableau (esdirk_gamma5) so y_n+1 itself
    collapses to ~0. If d ignored the null direction g of the singular At
    (the naive d = (1, 0, 0, 0) the module docstring in ark_ssp.py warns
    against), X(theta) would be free to diverge for theta strictly between
    0 and 1 even though the endpoints are fine. It does not: measured over
    a 999-point theta grid, X(theta) stays within [-0.016, 0.998] for
    y_n=1, y_n+1~-2e-11 -- comfortably inside the generous envelope
    asserted below, not diverging.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
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
