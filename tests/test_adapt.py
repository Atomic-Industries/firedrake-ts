"""Error-controlled stepping via TSADAPTBASIC and the embedded pair."""

import numpy as np
from firedrake import *

import firedrake_ts

EXACT = np.exp(-1.0)

ARK_SSP = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


def _decay(extra, dt=1e-2):
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
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


def test_shim_resolves_ts_adapt_symbols():
    from firedrake_ts._petsc_shim import ts_get_adapt

    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 0.1), G=-inner(u, v) * dx
    )
    solver = firedrake_ts.DAESolver(problem, options_prefix="")
    assert ts_get_adapt(solver.ts) is not None


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


def test_design_order_two_without_the_limiter():
    """Order 2, unlimited: halving dt should cut error by roughly 2**2 = 4."""
    _, coarse = _decay({"ts_adapt_type": "none"}, dt=4e-2)
    _, fine = _decay({"ts_adapt_type": "none"}, dt=2e-2)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"
