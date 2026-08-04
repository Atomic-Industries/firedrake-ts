"""``DAESolver(stage_callback=...)`` must fire per stage and actually take effect"""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts
from firedrake_ts._petsc_shim import stage_hook_propagates


def _problem(mesh_n=4):
    mesh = UnitIntervalMesh(mesh_n)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx
    G = -inner(u, v) * dx
    return u, firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.1), G=G)


def _solve(stage_callback=None, tableau="2c", nsteps=10):
    u, problem = _problem()
    solver = firedrake_ts.DAESolver(
        problem,
        stage_callback=stage_callback,
        solver_parameters={
            "ts_type": "arkimex",
            "ts_arkimex_type": tableau,
            "ts_adapt_type": "none",
            "ts_time_step": 0.1 / nsteps,
            "ts_exact_final_time": "stepover",
        },
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


def test_probe_reports_propagation():
    """If this fails, the shim is a no-op and everything below is meaningless."""
    assert stage_hook_propagates() is True


def test_callback_fires_once_per_implicit_stage():
    """2c has three stages, the first of which PETSc treats as explicit.

    Tie the count to the steps actually taken rather than to the steps
    requested: ``ts_exact_final_time = stepover`` overshoots the end time.
    """
    times = []
    solver, _ = _solve(stage_callback=lambda t, _u: times.append(t), nsteps=10)
    assert len(times) == 2 * solver.ts.getStepNumber()
    # Stage times are interior to the step, so they must straddle the grid.
    assert any(not (t / 0.01) == pytest.approx(round(t / 0.01)) for t in times)


def test_modification_changes_the_accepted_step():
    """A stage modification must not be silently discarded."""
    _, clean = _solve(stage_callback=None)

    def nudge(_t, u):
        u.dat.data[:] += 1e-3

    _, nudged = _solve(stage_callback=nudge)
    assert abs(clean - nudged) > 1e-6, (
        "modifying the stage vector had no effect on the accepted step -- the "
        "hook is firing at the wrong point in the step"
    )


def test_callback_sees_the_stage_value():
    """The callback runs mid-step, so it must see values away from the endpoints."""
    seen = []
    _solve(stage_callback=lambda _t, u: seen.append(float(u.dat.data_ro[0])), nsteps=1)
    assert len({round(s, 12) for s in seen}) > 1


def test_neutral_callback():
    """A callback that writes nothing must not perturb the answer at all."""
    _, clean = _solve(stage_callback=None)
    _, neutral = _solve(stage_callback=lambda _t, _u: None)
    assert clean == neutral


def test_mixed_space_subfunctions():
    """A limiter acts on one field of a mixed system."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1) * FunctionSpace(mesh, "DG", 0)
    w = Function(V)
    w_t = Function(V)
    a, b = split(w)
    adot, bdot = split(w_t)
    p, q = TestFunctions(V)
    w.subfunctions[0].assign(1.0)
    w.subfunctions[1].assign(1.0)
    F = inner(adot, p) * dx + inner(bdot, q) * dx
    G = -inner(a, p) * dx - inner(b, q) * dx

    touched = []

    def only_second_field(_t, u):
        touched.append(u.subfunctions[1].dat.data_ro.copy())
        u.subfunctions[1].dat.data[:] *= 0.5

    problem = firedrake_ts.DAEProblem(F, w, w_t, (0.0, 0.05), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        stage_callback=only_second_field,
        solver_parameters={
            "ts_type": "arkimex",
            "ts_adapt_type": "none",
            "ts_time_step": 0.05,
            "ts_exact_final_time": "stepover",
            "pc_type": "lu",
        },
    )
    solver.solve()
    assert touched
    first, second = (f.dat.data_ro for f in w.subfunctions)
    # The halved field must have fallen well below the untouched one.
    assert np.max(second) < 0.5 * np.min(first)
