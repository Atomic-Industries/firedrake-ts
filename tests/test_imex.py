"""The IMEX (``G``) path must actually advance the solution."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts

EXACT = np.exp(-1.0)  # solution of u' = -u at t = 1


def _decay(tableau="3", dt=1e-3, split=True):
    """Integrate ``u' = -u`` to t = 1, with ``-u`` explicit when ``split``."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    if split:
        F, G = inner(u_t, v) * dx, -inner(u, v) * dx
    else:
        F, G = inner(u_t, v) * dx + inner(u, v) * dx, None
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "arkimex",
            "ts_arkimex_type": tableau,
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "stepover",
        },
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


@pytest.mark.parametrize("tableau", ["3", "2c", "a2", "prssp2"])
def test_imex_advances_solution(tableau):
    """The explicit part must reach the state, not be computed and discarded."""
    _, value = _decay(tableau=tableau)
    assert abs(value - 1.0) > 0.1, (
        f"{tableau}: solution unchanged from its initial condition -- the "
        "explicit part G was evaluated but never applied"
    )
    assert abs(value - EXACT) < 1e-4, f"{tableau}: u(1) = {value}, expected {EXACT}"


def test_imex_snes_actually_solves():
    """Stage solves must do work; zero SNES iterations means a null residual."""
    solver, _ = _decay(dt=1e-2)
    assert solver.ts.getSNESIterations() > 0


def test_snes_residual_callback_survives_the_solve():
    """The TS's SNES must still be evaluating the DAE residual at the end."""
    solver, _ = _decay(dt=1e-2)
    _, callback = solver.snes.getFunction()
    assert callback is None, (
        "SNESTSFormFunction was displaced by a Python SNES residual; the DAE "
        f"residual is being evaluated by {callback[0]!r} against a stale udot"
    )


def test_implicit_path_matches_imex_path():
    """Splitting a term into G must not change the answer beyond method error."""
    _, split = _decay(split=True)
    _, monolithic = _decay(split=False)
    assert abs(split - monolithic) < 1e-3


def test_heat_explicit_example_diffuses():
    """The PDE of examples/heat-explicit.py must actually evolve."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    F = inner(u_t, v) * dx
    G = -(inner(grad(u), grad(v)) * dx - 1.0 * v * dx)
    bcs = [DirichletBC(V, 1.0, 1), DirichletBC(V, 0.0, 2)]
    x = SpatialCoordinate(mesh)
    u.interpolate(conditional(lt(x[0], 0.5), 1.0, 0.0))
    u0 = u.copy(deepcopy=True)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.4), bcs=bcs, G=G)
    firedrake_ts.DAESolver(problem, options_prefix="").solve()

    assert sqrt(abs(assemble((u - u0) ** 2 * dx))) > 0.1
