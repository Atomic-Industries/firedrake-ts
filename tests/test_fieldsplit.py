"""Fieldsplit preconditioning must work on mixed DAE problems."""

import numpy as np
import pytest
from conftest import ARKIMEX_2C, scalar_problem
from firedrake import *

import firedrake_ts

FIELDSPLIT = {
    **ARKIMEX_2C,
    "ts_adapt_type": "none",
    "ts_time_step": 0.05,
    "ts_exact_final_time": "stepover",
    "snes_type": "newtonls",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
}


def _coupled(with_G):
    """Two diffusing fields on a mixed space, optionally IMEX-split."""
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    w.sub(0).assign(1.0)

    F = (
        inner(adot, va) * dx
        + inner(grad(a), grad(va)) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    G = (-inner(a - b, va) * dx + inner(a - b, vb) * dx) if with_G else None
    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.2), G=G)
    return firedrake_ts.DAESolver(
        problem, solver_parameters=FIELDSPLIT, options_prefix=""
    ), w


@pytest.mark.parametrize("with_G", [False, True])
def test_fieldsplit_solve_runs(with_G):
    """A full solve under pc_type: fieldsplit must complete and advance.

    Also covers "split() must not die on DAEProblem's attribute names", which
    had its own test: the real PC-fieldsplit setup below drives
    ``_TSContext.split`` through ``DMCreateFieldDecomposition``, so a raising
    ``split()`` fails this test too. Only its ``len(splits) == 2`` assertion
    was not implied, so it is kept here.
    """
    solver, w = _coupled(with_G)
    assert len(solver._ctx.split([[0], [1]])) == 2
    solver.solve()
    assert solver.ts.getStepNumber() > 0
    assert norm(w.sub(0)) > 0.0


def test_supplied_jacobian_is_not_doubled():
    r"""A ``DAEProblem`` given an explicit ``J`` must use it as-is."""
    u, udot, v = scalar_problem(cells=8)
    du = TrialFunction(u.function_space())

    F = inner(udot, v) * dx + inner(grad(u), grad(v)) * dx
    mass = inner(du, v) * dx
    stiffness = inner(grad(du), grad(v)) * dx

    # Built through the documented callable-of-sigma route, which is the only
    # way a caller can express the shift dependency at all: the shift Constant
    # is created inside DAEProblem.__init__. This test used to have to borrow
    # `reference.J` from a second DAEProblem to get a J with a live shift in
    # it, which was the API telling us it had no front door.
    supplied = firedrake_ts.DAEProblem(
        F, u, udot, (0.0, 1.0), J=lambda sigma: sigma * mass + stiffness
    )
    supplied.shift.assign(3.0)

    expected = assemble(3.0 * mass + stiffness).petscmat
    actual = assemble(supplied.J).petscmat

    # THE assertion: a supplied J must survive unmodified.
    assert (actual - expected).norm() < 1e-10


def test_supplied_jacobian_of_the_shift_solves_a_pure_ode():
    """A caller-supplied J must be usable on a problem where dF/du is zero."""
    u, udot, v = scalar_problem()
    du = TrialFunction(u.function_space())

    problem = firedrake_ts.DAEProblem(
        inner(udot, v) * dx,
        u,
        udot,
        (0.0, 1.0),
        G=-inner(u, v) * dx,
        J=lambda sigma: sigma * inner(du, v) * dx,
    )
    firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARKIMEX_2C,
            ts_adapt_type="none",
            ts_time_step=1e-2,
            ts_exact_final_time="matchstep",
        ),
        options_prefix="",
    ).solve()

    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-4)
