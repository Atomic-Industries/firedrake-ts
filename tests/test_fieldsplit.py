"""Fieldsplit preconditioning must work on mixed DAE problems."""

import pytest
from firedrake import *

import firedrake_ts

FIELDSPLIT = {
    "ts_type": "arkimex",
    "ts_arkimex_type": "2c",
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
def test_split_does_not_raise(with_G):
    """split() must not die on DAEProblem's attribute names."""
    solver, _ = _coupled(with_G)
    splits = solver._ctx.split([[0], [1]])
    assert len(splits) == 2


@pytest.mark.parametrize("with_G", [False, True])
def test_fieldsplit_solve_runs(with_G):
    """A full solve under pc_type: fieldsplit must complete and advance."""
    solver, w = _coupled(with_G)
    solver.solve()
    assert solver.ts.getStepNumber() > 0
    assert norm(w.sub(0)) > 0.0


def test_supplied_jacobian_is_not_doubled():
    r"""A ``DAEProblem`` given an explicit ``J`` must use it as-is.

    ``DAEProblem.__init__`` documents ``J`` as the *complete* Jacobian
    ``sigma*dF/du_t + dF/du`` (see its ``:param J:`` docstring). A prior bug
    re-added ``shift*dF/du_t`` unconditionally even when ``J`` was supplied.
    Note this does NOT produce a doubled ``6*M + K`` from a supplied
    ``J = 3*M + K``: the stray term the old code added was
    ``supplied.shift * M``, where ``supplied.shift`` is a *fresh*, unassigned
    ``Constant(1.0)`` on the second ``DAEProblem`` below -- a different
    ``Constant`` from ``reference.shift`` (set to 3.0 to bake ``3*M + K``
    into ``reference.J`` in the first place). Reverting the fix on this
    exact test reproduces ``4*M + K``, not ``6*M + K``. ``_TSContext.split()``
    always supplies a ``J`` for its per-field sub-problems, so this
    corrupted the mass block of every fieldsplit sub-block Jacobian -- with
    no exception and no visible failure, just a preconditioner built from
    the wrong matrix. Regressing this fix would reintroduce that silent
    corruption.
    """
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    udot = Function(V)
    v = TestFunction(V)
    du = TrialFunction(V)

    F = inner(udot, v) * dx + inner(grad(u), grad(v)) * dx
    mass = inner(du, v) * dx
    stiffness = inner(grad(du), grad(v)) * dx

    # Derive the complete Jacobian the same way DAEProblem does internally
    # when no J is supplied -- this branch is untouched by the fix, so it
    # is a trustworthy "sigma*dF/du_t + dF/du" to hand to a second problem
    # as an already-complete, caller-supplied J.
    reference = firedrake_ts.DAEProblem(F, u, udot, (0.0, 1.0))
    reference.shift.assign(3.0)

    supplied = firedrake_ts.DAEProblem(F, u, udot, (0.0, 1.0), J=reference.J)

    expected = assemble(3.0 * mass + stiffness).petscmat
    actual = assemble(supplied.J).petscmat

    # THE assertion: a supplied J must survive unmodified. Discriminating on
    # its own -- the pre-fix code gave 4*M + K here, not 3*M + K -- so no
    # second assertion against an (inaccurate) doubled value is needed.
    assert (actual - expected).norm() < 1e-10
