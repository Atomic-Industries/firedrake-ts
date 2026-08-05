"""Structural predicates and singular-mass support."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts
from firedrake_ts.solving_utils import (
    algebraic_fields,
    explicitly_governed_fields,
    is_zero_form,
    nonzero_rows,
)


def _three_field():
    """Mimic the (f, p, T) character: explicit-only, algebraic, and both."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V * V
    w = Function(W)
    wdot = Function(W)
    f, p, T = split(w)
    fdot, _pdot, Tdot = split(wdot)
    vf, vp, vT = TestFunctions(W)
    F = (
        inner(fdot, vf) * dx  # mass only -> explicitly governed
        + inner(p - T, vp) * dx  # algebraic constraint
        + inner(Tdot, vT) * dx  # mass ...
        + inner(grad(T), grad(vT)) * dx  # ... plus diffusion
    )
    return F, w, wdot, (f, p, T), (vf, vp, vT)


def test_algebraic_fields_finds_the_constraint_row():
    F, w, wdot, _, _ = _three_field()
    assert algebraic_fields(F, w, wdot, 3) == (1,)


def test_explicitly_governed_fields_finds_the_mass_only_row():
    F, w, wdot, _, _ = _three_field()
    assert explicitly_governed_fields(F, w, wdot, 3) == (0,)


def test_partitions_are_distinct():
    """Differential/algebraic is NOT implicit/explicit. T is in both halves."""
    F, w, wdot, _, _ = _three_field()
    algebraic = algebraic_fields(F, w, wdot, 3)
    explicit_only = explicitly_governed_fields(F, w, wdot, 3)
    assert set(algebraic).isdisjoint(explicit_only)
    assert 2 not in algebraic
    assert 2 not in explicit_only


def test_nonsingular_two_field_has_no_algebraic_rows():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    F = (
        inner(adot, va) * dx
        + inner(grad(a), grad(va)) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    assert algebraic_fields(F, w, wdot, 2) == ()
    assert explicitly_governed_fields(F, w, wdot, 2) == ()


def test_mass_only_rows_are_all_explicitly_governed():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    F = inner(adot, va) * dx + inner(bdot, vb) * dx
    assert explicitly_governed_fields(F, w, wdot, 2) == (0, 1)
    assert algebraic_fields(F, w, wdot, 2) == ()


def test_nonzero_rows_locates_G():
    _F, _w, _wdot, (f, _p, T), (vf, vp, _vT) = _three_field()
    assert nonzero_rows(inner(f, vf) * dx, 3) == (0,)
    assert nonzero_rows(inner(T, vp) * dx, 3) == (1,)


def test_absent_residual_row_raises():
    """A component with no equation at all is an error, not an algebraic row.

    If this were silently folded into ``algebraic_fields`` it would be
    indistinguishable downstream from a genuine algebraic row -- including
    to code that gives algebraic rows a unit diagonal to make a projection
    invertible. That repair does not apply to a row that plain does not
    exist, and the failure would resurface later, further from the cause,
    as a zero pivot with no clue why.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    adot, _bdot = split(wdot)
    va, _vb = TestFunctions(W)
    F = inner(adot, va) * dx  # component 1's test function never appears
    with pytest.raises(ValueError, match=r"\[1\]"):
        algebraic_fields(F, w, wdot, 2)


def test_resolve_fields_prefers_an_explicit_override():
    """Structural default, runtime override -- PETSc's own idiom.

    Same pattern as -pc_fieldsplit_detect_saddle_point: detection is the
    default, not the only option.
    """
    from firedrake.petsc import PETSc

    from firedrake_ts.solving_utils import resolve_fields

    assert resolve_fields("ts_algebraic_fields", "probe_", (1,)) == (1,)
    opts = PETSc.Options()
    opts["probe_ts_algebraic_fields"] = "0,2"
    try:
        assert resolve_fields("ts_algebraic_fields", "probe_", (1,)) == (0, 2)
    finally:
        del opts["probe_ts_algebraic_fields"]


def test_is_zero_form_needs_expanded_derivatives():
    """derivative() with no dependence yields a symbolically-zero integral.

    Without expand_derivatives, Form.empty() is False and the predicate
    silently reports every row as non-zero.
    """
    from firedrake import ufl_expr

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    udot = Function(V)
    v = TestFunction(V)
    mass = inner(udot, v) * dx
    assert is_zero_form(ufl_expr.derivative(mass, u))
    assert not is_zero_form(ufl_expr.derivative(mass, udot))


RUNG_PARAMS = {
    "ts_adapt_type": "none",
    "ts_exact_final_time": "stepover",
}


def _rung1(stepper, dt=1e-3, tmax=1.0):
    """Index 1, one-cell DG0: ydot = z, 0 = z + y, z explicit. Exact y = e^-t.

    M = diag(1, 0) is genuinely singular and G vanishes on the algebraic row.
    No spatial discretisation error, so observed order is the tableau's alone.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)  # NOT "R": see the note in the brief
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, _zdot = split(wdot)
    vy, vz = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(-1.0)

    F = inner(ydot, vy) * dx + inner(z + y, vz) * dx
    G = inner(z, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    return float(w.sub(0).dat.data_ro[0]), float(w.sub(1).dat.data_ro[0])


ARKIMEX = {"ts_type": "arkimex", "ts_arkimex_type": "2c"}


def test_rung1_singular_mass_runs_under_arkimex():
    """A singular mass matrix must not break the RHS projection."""
    y, z = _rung1(ARKIMEX)
    assert y == pytest.approx(np.exp(-1.0), abs=1e-3)
    assert z == pytest.approx(-np.exp(-1.0), abs=1e-3)


def test_G_nonzero_on_an_algebraic_row_is_rejected():
    """The projection is undefined there; fail loudly, not silently."""
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)  # NOT "R": see the note in the brief
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, _zdot = split(wdot)
    vy, vz = TestFunctions(W)
    F = inner(ydot, vy) * dx + inner(z + y, vz) * dx
    G = inner(y, vz) * dx  # nonzero on the ALGEBRAIC row

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.1), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=0.05, **ARKIMEX),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="algebraic"):
        solver.solve()


def test_nonsingular_projection_is_unchanged():
    """The existing non-mixed path must be byte-identical."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=1e-3, **ARKIMEX),
        options_prefix="",
    ).solve()
    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-4)
