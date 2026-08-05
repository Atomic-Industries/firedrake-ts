"""Structural predicates and singular-mass support."""

from firedrake import *

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
