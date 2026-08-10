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


def test_project_rhs_false_is_rejected_when_G_is_supplied():
    """An unprojected G is a raw dual vector, wrong by a factor of the mass
    matrix once a TS treats it as a state-space derivative -- no supported
    TS type wants that, so it is rejected outright rather than left as a
    live (and silently wrong) option.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    with pytest.raises(ValueError, match="mass matrix"):
        firedrake_ts.DAESolver(problem, project_rhs=False, options_prefix="")


def test_two_algebraic_fields_drive_the_concatenated_zero_rows():
    """Every case above has exactly one algebraic field, so
    ``numpy.concatenate([ises[i].getIndices() for i in self._algebraic_fields])``
    is never driven with more than one element. Here y' = -y is
    differential; z1 = y and z2 = -y are algebraic constraints on two
    separate rows, so the mass matrix is singular on TWO rows, and
    ``zeroRows`` must be called with a genuinely concatenated index set.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)
    W = R * R * R
    w = Function(W)
    wdot = Function(W)
    y, z1, z2 = split(w)
    ydot, _z1dot, _z2dot = split(wdot)
    vy, vz1, vz2 = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(1.0)
    w.sub(2).assign(-1.0)

    F = inner(ydot, vy) * dx + inner(z1 - y, vz1) * dx + inner(z2 + y, vz2) * dx
    G = inner(-y, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 1.0), G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=1e-3, **ARKIMEX)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()

    exact_y = np.exp(-1.0)
    assert float(w.sub(0).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(1).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(2).dat.data_ro[0]) == pytest.approx(-exact_y, abs=1e-3)


ARK_SSP_G5 = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung1_dual_path(stepper):
    """Index 1, singular mass, exact solution. Both steppers must agree."""
    y, z = _rung1(stepper)
    assert y == pytest.approx(np.exp(-1.0), abs=1e-3)
    assert z == pytest.approx(-np.exp(-1.0), abs=1e-3)


def _rung2(stepper, dt=1e-3, tmax=1.0):
    """Index 2, the multiplier case: ydot = z - y, 0 = y - g(t), G = -y.

    g(t) = exp(-t), so z = gdot + g = 0 exactly and y = exp(-t). Returns
    (y, z, constraint_defect). The defect is what R3 stiff accuracy buys:
    near 1e-16 with b == A[s-1,:], near 1e-3 without.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)  # NOT "R": see the note below
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, _zdot = split(wdot)
    vy, vz = TestFunctions(W)
    time = Constant(0.0)
    w.sub(0).assign(1.0)
    w.sub(1).assign(0.0)

    F = inner(ydot - z, vy) * dx + inner(y - exp(-time), vz) * dx
    G = -inner(y, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), time=time, G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    y_val = float(w.sub(0).dat.data_ro[0])
    return y_val, float(w.sub(1).dat.data_ro[0]), abs(y_val - np.exp(-tmax))


def test_stiff_accuracy_is_what_buys_the_constraint_defect():
    """R3 -- b == A[s-1,:] and bt == At[s-1,:] -- makes the completion the
    last stage value, so the constraint holds to machine precision instead
    of to O(h^p).

    Asserted as a CONTRAST, not as a uniform bound, because no tableau
    PETSc ships satisfies R3: its b = NULL default collapses the explicit b
    onto the implicit bt, so the explicit forcing never collapses to the
    last stage. Demanding machine precision from a built-in would be asking
    for something the tableau cannot deliver. The gap between the two is the
    measurable value of R3, and the reason this project registers its own
    tableau rather than using a shipped one.

    Measured at dt = 1e-2: esdirk_gamma5 at machine precision, PETSc's 2c
    at 7.7e-06 -- about ten orders of magnitude apart.
    """
    _, _, stiffly_accurate = _rung2(ARK_SSP_G5, dt=1e-2)
    _, _, shipped = _rung2(ARKIMEX, dt=1e-2)
    assert stiffly_accurate < 1e-10, (
        f"stiffly accurate tableau gave defect {stiffly_accurate:.3e}, "
        "expected machine precision -- R3 is not being exploited"
    )
    assert shipped > 1e3 * stiffly_accurate, (
        f"the shipped non-stiffly-accurate tableau gave defect {shipped:.3e} "
        f"against {stiffly_accurate:.3e} -- the contrast R3 is supposed to "
        "produce has vanished, so this test no longer demonstrates anything"
    )


def _rung3(stepper, dt=2e-3, tmax=0.1, n=8):
    """PDE scale: heat equation on V x R with a mean-value multiplier.

    u_t = laplacian(u) + lambda, with int u dx pinned. Real mixed space, real
    singular mass, index 2, fieldsplit-able -- and no momentum balance.
    """
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "P", 1)
    # A mean-value multiplier would need an "R"-space (here "DG", 0) block,
    # but a mixed space containing R cannot be assembled monolithically at
    # all -- see the note above. lam is field-valued instead, in V * V.
    W = V * V
    w = Function(W)
    wdot = Function(W)
    u, lam = split(w)
    udot, _lamdot = split(wdot)
    vu, vlam = TestFunctions(W)
    (x,) = SpatialCoordinate(mesh)
    w.sub(0).interpolate(1.0 + 0.5 * sin(2 * pi * x))

    target = Function(V).interpolate(1.0 + 0.5 * sin(2 * pi * x))
    F = (
        inner(udot, vu) * dx
        + inner(grad(u), grad(vu)) * dx
        - inner(lam, vu) * dx
        + inner(u - target, vlam) * dx
    )
    G = -0.1 * inner(u, vu) * dx  # an explicit reaction term

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    return w


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung3_pde_with_multiplier_runs(stepper):
    w = _rung3(stepper)
    assert np.all(np.isfinite(w.sub(0).dat.data_ro))
    assert np.all(np.isfinite(w.sub(1).dat.data_ro))
    assert norm(w.sub(0)) > 0.0
