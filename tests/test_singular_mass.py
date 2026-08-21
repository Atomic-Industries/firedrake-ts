"""Structural predicates and singular-mass support."""

import numpy as np
import pytest
from conftest import ARK_SSP_G5, ARKIMEX_2C
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
    _f, p, T = split(w)
    fdot, _pdot, Tdot = split(wdot)
    vf, vp, vT = TestFunctions(W)
    F = (
        inner(fdot, vf) * dx  # mass only -> explicitly governed
        + inner(p - T, vp) * dx  # algebraic constraint
        + inner(Tdot, vT) * dx  # mass ...
        + inner(grad(T), grad(vT)) * dx  # ... plus diffusion
    )
    return F, w, wdot, 3


def _two_field(diffusive):
    """Two differential fields; ``diffusive`` gives each row an implicit part."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    F = inner(adot, va) * dx + inner(bdot, vb) * dx
    if diffusive:
        F += inner(grad(a), grad(va)) * dx + inner(grad(b), grad(vb)) * dx
    return F, w, wdot, 2


# The two predicates answer DIFFERENT questions over the same form, and the
# three cases below are what pins that rather than any one of them alone:
#
# mixed_character -- (1,) algebraic and (0,) explicitly governed, so the two
#   answers are disjoint, and T (row 2, which has both a time derivative and an
#   implicit operator) is in neither. A separate test asserting that
#   disjointness would follow deductively from these two exact tuples and could
#   not fail on its own.
# both_differential -- neither predicate fires: every row has a mass term AND
#   an implicit operator. This is also the form under which
#   test_rung3_pde_with_multiplier_runs[arkssp] exercises the empty-tuple path
#   through field_rows() end to end.
# both_mass_only -- (0, 1): mass-only rows are ALL explicitly governed, and a
#   singular mass matrix is not implied by one (algebraic is still empty).
#
# Each was its own test with one or two assertions and a copy of the same
# preamble; the forms are the data, so they are parameters here.
@pytest.mark.parametrize(
    "build,expected_algebraic,expected_explicit",
    [
        pytest.param(_three_field, (1,), (0,), id="mixed_character"),
        pytest.param(lambda: _two_field(True), (), (), id="both_differential"),
        pytest.param(lambda: _two_field(False), (), (0, 1), id="both_mass_only"),
    ],
)
def test_field_partitions(build, expected_algebraic, expected_explicit):
    """``algebraic_fields`` and ``explicitly_governed_fields`` on one form."""
    F, w, wdot, nfields = build()
    assert algebraic_fields(F, w, wdot, nfields) == expected_algebraic
    assert explicitly_governed_fields(F, w, wdot, nfields) == expected_explicit


def test_nonzero_rows_locates_G():
    _F, w, _wdot, nfields = _three_field()
    f, _p, T = split(w)
    vf, vp, _vT = TestFunctions(w.function_space())
    assert nonzero_rows(inner(f, vf) * dx, nfields) == (0,)
    assert nonzero_rows(inner(T, vp) * dx, nfields) == (1,)


def test_absent_residual_row_raises():
    """A component with no equation at all is an error, not an algebraic row."""
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
        assert resolve_fields("ts_algebraic_fields", "probe_", (1,), nfields=3) == (
            0,
            2,
        )
        with pytest.raises(ValueError, match=r"\[2\]"):
            resolve_fields("ts_algebraic_fields", "probe_", (1,), nfields=2)
    finally:
        del opts["probe_ts_algebraic_fields"]

    opts["probe_ts_algebraic_fields"] = "-1"
    try:
        with pytest.raises(ValueError, match=r"\[-1\]"):
            resolve_fields("ts_algebraic_fields", "probe_", (1,), nfields=3)
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


def _rung1(stepper, dt=1e-2, tmax=1.0):
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


@pytest.mark.parametrize(
    "which_row,extra",
    [
        pytest.param("algebraic", {}, id="detected"),
        pytest.param("differential", {"ts_algebraic_fields": "0,1"}, id="declared"),
    ],
)
def test_G_vanishes_on_algebraic_rows(which_row, extra):
    """``G`` on an algebraic row is undefined there; fail loudly, not silently.

    Two cases, because the row set has two sources and only the first was
    covered by the check's own test:

    ``detected`` -- ``G`` is nonzero on row 1, which ``algebraic_fields``
    detects structurally. The straightforward case.

    ``declared`` -- ``G`` is nonzero on row 0, which is DIFFERENTIAL, and
    ``ts_algebraic_fields`` declares rows 0 and 1 algebraic anyway.
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
    w.sub(1).assign(1.0)
    F = inner(ydot, vy) * dx + inner(z - y, vz) * dx
    # Row 1 is the algebraic one; row 0 is differential.
    G = inner(y, vz) * dx if which_row == "algebraic" else inner(-y, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.1), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=0.05, **ARKIMEX_2C, **extra),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="G is nonzero on algebraic"):
        solver.solve()


def test_project_rhs_false_is_rejected_when_G_is_supplied():
    """An unprojected G is a raw dual vector, wrong by a factor of the mass
    matrix once a TS treats it as a state-space derivative .
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    with pytest.raises(ValueError, match="raw dual G"):
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
    # dt=1e-2: order-2 arkimex against an abs=1e-3 tolerance, and this test is
    # about zeroRows being driven with a genuinely concatenated index set, not
    # about accuracy. 1e-2 divides tmax, so stepover lands on 1.0.
    parameters = dict(RUNG_PARAMS, ts_time_step=1e-2, **ARKIMEX_2C)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()

    exact_y = np.exp(-1.0)
    assert float(w.sub(0).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(1).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(2).dat.data_ro[0]) == pytest.approx(-exact_y, abs=1e-3)


@pytest.mark.parametrize("stepper", [ARKIMEX_2C, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung1_dual_path(stepper):
    """Index 1, singular mass, exact solution. Both steppers must agree."""
    y, z = _rung1(stepper)
    assert y == pytest.approx(np.exp(-1.0), abs=1e-3)
    assert z == pytest.approx(-np.exp(-1.0), abs=1e-3)


def _rung2(stepper, dt=1e-3, tmax=1.0):
    """Index 2, the multiplier case: ydot = z - y, 0 = y - g(t), G = -y.

    g(t) = exp(-t), so z = gdot + g = 0 exactly and y = exp(-t). Returns
    (y, z, constraint_defect).
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
    """
    _, _, stiffly_accurate = _rung2(ARK_SSP_G5, dt=1e-2)
    _, _, shipped = _rung2(ARKIMEX_2C, dt=1e-2)
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
    return w, target


@pytest.mark.parametrize("stepper", [ARKIMEX_2C, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung3_pde_with_multiplier_runs(stepper):
    """The PDE-scale rung: real accuracy, not just liveness."""
    w, target = _rung3(stepper)
    assert np.all(np.isfinite(w.sub(0).dat.data_ro))
    assert np.all(np.isfinite(w.sub(1).dat.data_ro))
    assert norm(w.sub(0)) > 0.0
    defect = np.max(np.abs(w.sub(0).dat.data_ro - target.dat.data_ro))
    assert defect < 1e-8, (
        f"u drifted from the pinned target by {defect:.3e} -- the "
        "algebraic constraint should hold to near machine precision"
    )


def test_stage0_ydot_is_zero_on_algebraic_rows():
    """Ẏ_0 must be zero where dF/du_t is, not -F_alg(t^n, y^n, 0)."""
    from firedrake import ufl_expr  # noqa: F401

    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, _zdot = split(wdot)
    vy, vz = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(0.0)  # inconsistent: z + y = 1, not 0

    F = inner(ydot, vy) * dx + inner(z + y, vz) * dx
    G = inner(z, vy) * dx
    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.05), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=0.01, **ARK_SSP_G5),
        options_prefix="",
    )
    stepper = solver.ts.getPythonContext()

    residual = assemble(replace(F, {wdot: Function(W)}))

    first = []
    original = stepper._prepare_stage0_ydot

    def spy(ts, t, x):
        original(ts, t, x)
        if not first:
            first.append(stepper._Ydot[0].getArray(readonly=True).copy())

    stepper._prepare_stage0_ydot = spy
    solver.solve()

    rows = stepper._algebraic_rows
    assert rows is not None, "the algebraic row was not detected at all"
    assert len(first) == 1

    with residual.dat.vec_ro as r:
        lo = r.getOwnershipRange()[0]
        unfixed = r.getArray(readonly=True)[rows - lo].copy()
    assert np.abs(unfixed).max() > 0.5, (
        f"initial condition is not inconsistent enough to detect the "
        f"defect; F_alg(t^0, y^0, 0) = {unfixed}"
    )

    assert np.all(first[0][rows - lo] == 0.0), (
        f"Ydot_0 is {first[0][rows - lo]} on algebraic rows, expected 0; "
        f"the unfixed value there is {-unfixed}"
    )
