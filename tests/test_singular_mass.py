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
    """Exactly the pressure row: (1,), not (0, 1) and not (1, 2).

    Together with the next test, this pins that differential/algebraic and
    implicit/explicit are DIFFERENT questions over the same form: the two
    answers here are (1,) and (0,), so they are disjoint, and T (row 2) --
    which has both a time derivative and an implicit operator -- is in
    neither. A separate test asserting that disjointness followed
    deductively from these two exact tuples and could not fail on its own.
    """
    F, w, wdot, _, _ = _three_field()
    assert algebraic_fields(F, w, wdot, 3) == (1,)


def test_explicitly_governed_fields_finds_the_mass_only_row():
    """Exactly the mass-only row: (0,). See the note above on the partitions."""
    F, w, wdot, _, _ = _three_field()
    assert explicitly_governed_fields(F, w, wdot, 3) == (0,)


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


def _rung1(stepper, dt=1e-2, tmax=1.0):
    """Index 1, one-cell DG0: ydot = z, 0 = z + y, z explicit. Exact y = e^-t.

    M = diag(1, 0) is genuinely singular and G vanishes on the algebraic row.
    No spatial discretisation error, so observed order is the tableau's alone.

    dt=1e-2 rather than 1e-3: this helper's callers assert abs=1e-3, and the
    measured errors at 1e-2 are 5.7e-6 (arkimex) and 3.1e-6 (arkssp) -- 176x
    and 325x inside that -- for a tenth of the steps. It divides tmax exactly,
    so stepover lands on 1.0 with no overshoot. No order test uses this
    default; the convergence runs pass dt explicitly.
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


# "A singular mass matrix must not break the RHS projection" under arkimex is
# test_rung1_dual_path[arkimex] below -- same helper, same tableau, same two
# abs=1e-3 assertions. A standalone copy of it here was a second 1000-step run
# of exactly that.


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


# "The existing non-mixed path must be unchanged" is test_imex.py's
# test_imex_advances_solution[2c]: same form, same tableau, same dt, same
# abs=1e-4 tolerance, and it additionally asserts liveness and covers three
# more tableaux. A copy of it here was strictly the weaker of the two.


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
    # dt=1e-2: order-2 arkimex against an abs=1e-3 tolerance, and this test is
    # about zeroRows being driven with a genuinely concatenated index set, not
    # about accuracy. 1e-2 divides tmax, so stepover lands on 1.0.
    parameters = dict(RUNG_PARAMS, ts_time_step=1e-2, **ARKIMEX)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()

    exact_y = np.exp(-1.0)
    assert float(w.sub(0).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(1).dat.data_ro[0]) == pytest.approx(exact_y, abs=1e-3)
    assert float(w.sub(2).dat.data_ro[0]) == pytest.approx(-exact_y, abs=1e-3)


def test_algebraic_fields_override_is_read_from_solver_parameters():
    """``ts_algebraic_fields`` must work from ``solver_parameters``, not just
    from the command line.

    ``_algebraic_fields`` is a ``cached_property`` that reads the option out
    of PETSc's database via ``resolve_fields``, and it is forced eagerly by
    ``solve()``'s ``_check_G_vanishes_on_algebraic_rows`` call. Options given
    in ``solver_parameters`` are only pushed into that database inside
    ``inserted_options()`` (and deleted again on exit), so forcing the
    property outside it resolved the option to its structural default AND
    cached that for the rest of the solve -- silently ignoring the documented
    override and stamping unit diagonals on the wrong rows of ``dF/du_t``.

    Declaring row 0 algebraic here is deliberately WRONG for this problem
    (row 0 is differential), because a correct override is indistinguishable
    from the detected default. ``G`` is nonzero on row 0, so if the override
    is honoured the ``G``-vanishes check must reject it; if it is dropped,
    the solve proceeds happily -- which is exactly the bug.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "DG", 0)
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, _zdot = split(wdot)
    vy, vz = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(1.0)
    F = inner(ydot, vy) * dx + inner(z - y, vz) * dx
    G = inner(-y, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.1), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            RUNG_PARAMS, ts_time_step=1e-2, ts_algebraic_fields="0,1", **ARKIMEX
        ),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="G is nonzero on algebraic"):
        solver.solve()


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
    return w, target


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung3_pde_with_multiplier_runs(stepper):
    """The PDE-scale rung: real accuracy, not just liveness.

    The constraint row (``u - target = 0``) pins ``u`` to ``target(x)``
    exactly at every stage -- it is algebraic, solved by the nonlinear
    solver's own tolerance, not subject to any O(h^p) truncation error.
    So ``u`` must match ``target`` to close to machine precision, not
    merely be finite and nonzero. Measured: max|u - target| ~ 1.3e-14
    (arkimex), ~2.2e-16 (arkssp); 1e-8 leaves ample margin against SNES's
    own default tolerances without demanding an exact discretisation
    result the test does not otherwise control.
    """
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
    """Ẏ_0 must be zero where dF/du_t is, not -F_alg(t^n, y^n, 0).

    _reassemble_stage0_mass gives the algebraic rows a unit diagonal so the
    mass matrix is invertible at all. That makes the stage-0 solve return
    -F_alg(t^n, y^n, 0) on those rows -- not a derivative, since there is no
    u_t in those equations -- and _build_offset then propagates it into every
    later stage as h At_ij Ydot_j. PETSc zeroes the same rows immediately
    after the solve that produces its own Ydot0
    (VecISSet(Ydot0, ark->alg_is, 0.0), arkimex.c:1389).

    The initial condition here is deliberately INCONSISTENT -- z(0) = 0 with
    y(0) = 1 violates the constraint 0 = z + y by exactly 1 -- because a
    consistent one makes the whole defect invisible: F_alg(t^0, y^0, 0) is
    then already zero and the unfixed code writes a zero it did not mean.
    _rung1's own y(0) = 1, z(0) = -1 is consistent, which is why no existing
    test caught this. The assembled residual is checked below so the test
    cannot pass vacuously.
    """
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

    # Non-vacuity, ASSEMBLED before the solve advances w: F(t^0, y^0, 0)
    # really is nonzero on the algebraic row, so the unfixed code wrote
    # -1.0 there rather than a zero it would have got for free. Assembling
    # after solve() would read the final state, where the constraint is
    # satisfied and the check would pass for the wrong reason. The row
    # indices only exist once setUp has run, so the values are read out
    # below; this Cofunction is unaffected by the solve.
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
