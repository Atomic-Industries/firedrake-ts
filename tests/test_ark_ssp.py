"""The ARKSSP TSPYTHON stepper."""

import numpy as np
import pytest
from conftest import ARK_SSP, ARK_SSP_G5, ARKIMEX_2C, EXACT_DECAY, scalar_problem
from firedrake import *

import firedrake_ts

EXACT = EXACT_DECAY  # solution of u' = -u at t = 1

#: Step sizes for the order-ratio tests. Four values, so three ratios: dt this
#: size on ODE-exact problems is the asymptotic regime, not resolution-limited.
_DTS = (0.1, 0.05, 0.025, 0.0125)


def _decay(tableau, dt=1e-3, tmax=1.0, extra=None):
    """Integrate u' = -u to tmax with -u explicit, under ARKSSP."""
    u, u_t, v = scalar_problem()
    F = inner(u_t, v) * dx
    G = -inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax), G=G)
    parameters = {
        **ARK_SSP,
        "ts_ark_ssp_type": tableau,
        "ts_adapt_type": "none",
        "ts_time_step": dt,
        "ts_exact_final_time": "matchstep",
    }
    parameters.update(extra or {})
    solver = firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


def test_setup_resolves_the_stepper_and_rebuilds_a_stale_mass_ksp():
    """``-ts_type python -ts_python_type`` resolves, and setUp is re-runnable."""
    from firedrake import dmhooks

    from firedrake_ts.ark_ssp import ARKSSP

    solver, _ = _decay("esdirk_gamma5", dt=0.1)
    ctx = solver.ts.getPythonContext()
    assert isinstance(ctx, ARKSSP)
    # setUp ran, asserted by artifacts only setUp produces rather than by a call
    # counter that existed solely to be read from a test.
    assert ctx._P is not None and ctx._q is not None
    old_ksp = ctx._mass_ksp
    assert old_ksp is not None
    assert old_ksp.handle != 0

    # setUp reads dmhooks.get_appctx(ts.getDM()) (via _setup_stage0_mass_solve
    # -> _find_frozen_rows), which only resolves inside the same add_hooks
    # context DAESolver itself opens around every real setUp()/solve() --
    # see ts_solver.py's __init__ and solve(). Without it this raises
    # AttributeError on a None appctx rather than exercising the KSP-reuse
    # path this test targets.
    dm = solver.ts.getDM()
    with dmhooks.add_hooks(dm, solver, appctx=solver._ctx):
        ctx.setUp(solver.ts)

    assert old_ksp.handle == 0, "the previous stage-0 mass KSP was leaked"
    assert ctx._mass_ksp is not None
    assert ctx._mass_ksp.handle != 0
    assert ctx._mass_ksp is not old_ksp


def test_imex_euler_advances_and_is_first_order():
    """Halving dt must halve the error, and the answer must be right."""
    _, coarse = _decay("imex_euler", dt=0.025)
    _, fine = _decay("imex_euler", dt=0.0125)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 1.7 < ratio < 2.3, f"observed order ratio {ratio}, expected ~2"
    assert abs(fine - 1.0) > 0.1, "solution never left its initial condition"
    # First order: error ~ C h with C ~ 0.185 measured, so ~2.3e-3 at h=0.0125.
    assert abs(fine - EXACT) < 5e-3, f"u(1) = {fine}, expected {EXACT}"


@pytest.mark.parametrize(
    "tableau,implicit",
    [("imex_euler", True), ("ssprk2", False)],
    ids=["implicit_part", "purely_explicit"],
)
def test_stage_solves_happen_only_when_the_tableau_has_an_implicit_part(
    tableau, implicit
):
    """Real work in the SNES for a tableau with At != 0, none for At == 0."""
    solver, _ = _decay(tableau, dt=1e-2)
    if implicit:
        assert solver.snes.getIterationNumber() > 0
        assert solver.snes.getConvergedReason() > 0
    else:
        assert solver.snes.getIterationNumber() == 0


def test_shu_osher_error_is_unwrapped_with_the_actionable_numbers():
    """DAESolver.solve must unwrap the PETSc.Error and surface ShuOsherError."""
    from firedrake_ts.tableaux import ShuOsherError

    with pytest.raises(ShuOsherError) as excinfo:
        _decay("ssprk2", dt=0.1, extra={"ts_ark_ssp_radius": 5.0})
    message = str(excinfo.value)
    assert "exceeds the radius" in message
    assert "r = 5.0" in message
    assert "R(A,b)" in message


def test_ssprk2_is_second_order():
    """Heun in Shu-Osher form must show design order 2."""
    _, coarse = _decay("ssprk2", dt=0.05)
    _, fine = _decay("ssprk2", dt=0.025)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"


def _esdirk_converge(build_F, u0, dt, tmax=1.0):
    """One esdirk_gamma5 run of a scalar problem in F, with G = None."""
    u, u_t, v = scalar_problem(u0=u0)
    time = Constant(0.0)
    problem = firedrake_ts.DAEProblem(
        build_F(u, u_t, v, time), u, u_t, (0.0, tmax), time=time
    )
    firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            **ARK_SSP_G5,
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    ).solve()
    return float(u.dat.data_ro[0])


# Every case below is a regression test, and the four are NOT interchangeable
#
# purely_implicit -- u' = -u with -u INSIDE F, G = None.
#   The class of problem ARKSSP exists to serve: a nontrivial implicit
#   operator, with esdirk_gamma5's explicit first stage (At[0, 0] == 0)
#   exercised against a real dF/du. Y_0 = y_n, so Ydot_0 must be evaluated
#   from F(t^n, y_n, 0).
#
# nonpolynomial_source -- u' = exp(-t) (u(0) = 0), source INSIDE F, G = None.
#   dF/du is structurally zero everywhere, but F(t, y, 0) != 0. A guard that
#   conflates "dF/du is structurally zero" with "Ydot_0 = 0 exactly" gets this
#   wrong -- those are not the same condition, and dF/du == 0 says nothing
#   about F(t, y, 0). exp(-t) rather than a constant so a real O(h^2)
#   truncation error remains to measure.
#   Exact: u(t) = 1 - exp(-t), u(1) = 1 - exp(-1).
#
# state_dependent_mass -- (1 + u) u' = 1 (u(0) = 0), G = None.
#   M = dF/du_t = (1 + u) v depends on the state itself: the variable
#   density/porosity/saturation class. A mass matrix cached once at setUp (the
#   initial condition u = 0) keeps solving Ydot_0 against M(0) forever however
#   far u has moved -- flat in dt, silent, no exception.
#   Exact: (1 + u) du = dt, so u + u^2/2 = t, u(t) = sqrt(1 + 2t) - 1.
#
# time_dependent_mass -- (1 + t) u' = 1 (u(0) = 0), G = None.
#   M = (1 + t) v depends on time but NOT the state, so dM/du is structurally
#   zero. That distinction is the whole point, and why this case cannot be
#   merged away in favour of the one above: a reassembly guarded on
#   dM/du != 0 skips every reassembly here and keeps inverting M(t^0) = 1,
#   while passing the M(u) test that motivated the guard.
_CONVERGENCE_CASES = [
    pytest.param(
        lambda u, u_t, v, time: inner(u_t, v) * dx + inner(u, v) * dx,
        1.0,
        EXACT,
        id="purely_implicit",
    ),
    pytest.param(
        lambda u, u_t, v, time: inner(u_t, v) * dx - inner(exp(-time), v) * dx,
        0.0,
        1.0 - np.exp(-1.0),
        id="nonpolynomial_source",
    ),
    pytest.param(
        lambda u, u_t, v, time: (
            inner((1.0 + u) * u_t, v) * dx - inner(Constant(1.0), v) * dx
        ),
        0.0,
        np.sqrt(3.0) - 1.0,
        id="state_dependent_mass",
    ),
    pytest.param(
        lambda u, u_t, v, time: (
            inner((1.0 + time) * u_t, v) * dx - inner(Constant(1.0), v) * dx
        ),
        0.0,
        np.log(2.0),
        id="time_dependent_mass",
    ),
]


@pytest.mark.parametrize("build_F,u0,exact", _CONVERGENCE_CASES)
def test_esdirk_gamma5_converges(build_F, u0, exact):
    """Design order 2 on each Ydot_0 defect case. See _CONVERGENCE_CASES above"""
    errors = [abs(_esdirk_converge(build_F, u0, dt) - exact) for dt in _DTS]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"


def test_esdirk_gamma5_converges_on_a_constant_source():
    """u' = 1 with the source INSIDE F: asserted EXACT, deliberately not a ratio"""
    for dt in _DTS:
        value = _esdirk_converge(
            lambda u, u_t, v, time: inner(u_t, v) * dx - inner(Constant(1.0), v) * dx,
            0.0,
            dt,
        )
        assert abs(value - 1.0) < 1e-8, (
            f"u(1) = {value} at dt={dt}, expected ~1.0 to near machine "
            "precision (u' = const is exact for any consistent RK method)"
        )


def test_esdirk_gamma5_refuses_f_nonlinear_in_udot():
    """``Ẏ_0 = -M^-1 F(t^n, y^n, 0)`` is exact only when F is affine in u̇."""
    u, u_t, v = scalar_problem(u0=0.0)
    F = inner(u_t * u_t - Constant(1.0), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP_G5,
            ts_adapt_type="none",
            ts_time_step=0.1,
            ts_exact_final_time="matchstep",
        ),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="nonlinear in"):
        solver.solve()


#: The message every induced divergence carries, so the tests below can assert
#: the original cause survives into whatever error finally surfaces.
_INDUCED = "induced stage divergence"


def _flaky_stage_solver(ctx, n_failures):
    """Wrap ``ctx._solve_stage`` to raise on its first ``n_failures`` calls.

    ``n_failures=None`` never stops failing, which is what the two
    persistent-divergence tests need; each previously defined its own
    identical always-raising stub inline.

    A reusable technique for exercising the SNES-divergence reject path
    without a pathological nonlinear problem (which tends to blow up
    numerically rather than diverge cleanly, testing the wrong failure
    mode). ``ConvergenceError`` is exactly what a real diverged SNES
    solve raises from ``_solve_stage``, so this reproduces the same
    exception the real path does, at a call site the test controls.
    """
    original = ctx._solve_stage
    calls = [0]

    def flaky(ts, tab, h, i):
        calls[0] += 1
        if n_failures is None or calls[0] <= n_failures:
            raise ConvergenceError(_INDUCED)
        return original(ts, tab, h, i)

    return flaky, calls, original


def _diverging_problem(max_step_rejections, dt=0.5, extra=None):
    """A trivial implicit problem, set up only to have its stage solver
    replaced -- the equation itself never needs to be hard to solve.
    """
    u, u_t, v = scalar_problem()
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, dt))
    parameters = dict(
        ARK_SSP_G5,
        ts_adapt_type="none",
        ts_time_step=dt,
        ts_exact_final_time="matchstep",
        ts_max_step_rejections=max_step_rejections,
    )
    parameters.update(extra or {})
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=parameters,
        options_prefix="",
    )
    return solver


def test_snes_divergence_is_retried_with_a_smaller_step():
    """A transient stage-SNES divergence must shrink h and retry to
    completion, not abort the solve.

    Mirrors PETSc's TSAdaptCheckStage: h *= ts_adapt_scale_solve_failed
    (default 0.25) on each induced failure.
    """
    solver = _diverging_problem(max_step_rejections=5)
    ctx = solver.ts.getPythonContext()
    flaky, calls, original = _flaky_stage_solver(ctx, n_failures=2)
    ctx._solve_stage = flaky
    dt0 = solver.ts.getTimeStep()
    try:
        solver.solve()
    finally:
        ctx._solve_stage = original
    assert calls[0] > 2, "the flaky wrapper's later, non-raising calls never ran"
    assert solver.ts.getTimeStep() < dt0, (
        f"h was never shrunk by the retry: {dt0} -> {solver.ts.getTimeStep()}"
    )


@pytest.mark.parametrize(
    "max_step_rejections,extra,match",
    [
        pytest.param(2, {}, "rejected", id="rejections_exhausted"),
        pytest.param(
            -1, {"ts_adapt_dt_min": 1e-3}, "ts_adapt_dt_min", id="step_floor_reached"
        ),
    ],
)
def test_persistent_snes_divergence_raises_a_convergence_error(
    max_step_rejections, extra, match
):
    """A stage that never converges must stop at one of the two limits, cleanly"""
    solver = _diverging_problem(max_step_rejections=max_step_rejections, extra=extra)
    ctx = solver.ts.getPythonContext()
    always, _calls, original = _flaky_stage_solver(ctx, n_failures=None)
    ctx._solve_stage = always
    try:
        with pytest.raises(ConvergenceError, match=match) as excinfo:
            solver.solve()
    finally:
        ctx._solve_stage = original
    assert _INDUCED in str(excinfo.value), (
        "the error that surfaces must name the cause of the last rejection"
    )
    assert solver.ts.getTimeStep() > 0.0, "h must not have underflowed to 0.0"


def test_shu_osher_matches_butcher_on_the_same_problem():
    """The Shu-Osher path and PETSc's Butcher-form TSRK must agree"""
    dt = 4e-3
    _, ours = _decay("ssprk2", dt=dt)
    u, u_t, v = scalar_problem()
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARKIMEX_2C,
            ts_adapt_type="none",
            ts_time_step=dt,
            ts_exact_final_time="matchstep",
        ),
        options_prefix="",
    ).solve()
    assert abs(ours - float(u.dat.data_ro[0])) < 1e-5


def test_limiter_fires_once_per_stage():
    """The limiter must see every explicit substage, not just the last."""
    calls = []

    def counting_limiter(vec):
        calls.append(vec.norm())

    u, u_t, v = scalar_problem()
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 0.05), G=-inner(u, v) * dx
    )
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            **ARK_SSP,
            "ts_ark_ssp_type": "ssprk2",
            "ts_adapt_type": "none",
            "ts_time_step": 0.01,
            "ts_exact_final_time": "stepover",
        },
        options_prefix="",
    )
    solver.ts.getPythonContext().set_stage_limiter(counting_limiter)
    solver.solve()
    # 2 stages x 5 steps
    assert len(calls) == 10, f"limiter fired {len(calls)} times, expected 10"


def test_complete_refuses_when_neither_stiffly_accurate_nor_explicit():
    """A tableau needing the implicit completion weights bt, but that is not
    stiffly accurate, must raise rather than silently drop them.

    esdirk_gamma5 is stiffly accurate on both its explicit and implicit
    parts. Perturbing b away from A[-1] breaks that without touching At, so
    the tableau is neither stiffly accurate nor purely explicit -- exactly
    the case the Shu-Osher completion row cannot express.
    """
    import dataclasses

    from firedrake_ts.ark_ssp import ARKSSP
    from firedrake_ts.tableaux import TABLEAUX

    tab = TABLEAUX["esdirk_gamma5"]
    bogus_b = tab.b.copy()
    bogus_b[0] += 0.05
    bogus_b[1] -= 0.05
    bogus = dataclasses.replace(tab, name="bogus_non_sa", b=bogus_b)

    stepper = ARKSSP()
    with pytest.raises(ValueError, match="bogus_non_sa"):
        stepper._complete(bogus, None, None)


def _freezable_problem(make_limiter, extra=None):
    """Two fields, one freezable, with a limiter registered on the stepper.

    Row 0 is mass only, so ``explicitly_governed_fields`` certifies it and it
    is freezable. Row 1 has mass plus diffusion, so the implicit operator acts
    there and it is not.

    ``make_limiter`` is called as ``make_limiter(W)`` with the mixed space, so
    a limiter needing scratch storage on it can be built here rather than
    having to reach back for a space this helper owns. Returns
    ``(solver, ctx, w)`` with the limiter registered and nothing solved yet.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(1.0)

    F = inner(adot, va) * dx + inner(bdot, vb) * dx + inner(grad(b), grad(vb)) * dx
    G = -inner(a, va) * dx - inner(b, vb) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.02), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP_G5,
            ts_adapt_type="none",
            ts_time_step=0.01,
            ts_exact_final_time="stepover",
            **(extra or {}),
        ),
        options_prefix="",
    )
    ctx = solver.ts.getPythonContext()
    ctx.set_stage_limiter(make_limiter(W))
    return solver, ctx, w


def _frozen_local(ctx, vec):
    """Ownership-relative indices of the frozen rows, as the stepper sees them."""
    return ctx._frozen_rows - vec.getOwnershipRange()[0]


def _row0_clamping_limiter(W, value_of_call):
    """Force the freezable row to ``value_of_call(n)`` on substage ``n``.

    A dedicated scratch Function, not the solution: ``w`` is also ``ctx._x``,
    which the TS callbacks write into, so borrowing it would work only by
    accident of ordering.
    """
    scratch = Function(W)
    calls = [0]

    def limiter(vec):
        with scratch.dat.vec_wo as target:
            vec.copy(target)
        scratch.sub(0).assign(value_of_call(calls[0]))
        calls[0] += 1
        with scratch.dat.vec_ro as source:
            source.copy(vec)

    return limiter


def test_frozen_rows_residual_constrains_the_stage_solve():
    """The frozen rows' residual must be ``x - target``, with a REAL target."""
    from firedrake import dmhooks

    solver, ctx, _w = _freezable_problem(lambda W: lambda vec: None)
    solver.solve()

    x = ctx._Y[ctx._stage]
    f = x.duplicate()
    local = _frozen_local(ctx, x)
    assert len(local) > 0, "no row was frozen, so there is nothing to constrain"

    def residual_at(vec):
        # computeIFunction resolves the problem through
        # dmhooks.get_appctx(dm), which is only populated inside the same
        # add_hooks context DAESolver opens around its own solve().
        with dmhooks.add_hooks(solver.ts.getDM(), solver, appctx=solver._ctx):
            ctx.formSNESFunction((solver.snes, vec, f, solver.ts))
        return f.getArray(readonly=True)[local].copy()

    # At the pinned value the constraint is satisfied, so its residual is zero.
    # Without this the assertion below could be met by a residual that is
    # simply wrong everywhere rather than by one that measures the offset.
    at_target = residual_at(x)
    assert abs(at_target).max() < 1e-12, (
        f"frozen residual is {at_target} at the pinned value itself, expected 0"
    )

    # Move the frozen rows off the target: the residual must report exactly
    # how far, which is what makes Newton correct them and what lets SNES's
    # convergence test see a broken pin.
    DELTA = 0.25
    x.getArray()[local] += DELTA
    perturbed = residual_at(x)
    assert np.allclose(perturbed, DELTA, atol=1e-12), (
        f"frozen residual is {perturbed} for an iterate {DELTA} off the "
        "pinned value, expected exactly that offset -- a residual of 0 means "
        "the target is being read out of the vector SNES iterates on, so the "
        "rows are unconstrained and the pin rests on the restore alone"
    )


def test_frozen_rows_are_pulled_back_by_a_solver_that_moves_them():
    """The constraint must actually shrink the drift when a solver moves the row."""

    def drift_with(self_referential):
        solver, ctx, _w = _freezable_problem(
            lambda W: _row0_clamping_limiter(W, lambda n: 0.5 + 0.1 * n),
            extra={"snes_type": "qn"},
        )
        if self_referential:
            # The pre-fix residual, verbatim: target read out of the very
            # vector SNES iterates on, making this x - x.
            def old_residual(x, f):
                if not ctx._freeze_active():
                    return
                local = ctx._frozen_local
                ya = ctx._Y[ctx._stage].getArray(readonly=True)
                f.getArray()[local] = x.getArray(readonly=True)[local] - ya[local]

            ctx._apply_freeze_residual = old_residual
        solver.solve()
        # Residual gap left AFTER _solve_stage's restore, which must be zero
        # bit-for-bit: that restore is what converts "pinned to the solver's
        # tolerance" into "pinned exactly", and the boundedness result this
        # freeze serves is a claim about the limited value surviving exactly.
        # Non-vacuous only because qn moves the row at all -- with newtonls
        # the drift is already 0 and the restore has nothing to close.
        x = ctx._Y[ctx._stage]
        residue = abs(
            x.getArray(readonly=True)[_frozen_local(ctx, x)] - ctx._frozen_target
        ).max()
        return ctx._frozen_drift, float(residue)

    constrained, residue = drift_with(False)
    unconstrained, _ = drift_with(True)

    assert residue == 0.0, (
        f"the frozen rows ended {residue:.3e} off their pinned value; "
        "_solve_stage's restore is what makes the pin exact rather than "
        "merely converged, and this solver moved the rows far enough that "
        "the difference is observable"
    )

    # Non-vacuity: if qn does not move the pinned row at all, there is nothing
    # for the constraint to pull back and this test proves nothing.
    assert unconstrained > 0.0, (
        "snes_type qn left the pinned rows untouched even with an "
        "unconstrained residual, so this test is not exercising the mechanism "
        "-- pick a solver whose iterate does not respect the identity row"
    )
    assert constrained < 0.1 * unconstrained, (
        f"the frozen-row constraint reduced the drift only from "
        f"{unconstrained:.3e} to {constrained:.3e}; expected at least an order "
        "of magnitude, so the residual is not pulling the pinned rows back"
    )


def test_frozen_component_survives_the_implicit_solve():
    """A limiter's change to an explicitly-governed row must not be undone."""
    drift = []

    # A value that DIFFERS on every call (0.5 + 0.1n), not the same constant
    # every stage. A stale pin left over from the wrong stage is otherwise
    # numerically indistinguishable from a correct one: that is the exact
    # blind spot which hid the aliasing bug this test was written to catch (a
    # corrupted warm-start guess reverting a frozen row to the PREVIOUS
    # stage's value survived undetected as long as every stage clamped to the
    # same number).
    solver, ctx, _w = _freezable_problem(
        lambda W: _row0_clamping_limiter(W, lambda n: 0.5 + 0.1 * n)
    )

    # Wrap _solve_stage so we can compare Y_i across the implicit solve. This
    # is diagnostic item 3 of the M4 gate, made into a test.
    original = ctx._solve_stage

    def checking_solve_stage(ts, tab, h, i):
        before = ctx._Y[i].getArray(readonly=True).copy()
        original(ts, tab, h, i)
        after = ctx._Y[i].getArray(readonly=True)
        rows = ctx._frozen_rows
        lo, _ = ctx._Y[i].getOwnershipRange()
        local = rows - lo
        drift.append(float(np.abs(after[local] - before[local]).max()))

    ctx._solve_stage = checking_solve_stage
    solver.solve()

    assert ctx._frozen_rows is not None, "row 0 was never detected as freezable"
    assert len(ctx._frozen_rows) > 0
    assert drift, "no implicit stage solve ran, so the freeze was never exercised"
    # THE assertion: the limited value must survive the solve bit-for-bit.
    assert max(drift) == 0.0, (
        f"frozen rows moved by up to {max(drift):.3e} during the implicit "
        "solve -- the solve is dragging the limited value back toward the "
        "unlimited Z_i, which is the defect COOL-193 describes"
    )
    # And the SOLVE held the pin, not just the restore afterwards. The
    # assertion above is measured across the whole of _solve_stage, so the
    # restore alone satisfies it and it says nothing about whether the
    # constraint was honoured. _frozen_drift is measured before that restore.
    assert ctx._frozen_drift == 0.0, (
        f"the stage solve moved the pinned rows by {ctx._frozen_drift:.3e} "
        "before _solve_stage restored them; the residual constraint is not "
        "holding and the restore is hiding it"
    )


@pytest.mark.parametrize("registered", ["before_setup", "after_setup"])
def test_limiter_on_an_implicit_component_is_rejected(registered):
    """Limiting a component with an implicit operator is unsound; say so."""
    u, u_t, v = scalar_problem()
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.02), G=-inner(u, v) * dx)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP_G5,
            ts_adapt_type="none",
            ts_time_step=0.01,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    ctx = solver.ts.getPythonContext()
    if registered == "before_setup":
        ctx.set_stage_limiter(lambda vec: None)
        with pytest.raises(ValueError, match="implicit operator"):
            solver.solve()
        return

    # No limiter yet: setUp must complete cleanly.
    solver.solve()
    assert ctx._tab is not None, "setUp never ran; the post-setUp guard never fires"
    assert ctx._frozen_rows is None, "a freezable row was found; expected none here"
    with pytest.raises(ValueError, match="implicit operator"):
        ctx.set_stage_limiter(lambda vec: None)


def _fsal_decay(fsal, mutate=False, tableau="esdirk_gamma5", dt=0.01, stiff=True):
    """Integrate ``kappa u' = -grad.grad u + u`` with a stiff implicit part"""
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    (x,) = SpatialCoordinate(mesh)
    u.interpolate(sin(pi * x))
    kappa = Constant(1.0)
    # stiff=False keeps F to the mass form alone, which a purely explicit
    # tableau requires -- _setup_stage0_mass_solve refuses a non-mass F
    # there, since its completion never reads Ydot and would drop the term.
    F = inner(kappa * u_t, v) * dx
    if stiff:
        F += inner(grad(u), grad(v)) * dx
    G = inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.1), G=G)
    kwargs = {}
    if mutate:
        kwargs["monitor_callback"] = lambda ts, step, time, U: kappa.assign(
            float(kappa) * 1.5
        )
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            **ARK_SSP,
            "ts_ark_ssp_type": tableau,
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
            "ts_ark_ssp_fsal": fsal,
        },
        options_prefix="",
        **kwargs,
    )
    stepper = solver.ts.getPythonContext()
    solver.solve()
    return u.dat.data_ro.copy(), stepper


def test_fsal_reuses_the_previous_step_last_stage_derivative():
    """Ẏ_0 at step n+1 is the last stage derivative of step n, exactly.

    For a stiffly accurate tableau with ct[-1] == 1, that stage solved
    F(t^{n+1}, y^{n+1}, Ẏ) = 0 -- which is Ẏ_0's defining equation at the
    next step -- so reusing it skips a mass assembly, an LU refactorisation
    and a solve. PETSc's own ARKIMEX does this (FSAL_implicit).

    Asserts the reuse actually happens (a passing answer alone would not
    distinguish it from silently always taking the fresh path) AND that it
    changes nothing.
    """
    off, _ = _fsal_decay(False)
    on, stepper = _fsal_decay(True)
    assert stepper._fsal_possible
    assert stepper._fsal_hits == 9, (
        f"expected 9 reuses over 10 steps, got {stepper._fsal_hits} hits "
        f"and {stepper._fsal_misses} misses"
    )
    assert stepper._fsal_misses == 0
    assert np.abs(on - off).max() < 1e-13 * np.abs(off).max()


def test_fsal_rejects_a_candidate_after_the_form_changes_under_it():
    """The FSAL guard must be a residual test, not a state comparison."""
    off, _ = _fsal_decay(False, mutate=True)
    on, stepper = _fsal_decay(True, mutate=True)
    assert stepper._fsal_hits == 0, (
        f"reused a candidate whose mass coefficient had changed under it "
        f"({stepper._fsal_hits} hits)"
    )
    assert stepper._fsal_misses == 9
    assert np.array_equal(on, off), (
        "falling back to the fresh solve must reproduce the FSAL-disabled "
        f"result exactly; max difference {np.abs(on - off).max():.3e}"
    )


def test_fsal_is_not_attempted_for_a_purely_explicit_tableau():
    """_Ydot[-1] is never populated when no stage is implicitly solved.

    ssprk2 has At identically zero, so _solve_stage never runs and
    _Ydot[-1] holds whatever it was allocated with. The structural
    pre-filter must exclude this before any residual evaluation is spent on
    it -- and the reuse must not happen even though the tableau is otherwise
    stiffly accurate with ct[-1] == 1.
    """
    _, stepper = _fsal_decay(True, tableau="ssprk2", dt=0.001, stiff=False)
    assert not stepper._fsal_possible
    assert stepper._fsal_hits == 0
    assert stepper._fsal_misses == 0


def test_purely_explicit_tableau_refuses_a_non_mass_f_term():
    """A purely explicit tableau has nothing to integrate F's non-mass terms.

    ssprk2 has At identically zero, so _complete takes the Shu-Osher branch,
    which reads only x^n, the stage values Y and the explicit slopes
    M^-1 G -- never Ydot. Any H in F = M u̇ + H is therefore computed by
    _prepare_stage0_ydot into Ydot[0] and dropped from every step.
    """
    u, u_t, v = scalar_problem()
    # The u term belongs in G for this tableau; putting it in F is the error.
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type="ssprk2",
            ts_adapt_type="none",
            ts_time_step=0.1,
            ts_exact_final_time="matchstep",
        ),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="purely explicit"):
        solver.solve()
