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
    """Integrate u' = -u to tmax with -u explicit, under ARKSSP.

    ``matchstep``, not ``stepover``: with ``stepover`` every accuracy and order
    assertion built on this helper silently depends on ``dt`` dividing ``tmax``
    in FLOATING POINT, which dt=0.1 does not -- 10 * 0.1 is
    0.9999999999999999, so an eleventh step runs and the reported error is the
    overshoot to t = 1.1, not method error. Measured at dt=0.1/0.05 under
    ``stepover``: order ratios 5.76 for imex_euler (design 1) and 215.8 for
    ssprk2 (design 2); under ``matchstep``, 2.04 and 4.16. The old dt~1e-3
    tunings hid this rather than avoiding it.
    """
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
    """``-ts_type python -ts_python_type`` resolves, and setUp is re-runnable.

    setUp may run more than once against the same stepper instance (a TS may be
    re-set-up, e.g. after an option change); the previous stage-0 mass ``KSP``
    -- built fresh each call by _setup_stage0_mass_solve -- must be destroyed
    rather than leaked.

    A destroyed ``PETSc.KSP``'s handle reads back as 0; calling any method on it
    beyond that would itself be unsafe (a destroyed PETSc object's methods are
    not guaranteed to fail cleanly), so this checks the handle only, not further
    behaviour of the stale object.

    The first three assertions were a separate "the stepper is selectable and
    sets up" test, running a second solve to reach the same point this one
    already stands at. Selectability is in any case implied by every other test
    in this file, all of which drive this stepper; asserting it once, here, is
    what keeps the failure legible if -ts_python_type stops resolving.
    """
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
    """Halving dt must halve the error, and the answer must be right.

    Order plus magnitude at one dt pins the whole error curve, so the two were
    one test's worth of claim run as two: a separate advances-and-converges
    test made a third solve to assert exactly these two bounds on a value the
    order test already had in hand.

    dt=0.025/0.0125 rather than 2e-3/1e-3 -- 120 steps instead of 1500 -- with
    every assertion below left as it was written. Measured: ratio 2.011, and
    the fine run's error is 2.31e-3 against the abs=5e-3 this has always
    asserted, a 2.2x margin.

    That margin is why the step is not the 0.05/0.025 the ratio alone would
    allow: the ratio there is a healthy 2.021, but first-order error is ~0.19h,
    so the fine run lands at 4.65e-3 -- inside the 5e-3 bound by 8%, tight
    enough that an unrelated change could trip it on accuracy while the order
    it is really testing was never in question.
    """
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
    """Real work in the SNES for a tableau with At != 0, none for At == 0.

    The positive and negative halves were two tests carrying the same
    paragraph of rationale: query the SNES directly rather than
    ``ts.getSNESIterations()``, because PETSc only accumulates ``ts->snes_its``
    inside its OWN step drivers, so a TSPYTHON type that owns ``step()``
    reports 0 there unconditionally and petsc4py binds no setter. Asserting
    ``ts.getSNESIterations() == 0`` for ssprk2 therefore could not fail --
    measured 0 for esdirk_gamma5 too, which solves a stage every step -- and
    the stale query is exactly the mistake the pairing is here to prevent.

    A converged-in-zero-iterations solve is what a residual that is
    identically zero looks like, which is why the positive half checks the
    iteration count and not merely the converged reason.
    """
    solver, _ = _decay(tableau, dt=1e-2)
    if implicit:
        assert solver.snes.getIterationNumber() > 0
        assert solver.snes.getConvergedReason() > 0
    else:
        assert solver.snes.getIterationNumber() == 0


# The ARKSSP-vs-PETSc cross-check lives in
# test_shu_osher_matches_butcher_on_the_same_problem, which makes the same
# comparison against the same arkimex-2c reference at a tighter tolerance
# (1e-5 vs 1e-3) and a fraction of the step count. An imex_euler version of it
# at dt=1e-4 cost 42s -- 37% of the whole suite at the time, 20k timesteps
# across two solves -- to assert a 1e-3 bound on a difference of ~2e-5. Its
# content is already covered analytically against exp(-1) by
# test_imex_euler_advances_and_is_first_order above; at dt=1e-3 the imex_euler
# error is ~5e-4 against that 1e-3 bound, too close to retune rather than drop.


def test_shu_osher_error_is_unwrapped_with_the_actionable_numbers():
    """DAESolver.solve must unwrap the PETSc.Error and surface ShuOsherError.

    libpetsc4py wraps any exception raised inside a TSPYTHON callback as
    PETSc.Error (code 101, PETSC_ERR_PYTHON). DAESolver.solve unwraps it back
    to the original ShuOsherError so the message -- naming both the
    offending r and the radius of absolute monotonicity R(A,b) -- reaches the
    caller directly instead of being hidden behind an opaque error code.

    Subsumes a separate rejection test that made the identical call and
    asserted only the "exceeds the radius" match, which is kept here so
    nothing is lost: refusing an over-large radius is the whole point, since
    silently losing the SSP guarantee is the failure mode we must not have.
    """
    from firedrake_ts.tableaux import ShuOsherError

    with pytest.raises(ShuOsherError) as excinfo:
        _decay("ssprk2", dt=0.1, extra={"ts_ark_ssp_radius": 5.0})
    message = str(excinfo.value)
    assert "exceeds the radius" in message
    assert "r = 5.0" in message
    assert "R(A,b)" in message


def test_ssprk2_is_second_order():
    """Heun in Shu-Osher form must show design order 2.

    dt=0.05/0.025 rather than 4e-3/2e-3: 45 steps instead of 750, measured
    ratio 4.077 against the same 3.4-4.6 band.
    """
    _, coarse = _decay("ssprk2", dt=0.05)
    _, fine = _decay("ssprk2", dt=0.025)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"


def _esdirk_converge(build_F, u0, dt, tmax=1.0):
    """One esdirk_gamma5 run of a scalar problem in F, with G = None.

    The five convergence helpers this replaces were 31-40 line copies of the
    same block -- mesh, P1 space, u/u_t/v, initial value, DAEProblem, DAESolver
    with a byte-identical six-key parameter dict, solve, return u[0] --
    differing only in the initial value and the UFL form. ``build_F`` is called
    as ``build_F(u, u_t, v, time)``; a ``time`` Constant is always created and
    always passed to DAEProblem, which is harmless for the forms that ignore it.

    ``matchstep`` rather than ``stepover`` so the comparison against the exact
    solution is not confounded by stepover's overshoot past t = tmax.
    """
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


# Every case below is a regression test for one member of the Ydot_0 defect
# family, and the four are NOT interchangeable -- each defeats a guard that the
# others pass. Keeping all four is the point; only the boilerplate is shared.
#
# purely_implicit -- u' = -u with -u INSIDE F, G = None.
#   The class of problem ARKSSP exists to serve: a nontrivial implicit
#   operator, with esdirk_gamma5's explicit first stage (At[0, 0] == 0)
#   exercised against a real dF/du. Y_0 = y_n, so Ydot_0 must be evaluated
#   from F(t^n, y_n, 0) rather than assumed zero: it is read with nonzero
#   weight in _build_offset's At[1, 0], evaluatestep's bt[0] and
#   interpolate's d[0]. Left unevaluated (Ydot_0 == 0, which is what
#   sol.duplicate() happens to leave it at) the error is FLAT in dt -- it
#   converges to the wrong limit rather than shrinking like h^2.
#
# nonpolynomial_source -- u' = exp(-t) (u(0) = 0), source INSIDE F, G = None.
#   dF/du is structurally zero everywhere, but F(t, y, 0) != 0. A guard that
#   conflates "dF/du is structurally zero" with "Ydot_0 = 0 exactly" gets this
#   wrong -- those are not the same condition, and dF/du == 0 says nothing
#   about F(t, y, 0). exp(-t) rather than a constant so a real O(h^2)
#   truncation error remains to measure (see the separate constant-source test
#   below for why that one cannot be a ratio test).
#   Exact: u(t) = 1 - exp(-t), u(1) = 1 - exp(-1).
#
# state_dependent_mass -- (1 + u) u' = 1 (u(0) = 0), G = None.
#   M = dF/du_t = (1 + u) v depends on the state itself: the variable
#   density/porosity/saturation class. A mass matrix cached once at setUp (the
#   initial condition u = 0) keeps solving Ydot_0 against M(0) forever however
#   far u has moved -- flat in dt, silent, no exception. Measured on the
#   unfixed code (mass reused from ctx._rhs_projection_mass_matrix, assembled
#   once): errors 6.9e-2, 7.2e-2, 7.4e-2, 7.5e-2, i.e. ratios 0.96-0.99.
#   Exact: (1 + u) du = dt, so u + u^2/2 = t, u(t) = sqrt(1 + 2t) - 1.
#
# time_dependent_mass -- (1 + t) u' = 1 (u(0) = 0), G = None.
#   M = (1 + t) v depends on time but NOT the state, so dM/du is structurally
#   zero. That distinction is the whole point, and why this case cannot be
#   merged away in favour of the one above: a reassembly guarded on
#   dM/du != 0 skips every reassembly here and keeps inverting M(t^0) = 1,
#   while passing the M(u) test that motivated the guard. Measured with that
#   guard in place: ratios 0.958/0.979/0.990, against arkimex's
#   4.011/4.006/4.003 on the same problem. Regression test for 29d335e. No
#   structural predicate on the form closes this in general either, since M
#   may close over a mutable auxiliary Function -- hence unconditional
#   reassembly. G is None, so this exercises the stage-0 mass solve alone, not
#   the RHS projection, which test_imex covers separately for both kinds of
#   non-constant mass.
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
    """Design order 2 on each Ydot_0 defect case. See _CONVERGENCE_CASES above.

    Every one of these was flat in dt on the code that motivated it, so the
    order ratio is the assertion that distinguishes fixed from broken -- an
    exactness check would not. dt = 0.1 ... 0.0125 on ODE-exact problems is
    the asymptotic regime, not resolution-limited, and the 3.4-4.6 band
    excludes both order 1 (ratio 2) and order 3 (ratio 8).
    """
    errors = [abs(_esdirk_converge(build_F, u0, dt) - exact) for dt in _DTS]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"


def test_esdirk_gamma5_converges_on_a_constant_source():
    """u' = 1 with the source INSIDE F: asserted EXACT, deliberately not a ratio.

    Same defect class as the nonpolynomial_source case above (dF/du
    structurally zero while F(t, y, 0) != 0), but kept as its own test rather
    than folded into the parametrized set, because the assertion is
    fundamentally different and folding it in for symmetry would weaken it.

    Any consistent (order >= 1) Runge-Kutta method reproduces a
    constant-coefficient ODE exactly: the local truncation error involves
    derivatives of u beyond the first, all zero for a linear exact solution.
    So halving dt does not systematically shrink an already-zero truncation
    error, and a ratio test here would be measuring solver-tolerance noise.
    Before the fix the error is flat at u(1) = bt[0] = 39/125 = 0.312, not
    noise-small -- that gap is what distinguishes fixed from broken here.
    """
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
    """``Ẏ_0 = -M^-1 F(t^n, y^n, 0)`` is exact only when F is affine in u̇.

    ``F = inner(u_t*u_t - Constant(1.0), v)*dx`` (i.e. ``u̇^2 = 1``) has a
    nonzero d^2F/du̇^2, so that formula is one Newton step from zero, not
    the true root -- silently wrong (measured: u(1) = -0.308 against an
    exact 1.0) rather than merely inaccurate. setUp must refuse this
    outright instead of handing back a plausible-looking wrong answer.
    """
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
    """A stage that never converges must stop at one of the two limits, cleanly.

    Either way it must be a ConvergenceError naming the original cause -- not a
    raw ``petsc4py.PETSc.Error``, which is what PETSc's own ``TSStep()``
    (``ts.c``) raises in C via TS_DIVERGED_STEP_REJECTED + errorifstepfailed if
    ``step()`` merely sets the converged reason and returns instead of raising
    itself. The two limits were two tests with the same body and the same pair
    of assertions, differing only in which limit they armed:

    ``rejections_exhausted`` -- ts_max_step_rejections=2 is hit first.

    ``step_floor_reached`` -- with rejections unlimited (-1), h must stop at
    ts_adapt_dt_min rather than grind all the way to 0.0, which would make
    ``_solve_stage``'s ``self._shift = 1 / (h * tab.At[i, i])`` infinite and
    fail in a way that has nothing to do with the original divergence.
    ts_adapt_dt_min is set well above PETSc's own default floor (1e-20) purely
    so the loop reaches it in a handful of *0.25 shrinks rather than ~60; the
    mechanism does not depend on which floor.
    """
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
    """The Shu-Osher path and PETSc's Butcher-form TSRK must agree.

    test_tableaux.py establishes algebraically, at machine precision, that P
    and q reproduce the Butcher stage map. What is left for an end-to-end run
    is that this STEPPER implements the recursion those coefficients describe,
    against an independent implementation of an independent order-2 tableau.

    dt=4e-3, not 1e-3: the difference scales like h^2, so cutting the step
    count 4x raises it from 3.16e-8 to 5.07e-7 against the same 1e-5 bound --
    16x tighter relative to the signal it is bounding, for a quarter of the
    work. This was the most expensive test in the suite at 1.86s.
    """
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


def test_frozen_component_survives_the_implicit_solve():
    """A limiter's change to an explicitly-governed row must not be undone.

    This is the defect COOL-193 measured: the implicit stage equation
    Ydot_i = (Y_i - Z_i)/(h At_ii) has no slot for a modified value, so the
    solve pulls Y_i back to Z_i and the correction re-enters later stages
    scaled by At_ji/At_ii.
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

    # Row 0: mass only -> explicitly governed, freezable.
    # Row 1: mass + diffusion -> implicit acts, not freezable.
    F = inner(adot, va) * dx + inner(bdot, vb) * dx + inner(grad(b), grad(vb)) * dx
    G = -inner(a, va) * dx - inner(b, vb) * dx

    # Dedicated scratch: w is also ctx._x, which the TS callbacks write into.
    scratch = Function(W)
    CLAMP = 0.5
    drift = []
    calls = [0]

    def clamping_limiter(vec):
        """Force the explicitly-governed row to a value that is different
        on every call (CLAMP + 0.1 * call count), not the same constant
        every stage. A stale pin left over from the wrong stage is
        otherwise numerically indistinguishable from a correct one: this
        is the exact blind spot that hid the aliasing bug this test was
        originally written to catch (a corrupted warm-start guess that
        reverted a frozen row to the *previous* stage's value survived
        undetected as long as every stage clamped to the same number).
        """
        with scratch.dat.vec_wo as target:
            vec.copy(target)
        scratch.sub(0).assign(CLAMP + 0.1 * calls[0])
        calls[0] += 1
        with scratch.dat.vec_ro as source:
            source.copy(vec)

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.02), G=G)
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
    ctx.set_stage_limiter(clamping_limiter)

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


# "Nothing is frozen when every row has an implicit operator" had its own
# 35-line test, running a solve on a two-field all-diffusive problem to assert
# `not ctx._frozen_rows`. Both halves of that are asserted elsewhere on the
# same form: test_singular_mass.py's test_field_partitions[both_differential]
# pins explicitly_governed_fields() == () on exactly this F, and
# test_rung3_pde_with_multiplier_runs[arkssp] drives a two-field space with no
# freezable row through a full ARKSSP solve -- so field_rows()'s empty-tuple
# path is exercised end to end there, on a mixed space, which is the condition
# the deleted test's docstring identified as its reason to exist.


@pytest.mark.parametrize("registered", ["before_setup", "after_setup"])
def test_limiter_on_an_implicit_component_is_rejected(registered):
    """Limiting a component with an implicit operator is unsound; say so.

    Both registration orders, because they are two different call sites of
    ``_check_limiter_soundness`` and only one of them fires from setUp:

    ``before_setup`` -- the limiter is registered on a stepper that has never
    been set up, so the check runs from setUp's own call during solve().

    ``after_setup`` -- ``set_stage_limiter`` re-runs the check itself (guarded
    on ``self._tab`` already being set) specifically so a limiter registered
    after setUp cannot bypass it. Nothing else in the suite registers a limiter
    late, so without this case that guard could be deleted with the whole suite
    still green.
    """
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
    """Integrate ``kappa u' = -grad.grad u + u`` with a stiff implicit part.

    ``mutate=True`` installs a monitor that scales ``kappa`` -- a coefficient
    of the mass term -- by 1.5 after every accepted step. That is the case a
    state-based FSAL guard gets wrong: the monitor runs after the step, so at
    the start of the next step ``t`` and ``x`` are both exactly what they were
    when the previous step's last stage derivative was computed, while ``M``
    is not. Returns the final state and the stepper's hit/miss counters.
    """
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
    changes nothing: 9 hits over 10 steps -- step 1 has no candidate -- with
    the result agreeing to roundoff, measured 3.3e-16.
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
    """The FSAL guard must be a residual test, not a state comparison.

    A monitor scaling the mass coefficient after each step leaves (t, x)
    identical at the next step's start while M is different, so the previous
    last stage derivative no longer satisfies the equation. A guard built on
    cached (t_end, x_end) plus VecEqual -- the obvious implementation, and
    what "reuse when the state has not moved" suggests -- reuses a stale
    derivative on every one of these steps with nothing raising. That is the
    same shape as the five staleness defects this stepper has already had.

    The residual test in _try_fsal_stage0_ydot has no claim about the form in
    it, so it catches all nine: measured 0 hits, 9 misses, and a result
    bit-identical to the same solve with FSAL disabled.
    """
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
    _prepare_stage0_ydot into Ydot[0] and dropped from every step, silently
    integrating M u̇ = G instead of the F = G that was posed.

    Measured before the refusal: u̇ + u = 0 with u(0) = 1 posed with the u
    term in F rather than G returned 0.0 at t = 1 where 1.0 is correct --
    the stiff term dropped entirely, the initial condition decayed to
    nothing by the mass-only equation u̇ = 0 it actually solved. Wrong in a
    direction that looks like a plausible answer, which is why this is a
    refusal rather than a warning.
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
