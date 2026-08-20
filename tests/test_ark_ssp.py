"""The ARKSSP TSPYTHON stepper."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts

EXACT = np.exp(-1.0)  # solution of u' = -u at t = 1

PYTHON_STEPPER = "firedrake_ts.ark_ssp.ARKSSP"


def _decay(tableau, dt=1e-3, tmax=1.0, extra=None):
    """Integrate u' = -u to tmax with -u explicit, under ARKSSP."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx
    G = -inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax), G=G)
    parameters = {
        "ts_type": "python",
        "ts_python_type": PYTHON_STEPPER,
        "ts_ark_ssp_type": tableau,
        "ts_adapt_type": "none",
        "ts_time_step": dt,
        "ts_exact_final_time": "stepover",
    }
    parameters.update(extra or {})
    solver = firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


def test_stepper_is_selectable_and_sets_up():
    """-ts_type python -ts_python_type must resolve and run setUp."""
    from firedrake_ts.ark_ssp import ARKSSP

    solver, _ = _decay("imex_euler", dt=0.1)
    ctx = solver.ts.getPythonContext()
    assert isinstance(ctx, ARKSSP)
    # Asserts setUp ran by an artifact only setUp produces, rather than by a
    # call counter that existed solely to be read here.
    assert ctx._mass_ksp is not None
    assert ctx._P is not None and ctx._q is not None


def test_setup_rerun_destroys_the_stale_stage0_mass_ksp():
    """setUp may run more than once against the same stepper instance (a TS
    may be re-set-up, e.g. after an option change); the previous stage-0
    mass ``KSP`` -- built fresh each call by _setup_stage0_mass_solve -- must
    be destroyed rather than leaked.

    A destroyed ``PETSc.KSP``'s handle reads back as 0; calling any method
    on it beyond that would itself be unsafe (a destroyed PETSc object's
    methods are not guaranteed to fail cleanly), so this checks the handle
    only, not further behaviour of the stale object.
    """
    from firedrake import dmhooks

    solver, _ = _decay("esdirk_gamma5", dt=0.1)
    ctx = solver.ts.getPythonContext()
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


def test_imex_euler_advances_and_converges():
    """The explicit part must reach the state, and the answer must be right."""
    _, value = _decay("imex_euler", dt=1e-3)
    assert abs(value - 1.0) > 0.1, "solution never left its initial condition"
    # First order: error ~ C h, so ~1e-3 at h=1e-3. Generous bound.
    assert abs(value - EXACT) < 5e-3, f"u(1) = {value}, expected {EXACT}"


def test_imex_euler_is_first_order():
    """Halving dt must halve the error."""
    _, coarse = _decay("imex_euler", dt=2e-3)
    _, fine = _decay("imex_euler", dt=1e-3)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 1.7 < ratio < 2.3, f"observed order ratio {ratio}, expected ~2"


def test_stage_solves_do_work():
    """Stage solves must do real work; a null residual would converge in zero.

    Query the SNES directly rather than ts.getSNESIterations(): PETSc only
    accumulates ts->snes_its inside its own step drivers, so a TSPYTHON type
    that owns step() always reports zero there, and petsc4py binds no setter.
    """
    solver, _ = _decay("imex_euler", dt=1e-2)
    assert solver.snes.getIterationNumber() > 0
    assert solver.snes.getConvergedReason() > 0


# The ARKSSP-vs-PETSc cross-check lives in
# test_shu_osher_matches_butcher_on_the_same_problem, which makes the same
# comparison against the same arkimex-2c reference at a tighter tolerance
# (1e-5 vs 1e-3) and a tenth of the step count. An imex_euler version of it
# at dt=1e-4 cost 42s -- 37% of the whole suite, 20k timesteps across two
# solves -- to assert a 1e-3 bound on a difference of ~2e-5. Its content is
# already covered analytically against exp(-1) by
# test_imex_euler_advances_and_converges and _is_first_order above; at
# dt=1e-3 the imex_euler error is ~5e-4 against that 1e-3 bound, too close to
# retune rather than drop.


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
    """Heun in Shu-Osher form must show design order 2."""
    _, coarse = _decay("ssprk2", dt=4e-3)
    _, fine = _decay("ssprk2", dt=2e-3)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"


def test_ssprk2_needs_no_implicit_solve():
    """At is identically zero, so no stage may enter the SNES.

    Queries the SNES directly, for the same reason
    ``test_stage_solves_do_work`` does: ``ts.getSNESIterations()`` is
    unconditionally 0 for a TSPYTHON type that owns ``step()`` (PETSc only
    accumulates ``ts->snes_its`` inside its own step drivers), so asserting
    it reads 0 here could not fail and did not test the claim -- measured 0
    for esdirk_gamma5 too, which solves a stage at every step.
    """
    solver, _ = _decay("ssprk2", dt=1e-2)
    assert solver.snes.getIterationNumber() == 0


def _pure_implicit_decay(dt, tmax=1.0):
    """Integrate u' = -u with -u INSIDE F (implicit), G = None.

    This is the class of problem ARKSSP exists to serve: a nontrivial
    implicit operator, with esdirk_gamma5's explicit first stage
    (At[0, 0] == 0) actually exercised against a real dF/du. ``matchstep``
    rather than ``stepover`` so the comparison against ``exp(-1)`` is not
    confounded by stepover's overshoot past t = 1.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
            "ts_ark_ssp_type": "esdirk_gamma5",
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


def test_esdirk_gamma5_converges_on_a_purely_implicit_problem():
    """The stepper must converge, at design order, with the operator
    entirely inside F and no G at all.

    esdirk_gamma5's first stage is explicit (At[0, 0] == 0), so Y_0 = y_n
    and Ydot_0 must be evaluated from F(t^n, y_n, 0) rather than assumed
    zero: read with nonzero weight in _build_offset's At[1, 0], in
    evaluatestep's bt[0], and in interpolate's d[0]. Left unevaluated
    (Ydot_0 == 0, the value sol.duplicate() happens to leave it at), the
    measured error is FLAT in dt -- it converges to the wrong limit, not
    to exp(-1) -- rather than shrinking like h^2.
    """
    errors = [
        abs(_pure_implicit_decay(dt) - EXACT) for dt in (0.1, 0.05, 0.025, 0.0125)
    ]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"


def _constant_source(dt, tmax=1.0):
    """Integrate u' = 1 (u(0) = 0) with the source written INSIDE F, G = None.

    dF/du is structurally zero everywhere here (F has no dependence on u at
    all), but F(t, y, 0) = -1 != 0 -- the state-independent Constant(1.0)
    term makes it so. A guard that skips the stage-0 mass solve whenever
    dF/du is zero (mistaking that for "F does not depend on the state, so
    Ydot_0 = 0 exactly") gets this wrong: Ydot_0 is stuck at 0, and the step
    is flat in dt at u(1) = bt[0] = 39/125 = 0.312 instead of converging to
    the exact answer, 1.0.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(0.0)
    F = inner(u_t, v) * dx - inner(Constant(1.0), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
            "ts_ark_ssp_type": "esdirk_gamma5",
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


def test_esdirk_gamma5_converges_on_a_constant_source():
    """A constant-in-time source in F, with G = None, must not reproduce
    the Ydot_0-never-computed defect.

    Regression test for a guard that conflated "dF/du is structurally
    zero" with "Ydot_0 = 0 exactly": those are NOT the same condition.
    dF/du == 0 says nothing about F(t, y, 0), and a state-independent
    source term (Constant(1.0) here) makes F(t, y, 0) nonzero while
    dF/du stays zero. See test_esdirk_gamma5_converges_on_a_purely_implicit_
    problem for the general (state-dependent) case this guard already
    covered.

    u' = 1 is checked against near-machine-precision closeness to 1.0,
    NOT an order-2 ratio: any consistent (order >= 1) Runge-Kutta method
    reproduces a constant-coefficient ODE exactly (its local truncation
    error involves derivatives of u beyond the first, all zero for a
    linear exact solution), so halving dt does not systematically shrink
    an already-zero truncation error -- the residual left is solver-
    tolerance noise, not the O(h^2) shrinking
    test_esdirk_gamma5_converges_on_a_nonpolynomial_source below checks.
    Before the fix, by contrast, the error is flat at 0.312, not
    noise-small: that gap is exactly what distinguishes "fixed" from
    "broken" here.
    """
    for dt in (0.1, 0.05, 0.025, 0.0125):
        value = _constant_source(dt)
        assert abs(value - 1.0) < 1e-8, (
            f"u(1) = {value} at dt={dt}, expected ~1.0 to near machine "
            "precision (u' = const is exact for any consistent RK method)"
        )


def _nonpolynomial_source(dt, tmax=1.0):
    """Integrate u' = exp(-t) (u(0) = 0) with the source INSIDE F, G = None.

    Same defect class as _constant_source (dF/du structurally zero,
    F(t, y, 0) != 0), but with a source that is NOT a polynomial in t --
    unlike u' = 1, no finite-order quadrature reproduces exp(-t) exactly, so
    a real, measurable O(h^2) local truncation error survives and halving
    dt should quarter it, giving actual evidence of design-order
    convergence rather than the exact-to-round-off answer _constant_source
    gets for a strictly polynomial right-hand side.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(0.0)
    time = Constant(0.0)
    F = inner(u_t, v) * dx - inner(exp(-time), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax), time=time)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
            "ts_ark_ssp_type": "esdirk_gamma5",
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


def test_esdirk_gamma5_converges_on_a_nonpolynomial_source():
    """A non-polynomial, purely time-dependent source in F, with G = None,
    must converge to the exact answer at design order.

    Same guard defect as test_esdirk_gamma5_converges_on_a_constant_source,
    but exp(-t) in place of Constant(1.0) leaves a real O(h^2) truncation
    error to measure, so this is the test that actually exercises "halving
    dt quarters the error" for this defect class -- exact solution
    u(t) = 1 - exp(-t), u(1) = 1 - exp(-1).
    """
    exact = 1.0 - np.exp(-1.0)
    errors = [
        abs(_nonpolynomial_source(dt) - exact) for dt in (0.1, 0.05, 0.025, 0.0125)
    ]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"


def _state_dependent_mass(dt, tmax=1.0):
    """Integrate ``(1 + u) u' = 1`` (``u(0) = 0``), G = None.

    The mass matrix ``M = dF/du̇ = (1 + u) v`` depends on the state ``u``
    itself -- exactly the class of problem (variable density, porosity,
    saturation, ...) that a mass matrix cached once at ``setUp`` time (the
    initial condition, ``u = 0``) gets wrong on every step after the
    first: ``_prepare_stage0_ydot`` would keep solving ``Ẏ_0`` against
    ``M(0)`` forever, no matter how far ``u`` had actually moved, giving a
    step that is FLAT in ``dt`` rather than converging -- silent, no
    exception. Exact solution: separating variables, ``(1 + u) du = dt``,
    so ``u + u^2/2 = t`` and ``u(t) = sqrt(1 + 2t) - 1``;
    ``u(1) = sqrt(3) - 1``.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(0.0)
    F = inner((1.0 + u) * u_t, v) * dx - inner(Constant(1.0), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
            "ts_ark_ssp_type": "esdirk_gamma5",
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


def test_esdirk_gamma5_converges_on_a_state_dependent_mass_matrix():
    """The mass matrix M = dF/du̇ depends on u itself, so a Ẏ_0 solved
    against a stale M(y^0) -- cached once, at setUp -- is flat in dt
    rather than converging. See _state_dependent_mass's docstring and
    _reassemble_stage0_mass, which _prepare_stage0_ydot calls
    unconditionally to reassemble the mass matrix at (t^n, y^n) every step
    precisely to avoid this. Not gated on a structural test for state
    dependence: such a test passes here but misses a purely time-dependent
    M -- see test_imex's
    test_rhs_projection_operator_is_assembled_at_the_stage_state.

    Measured on the unfixed code (mass matrix reused unconditionally from
    ctx._rhs_projection_mass_matrix, assembled once): errors of
    6.9e-2, 7.2e-2, 7.4e-2, 7.5e-2 at dt = 0.1, 0.05, 0.025, 0.0125 --
    ratios ~0.96-0.99, i.e. converging to the WRONG limit, not shrinking.
    """
    exact = np.sqrt(3.0) - 1.0
    errors = [
        abs(_state_dependent_mass(dt) - exact) for dt in (0.1, 0.05, 0.025, 0.0125)
    ]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"


def test_esdirk_gamma5_refuses_f_nonlinear_in_udot():
    """``Ẏ_0 = -M^-1 F(t^n, y^n, 0)`` is exact only when F is affine in u̇.

    ``F = inner(u_t*u_t - Constant(1.0), v)*dx`` (i.e. ``u̇^2 = 1``) has a
    nonzero d^2F/du̇^2, so that formula is one Newton step from zero, not
    the true root -- silently wrong (measured: u(1) = -0.308 against an
    exact 1.0) rather than merely inaccurate. setUp must refuse this
    outright instead of handing back a plausible-looking wrong answer.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(0.0)
    F = inner(u_t * u_t - Constant(1.0), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type="esdirk_gamma5",
            ts_adapt_type="none",
            ts_time_step=0.1,
            ts_exact_final_time="matchstep",
        ),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="nonlinear in"):
        solver.solve()


def _flaky_stage_solver(ctx, n_failures):
    """Wrap ``ctx._solve_stage`` to raise on its first ``n_failures`` calls.

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
        if calls[0] <= n_failures:
            raise ConvergenceError("induced stage divergence")
        return original(ts, tab, h, i)

    return flaky, calls, original


def _diverging_problem(max_step_rejections, dt=0.5, extra=None):
    """A trivial implicit problem, set up only to have its stage solver
    replaced -- the equation itself never needs to be hard to solve.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, dt))
    parameters = dict(
        ARK_SSP,
        ts_ark_ssp_type="esdirk_gamma5",
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


def test_snes_divergence_exhausts_retries_and_raises_convergence_error():
    """A persistent stage-SNES divergence must exhaust ts_max_step_rejections
    and raise a clean ConvergenceError naming the cause -- not surface a raw
    petsc4py.PETSc.Error, which is what PETSc's own TSStep() (ts.c) raises in
    C, via TS_DIVERGED_STEP_REJECTED + errorifstepfailed, if step() merely
    sets the converged reason and returns instead of raising itself.
    """
    solver = _diverging_problem(max_step_rejections=2)
    ctx = solver.ts.getPythonContext()

    def always_diverges(ts, tab, h, i):
        raise ConvergenceError("induced persistent stage divergence")

    original = ctx._solve_stage
    ctx._solve_stage = always_diverges
    try:
        with pytest.raises(ConvergenceError, match="rejected") as excinfo:
            solver.solve()
    finally:
        ctx._solve_stage = original
    assert "induced persistent stage divergence" in str(excinfo.value), (
        "the exhaustion error must name the cause of the last rejection"
    )


def test_snes_divergence_with_unlimited_rejections_stops_at_a_step_floor():
    """A persistently diverging stage, with ts_max_step_rejections unlimited
    (-1), must raise a clean ConvergenceError once h shrinks below
    ts_adapt_dt_min -- not grind h all the way to 0.0, which would make
    _solve_stage's self._shift = 1 / (h * tab.At[i, i]) infinite and fail in
    a way that has nothing to do with the original divergence.

    ts_adapt_dt_min is set well above PETSc's own default floor (1e-20)
    purely so the loop hits it in a handful of *0.25 shrinks rather than
    ~60, keeping the test fast; the mechanism being tested -- stop at the
    floor rather than underflow to zero -- does not depend on which floor.
    """
    solver = _diverging_problem(max_step_rejections=-1, extra={"ts_adapt_dt_min": 1e-3})
    ctx = solver.ts.getPythonContext()

    def always_diverges(ts, tab, h, i):
        raise ConvergenceError("induced persistent stage divergence")

    original = ctx._solve_stage
    ctx._solve_stage = always_diverges
    try:
        with pytest.raises(ConvergenceError, match="ts_adapt_dt_min") as excinfo:
            solver.solve()
    finally:
        ctx._solve_stage = original
    assert "induced persistent stage divergence" in str(excinfo.value), (
        "the floor error must still name the original cause"
    )
    assert solver.ts.getTimeStep() > 0.0, "h must not have underflowed to 0.0"


def test_shu_osher_matches_butcher_on_the_same_problem():
    """The Shu-Osher path and PETSc's Butcher-form TSRK must agree."""
    _, ours = _decay("ssprk2", dt=1e-3)
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
        solver_parameters={
            "ts_type": "arkimex",
            "ts_arkimex_type": "2c",
            "ts_adapt_type": "none",
            "ts_time_step": 1e-3,
            "ts_exact_final_time": "stepover",
        },
        options_prefix="",
    ).solve()
    assert abs(ours - float(u.dat.data_ro[0])) < 1e-5


def test_limiter_fires_once_per_stage():
    """The limiter must see every explicit substage, not just the last."""
    calls = []

    def counting_limiter(vec):
        calls.append(vec.norm())

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 0.05), G=-inner(u, v) * dx
    )
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
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


ARK_SSP = {"ts_type": "python", "ts_python_type": PYTHON_STEPPER}


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
            ARK_SSP,
            ts_ark_ssp_type="esdirk_gamma5",
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


def test_nothing_is_frozen_when_every_row_has_an_implicit_operator():
    """A genuinely mixed space where every row has diffusion: the branch
    the name describes. len(V) > 1 so _find_frozen_rows actually calls
    explicitly_governed_fields, rather than returning early on a
    single-field space where that call is never reached.
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

    F = (
        inner(adot, va) * dx
        + inner(grad(a), grad(va)) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    G = -inner(a, va) * dx - inner(b, vb) * dx
    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.02), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type="esdirk_gamma5",
            ts_adapt_type="none",
            ts_time_step=0.01,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    solver.solve()
    assert not solver.ts.getPythonContext()._frozen_rows


def test_limiter_on_an_implicit_component_is_rejected():
    """Limiting a component with an implicit operator is unsound; say so."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.02), G=-inner(u, v) * dx)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type="esdirk_gamma5",
            ts_adapt_type="none",
            ts_time_step=0.01,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    solver.ts.getPythonContext().set_stage_limiter(lambda vec: None)
    with pytest.raises(ValueError, match="implicit operator"):
        solver.solve()


def test_limiter_registered_after_setup_is_still_rejected():
    """The soundness guard must re-fire for a limiter registered late.

    set_stage_limiter re-runs _check_limiter_soundness itself (guarded on
    self._tab already being set) specifically so a limiter registered
    AFTER setUp has already run cannot silently bypass the check that
    fires from setUp's own call to it. Every other limiter-rejection test
    registers the limiter BEFORE the first solve() -- i.e. before setUp
    has run at all -- so none of them exercises this second call site;
    without a dedicated test, that guard could be deleted with nothing
    failing.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.02), G=-inner(u, v) * dx)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_ark_ssp_type="esdirk_gamma5",
            ts_adapt_type="none",
            ts_time_step=0.01,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    # No limiter yet: setUp must complete cleanly.
    solver.solve()
    ctx = solver.ts.getPythonContext()
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
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
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
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
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


def _time_dependent_mass(dt, tmax=1.0):
    """Integrate ``(1 + t) u' = 1`` (``u(0) = 0``), G = None.

    The mass matrix ``M = dF/du̇ = (1 + t) v`` depends on time but NOT on the
    state, so ``dM/du`` is structurally zero. That distinction is the whole
    point: a reassembly guarded on ``dM/du != 0`` -- which is what
    _prepare_stage0_ydot used to carry -- skips every reassembly here and
    keeps inverting ``M(t^0) = 1``, while passing the ``M(u)`` test that
    motivated the guard. No structural predicate on the form can close this
    in general either, since ``M`` may also close over a mutable auxiliary
    Function, which is why the reassembly is unconditional.

    Exact solution: ``du = dt / (1 + t)``, so ``u = ln(1 + t)`` and
    ``u(1) = ln 2``. G is None, so this exercises the stage-0 mass solve
    alone, not the RHS projection (which test_imex covers separately for
    both kinds of non-constant mass).
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(0.0)
    time = Constant(0.0)
    F = inner((1.0 + time) * u_t, v) * dx - inner(Constant(1.0), v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, tmax), time=time)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "python",
            "ts_python_type": PYTHON_STEPPER,
            "ts_ark_ssp_type": "esdirk_gamma5",
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


def test_esdirk_gamma5_converges_on_a_time_dependent_mass_matrix():
    """M depends on t but not on u, so a dM/du guard misses it entirely.

    Measured with the reassembly guarded on dM/du != 0: ratios
    0.958/0.979/0.990 -- converging to the wrong limit -- against arkimex's
    4.011/4.006/4.003 on the same problem. This is the regression test for
    the fix in 29d335e, which until now was verified only by hand.
    """
    exact = np.log(2.0)
    errors = [
        abs(_time_dependent_mass(dt) - exact) for dt in (0.1, 0.05, 0.025, 0.0125)
    ]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.4 < ratio < 4.6, f"observed order ratios {ratios}, expected ~4"
