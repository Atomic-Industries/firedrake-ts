"""DAESolver.step(): accepted-step-level driving loop (COOL-208)."""

from firedrake import *

import firedrake_ts

#: Fixed dt, adaptivity off -- makes a step()-driven run and a solve()-driven
#: run of the same problem path-identical, so they can be compared bit for
#: bit rather than to a numerical tolerance.
BDF2_FIXED = {
    "ts_type": "bdf",
    "ts_bdf_order": 2,
    "ts_time_step": 0.1,
    "ts_adapt_type": "none",
    "ts_exact_final_time": "stepover",
}


def _run_decay(step_driven, **extra_params):
    """Run ``u' = -u``, ``u(0) = 1`` via ``solve()`` or a manual ``step()`` loop.

    Reads the final value off ``solver.ts.getSolution()`` rather than the
    problem's own ``Function`` -- deliberately: ``DAESolver.step`` does not
    sync ``problem.u_restrict`` on every accepted step the way ``solve``
    does once at the end (see ``step``'s docstring), so ``ts.getSolution()``
    is the one read that means the same thing in both modes.

    :returns: ``(solver, final_u, accepted_flags)``, where ``accepted_flags``
        is the list of ``step()`` return values (empty when ``step_driven``
        is False).
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0))
    params = dict(BDF2_FIXED, **extra_params)
    solver = firedrake_ts.DAESolver(
        problem, solver_parameters=params, options_prefix=""
    )

    accepted = []
    if step_driven:
        while solver.ts.getTime() < problem.tspan[1]:
            accepted.append(solver.step())
    else:
        solver.solve()
    return solver, float(solver.ts.getSolution().getArray()[0]), accepted


def test_step_loop_matches_solve():
    """A manual ``while ... step()`` loop must reproduce ``solve()`` exactly.

    Fixed dt with adaptivity off makes both runs deterministic and
    path-identical, so this is bit-for-bit agreement, not a numerical
    tolerance: any divergence would mean ``step()``'s one-time setup (BC
    application, ``ts.setSolution``) does not actually reproduce what
    ``solve()`` does before its own ``ts.solve(work)`` call.
    """
    solver_solve, u_solve, _ = _run_decay(step_driven=False)
    solver_step, u_step, accepted = _run_decay(step_driven=True)

    assert accepted, "the loop never called step() -- the time span is wrong"
    assert all(accepted), (
        "every step here is expected to be accepted outright (fixed dt, "
        "generous tolerances); a False entry means ts_error_if_step_fails "
        "would have raised instead, so this should never fire"
    )
    assert solver_step.ts.getStepNumber() == solver_solve.ts.getStepNumber(), (
        "step() drove a different number of accepted steps than solve() "
        "took for the identical fixed-dt problem"
    )
    assert u_step == u_solve, (
        f"solve() gave u={u_solve!r}, the step() loop gave u={u_step!r} -- "
        "for a fixed-dt, non-adaptive BDF2 run these must match exactly, "
        "since step() is driving the identical TSStep() calls solve()'s "
        "own TSSolve() loop makes internally"
    )


def test_step_reports_rejection_without_raising():
    """``step()`` returns False on exhausted retries instead of raising.

    Mirrors test_adapt.py's ``test_exhausted_rejections_raises_convergence_error``,
    but for step()'s documented alternative to solve()'s exception-based
    failure reporting: with ``ts_error_if_step_fails=False``, a step sized
    to fail its one allowed attempt (``ts_max_reject=0``, an oversized
    initial guess against a tight tolerance) must come back as ``False``
    with a negative ``getConvergedReason()``, not raise.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0))
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "bdf",
            "ts_time_step": 0.9,
            "ts_adapt_type": "basic",
            "ts_adapt_dt_min": 1e-8,
            "ts_adapt_dt_max": 1.0,
            "ts_rtol": 1e-10,
            "ts_atol": 1e-12,
            "ts_max_reject": 0,
            "ts_error_if_step_fails": False,
        },
        options_prefix="",
    )

    accepted = solver.step()

    assert accepted is False
    assert solver.ts.getConvergedReason() < 0
