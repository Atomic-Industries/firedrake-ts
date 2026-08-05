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
    """-ts_type python -ts_python_type must resolve and run setUp once."""
    solver, _ = _decay("imex_euler", dt=0.1)
    ctx = solver.ts.getPythonContext()
    assert type(ctx).__name__ == "ARKSSP"
    assert ctx.setup_calls == 1


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


def test_matches_the_arkimex_path():
    """ARKSSP and PETSc's own arkimex must agree beyond method error."""
    _, ours = _decay("imex_euler", dt=1e-4)
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
            "ts_time_step": 1e-4,
            "ts_exact_final_time": "stepover",
        },
        options_prefix="",
    ).solve()
    assert abs(ours - float(u.dat.data_ro[0])) < 1e-3


def test_radius_above_the_ssp_limit_is_rejected():
    """Silently losing the SSP guarantee is the failure mode we must not have."""
    from firedrake_ts.tableaux import ShuOsherError

    with pytest.raises(ShuOsherError, match="exceeds the radius"):
        _decay("ssprk2", dt=0.1, extra={"ts_ark_ssp_radius": 5.0})


def test_shu_osher_error_is_unwrapped_with_the_actionable_numbers():
    """DAESolver.solve must unwrap the PETSc.Error and surface ShuOsherError.

    libpetsc4py wraps any exception raised inside a TSPYTHON callback as
    PETSc.Error (code 101, PETSC_ERR_PYTHON). DAESolver.solve unwraps it back
    to the original ShuOsherError so the message -- naming both the
    offending r and the radius of absolute monotonicity R(A,b) -- reaches the
    caller directly instead of being hidden behind an opaque error code.
    """
    from firedrake_ts.tableaux import ShuOsherError

    with pytest.raises(ShuOsherError) as excinfo:
        _decay("ssprk2", dt=0.1, extra={"ts_ark_ssp_radius": 5.0})
    message = str(excinfo.value)
    assert "r = 5.0" in message
    assert "R(A,b)" in message


def test_ssprk2_is_second_order():
    """Heun in Shu-Osher form must show design order 2."""
    _, coarse = _decay("ssprk2", dt=4e-3)
    _, fine = _decay("ssprk2", dt=2e-3)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"


def test_ssprk2_needs_no_implicit_solve():
    """At is identically zero, so no stage may enter the SNES."""
    solver, _ = _decay("ssprk2", dt=1e-2)
    assert solver.ts.getSNESIterations() == 0


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
