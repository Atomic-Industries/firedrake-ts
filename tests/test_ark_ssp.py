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
