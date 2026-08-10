"""The boundedness result that motivates COOL-193.

DG1 square-wave advection with a Zhang-Shu scaling limiter on every explicit
substage must hold 0 <= f <= 1 to machine precision. The reference
Butcher-form implementation gives -0.0099 / +1.0100, so the negative control
below must FAIL to bound -- otherwise this problem is not challenging and the
test proves nothing.
"""

from firedrake import *

import firedrake_ts

N = 40
CFL = 0.064
VELOCITY = 1.0


def _zhang_shu(V, V0):
    """Scale each cell about its mean so the cell lies in [0, 1].

    Zhang-Shu is a scaling limiter: it cannot repair an out-of-range cell
    MEAN, which is exactly why the stage value it acts on must already be a
    convex combination.

    Owns a dedicated scratch Function rather than borrowing the solution.
    The solution Function is also ``ctx._x``, which the TS callbacks write
    into; reusing it here would work only by accident of ordering.

    DG dof storage is cell-contiguous, so ``reshape(ncell, per_cell)`` gives
    one row per cell -- verified against the DG0 interpolant.
    """
    scratch = Function(V)
    mean = Function(V0)

    def limiter(vec):
        with scratch.dat.vec_wo as target:
            vec.copy(target)
        mean.project(scratch)
        fa = scratch.dat.data
        ma = mean.dat.data_ro
        ncell = len(ma)
        view = fa.reshape(ncell, len(fa) // ncell)
        for c in range(ncell):
            m = min(max(float(ma[c]), 0.0), 1.0)
            lo, hi = view[c].min(), view[c].max()
            theta = 1.0
            if hi > m:
                theta = min(theta, (1.0 - m) / (hi - m))
            if lo < m:
                theta = min(theta, m / (m - lo))
            theta = max(0.0, min(1.0, theta))
            view[c] = m + theta * (view[c] - m)
        with scratch.dat.vec_ro as source:
            source.copy(vec)

    return limiter


def _advect(stepper_parameters, limited):
    """DG1 upwind advection of a square wave on a periodic interval."""
    mesh = PeriodicUnitIntervalMesh(N)
    V = FunctionSpace(mesh, "DG", 1)
    V0 = FunctionSpace(mesh, "DG", 0)
    f = Function(V, name="f")
    f_t = Function(V)
    v = TestFunction(V)
    (x,) = SpatialCoordinate(mesh)
    f.interpolate(conditional(And(x > 0.25, x < 0.75), 1.0, 0.0))

    u = Constant(VELOCITY)
    n = FacetNormal(mesh)
    un = 0.5 * (u * n[0] + abs(u * n[0]))

    F = inner(f_t, v) * dx
    G = (
        f * u * v.dx(0) * dx
        - (un("+") * f("+") - un("-") * f("-")) * (v("+") - v("-")) * dS
    )

    dt = CFL / (N * VELOCITY)
    problem = firedrake_ts.DAEProblem(F, f, f_t, (0.0, 0.2), G=G)
    parameters = dict(
        stepper_parameters,
        ts_adapt_type="none",
        ts_time_step=dt,
        ts_exact_final_time="stepover",
    )
    solver = firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    )
    if limited:
        solver.set_stage_limiter(_zhang_shu(V, V0))
    solver.solve()
    data = f.dat.data_ro
    return float(data.min()), float(data.max())


ARK_SSP = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}
ARKIMEX = {"ts_type": "arkimex", "ts_arkimex_type": "2c"}


def test_shu_osher_form_holds_bounds_with_no_post_step_clamp():
    """The strong claim: bounds from the Shu-Osher induction alone."""
    lo, hi = _advect(ARK_SSP, limited=True)
    assert lo >= -1e-12, f"min = {lo}, expected >= 0 to machine precision"
    assert hi <= 1.0 + 1e-12, f"max = {hi}, expected <= 1 to machine precision"


def test_negative_control_butcher_form_does_not_bound():
    """Proves the test problem is actually challenging.

    If this ever starts passing, the bounds test above is not evidence for
    anything -- strengthen the problem before trusting it.
    """
    lo, hi = _advect(ARKIMEX, limited=False)
    assert lo < -1e-4 or hi > 1.0 + 1e-4, (
        f"the Butcher-form control stayed in bounds (min={lo}, max={hi}); "
        "this problem is not exercising the defect COOL-193 describes"
    )


def test_limiting_does_not_destroy_mass():
    """Zhang-Shu scales about the cell mean, so it must be mass-neutral.

    COOL-193 measured mass exact to 2e-16 with the limiter active -- the
    defect it describes is purely in boundedness. If this regresses, the
    limiter is not scaling about the mean and the bounds result above would
    be meaningless even if it passed.
    """
    mesh = PeriodicUnitIntervalMesh(N)
    V = FunctionSpace(mesh, "DG", 1)
    V0 = FunctionSpace(mesh, "DG", 0)
    f = Function(V, name="f")
    f_t = Function(V)
    v = TestFunction(V)
    (x,) = SpatialCoordinate(mesh)
    f.interpolate(conditional(And(x > 0.25, x < 0.75), 1.0, 0.0))
    initial_mass = assemble(f * dx)
    assert initial_mass > 0.0

    u = Constant(VELOCITY)
    n = FacetNormal(mesh)
    un = 0.5 * (u * n[0] + abs(u * n[0]))
    F = inner(f_t, v) * dx
    G = (
        f * u * v.dx(0) * dx
        - (un("+") * f("+") - un("-") * f("-")) * (v("+") - v("-")) * dS
    )

    dt = CFL / (N * VELOCITY)
    problem = firedrake_ts.DAEProblem(F, f, f_t, (0.0, 0.2), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_adapt_type="none",
            ts_time_step=dt,
            ts_exact_final_time="stepover",
        ),
        options_prefix="",
    )
    solver.set_stage_limiter(_zhang_shu(V, V0))
    solver.solve()

    final_mass = assemble(f * dx)
    assert abs(final_mass - initial_mass) < 1e-12 * abs(initial_mass), (
        f"mass drifted from {initial_mass} to {final_mass}"
    )
