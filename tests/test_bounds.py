"""The boundedness result that motivates COOL-193.

DG1 square-wave advection with a Zhang-Shu scaling limiter on every explicit
substage must hold 0 <= f <= 1 to machine precision. The reference
Butcher-form implementation gives -0.0099 / +1.0100, so the negative control
below must FAIL to bound -- otherwise this problem is not challenging and the
test proves nothing.

Two things here are load-bearing, and getting either wrong silently
manufactures a passing test.

The element must place its dofs at the cell vertices. Firedrake's default
``FunctionSpace(mesh, "DG", 1)`` puts them at the interior Gauss-Legendre
points (in-cell fractions 0.2113/0.7887). Zhang & Shu's Theorem 2.2 requires
the point values at a quadrature rule *including the two cell ends* -- their
Remark 2.7: "any quadrature rule will work as long as it includes the two
cell ends" -- because the identity the proof rests on, mean = sum(w_a * f_a),
needs the traces as explicit terms. A linear function held inside [0, 1] at
+-1/sqrt(3) reaches mean +- (sqrt(3)/2)|delta| at the endpoints, so traces
hit -0.366/+1.366 with every dof perfectly in range; those traces enter the
upwind flux and carry the cell mean out of [0, 1]. ``variant="equispaced"``
puts the dofs at fractions 0/1 -- the vertices, which for k=1 *are* the
2-point Gauss-Lobatto points -- and then a plain nodal limiter is exactly the
theorem's limiter, in any dimension.

The mean must not be clamped. Writing ``m = min(max(mean, 0, 1))`` forces
every cell into [0, 1] regardless of what the stepper did, so the bounds
test passes while the means underneath are out of range. That is a
post-step clamp in per-stage clothing -- precisely what this design claims
to be unnecessary. Scale about the *true* mean; if it is out of range the
limiter cannot help, and the test should say so.
"""

from firedrake import *

import firedrake_ts

N = 40
CFL = 0.064
VELOCITY = 1.0


def _zhang_shu(V, V0, mean_range=None):
    """Scale each cell about its mean so the cell lies in [0, 1].

    Zhang-Shu is a scaling limiter: it cannot repair an out-of-range cell
    MEAN, which is exactly why the stage value it acts on must already be a
    convex combination.

    Owns a dedicated scratch Function rather than borrowing the solution.
    The solution Function is also ``ctx._x``, which the TS callbacks write
    into; reusing it here would work only by accident of ordering.

    DG dof storage is cell-contiguous, so ``reshape(ncell, per_cell)`` gives
    one row per cell -- verified against the DG0 interpolant.

    :arg mean_range: optional list appended with ``(min, max)`` of the cell
        means seen on every call. Zhang-Shu can only scale a cell TOWARD its
        mean, never repair a mean that is already out of range -- so this is
        the mechanistic claim the bounds result rests on, not merely a
        property of the trace values the flux happens to see. Recording it
        here (rather than re-deriving it after the fact) is what would catch
        a reinstated clamp on ``m``, which would otherwise make the mean
        invisible to every assertion downstream.
    """
    scratch = Function(V)
    mean = Function(V0)

    def limiter(vec):
        with scratch.dat.vec_wo as target:
            vec.copy(target)
        mean.project(scratch)
        fa = scratch.dat.data
        ma = mean.dat.data_ro
        if mean_range is not None:
            mean_range.append((float(ma.min()), float(ma.max())))
        ncell = len(ma)
        view = fa.reshape(ncell, len(fa) // ncell)
        for c in range(ncell):
            m = float(ma[c])  # NOT clamped -- see the module docstring
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


def _advect(stepper_parameters, limited, mean_range=None):
    """DG1 upwind advection of a square wave on a periodic interval."""
    mesh = PeriodicUnitIntervalMesh(N)
    # variant="equispaced" is REQUIRED, not cosmetic: see the module docstring.
    V = FunctionSpace(mesh, "DG", 1, variant="equispaced")
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
        solver.set_stage_limiter(_zhang_shu(V, V0, mean_range=mean_range))
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
    mean_range = []
    lo, hi = _advect(ARK_SSP, limited=True, mean_range=mean_range)
    assert lo >= -1e-12, f"min = {lo}, expected >= 0 to machine precision"
    assert hi <= 1.0 + 1e-12, f"max = {hi}, expected <= 1 to machine precision"

    # The mechanistic claim, not just the trace it produces: Zhang-Shu can
    # only scale a cell TOWARD its mean, never repair a mean that is already
    # out of [0, 1]. If the mean ever left that range, the bounds above
    # could only hold as an artifact of clamping ``m`` -- a post-step clamp
    # in per-stage clothing, which this design forbids -- not as a property
    # the mechanism actually earned. Asserting on the recorded means is what
    # would catch a reinstated clamp; the trace-only asserts above would not.
    assert mean_range, "the limiter never ran; nothing was recorded"
    mean_lo = min(r[0] for r in mean_range)
    mean_hi = max(r[1] for r in mean_range)
    assert mean_lo >= -1e-12, f"cell mean min = {mean_lo}, expected >= 0"
    assert mean_hi <= 1.0 + 1e-12, f"cell mean max = {mean_hi}, expected <= 1"


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
    # variant="equispaced" is REQUIRED, not cosmetic: see the module docstring.
    V = FunctionSpace(mesh, "DG", 1, variant="equispaced")
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
