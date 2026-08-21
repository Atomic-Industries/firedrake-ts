"""The boundedness result that motivates the Shu-Osher timestepper.

DG1 square-wave advection with a Zhang-Shu scaling limiter on every explicit
substage must hold 0 <= f <= 1 to machine precision. The reference
Butcher-form implementation gives -0.0099 / +1.0100, so the negative control
below must FAIL to bound -- otherwise this problem is not challenging and the
test proves nothing.
"""

from collections import namedtuple

import numpy as np
from conftest import ARK_SSP_G5, ARKIMEX_2C
from firedrake import *

import firedrake_ts

N = 40
CFL = 0.064
VELOCITY = 1.0

#: What one advection run yields. ``initial_mass`` is measured before the solve
#: so the mass claim can be checked on the same run as the bounds claim rather
#: than by repeating it.
Advection = namedtuple("Advection", "lo hi f solver initial_mass")


def _zhang_shu(V, V0, mean_range=None, clipped=None):
    """Scale each cell about its mean so the cell lies in [0, 1].

    Zhang-Shu is a scaling limiter: it cannot repair an out-of-range cell
    MEAN, which is exactly why the stage value it acts on must already be a
    convex combination.

    :arg mean_range: optional list appended with ``(min, max)`` of the cell
        means seen on every call. Zhang-Shu can only scale a cell TOWARD its
        mean, never repair a mean that is already out of range -- so this is
        the mechanistic claim the bounds result rests on, not merely a
        property of the trace values the flux happens to see. Recording it
        here (rather than re-deriving it after the fact) is what would catch
        a reinstated clamp on ``m``, which would otherwise make the mean
        invisible to every assertion downstream.
    :arg clipped: optional single-element list incremented every time a cell
        is actually scaled (``theta < 1``). A limiter that never clips
        anything is consistent with a solution that never left the range the
        stepper handed it -- exactly the liveness gap the module docstring
        describes -- so this is exposed for the bounds test to check against.
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
        assert np.allclose(view.mean(axis=1), ma, atol=1e-10), (
            "the per-cell nodal average does not match the projected mean -- "
            "either the reshape is not grouping dofs by cell, or the element "
            "is no longer placing them where this limiter assumes"
        )
        for c in range(ncell):
            m = float(ma[c])  # NOT clamped -- see the module docstring
            lo, hi = view[c].min(), view[c].max()
            theta = 1.0
            if hi > m:
                theta = min(theta, (1.0 - m) / (hi - m))
            if lo < m:
                theta = min(theta, m / (m - lo))
            theta = max(0.0, min(1.0, theta))
            if clipped is not None and theta < 1.0:
                clipped[0] += 1
            view[c] = m + theta * (view[c] - m)
        with scratch.dat.vec_ro as source:
            source.copy(vec)

    return limiter


def _advect(stepper_parameters, limited, mean_range=None, clipped=None):
    """DG1 upwind advection of a square wave on a periodic interval.

    Returns an ``Advection``: the trace bounds, the solution Function itself
    (so a caller can check it against the exact translate), the solver (so a
    caller can check the accepted step count), and the pre-solve mass.
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
    initial_mass = float(assemble(f * dx))

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
        solver.set_stage_limiter(
            _zhang_shu(V, V0, mean_range=mean_range, clipped=clipped)
        )
    solver.solve()
    data = f.dat.data_ro
    return Advection(float(data.min()), float(data.max()), f, solver, initial_mass)


def test_shu_osher_form_holds_bounds_with_no_post_step_clamp():
    """The strong claim: bounds from the Shu-Osher induction alone."""
    mean_range = []
    clipped = [0]
    run = _advect(ARK_SSP_G5, limited=True, mean_range=mean_range, clipped=clipped)
    lo, hi, f, solver = run.lo, run.hi, run.f, run.solver
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

    # Liveness: the initial square wave, its exact translate at t=0.2, and a
    # stationary (unmoved) solution are all trivially bounded and trivially
    # mass-conserving, so none of the assertions above distinguish "bounded
    # because the Shu-Osher induction holds" from "bounded because nothing
    # moved" (or was over-limited to first-order FV). VELOCITY=1.0 for 0.2
    # time units on a periodic unit interval shifts the wave by exactly 0.2.
    mesh = f.function_space().mesh()
    (x,) = SpatialCoordinate(mesh)
    exact = Function(f.function_space()).interpolate(
        conditional(And(x > 0.45, x < 0.95), 1.0, 0.0)
    )
    l1_error = assemble(abs(f - exact) * dx)
    assert l1_error < 0.06, f"L1 error against the exact translate = {l1_error}."
    assert clipped[0] > 0, (
        "the limiter never actually scaled a cell (theta < 1 never "
        "occurred) -- consistent with a solution that never left the "
        "range the stepper handed it, which proves nothing about the "
        "Shu-Osher induction"
    )
    steps = solver.ts.getStepNumber()
    assert 120 <= steps <= 130, (
        f"expected ~126 accepted steps, got {steps} -- the run did not "
        "advance the way this test assumes"
    )

    # Mass neutrality. Zhang-Shu scales about the cell mean, so the limiter
    # must not move the mass at all
    assert run.initial_mass > 0.0
    final_mass = float(assemble(f * dx))
    assert abs(final_mass - run.initial_mass) < 1e-13 * abs(run.initial_mass), (
        f"mass drifted from {run.initial_mass} to {final_mass}"
    )


def test_negative_control_butcher_form_does_not_bound():
    """Proves the test problem is actually challenging.

    If this ever starts passing, the bounds test above is not evidence for
    anything -- strengthen the problem before trusting it.
    """
    run = _advect(ARKIMEX_2C, limited=False)
    assert run.lo < -1e-4 or run.hi > 1.0 + 1e-4, (
        f"the Butcher-form control stayed in bounds (min={run.lo}, "
        f"max={run.hi}); this problem is not exercising the defect COOL-193 "
        "describes"
    )
