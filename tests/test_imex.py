"""The IMEX (``G``) path must actually advance the solution."""

import pytest
from conftest import EXACT_DECAY, scalar_problem
from firedrake import *

import firedrake_ts

EXACT = EXACT_DECAY  # solution of u' = -u at t = 1


def _decay(tableau="3", dt=1e-2, split=True):
    """Integrate ``u' = -u`` to t = 1, with ``-u`` explicit when ``split``."""
    u, u_t, v = scalar_problem()
    if split:
        F, G = inner(u_t, v) * dx, -inner(u, v) * dx
    else:
        F, G = inner(u_t, v) * dx + inner(u, v) * dx, None
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "arkimex",
            "ts_arkimex_type": tableau,
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


@pytest.mark.parametrize("tableau", ["3", "2c", "a2", "prssp2"])
def test_imex_advances_solution(tableau):
    """The explicit part must reach the state, not be computed and discarded."""
    _, value = _decay(tableau=tableau)
    assert abs(value - 1.0) > 0.1, (
        f"{tableau}: solution unchanged from its initial condition -- the "
        "explicit part G was evaluated but never applied"
    )
    assert abs(value - EXACT) < 1e-4, f"{tableau}: u(1) = {value}, expected {EXACT}"


def test_imex_stage_solves_use_the_dae_callbacks():
    """The TS's SNES must still be evaluating the DAE at the end of the solve.

    ``None`` here means the callback is PETSc's own C entry point. A Python
    callback means the RHS projection displaced it on the shared DM: those are
    DM-scoped, so any auxiliary solver built on the TS's function space can
    silently take over the TS's residual and Jacobian.
    """
    solver, _ = _decay(dt=1e-2)

    assert solver.ts.getSNESIterations() > 0, (
        "stage solves did no work; a residual that is identically zero means "
        "SNESTSFormFunction was displaced and the solution never advances"
    )

    _, residual = solver.snes.getFunction()
    assert residual is None, (
        "SNESTSFormFunction was displaced by a Python SNES residual; the DAE "
        f"residual is being evaluated by {residual[0]!r} against a stale udot"
    )

    jacobian = solver.snes.getJacobian()[2]
    assert jacobian is None, (
        "SNESTSFormJacobian was displaced by a Python SNES Jacobian; stage "
        f"solves are linearising with {jacobian[0]!r} at a stale shift"
    )


def test_implicit_path_matches_imex_path():
    """Splitting a term into G must not change the answer beyond method error."""
    _, monolithic = _decay(split=False)
    assert abs(monolithic - EXACT) < 1e-3


def test_heat_explicit_example_diffuses():
    """The PDE of examples/heat-explicit.py must actually evolve."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    F = inner(u_t, v) * dx
    G = -(inner(grad(u), grad(v)) * dx - 1.0 * v * dx)
    bcs = [DirichletBC(V, 1.0, 1), DirichletBC(V, 0.0, 2)]
    x = SpatialCoordinate(mesh)
    u.interpolate(conditional(lt(x[0], 0.5), 1.0, 0.0))
    u0 = u.copy(deepcopy=True)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.4), bcs=bcs, G=G)
    firedrake_ts.DAESolver(problem, options_prefix="").solve()

    assert sqrt(abs(assemble((u - u0) ** 2 * dx))) > 0.1


def _nonconstant_mass(kind, dt, tableau="3"):
    """Integrate ``M u' = -u`` to ``t = 1`` with ``-u`` explicit, ``M != 1``.

    ``kind="u"`` uses ``M = (1 + u) v``, ``kind="t"`` uses ``M = (1 + t) v``.
    Both put the whole right-hand side in ``G``, so every stage slope is
    ``L_j = M^-1 G(t_j, Y_j)`` -- the projection whose operator must be
    assembled at the same ``(t_j, Y_j)`` that ``G`` is.

    Exact solutions. For ``M = 1 + u``: separating variables,
    ``(1 + u)/u du = -dt``, so ``ln u + u = 1 - t`` and ``u(1)`` is the root
    of ``ln u + u = 0``, i.e. the omega constant ``0.5671432904...``. For
    ``M = 1 + t``: ``du/u = -dt/(1 + t)`` gives ``u = 1/(1 + t)``, so
    ``u(1) = 1/2``.
    """
    u, u_t, v = scalar_problem()
    time = Constant(0.0)
    mass = (1.0 + u) if kind == "u" else (1.0 + time)
    F = inner(mass * u_t, v) * dx
    G = -inner(u, v) * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), G=G, time=time)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters={
            "ts_type": "arkimex",
            "ts_arkimex_type": tableau,
            "ts_adapt_type": "none",
            "ts_time_step": dt,
            "ts_exact_final_time": "matchstep",
        },
        options_prefix="",
    )
    solver.solve()
    return float(u.dat.data_ro[0])


@pytest.mark.parametrize(
    "kind,exact",
    [("u", 0.5671432904097838), ("t", 0.5)],
)
def test_rhs_projection_operator_is_assembled_at_the_stage_state(kind, exact):
    """``L_j = M^-1 G(Y_j)`` must invert ``M(t_j, Y_j)``, not ``M(t^0, y^0)``.

    This is the IMEX path in ``_TSContext``, so it applies to PETSc's own
    ``arkimex`` (used here) exactly as much as to the Python stepper; both
    ``kind`` cases matter because a structural test for state dependence
    (``dM/du != 0``) catches ``kind="u"`` and silently misses ``kind="t"``.
    See ``_reassemble_rhs_projection_mass_matrix``.
    """
    errors = [abs(_nonconstant_mass(kind, dt) - exact) for dt in (0.1, 0.05, 0.025)]
    assert all(e > 0.0 for e in errors)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 6.5 < ratio < 12.0, (
            f"M({kind}): observed order ratios {ratios} from errors "
            f"{errors}, expected ~8 (order 3)."
        )
