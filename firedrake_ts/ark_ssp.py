"""An additive Runge-Kutta stepper as a ``TSPYTHON`` type.

Selected with ``-ts_type python -ts_python_type firedrake_ts.ark_ssp.ARKSSP``.

The explicit part is advanced in canonical Shu-Osher form so that a limiter
applied to a stage value is mathematically sound: each stage is a convex
combination of forward-Euler steps taken from previously-limited stage values.
See ``docs/superpowers/specs/2026-08-04-shu-osher-tspython-stepper-design.md``.

Known limitation: PETSc only accumulates ``ts->snes_its`` and ``ts->ksp_its``
inside its own step drivers (``TSStep_ARKIMEX``, ``TSStep_Python_default``,
...). A ``TSPYTHON`` type that owns ``step`` itself, as this one does, never
populates those counters, and petsc4py binds no setter for them. As a result
``ts.getSNESIterations()`` and ``ts.getKSPIterations()`` -- and by extension
``-ts_monitor``'s reporting of those fields -- always read zero here. Query
``ts.getSNES()`` directly for the real iteration count.
"""

import numpy as np
from firedrake import dmhooks, ufl_expr
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import DEFAULT_KSP_PARAMETERS, PETSc
from petsctools import OptionsManager

from firedrake_ts._petsc_shim import (
    ts_adapt_candidate_add,
    ts_adapt_candidates_clear,
    ts_adapt_choose,
    ts_get_adapt,
)
from firedrake_ts.solving_utils import (
    explicitly_governed_fields,
    is_zero_form,
    resolve_fields,
)
from firedrake_ts.tableaux import TABLEAUX, kraaijevanger_radius, shu_osher

__all__ = ["ARKSSP"]


class ARKSSP:
    """Additive RK with the explicit part in Shu-Osher form."""

    def __init__(self):
        self.tableau_name = "esdirk_gamma5"
        self.radius = None
        self.setup_calls = 0
        # The exception raised by the most recent setUp/step call, if any.
        # DAESolver.solve reads this to recover the original exception:
        # PETSc's own error handler prints diagnostics as the error unwinds
        # through the C call stack, and under captured stdout/stderr (e.g.
        # pytest's default capture) that print clears the thread's pending
        # exception before it can be attached to the PETSc.Error PETSc4py
        # eventually raises. Stashing it here sidesteps that C boundary
        # entirely.
        self._error = None
        self._limiter = None
        self._frozen_rows = None
        self._tab = None
        self._P = None
        self._q = None
        self._r = None
        # Work vectors, allocated in setUp.
        self._Y = None  # stage values
        self._L = None  # explicit slopes, M^-1 G(Y_j)
        self._Ydot = None  # implicit stage derivatives
        self._Z = None  # Butcher stage offset
        self._rhs = None  # scratch for computeRHSFunction
        self._zero_xdot = None  # a permanent zero Xdot, for F(t, y_n, 0)
        # Stage-0 mass solve for an explicit first stage: built in setUp,
        # used once per step() by _prepare_stage0_ydot. Only built when the
        # first stage is actually explicit (tab.At[0, 0] == 0); an implicit
        # first stage needs none of this, since _solve_stage populates
        # Ydot[0] itself in _take_stages. Whether the first stage is
        # explicit is read directly off tab.At[0, 0] wherever it matters
        # (_setup_stage0_mass_solve, _prepare_stage0_ydot) rather than
        # cached in a separate flag.
        self._mass_ksp = None
        self._mass_options = None
        # Whether M = dF/du_dot depends on the state u -- see
        # _setup_stage0_mass_solve. When True, _prepare_stage0_ydot
        # reassembles self._mass_tensor at the current (t^n, y^n) every
        # step() rather than reusing a single assembly across the whole
        # solve.
        self._mass_is_state_dependent = False
        self._mass_tensor = None
        self._mass_form = None
        # Recorded at the top of each step() attempt, for evaluatestep --
        # the value the solution held at the START of the step. step()
        # overwrites ts.getSolution() with each attempt's order-p candidate
        # before TSAdaptChoose runs (and restores it here on rejection), so
        # evaluatestep must build its own answer from these rather than from
        # ts.getSolution().
        self._last_h = None
        self._last_x = None
        # Set per stage, read by the SNES callbacks.
        self._stage = None
        self._shift = None
        self._stage_time = None

    # -- options and lifecycle ------------------------------------------------

    def setFromOptions(self, ts):
        opts = PETSc.Options(ts.getOptionsPrefix() or "")
        self.tableau_name = opts.getString("ts_ark_ssp_type", self.tableau_name)
        radius = opts.getReal("ts_ark_ssp_radius", 0.0)
        self.radius = radius if radius > 0.0 else None

    def setUp(self, ts):
        # Clear any stale error before this call -- a failure from a
        # previous setUp/step must never be re-raised against this one.
        self._error = None
        try:
            self.setup_calls += 1
            try:
                self._tab = TABLEAUX[self.tableau_name]
            except KeyError:
                raise ValueError(
                    f"unknown tableau {self.tableau_name!r}; "
                    f"choose one of {sorted(TABLEAUX)}"
                ) from None
            tab = self._tab
            self._r = (
                self.radius
                if self.radius is not None
                else kraaijevanger_radius(tab.A, tab.b)
            )
            # Raises ShuOsherError if the radius is too large, naming both
            # numbers.
            self._P, self._q = shu_osher(tab.A, tab.b, self._r)

            sol = ts.getSolution()
            s = len(tab.b)
            self._Y = [sol.duplicate() for _ in range(s)]
            self._L = [sol.duplicate() for _ in range(s)]
            self._Ydot = [sol.duplicate() for _ in range(s)]
            self._Z = sol.duplicate()
            self._rhs = sol.duplicate()
            self._last_x = sol.duplicate()
            self._zero_xdot = sol.duplicate()
            self._zero_xdot.set(0.0)

            self._frozen_rows = self._find_frozen_rows(ts)
            self._check_limiter_soundness()
            self._setup_stage0_mass_solve(ts)
        except Exception as exc:
            self._error = exc
            raise

    def reset(self, ts):
        pass

    def view(self, ts, viewer):
        if viewer is None:
            return
        viewer.printfASCII(f"  ARK-SSP stepper, tableau {self.tableau_name}\n")
        viewer.printfASCII(f"  Shu-Osher radius r = {self._r}\n")
        viewer.printfASCII(
            f"  limiter: {'set' if self._limiter is not None else 'none'}\n"
        )

    def set_stage_limiter(self, limiter):
        """Register a callable fired on each explicit substage value.

        Known limitation: the soundness guard below is all-or-nothing at
        problem level, not per component. It refuses a limiter outright
        when NO component is freezable, but once at least one component
        IS freezable (``self._frozen_rows`` is not ``None``), it raises
        nothing further -- even if the limiter's callable also modifies a
        DIFFERENT, non-frozen component that has an implicit operator
        acting on it. That component's implicit stage solve silently
        reverts the limiter's change on exactly that row, reproducing the
        original COOL-193 defect this whole stepper exists to fix, with no
        error and no warning. Enforcing this per component would need to
        know, for an arbitrary callable, which sub-block(s) of the stage
        vector it actually touches -- real design work, out of scope here.
        """
        self._limiter = limiter
        # setUp runs this same check once the tableau and frozen rows are
        # known; re-run it here too, since a caller may register a limiter
        # AFTER setUp has already run (e.g. against an already-solved-once
        # TS). Without this, that ordering silently bypasses the guard
        # instead of raising, giving unsound behaviour with no warning.
        if self._tab is not None:
            self._check_limiter_soundness()

    def _check_limiter_soundness(self):
        """Refuse a limiter that a stage solve could undo without a trace.

        The soundness hazard only exists where an implicit stage actually
        runs (tab.At[i, i] > 0.0, matching step()'s own guard on
        _solve_stage): a purely-explicit tableau such as ssprk2 never
        solves a stage, so there is nothing for the limiter's change to be
        undone by, regardless of whether any component was found freezable.
        """
        has_implicit_stage = np.any(np.diagonal(self._tab.At) > 0.0)
        if (
            self._limiter is not None
            and self._frozen_rows is None
            and has_implicit_stage
        ):
            raise ValueError(
                "a stage limiter is registered, but no component of this "
                "problem is free of an implicit operator: every row of "
                "dF/du is structurally nonzero. Limiting a component "
                "that has an implicit operator is not sound -- the "
                "stage solve would undo the correction. Move the "
                "operator into G, or drop the limiter."
            )

    def _find_frozen_rows(self, ts):
        """Global row indices of components with no implicit operator.

        Those rows' stage equation reduces to Y_i = Z_i, so their value comes
        entirely from the explicit Shu-Osher recursion. Pinning them during
        the implicit solve is what stops the solve from undoing the limiter.

        A non-mixed space is one field (``nfields = 1``), not a structural
        exemption: ``explicitly_governed_fields`` answers correctly for it
        too (e.g. a scalar DG field advanced purely through G, whose F is a
        bare mass form with no dependence on the state at all -- exactly the
        DG1 bounds test in ``tests/test_bounds.py``). Detection is therefore
        unconditional and structural. Whether the freeze actually PINS
        anything during a solve is a separate question, gated in the
        application sites (``_apply_freeze_residual``, ``formSNESJacobian``,
        ``_solve_stage``) on a limiter being registered: with no limiter
        there is nothing for the implicit solve to undo, and for a
        single-field problem freezing every row would make the stage solve
        self-referentially trivial (zero Newton iterations) even though the
        real linear mass-matrix solve is doing legitimate work -- see
        ``test_stage_solves_do_work``.
        """
        ctx = dmhooks.get_appctx(ts.getDM())
        problem = ctx._problem
        V = problem.u_restrict.function_space()
        nfields = len(V) if len(V) > 1 else 1
        detected = explicitly_governed_fields(
            problem.F, problem.u_restrict, ctx._xdot, nfields
        )
        fields = resolve_fields(
            "ts_explicitly_governed_fields", ts.getOptionsPrefix(), detected
        )
        if not fields:
            return None
        # Taken from J's test space, matching _TSContext._rhs_projection_
        # mass_matrix's identical idiom in solving_utils.py (see that
        # method's comment): correct because J is derived from F when not
        # user-supplied, but this assumes a user-supplied J shares F's
        # test-function space.
        ises = problem.J.arguments()[0].function_space()._ises
        rows = np.concatenate([ises[i].getIndices() for i in fields])
        return rows.astype(PETSc.IntType)

    def _freeze_active(self):
        """Whether the freeze should actually pin rows this solve.

        Detection in ``_find_frozen_rows`` is unconditional and structural;
        application is gated on a limiter being registered, since freezing
        exists only to protect a limiter's correction from the implicit
        solve. Applying it with no limiter would be harmless where some
        other field remains for SNES to solve, but for a single, wholly
        explicitly-governed field it would make the stage solve trivial
        (see ``_find_frozen_rows``'s docstring), silently changing behaviour
        no limiter asked for.
        """
        return self._frozen_rows is not None and self._limiter is not None

    def _apply_freeze_residual(self, x, f):
        """Replace frozen rows of the residual with ``x - Y_i``."""
        if not self._freeze_active():
            return
        target = self._Y[self._stage]
        xa = x.getArray(readonly=True)
        ya = target.getArray(readonly=True)
        fa = f.getArray()
        lo, _ = x.getOwnershipRange()
        local = self._frozen_rows - lo
        fa[local] = xa[local] - ya[local]

    def _setup_stage0_mass_solve(self, ts):
        """Set up whatever an explicit first stage needs for ``Ẏ_0``.

        For an explicit first stage (``tab.At[0, 0] == 0``), ``Y_0 = y^n``
        and the implicit residual must still be satisfied there:
        ``F(t^n, y^n, Ẏ_0) = 0``. This is EXACT, not an approximation, when
        ``F`` is affine in ``u̇`` -- which is the overwhelmingly common
        case (any ``F`` built from a mass term plus terms with no ``u̇``
        dependence): solving the one linear equation
        ``M Ẏ_0 = -F(t^n, y^n, 0)`` with the mass matrix ``M = dF/du̇``
        (constant on that row precisely because ``F`` is affine in ``u̇``)
        gives ``Ẏ_0 = -M^-1 F(t^n, y^n, 0)`` as the unique root, not a
        linearisation of it. See ``_prepare_stage0_ydot``, which solves that
        every ``step()``. When ``F`` is NONLINEAR in ``u̇`` instead (a
        nonzero ``d^2F/du̇^2``), that same formula is only the first Newton
        step away from ``Ẏ_0 = 0``, not the true root -- silently wrong,
        with no exception -- so this refuses such a problem outright below
        rather than return a plausible-looking, wrong answer.

        Built unconditionally whenever the first stage is explicit --
        NOT gated on whether ``dF/du`` is structurally zero. A previous
        version skipped the solve there on the reasoning that a state-
        independent ``F`` gives ``Ẏ_0 = 0`` "exactly"; that reasoning is
        false. ``dF/du == 0`` says nothing about ``F(t^n, y^n, 0)`` itself
        -- a purely time-dependent or constant forcing term (e.g.
        ``F = inner(u_t, v)*dx - inner(Constant(1.0), v)*dx``, i.e.
        ``u̇ = 1``) has zero ``dF/du`` but nonzero ``F(t, y, 0)``, and
        skipping the solve there silently reproduces the exact defect this
        stage-0 handling exists to fix: ``Ẏ_0`` stuck at zero, giving a
        step that is flat in ``dt`` rather than converging. For a genuinely
        mass-only ``F`` (no source term at all), ``F(t^n, y^n, 0)`` really
        is zero, so this solve just returns zero at the cost of one extra
        ``IFunction`` evaluation and one mass solve per step -- cheap
        insurance against silently dropping a source term. This is NOT
        skipped merely because a limiter or the freeze is active -- those
        constrain what a limiter may do to a stage *value*, which is
        orthogonal to whether an explicit first stage's derivative needs
        solving for at all.

        Builds a bare ``PETSc.KSP`` on the assembled mass matrix, following
        ``_TSContext._rhs_projection_solver``'s idiom in
        ``solving_utils.py`` (see that method's comment): a Firedrake
        ``LinearSolver``/``NonlinearVariationalSolver`` would register a
        ``SNES`` on the TS's own DM, since this problem lives on the same
        function space as the TS's solution, and that would silently
        displace ``SNESTSFormFunction``/``SNESTSFormJacobian`` on the TS's
        real SNES. A ``KSP`` inverting an already-assembled operator needs
        no ``SNES`` and so cannot collide -- true whether or not that
        operator gets reassembled between solves.

        Whether ``M`` itself may change between steps is detected here via
        ``self._mass_is_state_dependent = not is_zero_form(dM/du)``:

        * When ``M`` does NOT depend on ``u`` (the common case -- e.g. any
          problem with a constant-coefficient mass term), the mass matrix
          is reused from ``ctx._rhs_projection_mass_matrix`` exactly as
          before: it is exactly the same ``dF/du̇`` (with the same
          algebraic-row diag=1 treatment) whether it is used to project
          ``G`` or to solve for ``Ẏ_0``, and ``_TSContext`` caches it once.
          Reassembling every step here would cost a real assembly for no
          benefit, since the answer would not change.
        * When ``M`` DOES depend on ``u`` (e.g. variable density, porosity,
          saturation -- any ``M(u)``), reusing that one assembly at every
          step is exactly the defect this method exists to fix: ``Ẏ_0`` was
          being solved against ``M(u^0)`` forever, on every step, no matter
          how far the state had moved from the initial condition -- silent,
          flat-in-``dt`` non-convergence, with no exception. In this case
          ``self._mass_form`` (``dF/du̇``, undifferentiated further) is kept
          so ``_prepare_stage0_ydot`` can reassemble it into
          ``self._mass_tensor`` in place at the current ``(t^n, y^n)``
          before every solve, and reset the ``KSP``'s operator to match.

        That property is safe to call even when ``G`` is ``None`` (it is
        not gated on ``G`` -- only ``_rhs_projection_solver`` is, which is
        why this method builds its own ``KSP`` instead of reusing that
        one).

        ``setUp`` may run more than once against the same stepper instance
        (a TS may be re-set-up after options change); destroy the previous
        ``KSP`` first rather than leaking it -- ``PETSc.KSP`` objects hold
        onto PETSc-side resources that Python's own garbage collector does
        not reliably reclaim promptly.
        """
        if self._mass_ksp is not None:
            self._mass_ksp.destroy()
            self._mass_ksp = None
        tab = self._tab
        if tab.At[0, 0] > 0.0:
            self._mass_is_state_dependent = False
            self._mass_form = None
            self._mass_tensor = None
            return
        ctx = dmhooks.get_appctx(ts.getDM())
        mass_form = ufl_expr.derivative(ctx.F, ctx._xdot)

        # F affine in u_dot? d^2F/du_dot^2 == 0 is required for
        # -M^-1 F(t^n, y^n, 0) to be the exact root rather than one Newton
        # step from zero -- see this method's docstring. Refuse rather than
        # hand back a plausible, silently wrong Ẏ_0.
        if not is_zero_form(ufl_expr.derivative(mass_form, ctx._xdot)):
            raise ValueError(
                "F is nonlinear in u̇ (d^2F/du̇^2 is not structurally "
                "zero), but this tableau's first stage is explicit "
                f"(tab.At[0, 0] == 0.0 for {tab.name!r}). "
                "Ẏ_0 = -M^-1 F(t^n, y^n, 0) is exact only when F is affine "
                "in u̇; otherwise it is one Newton step away from the true "
                "root of F(t^n, y^n, ·) = 0, and using it gives a silently "
                "wrong answer with no exception. Rewrite F to be affine in "
                "u̇, or choose a tableau with an implicit first stage "
                "(tab.At[0, 0] > 0.0)."
            )

        self._mass_is_state_dependent = not is_zero_form(
            ufl_expr.derivative(mass_form, ctx._x)
        )
        mass = ctx._rhs_projection_mass_matrix
        self._mass_tensor = mass
        self._mass_form = mass_form if self._mass_is_state_dependent else None
        ksp = PETSc.KSP().create(comm=mass.comm)
        ksp.setOperators(mass.petscmat)
        parameters = {
            k: v for k, v in DEFAULT_KSP_PARAMETERS.items() if k != "mat_type"
        }
        prefix = (ts.getOptionsPrefix() or "") + "ark_ssp_stage0_mass_solver_"
        self._mass_options = OptionsManager(parameters, prefix)
        self._mass_options.set_from_options(ksp)
        self._mass_ksp = ksp

    def _reassemble_stage0_mass(self, ctx):
        """Reassemble ``self._mass_tensor`` at the current state, in place.

        Only called from ``_prepare_stage0_ydot`` when
        ``self._mass_is_state_dependent`` -- see ``_setup_stage0_mass_solve``
        for why the common, state-independent case must NOT pay for this
        every step. ``ctx._x`` (== ``ctx._problem.u_restrict``, the same
        coefficient ``self._mass_form`` is built from) already holds
        ``y^n`` by the time this runs: ``_prepare_stage0_ydot`` calls
        ``ts.computeIFunction`` first, and that callback
        (``_TSContext.form_function``) copies the incoming state into
        ``ctx._x`` itself.

        ``tensor=self._mass_tensor`` reassembles into the existing
        ``Matrix``/``Mat`` in place, rather than allocating a fresh one each
        step: same sparsity, same PETSc handle the ``KSP`` already has as
        its operator, so no ``setOperators`` call is needed after this --
        PETSc's own assembly bumps the ``Mat``'s state counter, which is
        what tells the ``KSP`` its factorisation is stale.

        Mirrors ``_TSContext._rhs_projection_mass_matrix``'s algebraic-row
        handling (unit diagonal on any row with a structurally zero
        ``dF/du̇``), since a fresh assembly would otherwise zero those rows
        out again and reintroduce the singular pivot that property exists
        to avoid.
        """
        from firedrake import assemble

        assemble(self._mass_form, bcs=ctx.bcs_F, tensor=self._mass_tensor)
        if ctx._algebraic_fields:
            ises = ctx._problem.J.arguments()[0].function_space()._ises
            rows = np.concatenate([ises[i].getIndices() for i in ctx._algebraic_fields])
            self._mass_tensor.petscmat.zeroRows(rows.astype(PETSc.IntType), diag=1.0)

    def _prepare_stage0_ydot(self, ts, t, x):
        """Populate ``Ẏ_0`` once per ``step()``, before the retry loop.

        ``Ẏ_0 = -M(y^n)^-1 F(t^n, y^n, 0)`` depends only on ``(t^n, y^n)``,
        i.e. on ``(t, x)`` as passed in here -- neither of which changes
        across retries of the same step inside ``step()``'s reject loop
        (only ``h`` does). So this runs exactly once per ``step()`` call,
        not once per attempt inside ``_take_stages``, and it must be
        called with THIS step's ``t^n``/``y^n``: ``step()`` calls it right
        after recording ``self._last_x``, before entering the retry loop.
        """
        if self._tab.At[0, 0] > 0.0:
            return  # _solve_stage populates _Ydot[0] normally, in _take_stages.
        # _setup_stage0_mass_solve builds this KSP unconditionally whenever
        # the first stage is explicit -- see that method's docstring for
        # why skipping it based on dF/du alone is unsound.
        ts.computeIFunction(t, x, self._zero_xdot, self._rhs, True)
        if self._mass_is_state_dependent:
            # M depends on u: the assembly cached in setUp is stale by now
            # (built from whatever u happened to be at setUp time, e.g.
            # the initial condition) -- reassemble at THIS step's y^n
            # before solving, or Ẏ_0 solves against the wrong operator on
            # every step after the first. ctx._x already holds y^n, from
            # the computeIFunction call just above.
            ctx = dmhooks.get_appctx(ts.getDM())
            self._reassemble_stage0_mass(ctx)
        with self._mass_options.inserted_options():
            self._mass_ksp.solve(self._rhs, self._Ydot[0])
        self._Ydot[0].scale(-1.0)
        if self._freeze_active():
            # Same reasoning as _solve_stage's identical block at the end of
            # a real stage solve: a frozen row's implicit function is the
            # mass term alone, so M Ẏ = 0 there and the correct derivative
            # is exactly zero, not whatever -M^-1 F(t, y, 0) gives on a row
            # a limiter has no business perturbing further.
            lo, _ = self._Ydot[0].getOwnershipRange()
            local = self._frozen_rows - lo
            self._Ydot[0].getArray()[local] = 0.0

    # -- the step -------------------------------------------------------------

    def step(self, ts):
        # Clear any stale error before this call -- a failure from a
        # previous setUp/step must never be re-raised against this one.
        self._error = None
        try:
            tab = self._tab
            t = ts.getTime()
            h = ts.getTimeStep()
            x = ts.getSolution()
            s = len(tab.b)

            x.copy(self._last_x)
            # Once per step(), not once per retry: see _prepare_stage0_ydot's
            # docstring -- it depends only on (t^n, y^n), fixed for every
            # attempt below, but t^n does change between step() calls.
            self._prepare_stage0_ydot(ts, t, self._last_x)

            # petsc4py binds setMaxStepRejections but NOT a getter, so read
            # the option directly. Two traps, both verified against the
            # installed PETSc:
            #   * ts.c:133 does PetscOptionsDeprecated("-ts_max_reject",
            #     "-ts_max_step_rejections", "3.25", NULL), which REMOVES the
            #     old key from the database during TSSetFromOptions -- long
            #     before this runs. Reading "ts_max_reject" therefore always
            #     returns the default, silently ignoring both spellings. Read
            #     the new name, keeping the old one only as a legacy
            #     fallback (relevant only if TSSetFromOptions is somehow
            #     skipped for this TS).
            #   * PETSc's "no bound" sentinel is PETSC_UNLIMITED == -3
            #     (petscsys.h:367), not -1. Treated naively, max(1, n + 1)
            #     would turn a request for unlimited retries into exactly one
            #     attempt.
            opts = PETSc.Options(ts.getOptionsPrefix() or "")
            max_reject = opts.getInt(
                "ts_max_step_rejections", opts.getInt("ts_max_reject", 10)
            )
            attempts = 1 << 30 if max_reject < 0 else max(1, max_reject + 1)
            adapt = ts_get_adapt(ts)
            # Both kinds of rejection -- LTE-based (TSAdaptChoose declines
            # the completed candidate) and SNES-divergence-based (caught
            # below) -- share this one counter and its ts_max_step_rejections
            # cap. PETSc's own arkimex.c tracks stage-solve failures
            # separately against ts_max_snes_failures (TSAdaptCheckStage,
            # tsadapt.c), stopping with TS_DIVERGED_NONLINEAR_SOLVE once that
            # second cap is hit even if step rejections are unlimited. This
            # stepper does not reproduce that second counter: a single
            # attempts budget governs every retry regardless of which check
            # rejected it, so ts_max_snes_failures (DAESolver sets it to -1,
            # i.e. unlimited, at ts_solver.py:271) has no effect here -- the
            # reject loop's own cap is what actually bounds retries.
            # Named cause of the most recent rejection, for the exhaustion
            # message below -- there are two independent rejection sources
            # sharing this loop (SNES divergence, and TSAdaptChoose declining
            # a completed candidate), and once retries run out the caller
            # needs to know which one kept firing, not just that "some"
            # rejection happened `attempts` times.
            last_reject_cause = None
            for _ in range(attempts):
                self._last_h = h
                try:
                    self._take_stages(ts, tab, self._last_x, h, s)
                except ConvergenceError as exc:
                    # A stage's SNES diverged (_solve_stage). Reject this
                    # attempt to the adapt loop and retry with a smaller h,
                    # mirroring PETSc's own TSAdaptCheckStage (tsadapt.c
                    # reject_stage: dt *= adapt->scale_solve_failed, default
                    # 0.25) -- called from arkimex.c's own stage loop right
                    # after SNESSolve, at arkimex.c:1474-1479, before falling
                    # to reject_step. TSAdaptGetScaleSolveFailed has no
                    # petsc4py binding, so read the option directly, exactly
                    # as ts_max_step_rejections is read above. No completed
                    # candidate exists yet at this point (not every stage
                    # ran), so there is nothing to hand TSAdaptChoose; unlike
                    # a normal rejection this path chooses next_h itself
                    # rather than asking the adapt loop's error-based
                    # controller for one.
                    last_reject_cause = str(exc)
                    scale_solve_failed = opts.getReal(
                        "ts_adapt_scale_solve_failed", 0.25
                    )
                    # x (ts->vec_sol) is never written by _take_stages --
                    # only self._Y/_L/_Ydot are -- so this copy is a no-op
                    # today. Restoring it anyway keeps this path visibly
                    # symmetric with the completion-through-choose block's
                    # own restore-on-rejection/restore-on-exception below,
                    # so a future change to _take_stages that DOES touch x
                    # does not silently fall outside that guarantee.
                    self._last_x.copy(x)
                    # dt_min's magic number (1e-20) matches PETSc's own
                    # TSAdapt default floor (tsadapt.c:1155,
                    # adapt->dt_min = 1e-20) when ts_adapt_dt_min is not
                    # set, so a caller who already relies on that PETSc
                    # default for a real TSADAPT gets the same floor here.
                    # The floor is NOT applied the way PETSc's own reject
                    # path applies it, though: TSAdaptCheckStage's
                    # reject_stage (tsadapt.c:1110-1116) multiplies by
                    # scale_solve_failed unconditionally, with no floor at
                    # all in that path -- PETSc clamps to dt_min separately,
                    # inside TSAdaptChoose, not here. This loop clamps h
                    # itself (below) because, unlike PETSc's C code, a
                    # persistently diverging stage with
                    # ts_max_step_rejections unlimited (attempts ==
                    # 1 << 30) would otherwise shrink h by
                    # scale_solve_failed every attempt with nothing to stop
                    # it, underflowing past dt_min and eventually to
                    # exactly 0.0 -- at which point _solve_stage's
                    # self._shift = 1 / (h * tab.At[i, i]) raises a plain
                    # ZeroDivisionError (h is a Python float here, not a
                    # PETSc real with an IEEE infinity to fall back on),
                    # and the next stage solve fails in a way that has
                    # nothing to do with the original divergence.
                    #
                    # Checked BEFORE shrinking, and h clamped to the floor
                    # rather than left below it: checking only after
                    # shrinking, as a previous version did, raised with
                    # ZERO retries at the floor for any h already within
                    # one shrink of dt_min (h < dt_min / scale_solve_failed,
                    # e.g. h < 4 * dt_min at the default 0.25) -- the very
                    # case a floor should still get one last attempt at,
                    # not skip.
                    dt_min = opts.getReal("ts_adapt_dt_min", 1e-20)
                    if h <= dt_min:
                        ts.setConvergedReason(
                            PETSc.TS.ConvergedReason.DIVERGED_STEP_REJECTED
                        )
                        raise ConvergenceError(
                            f"stage solve diverged and h is already at "
                            f"the ts_adapt_dt_min floor ({h:.3g} <= "
                            f"{dt_min:.3g}); TS_DIVERGED_STEP_REJECTED. "
                            f"Last rejection: {last_reject_cause}."
                        ) from exc
                    h = max(h * scale_solve_failed, dt_min)
                    ts.setTimeStep(h)
                    continue
                # TSAdaptChoose reads ts->vec_sol as the completed, order-p
                # solution (TSErrorWeightedNorm's own docstring: "usually
                # ts->vec_sol"; PETSc's own TSStep_ARKIMEX writes the
                # completion into vec_sol before calling TSAdaptChoose, at
                # ts/impls/arkimex/arkimex.c, and restores a saved pre-step
                # copy on rejection). x IS ts->vec_sol here. Guard the whole
                # completion-through-choose block: if any of _complete,
                # ts_adapt_candidates_clear, ts_adapt_candidate_add or
                # ts_adapt_choose raises, x must still be restored to the
                # pre-step value before the exception propagates, or the TS
                # is left holding an unaccepted, never-validated candidate.
                try:
                    self._complete(tab, x, h)
                    ts_adapt_candidates_clear(adapt)
                    ts_adapt_candidate_add(
                        adapt, None, tab.order, tab.order, 1.0, float(s), True
                    )
                    _sc, next_h, accept = ts_adapt_choose(adapt, ts, h)
                except Exception:
                    self._last_x.copy(x)
                    raise
                if accept:
                    ts.setTime(t + h)
                    ts.setTimeStep(next_h)
                    return
                # Rejected: restore x (ts->vec_sol) to the pre-step value
                # before retrying with the smaller next_h -- _complete's
                # non-stiffly-accurate branch reads x as the step's starting
                # point, and would otherwise read back the just-rejected
                # candidate. self._last_x itself is untouched, so the retried
                # _take_stages still predicts from the correct x^n.
                last_reject_cause = (
                    f"the adapt controller declined the completed step "
                    f"(h {h:.6g} -> {next_h:.6g})"
                )
                self._last_x.copy(x)
                h = next_h
                ts.setTimeStep(h)
            # Retries exhausted. Setting the converged reason alone is NOT
            # enough to reach check_ts_convergence's clean ConvergenceError:
            # PETSc's own TSStep() (ts.c) checks `ts->reason < 0` itself,
            # right after (*ts->ops->step)(ts) returns, and -- because
            # TSSetErrorIfStepFails defaults to true -- immediately does its
            # own SETERRQ(PETSC_ERR_NOT_CONVERGED, "TSStep has failed due to
            # %s", ...) in C, before returning to TSSolve(), before
            # DAESolver.solve() ever gets to call check_ts_convergence.
            # That SETERRQ happens outside this method's Python frame (it
            # runs after TSStep_Python's call to step(ts) has already
            # returned success), so it is never caught by step()'s own
            # try/except above and self._error is never populated -- the
            # caller then sees a raw, unwrapped PETSc.Error(91) instead of
            # this stepper's usual clean exception. Raising here, instead of
            # just setting the reason and returning, is what actually routes
            # through the recorded-error path: this exception is caught by
            # step()'s own enclosing try/except (self._error = exc; raise),
            # surfaces to libpetsc4py as PETSC_ERR_PYTHON, and
            # DAESolver.solve() unwraps that back to this ConvergenceError --
            # the same mechanism test_shu_osher_error_is_unwrapped_with_the_
            # actionable_numbers already exercises for a setUp()-time error.
            ts.setConvergedReason(PETSc.TS.ConvergedReason.DIVERGED_STEP_REJECTED)
            raise ConvergenceError(
                f"step rejected {attempts} time(s) in a row "
                f"(ts_max_step_rejections); TS_DIVERGED_STEP_REJECTED. "
                f"Last rejection: {last_reject_cause}."
            )
        except Exception as exc:
            self._error = exc
            raise

    def _take_stages(self, ts, tab, x, h, s):
        """Run all s stages from ``x`` with step ``h``. Populates Y, L, Ydot."""
        t = ts.getTime()
        for i in range(s):
            self._shu_osher_predictor(x, h, i)
            # Limiter fires BEFORE _solve_stage, on rows the implicit
            # solve has not yet touched -- correct ordering, since a
            # limiter must act on the Shu-Osher predictor value, not on
            # whatever the stage solve does to it afterward. This ordering
            # is untested against a MIXED problem where the limiter acts
            # on a partially-frozen state: test_bounds.py's F is mass-only,
            # so every row there is frozen and the stage solve is a no-op
            # on pinned rows regardless of ordering -- moving the limiter
            # after _solve_stage would still pass that test with nothing
            # to distinguish the two. A test that actually exercises this
            # ordering needs a mixed problem with a limiter on a partially
            # frozen state (some rows implicit, some frozen), which is more
            # than this round takes on -- recorded here, not fixed.
            if self._limiter is not None:
                self._limiter(self._Y[i])
            self._build_offset(tab, x, h, i)
            if tab.At[i, i] > 0.0:
                self._solve_stage(ts, tab, h, i)
            ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._rhs)
            self._rhs.copy(self._L[i])

    def evaluatestep(self, ts, order, U):
        """Write the order-``order`` completion into ``U``.

        ``TSADAPTBASIC`` gets its lower-order solution through
        ``TSEvaluateStep``, which routes here. The error estimate
        ``|h sum (b - bhat)_j L(Y_j)|`` reuses the ``L(Y_j)`` the step
        computed anyway, so error control costs no extra evaluations.

        ``tab.bhat`` is a single embedded-weight array, not a pair -- there
        is no separate "implicit bhat". For a tableau with a genuine
        implicit part (``bt`` not identically zero, e.g. esdirk_gamma5),
        that one array is understood as embedding BOTH the implicit and
        explicit completions simultaneously, and reusing it for
        ``weights_i`` below is correct. For a purely explicit tableau
        (``bt`` identically zero, e.g. ssprk2), there is no implicit part
        to embed at all -- the embedded weight there must be zero too, not
        ``bhat``. Getting this wrong is currently invisible for ssprk2 only
        because its ``F`` in every existing test is mass-only, giving
        ``Ẏ_0 ≡ 0`` identically regardless of which weight multiplies it;
        the stage-0 fix above makes ``Ẏ_0`` non-zero for any ``F`` with a
        state-independent term, at which point a spurious ``h Ẏ_0``
        embedded-error contribution would appear from nowhere.
        """
        tab = self._tab
        h = self._last_h
        x = self._last_x
        has_implicit = np.any(tab.bt)
        weights_e = tab.b if order >= tab.order else tab.bhat
        weights_i = tab.bt if (order >= tab.order or not has_implicit) else tab.bhat
        x.copy(U)
        for j in range(len(tab.b)):
            if weights_i[j] != 0.0:
                U.axpy(h * weights_i[j], self._Ydot[j])
            if weights_e[j] != 0.0:
                U.axpy(h * weights_e[j], self._L[j])
        # petsc4py's TSEvaluateStep_Python treats a falsy return as failure
        # (PETSC_ERR_USER "Cannot evaluate step") whenever the caller -- as
        # TSAdaptChoose_Basic does -- passes a NULL `done` pointer.
        return True

    def interpolate(self, ts, t, U):
        """Dense output at time ``t`` within the step just taken.

        X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i,
        with w = bt for the implicit part and w = b for the explicit part.
        The coefficients d are chosen so that d . g = 0 for the null
        direction g of the singular At, which is what keeps the stiff
        limit bounded -- a naive d = (1, 0, 0, 0) has d . g != 0 and makes
        X(theta) diverge like z theta(theta - 1) as z -> -infinity.
        """
        tab = self._tab
        h = self._last_h
        if h is None:
            raise ValueError("interpolate called before any step was taken")
        theta = (t - (ts.getTime() - h)) / h
        self._last_x.copy(U)
        for i in range(len(tab.b)):
            impl = tab.d[i] * theta + (tab.bt[i] - tab.d[i]) * theta**2
            expl = tab.d[i] * theta + (tab.b[i] - tab.d[i]) * theta**2
            if impl != 0.0:
                U.axpy(h * impl, self._Ydot[i])
            if expl != 0.0:
                U.axpy(h * expl, self._L[i])
        # Unlike evaluatestep's, this return value is inert: petsc4py's
        # TSInterpolate_Python (libpetsc4py.pyx) calls interpolate(...) and
        # discards whatever it returns -- there is no flag out-param and no
        # truthiness check, in contrast to TSEvaluateStep_Python's `done`.
        # Kept only for symmetry with evaluatestep, not because anything
        # reads it.
        return True

    def _shu_osher_predictor(self, x, h, i):
        """Y_i = q_i x^n + sum_{j<i} P_ij (Y_j + (h/r) L_j).

        Every row of ``[P | q]`` is a nonnegative partition of unity, so this
        is a convex combination of forward-Euler steps taken from stage values
        the limiter has already seen. That is the whole reason for the
        stepper.
        """
        self._Y[i].set(0.0)
        if self._q[i] != 0.0:
            self._Y[i].axpy(self._q[i], x)
        for j in range(i):
            pij = self._P[i, j]
            if pij == 0.0:
                continue
            self._Y[i].axpy(pij, self._Y[j])
            self._Y[i].axpy(pij * h / self._r, self._L[j])

    def _build_offset(self, tab, x, h, i):
        """Z_i = x^n + h sum_{j<i} (At_ij Ydot_j + A_ij L_j)."""
        x.copy(self._Z)
        for j in range(i):
            if tab.At[i, j] != 0.0:
                self._Z.axpy(h * tab.At[i, j], self._Ydot[j])
            if tab.A[i, j] != 0.0:
                self._Z.axpy(h * tab.A[i, j], self._L[j])

    def _solve_stage(self, ts, tab, h, i):
        """Solve F(Ydot_i, Y_i, t_i) = 0 with Ydot_i = (Y_i - Z_i)/(h At_ii)."""
        self._stage = i
        self._shift = 1.0 / (h * tab.At[i, i])
        self._stage_time = ts.getTime() + tab.ct[i] * h
        snes = ts.getSNES()
        # _apply_freeze_residual reads self._Y[self._stage] itself as the
        # frozen target -- it is the same vector SNES iterates on, so the
        # residual it builds is self-referentially zero on those rows.
        # That only pins the *right* value if the vector already holds it
        # before the solve starts: with a zero residual and an identity
        # Jacobian row, Newton's own step contributes exactly zero there
        # every iteration, so whatever the frozen rows hold going in is
        # what comes out. Snapshot them before the warm-start overwrite
        # below replaces the whole vector with the previous stage's value
        # (or x^n), which would otherwise silently discard the predictor's
        # -- and any limiter's -- value on exactly the rows the freeze
        # exists to protect.
        freeze = self._freeze_active()
        if freeze:
            lo, _ = self._Y[i].getOwnershipRange()
            local = self._frozen_rows - lo
            frozen_values = self._Y[i].getArray(readonly=True)[local].copy()
        # Initial guess: the previous stage value, or x^n for the first
        # implicit stage -- matching PETSc's own ARKIMEX. Guessing Z_i itself
        # would make Ydot_i identically zero already for any tableau whose
        # implicit form has no dependence on the state (as in this task's
        # decay shakedown), so SNES would report zero iterations even with
        # the callbacks wired correctly.
        #
        # The i == 0 branch is unreachable for every tableau in TABLEAUX
        # today: this method only runs when tab.At[i, i] > 0.0 (_take_stages'
        # own guard), and every registry tableau has an explicit first stage
        # (At[0, 0] == 0.0). It is retained rather than removed because it
        # is what a fully-implicit tableau (At[0, 0] > 0.0, none of which
        # exist in TABLEAUX yet) would need, and it matches PETSc's own
        # ARKIMEX, which carries the identical case for the same reason.
        if i > 0:
            self._Y[i - 1].copy(self._Y[i])
        else:
            ts.getSolution().copy(self._Y[i])
        if freeze:
            self._Y[i].getArray()[local] = frozen_values
        snes.solve(None, self._Y[i])
        # The residual on frozen rows is self-referential (x - Y_i, the same
        # vector SNES is iterating on) so it cannot itself enforce the pin --
        # it is identically zero regardless of what value the row holds.
        # That only *looks* like a pin because Newton's own step happens to
        # contribute zero there too, for a plain Newton/KSP iteration with a
        # preconditioner that preserves the identity row. It does not hold
        # for a fieldsplit that solves the non-frozen block approximately,
        # nor for ngmres/anderson, whose iterate mixes past iterates -- there
        # the row can drift by roughly the inner-solve tolerance with
        # nothing raising, since SNES's own convergence test also sees a
        # zero residual there by construction. Restoring the snapshot here,
        # unconditionally, is what actually holds the pin.
        if freeze:
            self._Y[i].getArray()[local] = frozen_values
        reason = snes.getConvergedReason()
        if reason < 0:
            # petsc4py exposes no PETSc.ERR_* constants, so signal with
            # Firedrake's own exception. step()'s attempt loop catches this
            # specific exception around _take_stages, rejects the attempt
            # to the adapt loop (shrinking h by ts_adapt_scale_solve_failed)
            # and retries, only letting it propagate once retries are
            # exhausted -- see the comment there for the PETSc mechanism
            # this mirrors.
            raise ConvergenceError(f"stage {i} SNES diverged, reason {reason}")
        self._Y[i].copy(self._Ydot[i])
        self._Ydot[i].axpy(-1.0, self._Z)
        self._Ydot[i].scale(self._shift)
        if freeze:
            # A frozen row's implicit function is M_k Ydot_k alone (that is
            # what explicitly_governed_fields certifies), so its stage
            # equation is M_k Ydot_k = 0 and the correct derivative is
            # exactly zero. The generic (Y_i - Z_i) * shift formula above
            # instead gives the discarded limiter correction amplified by
            # 1/h -- the same At_ji/At_ii-style amplification this project
            # exists to eliminate, re-entering through Ydot rather than Y.
            # It would otherwise reach other rows' dF/du_t dependence on
            # this field via xdot in formSNESFunction/formSNESJacobian.
            self._Ydot[i].getArray()[local] = 0.0

    def _complete(self, tab, x, h):
        """x^{n+1}.

        Three cases, and the distinction is not cosmetic:

        * Stiffly accurate (b == A[-1] and bt == At[-1]): the completion IS the
          last stage value, which already carries the implicit contribution.
          Boundedness still holds, because on the components a limiter acts on
          Y[-1] was produced by convex-combination row s of [P | q] -- via the
          stage, not via a separate completion formula.
        * Purely explicit (At identically zero): no implicit contribution
          exists to drop, so the Shu-Osher completion row applies directly and
          is manifestly a convex combination.
        * Neither: a non-stiffly-accurate tableau WITH an implicit part would
          need the Butcher implicit completion, which the Shu-Osher form cannot
          express. Refuse rather than silently drop it.
        """
        s = len(tab.b)
        if np.allclose(tab.b, tab.A[-1], atol=1e-14) and np.allclose(
            tab.bt, tab.At[-1], atol=1e-14
        ):
            self._Y[-1].copy(x)
            return
        if not np.any(tab.At):
            result = self._Z  # reuse as scratch; Z is dead at this point
            result.set(0.0)
            if self._q[s] != 0.0:
                result.axpy(self._q[s], x)
            for j in range(s):
                psj = self._P[s, j]
                if psj == 0.0:
                    continue
                result.axpy(psj, self._Y[j])
                result.axpy(psj * h / self._r, self._L[j])
            result.copy(x)
            return
        raise ValueError(
            f"tableau {tab.name!r} is neither stiffly accurate nor purely "
            "explicit. Its completion needs the implicit weights bt, which the "
            "Shu-Osher form cannot express, so the implicit contribution would "
            "be silently dropped. Use a stiffly accurate tableau."
        )

    # -- SNES callbacks -------------------------------------------------------

    def formSNESFunction(self, args):
        """Stage residual.

        NOTE the signature: libpetsc4py calls this as ``formSNESFunction(args)``
        with the four-tuple as ONE positional argument, unlike
        ``formSNESJacobian(*args)``.
        """
        _snes, x, f, ts = args
        xdot = self._rhs
        x.copy(xdot)
        xdot.axpy(-1.0, self._Z)
        xdot.scale(self._shift)
        ts.computeIFunction(self._stage_time, x, xdot, f, True)
        self._apply_freeze_residual(x, f)

    def formSNESJacobian(self, snes, x, A, B, ts):
        xdot = self._rhs
        x.copy(xdot)
        xdot.axpy(-1.0, self._Z)
        xdot.scale(self._shift)
        ts.computeIJacobian(self._stage_time, x, xdot, self._shift, A, B, True)
        if self._freeze_active():
            # zeroRows clears the row but leaves the column alone -- correct
            # here, since other (non-frozen) rows must still see the pinned
            # unknown's coefficient; zeroing the column too would silently
            # change the equations those rows solve. But it does break
            # symmetry even where the caller's un-frozen operator was
            # symmetric, and there is no check for that: a caller running
            # -ksp_type cg against a problem that was symmetric before this
            # freeze applied gets an asymmetric operator with no error, just
            # a solver that may stagnate or converge to the wrong answer.
            A.zeroRows(self._frozen_rows, diag=1.0)
            if B is not None and B.handle != A.handle:
                B.zeroRows(self._frozen_rows, diag=1.0)
