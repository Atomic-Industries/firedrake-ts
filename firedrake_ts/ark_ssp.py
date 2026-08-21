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

import ufl
from firedrake import dmhooks, ufl_expr
from firedrake.assemble import get_assembler
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import DEFAULT_KSP_PARAMETERS, PETSc
from petsctools import OptionsManager

from firedrake_ts._petsc_shim import (
    ts_adapt_candidate_add,
    ts_adapt_candidates_clear,
    ts_adapt_choose,
    ts_get_adapt,
)
from firedrake_ts.solving_utils import field_rows, is_zero_form
from firedrake_ts.tableaux import TABLEAUX, kraaijevanger_radius, shu_osher

__all__ = ["ARKSSP"]


class ARKSSP:
    """Additive RK with the explicit part in Shu-Osher form.

    Options, all under the ``TS``'s own options prefix:

    ``-ts_ark_ssp_type <name>`` (default ``esdirk_gamma5``)
        Tableau, one of :data:`~firedrake_ts.tableaux.TABLEAUX`.

    ``-ts_ark_ssp_radius <float>`` (default: the Kraaijevanger radius
    ``R(A, b)`` of the chosen tableau)
        Radius of absolute monotonicity used to build the Shu-Osher form.
        Raises :class:`~firedrake_ts.tableaux.ShuOsherError`, naming both
        numbers, if the requested radius exceeds what the tableau admits.

    ``-ts_ark_ssp_fsal <bool>`` (default ``true``)
        Reuse the previous step's last stage derivative as ``Ẏ_0`` instead of
        assembling the mass matrix and solving for it again, when the
        candidate is verified to satisfy ``Ẏ_0``'s defining equation. Saves
        a mass assembly, an LU refactorisation and a solve per step --
        measured 8-21% of solve time, larger share on smaller problems. Has
        no effect for a tableau that is not stiffly accurate with
        ``ct[-1] == 1``, or whose last stage is not implicitly solved. See
        ``_try_fsal_stage0_ydot``.

    ``-ts_ark_ssp_fsal_rtol <float>`` (default ``1e-8``)
        Relative tolerance for that verification: the candidate is accepted
        when ``|F(t^n, y^n, Ẏ_cand)| <= rtol |F(t^n, y^n, 0)|``. The reused
        derivative is itself only as accurate as the stage solve that
        produced it, so this bounds how much of that error is inherited
        rather than recomputed; tighten it to force more fresh solves.
    """

    def __init__(self):
        self.tableau_name = "esdirk_gamma5"
        self.radius = None
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
        # Ownership-relative form of _frozen_rows, derived once in setUp.
        self._frozen_local = None
        # The values the frozen rows must hold through the stage currently
        # being solved, captured by _solve_stage before its warm start
        # overwrites them. This is the freeze's actual mechanism: it is what
        # makes the residual on those rows a constraint rather than the
        # identically-zero x - x it was when the target was read back out of
        # the very vector SNES iterates on. See _apply_freeze_residual.
        self._frozen_target = None
        # max |Y_i[frozen] - target| observed across the last stage solve,
        # BEFORE _solve_stage restores exactness. Nonzero means the solver
        # moved a pinned row and the residual constraint had to pull it back;
        # large means it did not manage to. Read by the freeze tests.
        self._frozen_drift = 0.0
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
        self._fsal_residual = None  # scratch for the FSAL candidate's residual
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
        self._mass_tensor = None
        self._mass_form = None
        self._mass_assembler = None
        # Global row indices whose dF/du_t is structurally zero. Distinct
        # from _frozen_rows, which is about dF/du -- see _find_algebraic_rows.
        self._algebraic_rows = None
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

        # FSAL reuse of the previous step's last stage derivative as Ydot_0.
        # _fsal_possible is a cheap structural pre-filter, set at setUp;
        # _fsal_valid says a step has completed since setUp. NEITHER decides
        # correctness -- the residual check in _try_fsal_stage0_ydot does.
        self.fsal = True
        self.fsal_rtol = 1e-8
        self._fsal_possible = False
        self._fsal_valid = False
        self._fsal_hits = 0
        self._fsal_misses = 0

    # -- options and lifecycle ------------------------------------------------

    def setFromOptions(self, ts):
        opts = PETSc.Options(ts.getOptionsPrefix() or "")
        self.tableau_name = opts.getString("ts_ark_ssp_type", self.tableau_name)
        radius = opts.getReal("ts_ark_ssp_radius", 0.0)
        self.radius = radius if radius > 0.0 else None
        self.fsal = opts.getBool("ts_ark_ssp_fsal", self.fsal)
        self.fsal_rtol = opts.getReal("ts_ark_ssp_fsal_rtol", self.fsal_rtol)

    def setUp(self, ts):
        # Clear any stale error before this call -- a failure from a
        # previous setUp/step must never be re-raised against this one.
        self._error = None
        try:
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
            # Separate from _rhs so an FSAL miss does not have to recompute
            # F(t^n, y^n, 0): storage, not a cached operator value.
            self._fsal_residual = sol.duplicate()
            self._last_x = sol.duplicate()
            self._zero_xdot = sol.duplicate()
            self._zero_xdot.set(0.0)

            # Structural pre-filter for FSAL: conditions under which
            # _Ydot[-1] could possibly be Ydot_0 for the next step. Only a
            # filter -- it exists to avoid spending a residual evaluation on
            # a tableau where reuse can never work, NOT to establish that
            # reuse is valid. _try_fsal_stage0_ydot verifies that itself.
            #   * At[-1, -1] > 0: the last stage is implicitly solved, so
            #     _solve_stage populated _Ydot[-1] at all. For a purely
            #     explicit tableau it never runs and _Ydot[-1] is garbage.
            #   * stiffly accurate: x^{n+1} == Y[-1], so _Ydot[-1] is the
            #     derivative at the state the next step starts from.
            #   * ct[-1] == 1: that stage sits at t^n + h == t^{n+1}.
            self._fsal_possible = bool(
                tab.At[-1, -1] > 0.0
                and tab.stiffly_accurate
                and abs(tab.ct[-1] - 1.0) <= 1e-14
            )
            # Vectors were just reallocated, so any candidate from a previous
            # setUp is gone regardless of what it held.
            self._fsal_valid = False

            self._check_stage_pattern_is_supported(tab)
            self._frozen_rows = self._find_frozen_rows(ts)
            # Ownership-relative indices, derived once here rather than per
            # residual evaluation. Safe to precompute -- unlike the operator
            # values this stepper has repeatedly been burned by caching --
            # because both inputs are fixed for the life of this setUp: the
            # row set is what was just detected, and the ownership range
            # belongs to the layout every work vector was duplicated from.
            self._frozen_local = (
                None
                if self._frozen_rows is None
                else self._frozen_rows - sol.getOwnershipRange()[0]
            )
            # No stage is in flight, so there is no pinned value yet. Left
            # None rather than stale: _apply_freeze_residual refuses to run
            # without one instead of silently pinning to a previous solve's.
            self._frozen_target = None
            self._frozen_drift = 0.0
            # Read by _reassemble_stage0_mass, so computed before it runs.
            self._algebraic_rows = self._find_algebraic_rows(ts)
            self._check_limiter_soundness()
            self._setup_stage0_mass_solve(ts)
        except Exception as exc:
            self._error = exc
            raise

    def reset(self, ts):
        # TSReset means the problem may change under this stepper, so the
        # carried-over FSAL candidate no longer describes the state the next
        # step will start from. Belt-and-braces: _try_fsal_stage0_ydot's
        # residual test would reject a stale candidate anyway.
        self._fsal_valid = False

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

    @staticmethod
    def _check_stage_pattern_is_supported(tab):
        """Refuse a tableau whose stages this stepper cannot actually take.

        ``_take_stages`` calls ``_solve_stage`` only when ``At[i, i] > 0``,
        and otherwise takes the stage value straight from the Shu-Osher
        predictor -- which encodes ``A`` alone. So for a stage with a zero
        diagonal but a nonzero row, two things go wrong silently:

        * ``Y_i`` is missing its ``h sum_{j<i} At_ij Ydot_j`` offset, so the
          stage value is simply wrong; and
        * ``_Ydot[i]`` is never written, while ``_build_offset`` reads it
          whenever ``At[k, i] != 0``, ``evaluatestep`` whenever
          ``bt[i] != 0`` and ``interpolate`` whenever ``d[i] != 0`` -- so
          those read whatever ``VecDuplicate`` left in the vector.

        This pattern (an explicit first stage followed by a zero-diagonal
        row that still couples to earlier stages) is common in published
        ARK-IMEX tableaux, and ``TABLEAUX`` is the documented extension
        point for ``-ts_ark_ssp_type``. Every other structural precondition
        here is checked at ``setUp``; this one was assumed.

        Stage 0 is exempt: ``At[0, 0] == 0`` with an all-zero row is the
        normal explicit-first-stage case that ``_prepare_stage0_ydot``
        handles.
        """
        offenders = [
            i
            for i in range(1, len(tab.b))
            if tab.At[i, i] <= 0.0 and bool(tab.At[i, :i].any())
        ]
        if offenders:
            raise ValueError(
                f"tableau {tab.name!r} has stage(s) {offenders} with a zero "
                f"implicit diagonal but a nonzero implicit row. This stepper "
                f"takes such a stage from the Shu-Osher predictor, which "
                f"encodes only the explicit tableau A, so the stage would "
                f"silently lose its h*sum(At_ij Ydot_j) offset and leave "
                f"Ydot[i] unwritten while later stages read it. Only stages "
                f"with At[i, i] > 0, or with an entirely zero At row, are "
                f"supported."
            )

    def _check_limiter_soundness(self):
        """Refuse a limiter that a stage solve could undo without a trace.

        The soundness hazard only exists where an implicit stage actually
        runs (tab.At[i, i] > 0.0, matching step()'s own guard on
        _solve_stage): a purely-explicit tableau such as ssprk2 never
        solves a stage, so there is nothing for the limiter's change to be
        undone by, regardless of whether any component was found freezable.
        """
        has_implicit_stage = self._tab.has_implicit_stage
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
        # Both halves of the partition are owned and cached by the context
        # (_algebraic_fields / _explicitly_governed_fields), which also spares
        # a second _classify_rows pass and a second derivation of nfields.
        # ctx.options_prefix is ts.getOptionsPrefix() modulo None vs "", which
        # resolve_fields normalises.
        ctx = dmhooks.get_appctx(ts.getDM())
        return field_rows(ctx._problem, ctx._explicitly_governed_fields)

    def _find_algebraic_rows(self, ts):
        """Global row indices of components with no time derivative.

        These are the rows where ``dF/du_t`` is structurally zero -- the
        algebraic constraints of a DAE. NOT the same set as
        ``_frozen_rows``, which is the rows with no ``dF/du``: one is about
        the mass operator, the other about the implicit operator, and a
        component can be in either, both or neither.

        ``_reassemble_stage0_mass`` gives these rows a unit diagonal so the
        mass matrix is invertible, which means the stage-0 solve returns
        ``-F_alg(t^n, y^n, 0)`` on them rather than a derivative. That value
        is meaningless -- there is no ``u_t`` in those equations -- and it
        does not stay put: ``_build_offset`` propagates it into every later
        stage as ``h At_ij Ydot_j``. ``_prepare_stage0_ydot`` therefore zeroes
        it, matching PETSc, which does exactly this immediately after the
        solve that produces its own ``Ydot0``
        (``VecISSet(Ydot0, ark->alg_is, 0.0)``, ``arkimex.c:1389``).
        """
        ctx = dmhooks.get_appctx(ts.getDM())
        return field_rows(ctx._problem, ctx._algebraic_fields)

    @staticmethod
    def _zero_rows(vec, rows):
        """Set ``vec`` to zero on the given global row indices, in place."""
        if rows is None:
            return
        lo, _ = vec.getOwnershipRange()
        vec.getArray()[rows - lo] = 0.0

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
        """Replace frozen rows of the residual with ``x - target``.

        ``target`` is ``self._frozen_target``, the snapshot ``_solve_stage``
        took before its warm start, NOT ``self._Y[self._stage]``. That
        distinction is the whole content of this method.

        ``self._Y[self._stage]`` is the very vector SNES iterates on -- it is
        the ``x`` handed to this callback -- so reading the target out of it
        made this ``x - x``: identically zero for whatever value the row
        happened to hold, carrying no information and constraining nothing.
        The pin then rested entirely on Newton's step being zero there too
        (zero residual, identity Jacobian row) plus ``_solve_stage``'s
        restore, which holds for plain Newton with an identity-preserving
        preconditioner and quietly does not for ``pc_type fieldsplit``
        solving the non-frozen block approximately, or for ``snes_type
        ngmres``/``anderson``, whose accepted iterate mixes past iterates.
        In those cases the row could drift by roughly the inner-solve
        tolerance with nothing raising, because SNES's own convergence test
        was looking at a residual that was zero there BY CONSTRUCTION.

        Against a real target the row's residual entry IS its drift, and two
        things follow with no new tolerance to pick. Newton's step on that row
        becomes ``delta = target - x``, so the solve actively corrects the row
        instead of passively leaving it alone -- measured under ``snes_type
        qn``, which does move it, the drift falls from 1.87e-06 to 2.48e-08.
        And the drift stops being invisible: the frozen entries are part of
        the residual vector SNES measures, so ``|drift| <= ||f||`` at the
        accepted iterate.

        That bound is worth stating carefully rather than overstating. It
        constrains the drift directly under an absolute tolerance, but only
        relative to the initial residual under a relative one -- the same
        ``qn`` run reports ``||f|| = 16.96`` at convergence with the OLD
        residual, so "SNES converged" was never on its own a bound on
        anything. What changed is that the frozen rows now contribute to the
        norm SNES tests at all, where before they contributed zero by
        construction.
        """
        if not self._freeze_active():
            return
        if self._frozen_target is None:
            raise ValueError(
                "the stage residual was evaluated with a live freeze but no "
                "pinned value; _apply_freeze_residual must only run inside "
                "_solve_stage, which captures self._frozen_target"
            )
        local = self._frozen_local
        f.getArray()[local] = x.getArray(readonly=True)[local] - self._frozen_target

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
        rather than return a plausible-looking, wrong answer. That refusal
        is checked unconditionally, for every tableau, NOT only when the
        first stage is explicit: ``M^-1 G``
        (``_TSContext._rhs_projection_solver``, used whenever ``G`` is not
        ``None``) rests on exactly the same affine-in-``u̇`` assumption, for
        every tableau, whether or not its first stage is explicit.

        Built unconditionally whenever the first stage is explicit --
        NOT gated on whether ``dF/du`` is structurally zero, and NOT gated
        on whether ``M`` itself depends on the state. Two previous versions
        each narrowed this on a structural predicate that looked sufficient
        and was not: skipping the solve when ``dF/du`` was structurally
        zero (false -- ``dF/du == 0`` says nothing about ``F(t^n, y^n, 0)``
        itself, e.g. a purely time-dependent or constant source), and
        reusing a single assembly of ``M`` across the whole solve unless
        ``dM/du`` was structurally nonzero (also false -- a structurally
        zero ``dM/du`` says nothing about ``M``'s dependence on ``t``, e.g.
        ``M = (1 + t) v``, or on some other mutable coefficient the form
        closes over; no structural predicate on the UFL form can rule that
        out in general). ``_prepare_stage0_ydot`` therefore reassembles
        ``M`` at the current ``(t^n, y^n)`` every ``step()``,
        unconditionally -- one extra assembly per step, not per stage,
        alongside the ``IFunction`` evaluation and mass solve that path
        already performs every step.

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

        ``self._mass_tensor`` is this method's OWN assembly -- never
        ``ctx._rhs_projection_mass_matrix``, even though the two are the
        same form. Both are now reassembled in place, but at DIFFERENT
        states: this one at ``(t^n, y^n)`` once per ``step()``, that one at
        ``(t_j, Y_j)`` before every stage's projection solve. Sharing one
        ``Mat`` would therefore have each reassembly silently overwrite the
        operator the other's ``KSP`` inverts, with whichever ran last
        deciding the state both solves see -- coupling two solves through a
        handle neither one's caller can see. This is safe to build even when
        ``G`` is ``None`` (it is not gated on ``G`` -- only
        ``_rhs_projection_solver`` is, which is why this method builds its
        own ``KSP`` instead of reusing that one).

        ``setUp`` may run more than once against the same stepper instance
        (a TS may be re-set-up after options change); destroy the previous
        ``KSP`` first rather than leaking it -- ``PETSc.KSP`` objects hold
        onto PETSc-side resources that Python's own garbage collector does
        not reliably reclaim promptly. ``self._mass_tensor``,
        ``self._mass_form`` and ``self._mass_assembler`` are reset before any
        check below can raise, so a failed second ``setUp`` does not leave
        them pointing at the previous run's objects.
        """
        if self._mass_ksp is not None:
            self._mass_ksp.destroy()
            self._mass_ksp = None
        self._mass_tensor = None
        self._mass_form = None
        self._mass_assembler = None
        tab = self._tab
        ctx = dmhooks.get_appctx(ts.getDM())
        mass_form = ufl_expr.derivative(ctx.F, ctx._xdot)

        # F affine in u_dot? d^2F/du_dot^2 == 0 is required both for
        # -M^-1 F(t^n, y^n, 0) to be the exact root rather than one Newton
        # step from zero (only relevant when the first stage is explicit),
        # and for M^-1 G to mean anything at all for ANY tableau whenever G
        # is not None -- see this method's docstring. Refuse rather than
        # hand back a plausible, silently wrong answer.
        if not is_zero_form(ufl_expr.derivative(mass_form, ctx._xdot)):
            # The Ẏ_0 half of this only applies when the first stage is
            # actually explicit; the M^-1 G half applies to every tableau,
            # which is why the check itself is unconditional (it runs before
            # the At[0, 0] > 0.0 early return below). Naming both as though
            # both were in play -- as this message used to -- asserts
            # tab.At[0, 0] == 0.0 as a fact even for a tableau where it is
            # false.
            affected = ["M^-1 G (whenever G is not None)"]
            if tab.At[0, 0] == 0.0:
                affected.insert(
                    0,
                    "Ẏ_0 = -M^-1 F(t^n, y^n, 0) (this tableau's first stage "
                    f"is explicit, tab.At[0, 0] == 0.0 for {tab.name!r})",
                )
            raise ValueError(
                "F is nonlinear in u̇ (d^2F/du̇^2 is not structurally "
                "zero). The mass matrix M = dF/du̇ is then not a "
                "well-defined constant operator, so "
                + " and ".join(affected)
                + (" are" if len(affected) > 1 else " is")
                + " one Newton step away from the true root rather than an "
                "exact answer, with no exception raised otherwise. Rewrite "
                "F to be affine in u̇."
            )

        # A purely explicit tableau (At identically zero) has no implicit
        # part at all: its completion (_complete's second branch) is the
        # Shu-Osher row, which reads only x^n, the stage values Y and the
        # explicit slopes L = M^-1 G. It never reads Ydot. So any term of F
        # beyond the mass form dF/du̇ -- i.e. any H in F = M u̇ + H -- is
        # computed by _prepare_stage0_ydot into Ydot[0] and then dropped
        # from every step. The method integrates M u̇ = G instead of the
        # requested M u̇ = G - H, with no exception and a plausible answer:
        # measured on u̇ + u = 0 posed with u in F rather than G, ssprk2
        # returns 0.0 where 1.0 is correct.
        #
        # Detected structurally, as "F with u̇ replaced by zero is not the
        # zero form", and refused here at setUp rather than per step: the
        # condition depends only on the tableau and the form, both fixed for
        # the whole solve, so there is nothing a later check could learn.
        if tab.purely_explicit:
            residual = ufl.replace(ctx.F, {ctx._xdot: ufl.zero(ctx._xdot.ufl_shape)})
            if not is_zero_form(residual):
                raise ValueError(
                    f"tableau {tab.name!r} is purely explicit (At is "
                    "identically zero), so it has no implicit part to "
                    "integrate F's non-mass terms with: its completion reads "
                    "only the stage values and the explicit slopes M^-1 G, "
                    "never Ydot. F here is not the mass form alone -- "
                    "F with u̇ set to zero is a nonzero form -- so those "
                    "terms would be silently dropped from every step, "
                    "integrating M u̇ = G rather than the F = G you posed. "
                    "Move them into G, or choose a tableau with an implicit "
                    "part (e.g. esdirk_gamma5)."
                )

        if tab.At[0, 0] > 0.0:
            return  # _solve_stage populates Ydot[0] normally, in _take_stages.

        self._mass_form = mass_form
        self._reassemble_stage0_mass(ctx)  # first assembly; builds the Mat.
        ksp = PETSc.KSP().create(comm=self._mass_tensor.comm)
        ksp.setOperators(self._mass_tensor.petscmat)
        parameters = {
            k: v for k, v in DEFAULT_KSP_PARAMETERS.items() if k != "mat_type"
        }
        prefix = (ts.getOptionsPrefix() or "") + "ark_ssp_stage0_mass_solver_"
        self._mass_options = OptionsManager(parameters, prefix)
        self._mass_options.set_from_options(ksp)
        self._mass_ksp = ksp

    def _reassemble_stage0_mass(self, ctx):
        """(Re)assemble ``self._mass_tensor`` at the current state.

        Called from ``_setup_stage0_mass_solve`` (``self._mass_tensor`` is
        ``None`` there, so this performs the first, allocating assembly)
        and, unconditionally, from ``_prepare_stage0_ydot`` on every
        ``step()`` (``self._mass_tensor`` already exists there, so this
        reassembles in place). ``ctx._x`` (== ``ctx._problem.u_restrict``,
        the same coefficient ``self._mass_form`` is built from) already
        holds the state this must be assembled at: the initial condition on
        the first call, and ``y^n`` on every later call, because
        ``_prepare_stage0_ydot`` calls ``ts.computeIFunction`` first, and
        that callback (``_TSContext.form_function``) copies the incoming
        state into ``ctx._x`` itself.

        ``tensor=self._mass_tensor`` reassembles into the existing
        ``Matrix``/``Mat`` in place once it exists, rather than allocating a
        fresh one each step: same sparsity, same PETSc handle the ``KSP``
        already has as its operator, so no ``setOperators`` call is needed
        after this -- PETSc's own assembly bumps the ``Mat``'s state
        counter, which is what tells the ``KSP`` its factorisation is
        stale.

        The assembler is built once for the same reason the form is, and
        caches only compiled-kernel and parloop wiring, never a value of the
        operator -- see the comment at the call below.

        Mirrors ``_TSContext._rhs_projection_mass_matrix``'s algebraic-row
        handling (unit diagonal on any row with a structurally zero
        ``dF/du̇``), since a fresh assembly would otherwise zero those rows
        out again and reintroduce the singular pivot that property exists
        to avoid.
        """
        if self._mass_assembler is None:
            # Built once, then called per step: the top-level ``assemble``
            # entry point re-runs form preprocessing (signature hashing,
            # function-space reconstruction, pyop2 cache probes) on every
            # call -- 365us against 27.5us for a prebuilt assembler, and
            # flat in problem size. This caches no value of the operator:
            # each ``assemble`` re-reads the coefficients' current ``dat``
            # values and re-executes the parloop, so the reassembly below
            # stays exactly as unconditional as it was. Mirrors
            # ``_TSContext._rhs_projection_mass_assembler``.
            self._mass_assembler = get_assembler(self._mass_form, bcs=ctx.bcs_F)
            self._mass_tensor = self._mass_assembler.allocate()
        self._mass_assembler.assemble(tensor=self._mass_tensor)
        if self._algebraic_rows is not None:
            self._mass_tensor.petscmat.zeroRows(self._algebraic_rows, diag=1.0)

    def _try_fsal_stage0_ydot(self, ts, t, x, reference):
        r"""Try the previous step's last stage derivative as ``Ẏ_0``.

        FSAL ("first same as last"): for a stiffly accurate tableau with
        ``ct[-1] == 1``, the last stage solve of step ``n`` already produced a
        derivative satisfying ``F(t^{n+1}, y^{n+1}, Ẏ) = 0`` -- which is
        exactly the equation ``Ẏ_0`` is defined by at step ``n+1``. So the
        value is reusable, and reusing it skips a mass assembly, an LU
        refactorisation and a solve. PETSc's own ARKIMEX does this
        (``arkimex.c:1356-1359``, ``FSAL_implicit``), recomputing only when
        ``ts->steprestart`` or ``ts->stepresize`` is set. Measured share of
        solve time for the path this replaces: 13-16% on a 2D reaction-
        diffusion problem with a state-dependent mass matrix, over
        2.4k-26k dofs.

        VERIFIED, not assumed. The reuse is accepted only if the candidate
        actually satisfies its defining equation: one ``IFunction``
        evaluation -- a vector assembly, no matrix assembly and no
        factorisation -- compared against ``reference``, the norm of
        ``F(t^n, y^n, 0)``, which is the right-hand side the fresh solve
        would have used and so the natural scale for this residual.

        Checking rather than enumerating preconditions is the point. A guard
        on ``(t, x)`` equality -- cache the end-of-step state, compare with
        ``VecEqual`` -- looks sufficient and is not: it establishes that the
        STATE is unchanged, not that the FORM is. ``DAEProblem`` explicitly
        supports a callback that mutates coefficients the form closes over
        between steps (``ts_solver.py``'s ``update_diffusivity`` example), and
        after such a mutation ``t`` and ``x`` are both unchanged while ``M``
        and ``F`` are not, so a state guard would reuse a stale derivative
        with nothing raising. That is the same shape as the five staleness
        defects this stepper has already had, every one of them a narrowing
        resting on an unverifiable claim about the form. The residual test
        has no such claim in it: it is robust against any reason the reuse
        could be invalid, including reasons not enumerated here, because the
        residual IS the definition. It also subsumes the state guard
        entirely, so no end-of-step copy of the solution is kept -- the reuse
        costs no additional memory, since ``_Ydot[-1]`` persists anyway.

        A miss costs one extra vector assembly and falls through to the fresh
        path. The candidate's residual goes into ``self._fsal_residual``
        rather than reusing ``self._rhs``, so ``F(t^n, y^n, 0)`` -- which the
        caller assembled to get ``reference``, and which is the right-hand
        side of the fresh path's mass solve -- survives a miss intact. It
        used to be reassembled, making a miss cost two assemblies where the
        docstring claimed one; nothing between the two calls can change the
        value, since neither touches a coefficient the form closes over.

        ``reference == 0.0`` (i.e. ``F(t^n, y^n, 0) == 0``, so ``Ẏ_0 = 0``)
        makes the test unsatisfiable for any nonzero candidate and it falls
        through -- correct, and the fresh path is trivial in that case.
        """
        if not (self.fsal and self._fsal_possible and self._fsal_valid):
            return False
        self._Ydot[-1].copy(self._Ydot[0])
        ts.computeIFunction(t, x, self._Ydot[0], self._fsal_residual, True)
        if self._fsal_residual.norm() > self.fsal_rtol * reference:
            self._fsal_misses += 1
            return False
        self._fsal_hits += 1
        # The algebraic- and frozen-row postcondition is applied by the
        # caller, on both paths at once. _Ydot[-1] came from _solve_stage's
        # (Y - Z) * shift, which is no more meaningful on an algebraic row
        # than the fresh path's -F_alg is, so it does need applying here too
        # -- but hoisting it makes the two paths' postconditions identical by
        # construction rather than by two blocks agreeing.
        return True

    def _prepare_stage0_ydot(self, ts, t, x):
        """Populate ``Ẏ_0`` once per ``step()``, before the retry loop.

        ``Ẏ_0 = -M(t^n, y^n)^-1 F(t^n, y^n, 0)`` depends only on
        ``(t^n, y^n)``, i.e. on ``(t, x)`` as passed in here -- neither of
        which changes across retries of the same step inside ``step()``'s
        reject loop (only ``h`` does). So this runs exactly once per
        ``step()`` call, not once per attempt inside ``_take_stages``, and
        it must be called with THIS step's ``t^n``/``y^n``: ``step()``
        calls it right after recording ``self._last_x``, before entering
        the retry loop.
        """
        if self._tab.At[0, 0] > 0.0:
            return  # _solve_stage populates _Ydot[0] normally, in _take_stages.
        # _setup_stage0_mass_solve builds this KSP unconditionally whenever
        # the first stage is explicit -- see that method's docstring for
        # why skipping it based on dF/du alone is unsound.
        ts.computeIFunction(t, x, self._zero_xdot, self._rhs, True)
        # F(t^n, y^n, 0) is the right-hand side of the mass solve below, so
        # its norm is already the natural scale for the residual test in
        # _try_fsal_stage0_ydot -- computed here, before that call, because
        # that call overwrites self._rhs.
        if not self._try_fsal_stage0_ydot(ts, t, x, self._rhs.norm()):
            # Reassemble M at THIS step's (t^n, y^n), unconditionally -- see
            # _setup_stage0_mass_solve's docstring for why no structural
            # predicate on the form is a sound basis for skipping this. ctx._x
            # already holds y^n, from the computeIFunction call above; the
            # form itself may also depend on t directly (e.g. M = (1 + t) v),
            # which ctx._time -- updated by that same computeIFunction call --
            # already reflects.
            ctx = dmhooks.get_appctx(ts.getDM())
            self._reassemble_stage0_mass(ctx)
            with self._mass_options.inserted_options():
                self._mass_ksp.solve(self._rhs, self._Ydot[0])
            self._Ydot[0].scale(-1.0)
        # Postcondition for BOTH paths, applied once so the two cannot drift.
        #
        # Algebraic rows: the unit diagonal _reassemble_stage0_mass put there
        # makes the fresh solve return -F_alg(t^n, y^n, 0) rather than a
        # derivative, and the FSAL copy carries _solve_stage's (Y - Z) * shift,
        # which is no more meaningful there. See _find_algebraic_rows; PETSc
        # does the same at arkimex.c:1389.
        #
        # Frozen rows: a frozen row's implicit function is the mass term
        # alone, so M Ẏ = 0 there and the correct derivative is exactly zero,
        # not whatever -M^-1 F(t, y, 0) gives on a row a limiter has no
        # business perturbing further.
        self._zero_rows(self._Ydot[0], self._algebraic_rows)
        if self._freeze_active():
            self._zero_rows(self._Ydot[0], self._frozen_rows)

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
            # IN/OUT for TSAdaptChoose: whether the PREVIOUS attempt at this
            # step was accepted. PETSc's TSStep_ARKIMEX declares the same
            # thing once per step (accept = PETSC_TRUE, arkimex.c:1343) and
            # clears it at its reject_step label (:1529), which both of this
            # loop's rejection paths correspond to. See ts_adapt_choose.
            last_accepted = True
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
                    # Reaches PETSc's reject_step, which clears accept -- so
                    # the next TSAdaptChoose in this step must see false even
                    # though this path never calls it.
                    last_accepted = False
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
                        adapt, tab.order, tab.order, 1.0, float(s), True
                    )
                    next_h, accept = ts_adapt_choose(adapt, ts, h, last_accepted)
                except Exception:
                    self._last_x.copy(x)
                    raise
                if accept:
                    ts.setTime(t + h)
                    ts.setTimeStep(next_h)
                    # _Ydot[-1] now belongs to the accepted attempt, so it is
                    # a candidate for the next step's Ydot_0. Set here rather
                    # than anywhere earlier because a rejected attempt's
                    # _Ydot[-1] is not a derivative at the state the next
                    # step starts from. Whether the candidate is actually
                    # usable is still decided by _try_fsal_stage0_ydot's
                    # residual test, not by this flag.
                    self._fsal_valid = True
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
                last_accepted = False
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
            # Written straight into _L[i]: form_rhs_function copies out of
            # ctx._G_or_projected_G into whatever Vec it is handed, so the
            # _rhs round trip was a full-state VecCopy per stage for nothing.
            ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._L[i])

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
        if h is None:
            # Same guard, and same message shape, as interpolate's. Without
            # it the axpy below raised an unactionable "unsupported operand
            # type(s) for *: 'NoneType' and 'numpy.float64'".
            raise ValueError("evaluatestep called before any step was taken")
        x = self._last_x
        has_implicit = tab.has_implicit_part
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
        # Same guard evaluatestep carries, for the same reason. A purely
        # explicit tableau (bt identically zero) has no implicit part, so the
        # interpolant must have no Ydot term: the implicit dense-output
        # coefficient collapses to d_i theta + (0 - d_i) theta^2 =
        # d_i theta (1 - theta), which is NOT zero inside the step even
        # though it vanishes at both ends. For ssprk2 (d = [1, 0]) that puts
        # a spurious 0.25 h Ydot_0 at theta = 0.5. Consistency is unaffected:
        # at theta = 1 the explicit coefficient is already b_i on its own, so
        # X(1) = y^n + h sum b_i L_i is still the completion.
        #
        # Dropping the term also stops this reading _Ydot[i] for i >= 1 at
        # all, which for such a tableau _solve_stage never writes -- those
        # Vecs hold whatever VecDuplicate left in them. ssprk2 escapes that
        # today only because its d[1] is 0.0, i.e. by accident of one
        # coefficient rather than by construction.
        has_implicit = tab.has_implicit_part
        self._last_x.copy(U)
        for i in range(len(tab.b)):
            impl = (
                tab.d[i] * theta + (tab.bt[i] - tab.d[i]) * theta**2
                if has_implicit
                else 0.0
            )
            expl = tab.d[i] * theta + (tab.b[i] - tab.d[i]) * theta**2
            if impl != 0.0:
                U.axpy(h * impl, self._Ydot[i])
            if expl != 0.0:
                U.axpy(h * expl, self._L[i])
        # No return value: petsc4py's TSInterpolate_Python discards whatever
        # interpolate() returns -- there is no flag out-param and no
        # truthiness check, in contrast to TSEvaluateStep_Python's `done`.

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
        # Capture the value the frozen rows must keep, BEFORE the warm start
        # below replaces the whole vector with the previous stage's value (or
        # x^n) and discards the predictor's -- and any limiter's -- value on
        # exactly the rows the freeze exists to protect.
        #
        # This snapshot is the freeze's mechanism, not merely its bookkeeping:
        # _apply_freeze_residual builds the frozen rows' residual against it,
        # which is what makes those rows a real constraint on the solve. See
        # that method for why reading the target back out of self._Y[i]
        # instead made the residual identically zero.
        freeze = self._freeze_active()
        if freeze:
            local = self._frozen_local
            # No .copy(): numpy advanced indexing already returns a new array.
            self._frozen_target = self._Y[i].getArray(readonly=True)[local]
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
            # Start the frozen rows AT the target, so their residual is zero
            # to begin with and the solve has nothing to correct there unless
            # it moves them itself. Without this the warm start would hand
            # SNES an initial residual equal to the whole limiter correction,
            # which is a real constraint now but a needless one to impose.
            self._Y[i].getArray()[local] = self._frozen_target
        snes.solve(None, self._Y[i])
        # Record how far the solve moved the pinned rows, then restore them
        # exactly.
        #
        # The residual built in _apply_freeze_residual is what enforces the
        # pin; this restore is no longer that mechanism. What it still buys is
        # EXACTNESS: a converged solve holds the pin only to whatever residual
        # norm SNES accepted, and the boundedness result this freeze serves is
        # a claim about the limited value surviving bit-for-bit, not to
        # 1e-8. So the row is written back rather than left at "close".
        #
        # What the drift actually threatens is worth being precise about, and
        # it is NOT the pinned rows: those are restored, so the boundedness
        # claim holds whatever the solver did to them. It is the OTHER rows.
        # If the solve moved a frozen row and the non-frozen rows equilibrated
        # against the moved value, writing the row back leaves those rows
        # inconsistent by the drift -- so _frozen_drift is the size of the
        # inconsistency this restore introduces, bounded by the residual norm
        # SNES accepted. Benign at solver tolerance, which is why this records
        # the number rather than raising on it: any threshold to raise at would
        # be invented, and the answer is correct either way. It is exposed so a
        # solver that fights the pin can be diagnosed instead of hidden, which
        # is what the old self-referential residual made impossible.
        if freeze:
            ya = self._Y[i].getArray()
            # abs() rather than numpy.abs: the builtin dispatches elementwise
            # on the array, so this needs no import. len(local) can be 0 on a
            # rank owning none of the field's dofs, where .max() would raise.
            self._frozen_drift = (
                float(abs(ya[local] - self._frozen_target).max()) if len(local) else 0.0
            )
            ya[local] = self._frozen_target
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
        if tab.stiffly_accurate:
            self._Y[-1].copy(x)
            return
        if tab.purely_explicit:
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
