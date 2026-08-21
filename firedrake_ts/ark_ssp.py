"""An additive Runge-Kutta stepper as a ``TSPYTHON`` type.

Selected with ``-ts_type python -ts_python_type firedrake_ts.ark_ssp.ARKSSP``.

The explicit part is advanced in canonical Shu-Osher form so that a limiter
applied to a stage value is mathematically sound: each stage is a convex
combination of forward-Euler steps taken from previously-limited stage values.

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
        # makes the residual on those rows a constraint rather.
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
        # (_setup_stage0_mass_solve, _prepare_stage0_ydot).
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
        """Register a callable fired on each explicit substage value."""
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
        diagonal but a nonzero row, two things go wrong:

        * ``Y_i`` is missing its ``h sum_{j<i} At_ij Ydot_j`` offset, so the
          stage value is simply wrong; and
        * ``_Ydot[i]`` is never written, while ``_build_offset`` reads it
          whenever ``At[k, i] != 0``, ``evaluatestep`` whenever
          ``bt[i] != 0`` and ``interpolate`` whenever ``d[i] != 0`` -- so
          those read whatever ``VecDuplicate`` left in the vector.

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
        """Refuse a limiter that a stage solve could undo without a trace."""
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
        """Whether the freeze should actually pin rows this solve."""
        return self._frozen_rows is not None and self._limiter is not None

    def _apply_freeze_residual(self, x, f):
        """Replace frozen rows of the residual with ``x - target``.

        ``target`` is ``self._frozen_target``, the snapshot ``_solve_stage``
        took before its warm start, NOT ``self._Y[self._stage]``. That
        distinction is the whole content of this method.
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

        Built unconditionally whenever the first stage is explicit --
        NOT gated on whether ``dF/du`` is structurally zero, and NOT gated
        on whether ``M`` itself depends on the state.

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
        # is not None -- see this method's docstring.
        if not is_zero_form(ufl_expr.derivative(mass_form, ctx._xdot)):
            # The Ẏ_0 half of this only applies when the first stage is
            # actually explicit; the M^-1 G half applies to every tableau,
            # which is why the check itself is unconditional (it runs before
            # the At[0, 0] > 0.0 early return below).
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
                + " one Newton step away from the true root. Rewrite "
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
                    "part."
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
        """(Re)assemble ``self._mass_tensor`` at the current state."""
        if self._mass_assembler is None:
            # Built once, then called per step: the top-level ``assemble``
            # entry point re-runs form preprocessing (signature hashing,
            # function-space reconstruction, pyop2 cache probes) on every
            # call.
            self._mass_assembler = get_assembler(self._mass_form, bcs=ctx.bcs_F)
            self._mass_tensor = self._mass_assembler.allocate()
        self._mass_assembler.assemble(tensor=self._mass_tensor)
        if self._algebraic_rows is not None:
            self._mass_tensor.petscmat.zeroRows(self._algebraic_rows, diag=1.0)

    def _try_fsal_stage0_ydot(self, ts, t, x, reference):
        r"""Try the previous step's last stage derivative as ``Ẏ_0``.

        FSAL ("first same as last"): for a stiffly accurate tableau with
        ``ct[-1] == 1``, the last stage solve of step ``n`` already produced a
        derivative satisfying ``F(t^{n+1}, y^{n+1}, Ẏ) = 0``.

        ``reference == 0.0`` (i.e. ``F(t^n, y^n, 0) == 0``, so ``Ẏ_0 = 0``)
        makes the test unsatisfiable for any nonzero candidate and it falls
        through.
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
        # construction.
        return True

    def _prepare_stage0_ydot(self, ts, t, x):
        """Populate ``Ẏ_0`` once per ``step()``, before the retry loop."""
        if self._tab.At[0, 0] > 0.0:
            return  # _solve_stage populates _Ydot[0] normally, in _take_stages.
        # _setup_stage0_mass_solve builds this KSP unconditionally whenever
        # the first stage is explicit.
        ts.computeIFunction(t, x, self._zero_xdot, self._rhs, True)
        # F(t^n, y^n, 0) is the right-hand side of the mass solve below, so
        # its norm is already the natural scale for the residual test in
        # _try_fsal_stage0_ydot.
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
        # does the same.
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
            # the option directly.
            opts = PETSc.Options(ts.getOptionsPrefix() or "")
            max_reject = opts.getInt(
                "ts_max_step_rejections", opts.getInt("ts_max_reject", 10)
            )
            attempts = 1 << 30 if max_reject < 0 else max(1, max_reject + 1)
            adapt = ts_get_adapt(ts)
            last_reject_cause = None
            last_accepted = True
            for _ in range(attempts):
                self._last_h = h
                try:
                    self._take_stages(ts, tab, self._last_x, h, s)
                except ConvergenceError as exc:
                    # A stage's SNES diverged (_solve_stage). Reject this
                    # attempt to the adapt loop and retry with a smaller h,
                    # mirroring PETSc's own TSAdaptCheckStage.
                    last_reject_cause = str(exc)
                    # Reaches PETSc's reject_step, which clears accept -- so
                    # the next TSAdaptChoose in this step must see false even
                    # though this path never calls it.
                    last_accepted = False
                    scale_solve_failed = opts.getReal(
                        "ts_adapt_scale_solve_failed", 0.25
                    )
                    self._last_x.copy(x)
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
                    # a candidate for the next step's Ydot_0.
                    self._fsal_valid = True
                    return
                # Rejected: restore x (ts->vec_sol) to the pre-step value
                # before retrying with the smaller next_h. self._last_x itself
                # is untouched, so the retried _take_stages still predicts from
                # the correct x^n.
                last_reject_cause = (
                    f"the adapt controller declined the completed step "
                    f"(h {h:.6g} -> {next_h:.6g})"
                )
                last_accepted = False
                self._last_x.copy(x)
                h = next_h
                ts.setTimeStep(h)
            # Retries exhausted.
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
            # whatever the stage solve does to it afterward.
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
        """Write the order-``order`` completion into ``U``."""
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
        limit bounded.
        """
        tab = self._tab
        h = self._last_h
        if h is None:
            raise ValueError("interpolate called before any step was taken")
        theta = (t - (ts.getTime() - h)) / h
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
        the limiter has already seen.
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
        # x^n) and discards the predictor's value on the rows the freeze exists
        # to protect.
        freeze = self._freeze_active()
        if freeze:
            local = self._frozen_local
            # No .copy(): numpy advanced indexing already returns a new array.
            self._frozen_target = self._Y[i].getArray(readonly=True)[local]
        # Initial guess: the previous stage value, or x^n for the first
        # implicit stage -- matching PETSc's own ARKIMEX.
        if i > 0:
            self._Y[i - 1].copy(self._Y[i])
        else:
            ts.getSolution().copy(self._Y[i])
        if freeze:
            # Start the frozen rows AT the target, so their residual is zero
            # to begin with and the solve has nothing to correct there unless
            # it moves them itself.
            self._Y[i].getArray()[local] = self._frozen_target
        snes.solve(None, self._Y[i])
        # Record how far the solve moved the pinned rows, then restore them
        # exactly.
        if freeze:
            ya = self._Y[i].getArray()
            self._frozen_drift = (
                float(abs(ya[local] - self._frozen_target).max()) if len(local) else 0.0
            )
            ya[local] = self._frozen_target
        reason = snes.getConvergedReason()
        if reason < 0:
            # petsc4py exposes no PETSc.ERR_* constants, so signal with
            # Firedrake's own exception.
            raise ConvergenceError(f"stage {i} SNES diverged, reason {reason}")
        self._Y[i].copy(self._Ydot[i])
        self._Ydot[i].axpy(-1.0, self._Z)
        self._Ydot[i].scale(self._shift)
        if freeze:
            # A frozen row's implicit function is M_k Ydot_k alone (that is
            # what explicitly_governed_fields certifies), so its stage
            # equation is M_k Ydot_k = 0 and the correct derivative is
            # exactly zero.
            self._Ydot[i].getArray()[local] = 0.0

    def _complete(self, tab, x, h):
        """x^{n+1}.

        Three cases:

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
          express.
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
            "be dropped. Use a stiffly accurate tableau."
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
            # unknown's coefficient.
            A.zeroRows(self._frozen_rows, diag=1.0)
            if B is not None and B.handle != A.handle:
                B.zeroRows(self._frozen_rows, diag=1.0)
