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
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc

from firedrake_ts.tableaux import TABLEAUX, kraaijevanger_radius, shu_osher

__all__ = ["ARKSSP"]


class ARKSSP:
    """Additive RK with the explicit part in Shu-Osher form."""

    def __init__(self):
        self.tableau_name = "esdirk_gamma5"
        self.radius = None
        self.setup_calls = 0
        self.step_calls = 0
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
        """Register a callable fired on each explicit substage value."""
        self._limiter = limiter

    # -- the step -------------------------------------------------------------

    def step(self, ts):
        # Clear any stale error before this call -- a failure from a
        # previous setUp/step must never be re-raised against this one.
        self._error = None
        try:
            self.step_calls += 1
            tab = self._tab
            t = ts.getTime()
            h = ts.getTimeStep()
            x = ts.getSolution()
            s = len(tab.b)

            for i in range(s):
                self._shu_osher_predictor(x, h, i)
                if self._limiter is not None:
                    self._limiter(self._Y[i])
                self._build_offset(tab, x, h, i)
                if tab.At[i, i] > 0.0:
                    self._solve_stage(ts, tab, h, i)
                ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._rhs)
                self._rhs.copy(self._L[i])

            self._complete(tab, x, h)
            ts.setTime(t + h)
        except Exception as exc:
            self._error = exc
            raise

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
        # Initial guess: the previous stage value, or x^n for the first
        # implicit stage -- matching PETSc's own ARKIMEX. Guessing Z_i itself
        # would make Ydot_i identically zero already for any tableau whose
        # implicit form has no dependence on the state (as in this task's
        # decay shakedown), so SNES would report zero iterations even with
        # the callbacks wired correctly.
        if i > 0:
            self._Y[i - 1].copy(self._Y[i])
        else:
            ts.getSolution().copy(self._Y[i])
        snes.solve(None, self._Y[i])
        reason = snes.getConvergedReason()
        if reason < 0:
            # petsc4py exposes no PETSc.ERR_* constants, so signal with
            # Firedrake's own exception. Task 10 replaces this with a step
            # rejection routed through TSAdapt.
            raise ConvergenceError(f"stage {i} SNES diverged, reason {reason}")
        self._Y[i].copy(self._Ydot[i])
        self._Ydot[i].axpy(-1.0, self._Z)
        self._Ydot[i].scale(self._shift)

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
        xdot = self._Z.duplicate()
        x.copy(xdot)
        xdot.axpy(-1.0, self._Z)
        xdot.scale(self._shift)
        ts.computeIFunction(self._stage_time, x, xdot, f, True)

    def formSNESJacobian(self, snes, x, A, B, ts):
        xdot = self._Z.duplicate()
        x.copy(xdot)
        xdot.axpy(-1.0, self._Z)
        xdot.scale(self._shift)
        ts.computeIJacobian(self._stage_time, x, xdot, self._shift, A, B, True)
