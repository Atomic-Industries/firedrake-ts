# Shu–Osher ARK-IMEX `TSPYTHON` Stepper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TSPYTHON` time stepper to `firedrake_ts` that runs the explicit part of an additive Runge–Kutta method in canonical Shu–Osher form, so a per-stage limiter acts on stage *values* and the SSP boundedness proof actually applies.

**Architecture:** Butcher tableaux live in a Firedrake-free numpy module that also does the Shu–Osher conversion and the `rk-method-spec.md` acceptance predicates. A `TSPYTHON` class owns the step loop, deriving Shu–Osher coefficients at `setUp`. `_TSContext` gains two UFL-structural predicates — which components are algebraic, and which have no implicit operator — used to restrict the mass-matrix projection and to freeze limited components during the implicit solve.

**Tech Stack:** Python 3.12, Firedrake (`/opt/firedrake`), petsc4py 3.25.0 / PETSc 3.25.0, numpy, pytest, ruff, uv, ctypes.

**Spec:** `docs/superpowers/specs/2026-08-04-shu-osher-tspython-stepper-design.md`
**Issue:** COOL-193. **Tableau requirements:** `/sundrake/local/fill/rk-method-spec.md`

## Global Constraints

- **Test command is `uv run pytest --verbose <path>`.** PETSc parses `sys.argv` and will emit "Option left: name:-q" warnings for short pytest flags. Use long forms (`--verbose`, not `-v`; no `-x`, `-q`, `-k`). Warnings about unused options are harmless noise, not failures.
- **`firedrake_ts/tableaux.py` must not import Firedrake, PETSc, or petsc4py.** numpy only. A test asserts this.
- Python 3.12 target, ruff `line-length = 88`. A pre-commit hook runs `ruff check` and `ruff format`; both must pass before a commit succeeds.
- **Never construct a Firedrake `LinearSolver` or `NonlinearVariationalSolver` on the TS's function space.** Both are `NonlinearVariationalSolver`s, which register `_SNESContext.form_function` on the shared DM. `SNESSetFunction` is DM-scoped, so this displaces `SNESTSFormFunction` on the TS's own SNES and every stage residual silently becomes zero. Use a bare `PETSc.KSP` (see `_rhs_projection_solver` in `firedrake_ts/solving_utils.py` and its comment).
- Commit after every task. Never amend a previous task's commit.
- Do **not** revive `repair_ts_snes_callbacks`, `set_stage_hook`, or `stage_hook_propagates` from git history. The first was retired by PR #6; the latter two are the abandoned COOL-189 approach.

## File Structure

| File | Responsibility |
|---|---|
| `firedrake_ts/tableaux.py` (new) | Butcher tableau data, Shu–Osher conversion, Kraaijevanger radius, acceptance predicates. Firedrake-free. |
| `firedrake_ts/ark_ssp.py` (new) | The `ARKSSP` `TSPYTHON` class: step loop, stage solves, limiter, dense output. |
| `firedrake_ts/_petsc_shim.py` (new, Task 10) | ctypes for the four `TSAdapt` entry points petsc4py does not bind. |
| `firedrake_ts/solving_utils.py` (modify) | `split()` fix (Task 1); structural predicates (Task 6); restricted projection (Task 7). |
| `firedrake_ts/ts_solver.py` (modify) | `DAESolver.set_stage_limiter` forwarding (Task 9). |
| `tests/test_fieldsplit.py` (new) | Task 1. |
| `tests/test_tableaux.py` (new) | Tasks 2, 3. |
| `tests/test_ark_ssp.py` (new) | Tasks 4, 5, 8. |
| `tests/test_singular_mass.py` (new) | Tasks 6, 7, 8. |
| `tests/test_bounds.py` (new) | Task 9. |
| `tests/test_adapt.py` (new) | Task 10. |
| `tests/test_interpolate.py` (new) | Task 11. |

## Milestone Map and the M4 Gate

| Task | Milestone | Deliverable |
|---|---|---|
| 1 | M0 | `split()` fix — unblocks fieldsplit on mixed problems |
| 2–3 | M1 | `tableaux.py` |
| 4 | M2a | `ARKSSP` skeleton + IMEX Euler (Butcher path) |
| 5 | M2b | Shu–Osher loop, explicit-only tableau |
| 6–7 | M3 | Structural predicates + restricted projection |
| 8 | M3 | Combined ARK with freeze; singular-mass rungs |
| **9** | **M4** | **Limiter + DG1 bounds test — STOP AND TAKE STOCK** |
| 10 | M5 | `_petsc_shim.py` + `evaluatestep` + adaptivity |
| 11 | M6 | `interpolate` / dense output |

> ### ⛔ CHECKPOINT AFTER TASK 9 — DO NOT PROCEED AUTOMATICALLY
>
> Task 9 is where the entire premise of COOL-193 is either confirmed or refuted. **Stop after Task 9 and report to the user, whether it passes or fails.** Do not start Task 10.
>
> **If the bounds test passes:** report the measured `min`/`max`, and confirm the negative control genuinely failed (reproducing roughly `−0.0099 / +1.0100`). A bounds test that passes while the control *also* passes proves nothing — the problem was not challenging, and the test needs strengthening before M5/M6 are worth building.
>
> **If the bounds test fails:** stop and debug rather than proceeding. Diagnostic order:
> 1. Print `min(P)`, `min(q)`, and each row sum of `[P | q]`. Every row must be a nonnegative partition of unity. A negative entry means `r > R(A,b)` — Task 3's gate should have caught it.
> 2. Confirm the limiter is firing on *every* explicit substage, and **before** the value is consumed. Log the call count per step; for `esdirk_gamma5` expect 4.
> 3. Confirm the frozen components really are frozen: after the implicit solve, the frozen rows of `Y_i` must equal the limited Shu–Osher predictor bit-for-bit. If they drifted, the identity-row freeze (Task 8) is not being applied and the solve dragged the value back to the unlimited `Z_i` — the exact `ã_ji/ã_ii` failure COOL-193 measured at 1.879988.
> 4. Check `Δt ≤ R(A,b)·τ_ZS`. The bound is conditional; a too-large step legitimately violates bounds without any code defect.
>
> Report which of the four it was before changing anything.

---

### Task 1: Fix `_TSContext.split()` so fieldsplit works at all

`_TSContext.split()` reads `problem.u`, but `DAEProblem` defines only `u_restrict`. Any mixed problem with `pc_type: fieldsplit` dies in `DMCreateFieldDecomposition` with an unhandled `AttributeError`. Firedrake's own `_SNESContext.split()` (`/opt/firedrake/firedrake/solving_utils.py:379`) uses `problem.u_restrict`; ours is a stale copy predating Firedrake's restricted-function-space rename.

This is first because Task 7's singular-mass verification needs fieldsplit to work.

**Files:**
- Modify: `firedrake_ts/solving_utils.py:185,229,230,233,238`
- Test: `tests/test_fieldsplit.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: no new API. `_TSContext.split(fields)` stops raising.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fieldsplit.py`:

```python
"""Fieldsplit preconditioning must work on mixed DAE problems."""

import pytest
from firedrake import *

import firedrake_ts

FIELDSPLIT = {
    "ts_type": "arkimex",
    "ts_arkimex_type": "2c",
    "ts_adapt_type": "none",
    "ts_time_step": 0.05,
    "ts_exact_final_time": "stepover",
    "snes_type": "newtonls",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
}


def _coupled(with_G):
    """Two diffusing fields on a mixed space, optionally IMEX-split."""
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    w.sub(0).assign(1.0)

    F = (
        inner(adot, va) * dx
        + inner(grad(a), grad(va)) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    G = (-inner(a - b, va) * dx + inner(a - b, vb) * dx) if with_G else None
    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.2), G=G)
    return firedrake_ts.DAESolver(
        problem, solver_parameters=FIELDSPLIT, options_prefix=""
    ), w


@pytest.mark.parametrize("with_G", [False, True])
def test_split_does_not_raise(with_G):
    """split() must not die on DAEProblem's attribute names."""
    solver, _ = _coupled(with_G)
    splits = solver._ctx.split([[0], [1]])
    assert len(splits) == 2


@pytest.mark.parametrize("with_G", [False, True])
def test_fieldsplit_solve_runs(with_G):
    """A full solve under pc_type: fieldsplit must complete and advance."""
    solver, w = _coupled(with_G)
    solver.solve()
    assert solver.ts.getStepNumber() > 0
    assert norm(w.sub(0)) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_fieldsplit.py`
Expected: all four FAIL. `test_split_does_not_raise` with `AttributeError: 'DAEProblem' object has no attribute 'u'`; `test_fieldsplit_solve_runs` with a PETSc "Unhandled Python Exception" traced through `DMCreateFieldDecomposition`.

- [ ] **Step 3: Rename the five occurrences**

In `firedrake_ts/solving_utils.py`, inside `split()`, replace `problem.u` with `problem.u_restrict` at all five sites. They are:

```python
            us = problem.u_restrict.subfunctions          # was line 185
...
            F = replace(F, {problem.u_restrict: u})       # was line 229
            J = replace(J, {problem.u_restrict: u})       # was line 230
            if problem.Jp is not None:
                Jp = splitter.split(problem.Jp, argument_indices=(field, field))
                Jp = replace(Jp, {problem.u_restrict: u})  # was line 233
            else:
                Jp = None
            if problem.G is not None:
                G = splitter.split(problem.G, argument_indices=(field,))
                G = replace(G, {problem.u_restrict: u})    # was line 238
```

Do **not** add `self.u = u` to `DAEProblem` instead. Matching Firedrake's current `_SNESContext.split()` keeps the two implementations in step; adding an alias hides the divergence.

**While you are in this method, confirm one thing** (spec §10). `split()` constructs sub-contexts without passing `options_prefix`, `project_rhs`, or `rhs_projection_parameters`, so a split sub-context carrying a `G` would build a default-parameter projection KSP with no prefix. It should be unreachable — the projection is only triggered from `form_rhs_function`, which split contexts do not serve. Verify that by adding a temporary `raise` inside `_rhs_projection_solver` and running `tests/test_fieldsplit.py::test_fieldsplit_solve_runs[True]`; if it does not fire, remove the `raise` and note the confirmation in the commit message. If it *does* fire, stop and report — the sub-context needs those kwargs forwarded, which is a change in scope.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_fieldsplit.py`
Expected: 4 passed.

Then confirm nothing regressed: `uv run pytest --verbose tests/`
Expected: all previously-passing tests still pass (7 in `test_imex.py` plus the rest).

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/solving_utils.py tests/test_fieldsplit.py
git commit -m "Fix split() to use DAEProblem.u_restrict

_TSContext.split() read problem.u, which DAEProblem never defines --
it sets u_restrict. Any mixed problem with pc_type: fieldsplit died in
DMCreateFieldDecomposition with an unhandled AttributeError, so
fieldsplit preconditioning was unavailable on every mixed DAE.

Firedrake's own _SNESContext.split() uses u_restrict; ours was a stale
copy from before the restricted-function-space rename."
```

---

### Task 2: Shu–Osher conversion and Kraaijevanger radius

Pure numpy. This is where the mathematical risk of the whole project lives, so it gets tested at machine precision with no PDE machinery.

**Files:**
- Create: `firedrake_ts/tableaux.py`
- Test: `tests/test_tableaux.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `shu_osher(A: np.ndarray, b: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]` returning `(P, q)`, both sized `(s+1, s+1)` and `(s+1,)`.
  - `butcher_to_K(A: np.ndarray, b: np.ndarray) -> np.ndarray` returning the `(s+1, s+1)` matrix `K = [[A, 0], [bᵀ, 0]]`.
  - `kraaijevanger_radius(A: np.ndarray, b: np.ndarray, hi: float = 50.0) -> float`.
  - `ShuOsherError(ValueError)` raised when `r` exceeds the radius.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tableaux.py`:

```python
"""Tableau algebra. No Firedrake, no PETSc -- pure numpy at machine precision."""

import numpy as np
import pytest

from firedrake_ts.tableaux import (
    ShuOsherError,
    butcher_to_K,
    kraaijevanger_radius,
    shu_osher,
)

# Ketcheson's optimal SSPRK(3,2) in stiffly accurate form, the explicit part of
# rk-method-spec.md 5.1. Four stages, R(A,b) = 2.
SSPRK32_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
    ]
)
SSPRK32_B = np.array([1 / 3, 1 / 3, 1 / 3, 0.0])

HEUN_A = np.array([[0.0, 0.0], [1.0, 0.0]])
HEUN_B = np.array([0.5, 0.5])


def _stages_shu_osher(z, P, q, r):
    """Stage values of the Shu-Osher recursion on y' = z y, y(0) = 1."""
    n = len(q)
    Y = np.zeros(n, dtype=complex)
    for i in range(n):
        Y[i] = q[i]
        for j in range(i):
            Y[i] += P[i, j] * Y[j] * (1.0 + z / r)
    return Y


def _stages_butcher(z, A, b):
    """The same stage values from the Butcher map."""
    s = len(b)
    Y = np.zeros(s + 1, dtype=complex)
    for i in range(s):
        Y[i] = 1.0 + z * sum(A[i, j] * Y[j] for j in range(i))
    Y[s] = 1.0 + z * sum(b[j] * Y[j] for j in range(s))
    return Y


def test_ssprk32_gives_the_textbook_form():
    """At r = 2 the conversion must reproduce the hand-derived coefficients."""
    P, q = shu_osher(SSPRK32_A, SSPRK32_B, 2.0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 1 / 3, 1 / 3], atol=1e-14)
    expected_P = np.zeros((5, 5))
    expected_P[1, 0] = 1.0
    expected_P[2, 1] = 1.0
    expected_P[3, 2] = 2 / 3
    expected_P[4, 2] = 2 / 3
    np.testing.assert_allclose(P, expected_P, atol=1e-14)


def test_heun_gives_the_textbook_form():
    """u1 = un + h L(un);  u2 = 1/2 un + 1/2 (u1 + h L(u1))."""
    P, q = shu_osher(HEUN_A, HEUN_B, 1.0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.5], atol=1e-14)
    expected_P = np.zeros((3, 3))
    expected_P[1, 0] = 1.0
    expected_P[2, 1] = 0.5
    np.testing.assert_allclose(P, expected_P, atol=1e-14)


@pytest.mark.parametrize(
    "A,b,r", [(SSPRK32_A, SSPRK32_B, 2.0), (HEUN_A, HEUN_B, 1.0)]
)
def test_shu_osher_reproduces_the_butcher_map(A, b, r):
    """The two representations must agree to machine precision."""
    P, q = shu_osher(A, b, r)
    for z in [-0.3, -1.0 + 0.4j, 0.7, -2.5, 1.5 - 2.0j]:
        np.testing.assert_allclose(
            _stages_shu_osher(z, P, q, r), _stages_butcher(z, A, b), atol=1e-14
        )


@pytest.mark.parametrize(
    "A,b,r", [(SSPRK32_A, SSPRK32_B, 2.0), (HEUN_A, HEUN_B, 1.0)]
)
def test_rows_are_nonnegative_partitions_of_unity(A, b, r):
    """Every row, INCLUDING the completion row, must be a convex combination.

    This is what makes the accepted step bounded and retires the post-step
    clamp of rk-method-spec.md R8.
    """
    P, q = shu_osher(A, b, r)
    assert P.min() >= -1e-14
    assert q.min() >= -1e-14
    np.testing.assert_allclose(P.sum(axis=1) + q, 1.0, atol=1e-14)


def test_completion_row_equals_last_stage_row_under_stiff_accuracy():
    """b == A[s-1,:] means the completion IS the last stage."""
    P, q = shu_osher(SSPRK32_A, SSPRK32_B, 2.0)
    np.testing.assert_allclose(P[-1], P[-2], atol=1e-14)
    np.testing.assert_allclose(q[-1], q[-2], atol=1e-14)


def test_kraaijevanger_radius():
    assert kraaijevanger_radius(SSPRK32_A, SSPRK32_B) == pytest.approx(2.0, abs=1e-9)
    assert kraaijevanger_radius(HEUN_A, HEUN_B) == pytest.approx(1.0, abs=1e-9)


def test_radius_is_sharp():
    """Just above the radius the conversion must be rejected, not silently wrong."""
    shu_osher(SSPRK32_A, SSPRK32_B, 2.0)  # must not raise
    with pytest.raises(ShuOsherError, match="2.0"):
        shu_osher(SSPRK32_A, SSPRK32_B, 2.0001)


def test_butcher_to_K_shape():
    K = butcher_to_K(HEUN_A, HEUN_B)
    np.testing.assert_allclose(K, [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]], atol=1e-14)


def test_module_does_not_import_firedrake():
    """tableaux.py must stay usable without Firedrake or PETSc importable.

    Tests the real property in a subprocess with both blocked, rather than
    scanning the source for a substring.
    """
    import pathlib
    import subprocess
    import sys
    import textwrap

    # Load the file directly, NOT as firedrake_ts.tableaux: the package
    # __init__ imports Firedrake, so importing through the package would
    # always fail regardless of what tableaux.py itself does.
    target = pathlib.Path(__file__).parent.parent / "firedrake_ts" / "tableaux.py"
    assert target.exists(), target

    program = textwrap.dedent(
        """
        import importlib.util
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("firedrake", "petsc4py"):
                    raise ImportError(f"{name} is blocked for this test")
                return None

        sys.meta_path.insert(0, Blocker())
        spec = importlib.util.spec_from_file_location("_isolated", sys.argv[1])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.kraaijevanger_radius is not None
        assert mod.shu_osher is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"tableaux.py cannot load without Firedrake/PETSc importable:\n"
        f"{result.stdout}\n{result.stderr}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_tableaux.py`
Expected: collection error — `ModuleNotFoundError: No module named 'firedrake_ts.tableaux'`.

- [ ] **Step 3: Write the implementation**

Create `firedrake_ts/tableaux.py`:

```python
"""Runge-Kutta tableau algebra: Butcher data, Shu-Osher form, SSP radii.

Deliberately free of Firedrake and PETSc imports. Everything here is testable
at machine precision with no PDE machinery, which is where the mathematical
risk of the Shu-Osher stepper lives.

References
----------
Shu-Osher form and the absolute monotonicity radius: Kraaijevanger (1991);
Ketcheson's optimal SSPRK(3,2). The acceptance predicates R1-R9 come from
``local/fill/rk-method-spec.md``.
"""

import numpy as np

__all__ = [
    "ShuOsherError",
    "butcher_to_K",
    "kraaijevanger_radius",
    "shu_osher",
]


class ShuOsherError(ValueError):
    """Raised when a Shu-Osher conversion would produce negative coefficients."""


def butcher_to_K(A, b):
    """Return ``K = [[A, 0], [b^T, 0]]``, the tableau with its completion row.

    Folding the completion into ``K`` is what makes the accepted step subject
    to the same convex-combination argument as the stages.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    s = len(b)
    K = np.zeros((s + 1, s + 1))
    K[:s, :s] = A
    K[s, :s] = b
    return K


def _monotone_at(K, r, tol=1e-13):
    """Is ``I + rK`` invertible with ``M^-1 K >= 0`` and ``M^-1 e >= 0``?"""
    n = K.shape[0]
    M = np.eye(n) + r * K
    if abs(np.linalg.det(M)) < 1e-14:
        return False
    Minv = np.linalg.inv(M)
    return bool(
        (Minv @ K >= -tol).all() and (Minv @ np.ones(n) >= -tol).all()
    )


def kraaijevanger_radius(A, b, hi=50.0, iterations=200):
    """The radius of absolute monotonicity ``R(A, b)``, by bisection.

    ``hi`` bounds the search; methods with an unbounded radius return ``hi``.
    """
    K = butcher_to_K(A, b)
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _monotone_at(K, mid):
            lo = mid
        else:
            hi = mid
    return lo


def shu_osher(A, b, r):
    """Convert a Butcher tableau to canonical Shu-Osher form at radius ``r``.

    Returns ``(P, q)`` with ``P = r M^-1 K`` and ``q = M^-1 e``, where
    ``M = I + rK``. The stage recursion is then

        Y_i = q_i x^n + sum_{j<i} P_ij (Y_j + (h/r) L(Y_j))

    with the last row giving ``x^{n+1}``. Every row of ``[P | q]`` is a
    nonnegative partition of unity exactly when ``r <= R(A, b)``.

    Raises
    ------
    ShuOsherError
        If ``r`` exceeds ``R(A, b)``, so some coefficient is negative and the
        SSP guarantee would silently not hold.
    """
    if r <= 0.0:
        raise ShuOsherError(f"r must be positive, got {r!r}")
    K = butcher_to_K(A, b)
    n = K.shape[0]
    M = np.eye(n) + r * K
    if abs(np.linalg.det(M)) < 1e-14:
        raise ShuOsherError(f"I + rK is singular at r = {r!r}")
    Minv = np.linalg.inv(M)
    P = r * (Minv @ K)
    q = Minv @ np.ones(n)
    if P.min() < -1e-13 or q.min() < -1e-13:
        radius = kraaijevanger_radius(A, b)
        raise ShuOsherError(
            f"r = {r!r} exceeds the radius of absolute monotonicity "
            f"R(A,b) = {radius:.10g}: min(P) = {P.min():.3e}, "
            f"min(q) = {q.min():.3e}. The SSP bound would not hold."
        )
    return P, q
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_tableaux.py`
Expected: all pass. Reference values to expect if you debug: `q = (1, 0, 0, ⅓, ⅓)`, `P` non-zeros `P[1,0]=P[2,1]=1`, `P[3,2]=P[4,2]=⅔`, Butcher agreement `4.44e-16`, `R(A,b) = 2.0000000000002`.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/tableaux.py tests/test_tableaux.py
git commit -m "Add Shu-Osher conversion and Kraaijevanger radius

Pure numpy, no Firedrake or PETSc import, so the mathematically subtle
part is testable at machine precision without any PDE machinery.

Verified: SSPRK(3,2) at r=2 gives the textbook q=(1,0,0,1/3,1/3) with P
carrying 1,1,2/3,2/3; the recursion reproduces the Butcher map to
4.4e-16; R(A,b)=2 and r=2.0001 is rejected rather than silently
producing negative coefficients.

Every row of [P|q] is a nonnegative partition of unity, including the
completion row -- which is why the accepted step needs no post-step
clamp."
```

---

### Task 3: Tableau registry and the acceptance report

Turns `rk-method-spec.md` §4's R1–R9 predicates from prose into an executable test.

**Files:**
- Modify: `firedrake_ts/tableaux.py`
- Test: `tests/test_tableaux.py`

**Interfaces:**
- Consumes: `shu_osher`, `kraaijevanger_radius`, `butcher_to_K` from Task 2.
- Produces:
  - `ARKTableau` dataclass with fields `name: str`, `A`, `b`, `bhat`, `At`, `bt`, `c`, `ct`, `d` (all `np.ndarray`), `order: int`.
  - `TABLEAUX: dict[str, ARKTableau]` with keys `"imex_euler"`, `"ssprk2"`, `"esdirk_gamma5"`, `"ssp2_444_lsa"`.
  - `stability_function(At, w, z) -> complex` computing `1 + z·wᵀ(I − zÃ)⁻¹e`.
  - `acceptance_report(tab: ARKTableau) -> dict` with keys `r3_explicit`, `r3_implicit`, `r4_r_infinity`, `r5_bhat_sum`, `r5_bhat_dot_c`, `r6_radius`, `r8_min_diagonal`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tableaux.py`:

```python
from firedrake_ts.tableaux import (  # noqa: E402
    TABLEAUX,
    acceptance_report,
    stability_function,
)


def test_registry_has_the_expected_tableaux():
    assert set(TABLEAUX) == {
        "imex_euler",
        "ssprk2",
        "esdirk_gamma5",
        "ssp2_444_lsa",
    }


@pytest.mark.parametrize("name", ["imex_euler", "esdirk_gamma5", "ssp2_444_lsa"])
def test_stiff_accuracy_r3(name):
    """b == A[s-1,:] and bt == At[s-1,:]. Required for the algebraic variables."""
    tab = TABLEAUX[name]
    np.testing.assert_allclose(tab.b, tab.A[-1], atol=1e-14)
    np.testing.assert_allclose(tab.bt, tab.At[-1], atol=1e-14)


@pytest.mark.parametrize("name", ["esdirk_gamma5", "ssp2_444_lsa"])
def test_l_stability_r4(name):
    """R(inf) == 0 and sup|R(z)| <= 1 on the left half-plane.

    Note At is singular for these tableaux (explicit first stage), so R(inf)
    must come from the limit of 1 + z bt^T (I - z At)^-1 e, NOT from
    1 + bt^T At^-1 e. Probe at z = -1e8; -1e10 is dominated by roundoff.
    """
    tab = TABLEAUX[name]
    assert abs(stability_function(tab.At, tab.bt, -1e8)) < 1e-6
    assert abs(stability_function(tab.At, tab.bhat, -1e8)) < 1e-5
    grid = [
        complex(re, im)
        for re in np.linspace(-40.0, 0.0, 200)
        for im in np.linspace(0.0, 40.0, 200)
    ]
    assert max(abs(stability_function(tab.At, tab.bt, z)) for z in grid) <= 1.0 + 1e-9


def test_esdirk_gamma5_matches_the_spec_closed_form():
    """R(z) = -5(z^2 + 20z + 50) / (2(z-5)^3), rk-method-spec.md 5.1."""
    tab = TABLEAUX["esdirk_gamma5"]
    for z in [-1.0, -10.0, -100.0]:
        expected = -5 * (z**2 + 20 * z + 50) / (2 * (z - 5) ** 3)
        assert stability_function(tab.At, tab.bt, z) == pytest.approx(
            expected, rel=1e-10
        )


def test_acceptance_report_reproduces_the_spec_table():
    """rk-method-spec.md 5, esdirk_gamma5 row."""
    report = acceptance_report(TABLEAUX["esdirk_gamma5"])
    assert report["r3_explicit"] is True
    assert report["r3_implicit"] is True
    assert abs(report["r4_r_infinity"]) < 1e-6
    assert report["r5_bhat_sum"] == pytest.approx(1.0, abs=1e-14)
    assert report["r5_bhat_dot_c"] == pytest.approx(16 / 25, abs=1e-12)
    assert report["r6_radius"] == pytest.approx(2.0, abs=1e-9)
    # Explicit first stage: At[0,0] == 0, the "circle" entry in the spec table.
    assert report["r8_min_diagonal"] == pytest.approx(0.0, abs=1e-14)


def test_ssp2_444_lsa_radius_and_embedding():
    report = acceptance_report(TABLEAUX["ssp2_444_lsa"])
    assert report["r6_radius"] == pytest.approx(2.0, abs=1e-9)
    assert report["r5_bhat_sum"] == pytest.approx(1.0, abs=1e-14)
    assert report["r5_bhat_dot_c"] == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize("name", ["imex_euler", "ssprk2"])
def test_shakedown_tableaux_have_unit_radius(name):
    tab = TABLEAUX[name]
    assert kraaijevanger_radius(tab.A, tab.b) == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_tableaux.py`
Expected: collection error — `ImportError: cannot import name 'TABLEAUX'`.

- [ ] **Step 3: Write the implementation**

Append to `firedrake_ts/tableaux.py`. Add `dataclass` to the imports and extend `__all__` with `"ARKTableau"`, `"TABLEAUX"`, `"acceptance_report"`, `"stability_function"`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ARKTableau:
    """An additive Runge-Kutta pair.

    ``A, b, bhat`` are the explicit tableau, its completion weights and its
    embedded weights; ``At, bt`` the implicit tableau and completion; ``c, ct``
    the abscissae; ``d`` the dense-output theta-coefficients.
    """

    name: str
    A: np.ndarray
    b: np.ndarray
    bhat: np.ndarray
    At: np.ndarray
    bt: np.ndarray
    c: np.ndarray
    ct: np.ndarray
    d: np.ndarray
    order: int


def stability_function(At, w, z):
    """``R(z) = 1 + z w^T (I - z At)^-1 e``.

    Use this rather than ``1 + w^T At^-1 e`` for ``R(infinity)``: ``At`` is
    singular whenever the first stage is explicit, which is the case for every
    stiffly accurate tableau here.
    """
    At = np.asarray(At, dtype=float)
    n = At.shape[0]
    e = np.ones(n)
    return 1.0 + z * (np.asarray(w, dtype=float) @ np.linalg.solve(np.eye(n) - z * At, e))


def acceptance_report(tab):
    """Evaluate the R1-R9 predicates of ``local/fill/rk-method-spec.md`` 4."""
    return {
        "r3_explicit": bool(np.allclose(tab.b, tab.A[-1], atol=1e-14)),
        "r3_implicit": bool(np.allclose(tab.bt, tab.At[-1], atol=1e-14)),
        "r4_r_infinity": stability_function(tab.At, tab.bt, -1e8).real,
        "r5_bhat_sum": float(tab.bhat.sum()),
        "r5_bhat_dot_c": float(tab.bhat @ tab.c),
        "r6_radius": kraaijevanger_radius(tab.A, tab.b),
        "r8_min_diagonal": float(np.diag(tab.At).min()),
    }


def _t(name, A, b, bhat, At, bt, c, ct, d, order):
    return ARKTableau(
        name=name,
        A=np.array(A, dtype=float),
        b=np.array(b, dtype=float),
        bhat=np.array(bhat, dtype=float),
        At=np.array(At, dtype=float),
        bt=np.array(bt, dtype=float),
        c=np.array(c, dtype=float),
        ct=np.array(ct, dtype=float),
        d=np.array(d, dtype=float),
        order=order,
    )


# Shakedown tableau: explicit Euler on G, backward Euler on F. Two stages so the
# explicit part stays strictly lower triangular while the implicit part is
# stiffly accurate. Stage 0 is x^n exactly (c_0 = 0, both rows zero).
_IMEX_EULER = _t(
    "imex_euler",
    A=[[0.0, 0.0], [1.0, 0.0]],
    b=[1.0, 0.0],
    bhat=[1.0, 0.0],
    At=[[0.0, 0.0], [0.0, 1.0]],
    bt=[0.0, 1.0],
    c=[0.0, 1.0],
    ct=[0.0, 1.0],
    d=[1.0, 0.0],
    order=1,
)

# Shakedown tableau: Heun / SSPRK(2,2), explicit only. At is identically zero,
# so no stage requires an implicit solve.
_SSPRK2 = _t(
    "ssprk2",
    A=[[0.0, 0.0], [1.0, 0.0]],
    b=[0.5, 0.5],
    bhat=[1.0, 0.0],
    At=[[0.0, 0.0], [0.0, 0.0]],
    bt=[0.0, 0.0],
    c=[0.0, 1.0],
    ct=[0.0, 1.0],
    d=[1.0, 0.0],
    order=2,
)

# rk-method-spec.md 5.1. Explicit part is Ketcheson's optimal SSPRK(3,2) in
# stiffly accurate form; implicit part a stiffly accurate, L-stable ESDIRK with
# uniform diagonal gamma = 1/5, so PETSc passes a single shift and the shifted
# operator is reusable across all three implicit solves.
_ESDIRK_GAMMA5 = _t(
    "esdirk_gamma5",
    A=[
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
    ],
    b=[1 / 3, 1 / 3, 1 / 3, 0.0],
    bhat=[27 / 125, 36 / 125, 12 / 125, 2 / 5],
    At=[
        [0.0, 0.0, 0.0, 0.0],
        [3 / 10, 1 / 5, 0.0, 0.0],
        [3 / 10, 1 / 2, 1 / 5, 0.0],
        [39 / 125, 47 / 125, 14 / 125, 1 / 5],
    ],
    bt=[39 / 125, 47 / 125, 14 / 125, 1 / 5],
    c=[0.0, 0.5, 1.0, 1.0],
    ct=[0.0, 0.5, 1.0, 1.0],
    d=[573 / 875, 604 / 875, 148 / 875, -18 / 35],
    order=2,
)

# rk-method-spec.md 5.2. Same explicit part; implicit diagonals are distinct
# (1/6, 1/5, 1/4), so there is no operator reuse across stages. Larger joint
# region (1.200 vs 1.050) but smaller explicit-axis radius. Kept as the
# fallback if the uniform-gamma part conditions badly in practice.
_SSP2_444_LSA = _t(
    "ssp2_444_lsa",
    A=[
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
    ],
    b=[1 / 3, 1 / 3, 1 / 3, 0.0],
    bhat=[11 / 24, 5 / 12, 1 / 8, 0.0],
    At=[
        [0.0, 0.0, 0.0, 0.0],
        [1 / 3, 1 / 6, 0.0, 0.0],
        [1 / 3, 7 / 15, 1 / 5, 0.0],
        [11 / 32, 5 / 16, 3 / 32, 1 / 4],
    ],
    bt=[11 / 32, 5 / 16, 3 / 32, 1 / 4],
    c=[0.0, 0.5, 1.0, 1.0],
    ct=[0.0, 0.5, 1.0, 1.0],
    d=[11 / 16, 5 / 8, 3 / 16, -1 / 2],
    order=2,
)

TABLEAUX = {
    t.name: t
    for t in (_IMEX_EULER, _SSPRK2, _ESDIRK_GAMMA5, _SSP2_444_LSA)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_tableaux.py`
Expected: all pass.

If `test_l_stability_r4` fails for `ssp2_444_lsa` on `bhat`: its `bhat` has a zero last entry, so `R̂(∞)` decays more slowly. The tolerance `1e-5` accounts for that. Do not loosen it further without checking `stability_function(tab.At, tab.bhat, -1e10)` also trends to zero.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/tableaux.py tests/test_tableaux.py
git commit -m "Add ARK tableau registry and rk-method-spec acceptance report

Four tableaux: imex_euler and ssprk2 for shakedown, esdirk_gamma5
(rk-method-spec.md 5.1) and ssp2_444_lsa (5.2) for production.

acceptance_report evaluates R3-R8 so the spec's acceptance table is an
executable test rather than prose. Reproduces the published values:
R(A,b)=2, bhat.c=16/25, R(inf)=0, and the closed-form R(z) to ten
digits.

stability_function uses 1 + z bt^T (I - z At)^-1 e rather than
1 + bt^T At^-1 e: At is singular whenever the first stage is explicit,
which holds for every stiffly accurate tableau here."
```

---

### Task 4: `ARKSSP` skeleton and IMEX Euler (M2a)

Proves the stage solve — `Z_i`, the shift, the SNES path — with the simplest possible tableau and no Shu–Osher machinery. Butcher form only.

**Files:**
- Create: `firedrake_ts/ark_ssp.py`
- Test: `tests/test_ark_ssp.py` (create)

**Interfaces:**
- Consumes: `TABLEAUX` from Task 3.
- Produces:
  - `ARKSSP` class, constructible with no arguments, usable via `-ts_python_type firedrake_ts.ark_ssp.ARKSSP`.
  - `ARKSSP.setUp(ts)`, `ARKSSP.step(ts)`, `ARKSSP.reset(ts)`, `ARKSSP.setFromOptions(ts)`, `ARKSSP.view(ts, viewer)`.
  - Options: `-ts_ark_ssp_type <name>` (default `esdirk_gamma5`), `-ts_ark_ssp_radius <float>` (default: `R(A,b)`).
  - `ARKSSP.formSNESFunction(args)` — **one positional tuple**, `(snes, x, f, ts)`. libpetsc4py calls this as `formSNESFunction(args)`, unlike `formSNESJacobian(*args)`.
  - `ARKSSP.formSNESJacobian(snes, x, A, B, ts)` — splatted.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ark_ssp.py`:

```python
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
    """Zero SNES iterations would mean a null residual."""
    solver, _ = _decay("imex_euler", dt=1e-2)
    assert solver.ts.getSNESIterations() > 0


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_ark_ssp.py`
Expected: all fail. The PETSc error will mention it cannot import `firedrake_ts.ark_ssp`.

- [ ] **Step 3: Write the implementation**

Create `firedrake_ts/ark_ssp.py`. This task implements the Butcher path only; Task 5 adds Shu–Osher, Task 8 adds the freeze.

```python
"""An additive Runge-Kutta stepper as a ``TSPYTHON`` type.

Selected with ``-ts_type python -ts_python_type firedrake_ts.ark_ssp.ARKSSP``.

The explicit part is advanced in canonical Shu-Osher form so that a limiter
applied to a stage value is mathematically sound: each stage is a convex
combination of forward-Euler steps taken from previously-limited stage values.
See ``docs/superpowers/specs/2026-08-04-shu-osher-tspython-stepper-design.md``.
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
        self._limiter = None
        self._tab = None
        self._P = None
        self._q = None
        self._r = None
        # Work vectors, allocated in setUp.
        self._Y = None      # stage values
        self._L = None      # explicit slopes, M^-1 G(Y_j)
        self._Ydot = None   # implicit stage derivatives
        self._Z = None      # Butcher stage offset
        self._rhs = None    # scratch for computeRHSFunction
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
        # Raises ShuOsherError if the radius is too large, naming both numbers.
        self._P, self._q = shu_osher(tab.A, tab.b, self._r)

        sol = ts.getSolution()
        s = len(tab.b)
        self._Y = [sol.duplicate() for _ in range(s)]
        self._L = [sol.duplicate() for _ in range(s)]
        self._Ydot = [sol.duplicate() for _ in range(s)]
        self._Z = sol.duplicate()
        self._rhs = sol.duplicate()

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
        self.step_calls += 1
        tab = self._tab
        t = ts.getTime()
        h = ts.getTimeStep()
        x = ts.getSolution()
        s = len(tab.b)

        for i in range(s):
            self._build_offset(tab, x, h, i)
            if tab.At[i, i] > 0.0:
                self._solve_stage(ts, tab, h, i)
            else:
                # Purely explicit stage: the value IS the offset.
                self._Z.copy(self._Y[i])
                self._Ydot[i].set(0.0)
            ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._rhs)
            self._rhs.copy(self._L[i])

        self._complete(tab, x, h)
        ts.setTime(t + h)

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
        self._Z.copy(self._Y[i])  # initial guess
        snes.solve(None, self._Y[i])
        reason = snes.getConvergedReason()
        if reason < 0:
            # petsc4py exposes no PETSc.ERR_* constants, so signal with
            # Firedrake's own exception. Task 10 replaces this with a step
            # rejection routed through TSAdapt.
            raise ConvergenceError(
                f"stage {i} SNES diverged, reason {reason}"
            )
        self._Y[i].copy(self._Ydot[i])
        self._Ydot[i].axpy(-1.0, self._Z)
        self._Ydot[i].scale(self._shift)

    def _complete(self, tab, x, h):
        """x^{n+1}.

        Under stiff accuracy (b == A[s-1,:] and bt == At[s-1,:]) the completion
        is exactly the last stage, so copying it is not a shortcut but the
        definition. Fall back to the weighted sum otherwise.
        """
        if np.allclose(tab.b, tab.A[-1], atol=1e-14) and np.allclose(
            tab.bt, tab.At[-1], atol=1e-14
        ):
            self._Y[-1].copy(x)
            return
        for j, (bj, btj) in enumerate(zip(tab.b, tab.bt)):
            if btj != 0.0:
                x.axpy(h * btj, self._Ydot[j])
            if bj != 0.0:
                x.axpy(h * bj, self._L[j])

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_ark_ssp.py`
Expected: all pass.

If `test_imex_euler_advances_and_converges` reports a value of exactly 1.0, the explicit part is not reaching the state — check that `computeRHSFunction` results are being copied into `self._L[i]` and that `_build_offset` uses them.

If the SNES reports zero iterations, `formSNESFunction` is not being called; confirm the single-tuple signature.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/ark_ssp.py tests/test_ark_ssp.py
git commit -m "Add ARKSSP TSPYTHON stepper with the IMEX Euler shakedown

Butcher path only: stage offsets Z_i, the 1/(h At_ii) shift, and the SNES
stage solve, with no Shu-Osher machinery yet. Proves the stage solve in
isolation before the convex-combination loop lands.

Completion exploits stiff accuracy: when b == A[s-1,:] and
bt == At[s-1,:] the completion IS the last stage value, so it is copied
rather than re-summed.

formSNESFunction takes ONE positional tuple -- libpetsc4py calls it as
formSNESFunction(args) while splatting formSNESJacobian(*args)."
```

---

### Task 5: Shu–Osher loop on a purely explicit tableau (M2b)

Proves the convex-combination recursion with no implicit part, so a bug here cannot hide behind a stage solve.

**Files:**
- Modify: `firedrake_ts/ark_ssp.py`
- Test: `tests/test_ark_ssp.py`

**Interfaces:**
- Consumes: `ARKSSP` from Task 4.
- Produces: `ARKSSP._shu_osher_predictor(x, h, i)` populating `self._Y[i]`; `step` uses it for every stage.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ark_ssp.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_ark_ssp.py`
Expected: `test_ssprk2_is_second_order` fails (the Butcher path with `At = 0` gives `Y_i = Z_i`, which is *also* correct for this tableau, so this may pass — if so, `test_limiter_fires_once_per_stage` will still fail with 0 calls). At minimum the limiter test must fail before you implement.

- [ ] **Step 3: Replace the stage loop with the Shu–Osher predictor**

In `firedrake_ts/ark_ssp.py`, add the predictor and call it from `step`:

```python
    def _shu_osher_predictor(self, x, h, i):
        """Y_i = q_i x^n + sum_{j<i} P_ij (Y_j + (h/r) L_j).

        Every row of ``[P | q]`` is a nonnegative partition of unity, so this
        is a convex combination of forward-Euler steps taken from stage values
        the limiter has already seen. That is the whole reason for the stepper.
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
```

Then rewrite the loop body of `step`:

```python
        for i in range(s):
            self._shu_osher_predictor(x, h, i)
            if self._limiter is not None:
                self._limiter(self._Y[i])
            self._build_offset(tab, x, h, i)
            if tab.At[i, i] > 0.0:
                self._solve_stage(ts, tab, h, i)
            ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._rhs)
            self._rhs.copy(self._L[i])
```

Note what changed: the predictor now supplies `Y_i` for every stage, the limiter fires **before** anything consumes it, and the purely-explicit branch no longer overwrites `Y_i` from `Z_i` — the Shu–Osher value stands. `_build_offset` is still needed because `_solve_stage` requires `Z_i`.

Also change `_complete` to use the Shu–Osher completion row, which is what makes the accepted step bounded:

```python
    def _complete(self, tab, x, h):
        """x^{n+1} from the Shu-Osher completion row.

        The last row of ``[P | q]`` is a nonnegative partition of unity like
        any other, so the accepted step is bounded by the same induction as
        the stages -- no post-step clamp is needed. Under stiff accuracy this
        row is identical to the last stage row.
        """
        s = len(tab.b)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_ark_ssp.py`
Expected: all pass, including Task 4's tests. `test_imex_euler_is_first_order` must still hold — the Shu–Osher path has to reproduce IMEX Euler too.

If `test_imex_euler_advances_and_converges` breaks here, the predictor and the implicit solve are disagreeing on `Y_i`. For `imex_euler` at `r = 1`, `q = (1, 0, 0)` and `P[1,0] = P[2,0] = 1`, so the predictor gives `Y_1 = x + h L_0`, which must equal `Z_1` before the implicit correction.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/ark_ssp.py tests/test_ark_ssp.py
git commit -m "Advance the explicit part in Shu-Osher form

Stage values now come from the convex-combination recursion
Y_i = q_i x^n + sum_j P_ij (Y_j + (h/r) L_j), with the limiter firing on
each substage before anything consumes it. The completion uses the last
row of [P|q], which is a nonnegative partition of unity like any other,
so the accepted step is bounded without a post-step clamp.

Verified with explicit-only SSPRK2: design order 2, zero SNES
iterations, agreement with PETSc's Butcher-form path to 1e-5, and the
limiter firing on every stage of every step."
```

---

### Task 6: Structural predicates on the residual

Two UFL-structural facts, discovered from the caller's forms rather than declared: which components are algebraic, and which have no implicit operator acting on them.

**Files:**
- Modify: `firedrake_ts/solving_utils.py`
- Test: `tests/test_singular_mass.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces, as module-level functions in `firedrake_ts/solving_utils.py`:
  - `is_zero_form(form) -> bool`
  - `differential_fields(F, u, udot, nfields) -> tuple[int, ...]` — components *with* a time derivative.
  - `algebraic_fields(F, u, udot, nfields) -> tuple[int, ...]` — components without.
  - `explicitly_governed_fields(F, u, udot, nfields) -> tuple[int, ...]` — components with no implicit operator.
  - `nonzero_rows(form, nfields) -> tuple[int, ...]`
  - `resolve_fields(option, prefix, detected) -> tuple[int, ...]` — the runtime override: returns the comma-separated option value if set, else `detected`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_singular_mass.py`:

```python
"""Structural predicates and singular-mass support."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts
from firedrake_ts.solving_utils import (
    algebraic_fields,
    explicitly_governed_fields,
    is_zero_form,
    nonzero_rows,
)


def _three_field():
    """Mimic the (f, p, T) character: explicit-only, algebraic, and both."""
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V * V
    w = Function(W)
    wdot = Function(W)
    f, p, T = split(w)
    fdot, pdot, Tdot = split(wdot)
    vf, vp, vT = TestFunctions(W)
    F = (
        inner(fdot, vf) * dx           # mass only -> explicitly governed
        + inner(p - T, vp) * dx        # algebraic constraint
        + inner(Tdot, vT) * dx         # mass ...
        + inner(grad(T), grad(vT)) * dx  # ... plus diffusion
    )
    return F, w, wdot, (f, p, T), (vf, vp, vT)


def test_algebraic_fields_finds_the_constraint_row():
    F, w, wdot, _, _ = _three_field()
    assert algebraic_fields(F, w, wdot, 3) == (1,)


def test_explicitly_governed_fields_finds_the_mass_only_row():
    F, w, wdot, _, _ = _three_field()
    assert explicitly_governed_fields(F, w, wdot, 3) == (0,)


def test_partitions_are_distinct():
    """Differential/algebraic is NOT implicit/explicit. T is in both halves."""
    F, w, wdot, _, _ = _three_field()
    algebraic = algebraic_fields(F, w, wdot, 3)
    explicit_only = explicitly_governed_fields(F, w, wdot, 3)
    assert set(algebraic).isdisjoint(explicit_only)
    assert 2 not in algebraic
    assert 2 not in explicit_only


def test_nonsingular_two_field_has_no_algebraic_rows():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    a, b = split(w)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    F = (
        inner(adot, va) * dx
        + inner(grad(a), grad(va)) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    assert algebraic_fields(F, w, wdot, 2) == ()
    assert explicitly_governed_fields(F, w, wdot, 2) == ()


def test_mass_only_rows_are_all_explicitly_governed():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    W = V * V
    w = Function(W)
    wdot = Function(W)
    adot, bdot = split(wdot)
    va, vb = TestFunctions(W)
    F = inner(adot, va) * dx + inner(bdot, vb) * dx
    assert explicitly_governed_fields(F, w, wdot, 2) == (0, 1)
    assert algebraic_fields(F, w, wdot, 2) == ()


def test_nonzero_rows_locates_G():
    F, w, wdot, (f, p, T), (vf, vp, vT) = _three_field()
    assert nonzero_rows(inner(f, vf) * dx, 3) == (0,)
    assert nonzero_rows(inner(T, vp) * dx, 3) == (1,)


def test_resolve_fields_prefers_an_explicit_override():
    """Structural default, runtime override -- PETSc's own idiom.

    Same pattern as -pc_fieldsplit_detect_saddle_point: detection is the
    default, not the only option.
    """
    from firedrake.petsc import PETSc

    from firedrake_ts.solving_utils import resolve_fields

    assert resolve_fields("ts_algebraic_fields", "probe_", (1,)) == (1,)
    opts = PETSc.Options()
    opts["probe_ts_algebraic_fields"] = "0,2"
    try:
        assert resolve_fields("ts_algebraic_fields", "probe_", (1,)) == (0, 2)
    finally:
        del opts["probe_ts_algebraic_fields"]


def test_is_zero_form_needs_expanded_derivatives():
    """derivative() with no dependence yields a symbolically-zero integral.

    Without expand_derivatives, Form.empty() is False and the predicate
    silently reports every row as non-zero.
    """
    from firedrake import ufl_expr

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    udot = Function(V)
    v = TestFunction(V)
    mass = inner(udot, v) * dx
    assert is_zero_form(ufl_expr.derivative(mass, u))
    assert not is_zero_form(ufl_expr.derivative(mass, udot))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_singular_mass.py`
Expected: collection error — `ImportError: cannot import name 'algebraic_fields'`.

- [ ] **Step 3: Write the implementation**

Add to `firedrake_ts/solving_utils.py`. Add `import ufl` and `from ufl.algorithms import expand_derivatives` to the imports; `ExtractSubBlock` is already imported.

```python
def is_zero_form(form):
    """Is ``form`` structurally zero?

    ``derivative()`` applied to a form with no dependence on the coefficient
    returns a ``Form`` carrying one symbolically-zero integral, so ``empty()``
    is False until ``expand_derivatives`` has folded it away. Skipping the
    expansion makes every predicate below report "non-zero" for everything.

    Structural, not numerical: a coefficient that happens to vanish at t = 0
    must not be mistaken for an absent operator.
    """
    if form is None or isinstance(form, ufl.ZeroBaseForm):
        return True
    expanded = expand_derivatives(form)
    if isinstance(expanded, ufl.ZeroBaseForm):
        return True
    return bool(expanded.empty())


def nonzero_rows(form, nfields):
    """Which test-function components of ``form`` carry any integral."""
    splitter = ExtractSubBlock()
    return tuple(
        i
        for i in range(nfields)
        if not is_zero_form(splitter.split(form, argument_indices=(i,)))
    )


def _classify_rows(F, u, udot, nfields):
    """Per row: (has time derivative, has implicit operator)."""
    from firedrake import ufl_expr

    splitter = ExtractSubBlock()
    classified = {}
    for i in range(nfields):
        row = splitter.split(F, argument_indices=(i,))
        if is_zero_form(row):
            continue
        classified[i] = (
            not is_zero_form(ufl_expr.derivative(row, udot)),
            not is_zero_form(ufl_expr.derivative(row, u)),
        )
    return classified


def differential_fields(F, u, udot, nfields):
    """Components whose residual row contains a time derivative."""
    rows = _classify_rows(F, u, udot, nfields)
    return tuple(i for i, (has_dot, _) in rows.items() if has_dot)


def algebraic_fields(F, u, udot, nfields):
    """Components whose residual row has no time derivative.

    The mass matrix ``dF/du_t`` is structurally zero on these rows, so the
    right-hand-side projection is undefined there. ``G`` must vanish on them.
    """
    rows = _classify_rows(F, u, udot, nfields)
    return tuple(i for i, (has_dot, _) in rows.items() if not has_dot)


def explicitly_governed_fields(F, u, udot, nfields):
    """Components with no implicit operator acting on them.

    ``dF/du`` is structurally zero on these rows, so the implicit stage
    equation reduces to ``Y_i = Z_i``: the stage value is fully determined by
    the explicit recursion and can be limited soundly. This is exactly the
    condition under which a limiter preserves monotonicity.
    """
    rows = _classify_rows(F, u, udot, nfields)
    return tuple(i for i, (_, has_implicit) in rows.items() if not has_implicit)


def resolve_fields(option, prefix, detected):
    """``detected``, unless the options database overrides it.

    Structural default plus runtime override, following PETSc's own idiom --
    ``-pc_fieldsplit_detect_saddle_point`` detects the algebraic block from
    zero diagonals but does not force the choice. Set e.g.
    ``-ts_algebraic_fields 1,3`` to declare the partition instead.
    """
    value = PETSc.Options(prefix or "").getString(option, "")
    if not value:
        return tuple(detected)
    return tuple(int(part) for part in value.replace(" ", "").split(",") if part)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_singular_mass.py`
Expected: all pass.

If every predicate returns `()`, `expand_derivatives` is missing from `is_zero_form`.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/solving_utils.py tests/test_singular_mass.py
git commit -m "Add UFL-structural predicates for algebraic and explicit rows

Two facts discovered from the caller's forms rather than declared:
structurally zero rows of dF/du_t are algebraic (the mass projection is
undefined there), and structurally zero rows of dF/du have no implicit
operator (so their stage value is fully determined by the explicit
recursion and can be limited soundly).

The two partitions are deliberately different: a variable with both
diffusion and advection is differential but not explicitly governed.

is_zero_form must expand derivatives first. derivative() on a form with
no dependence returns a Form with one symbolically-zero integral, whose
empty() is False -- without the expansion every predicate reports
'non-zero' for every row."
```

---

### Task 7: Restrict the RHS projection to the differential block

Makes `_TSContext` handle a singular mass matrix. Tested through **`arkimex`**, with no ARKSSP involved, so the projection is proven independently of the new stepper.

**Files:**
- Modify: `firedrake_ts/solving_utils.py`
- Test: `tests/test_singular_mass.py`

**Interfaces:**
- Consumes: `algebraic_fields`, `nonzero_rows` from Task 6.
- Produces: `_TSContext._algebraic_fields` (cached property, tuple of ints); `_TSContext._rhs_projection_mass_matrix` restricted to the differential block; a `ValueError` when `G` is non-zero on an algebraic row.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_singular_mass.py`:

```python
RUNG_PARAMS = {
    "ts_adapt_type": "none",
    "ts_exact_final_time": "stepover",
}


def _rung1(stepper, dt=1e-3, tmax=1.0):
    """Index 1, R-space: ydot = z, 0 = z + y, z explicit. Exact y = e^-t.

    M = diag(1, 0) is genuinely singular and G vanishes on the algebraic row.
    No spatial discretisation error, so observed order is the tableau's alone.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "R", 0)
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, zdot = split(wdot)
    vy, vz = TestFunctions(W)
    w.sub(0).assign(1.0)
    w.sub(1).assign(-1.0)

    F = inner(ydot, vy) * dx + inner(z + y, vz) * dx
    G = inner(z, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    return float(w.sub(0).dat.data_ro[0]), float(w.sub(1).dat.data_ro[0])


ARKIMEX = {"ts_type": "arkimex", "ts_arkimex_type": "2c"}


def test_rung1_singular_mass_runs_under_arkimex():
    """A singular mass matrix must not break the RHS projection."""
    y, z = _rung1(ARKIMEX)
    assert y == pytest.approx(np.exp(-1.0), abs=1e-3)
    assert z == pytest.approx(-np.exp(-1.0), abs=1e-3)


def test_G_nonzero_on_an_algebraic_row_is_rejected():
    """The projection is undefined there; fail loudly, not silently."""
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "R", 0)
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, zdot = split(wdot)
    vy, vz = TestFunctions(W)
    F = inner(ydot, vy) * dx + inner(z + y, vz) * dx
    G = inner(y, vz) * dx  # nonzero on the ALGEBRAIC row

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 0.1), G=G)
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=0.05, **ARKIMEX),
        options_prefix="",
    )
    with pytest.raises(ValueError, match="algebraic"):
        solver.solve()


def test_nonsingular_projection_is_unchanged():
    """The existing non-mixed path must be byte-identical."""
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
        solver_parameters=dict(RUNG_PARAMS, ts_time_step=1e-3, **ARKIMEX),
        options_prefix="",
    ).solve()
    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_singular_mass.py`
Expected: `test_rung1_singular_mass_runs_under_arkimex` fails — the mass matrix is singular, so the projection KSP diverges or errors (LU on a structurally singular matrix gives a zero-pivot error). `test_G_nonzero_on_an_algebraic_row_is_rejected` fails because no check exists yet. `test_nonsingular_projection_is_unchanged` should already pass.

- [ ] **Step 3: Write the implementation**

In `firedrake_ts/solving_utils.py`, inside `_TSContext`:

```python
    @cached_property
    def _nfields(self):
        V = self._problem.u_restrict.function_space()
        return len(V) if len(V) > 1 else 1

    @cached_property
    def _algebraic_fields(self):
        """Components with no time derivative, so no invertible mass block."""
        if self._nfields == 1:
            return ()
        detected = algebraic_fields(
            self._problem.F,
            self._problem.u_restrict,
            self._xdot,
            self._nfields,
        )
        return resolve_fields(
            "ts_algebraic_fields", self.options_prefix, detected
        )

    def _check_G_vanishes_on_algebraic_rows(self):
        if self.G is None or not self._algebraic_fields:
            return
        offending = set(nonzero_rows(self.G, self._nfields)) & set(
            self._algebraic_fields
        )
        if offending:
            raise ValueError(
                f"G is nonzero on algebraic component(s) {sorted(offending)}, "
                f"whose rows of dF/du_t are structurally zero. The mass "
                f"projection M^-1 G is undefined there. Move those terms into "
                f"the implicit residual F, or give those components a time "
                f"derivative."
            )
```

Then restrict the mass matrix. Replace `_rhs_projection_mass_matrix` with:

```python
    @cached_property
    def _rhs_projection_mass_matrix(self):
        r"""The mass matrix ``dF/du_t``, assembled once.

        On algebraic components this block is structurally zero, so a plain
        solve would hit a zero pivot. Those rows are given a unit diagonal
        instead: ``G`` is guaranteed zero there (checked separately), so the
        projected result is zero on them either way, and the operator becomes
        invertible.

        Held on the context so it outlives the ``KSP`` that takes it as an
        operator.
        """
        from firedrake import assemble, ufl_expr

        self._check_G_vanishes_on_algebraic_rows()
        mass = assemble(ufl_expr.derivative(self.F, self._xdot), bcs=self.bcs_F)
        if self._algebraic_fields:
            ises = self._problem.J.arguments()[0].function_space()._ises
            rows = numpy.concatenate(
                [ises[i].getIndices() for i in self._algebraic_fields]
            )
            mass.petscmat.zeroRows(rows.astype(PETSc.IntType), diag=1.0)
        return mass
```

Add the imports used above to the module's own namespace — `algebraic_fields` and `nonzero_rows` are defined in this same file, so no import is needed, but confirm `numpy` is imported (it is, line 4).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_singular_mass.py`
Expected: all pass.

Then confirm no regression: `uv run pytest --verbose tests/`
Expected: everything from Tasks 1–6 still passes.

If `zeroRows` raises about a missing preallocation, assemble with `mat_type="aij"` — `zeroRows` needs an assembled AIJ matrix, not a nested or matrix-free one.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/solving_utils.py tests/test_singular_mass.py
git commit -m "Support a singular mass matrix in the RHS projection

dF/du_t is structurally zero on algebraic components, so the projection
KSP hit a zero pivot on any DAE with algebraic variables. Those rows now
get a unit diagonal: G is guaranteed zero on them, so the projected
result is unchanged and the operator becomes invertible.

G nonzero on an algebraic row is now a clear ValueError naming the
components, rather than a KSP divergence.

Verified through arkimex alone, with no new stepper involved, on an
index-1 R-space DAE with M = diag(1, 0) and an exact solution."
```

---

### Task 8: Freeze limited components during the implicit solve (M3)

The heart of the issue. Without this, the implicit solve drags a limited component straight back to the unlimited `Z_i`.

**Files:**
- Modify: `firedrake_ts/ark_ssp.py`
- Test: `tests/test_singular_mass.py`, `tests/test_ark_ssp.py`

**Interfaces:**
- Consumes: `explicitly_governed_fields` (Task 6), `ARKSSP` (Task 5).
- Produces: `ARKSSP._frozen_rows` (numpy array of global row indices, or `None`); freeze applied in `formSNESFunction` / `formSNESJacobian`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ark_ssp.py`:

```python
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
    F = (
        inner(adot, va) * dx
        + inner(bdot, vb) * dx
        + inner(grad(b), grad(vb)) * dx
    )
    G = -inner(a, va) * dx - inner(b, vb) * dx

    # Dedicated scratch: w is also ctx._x, which the TS callbacks write into.
    scratch = Function(W)
    CLAMP = 0.5
    drift = []

    def clamping_limiter(vec):
        """Force the explicitly-governed row to exactly CLAMP."""
        with scratch.dat.vec_wo as target:
            vec.copy(target)
        scratch.sub(0).assign(CLAMP)
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
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx
    problem = firedrake_ts.DAEProblem(
        F, u, u_t, (0.0, 0.02), G=-inner(u, v) * dx
    )
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
    problem = firedrake_ts.DAEProblem(
        F, u, u_t, (0.0, 0.02), G=-inner(u, v) * dx
    )
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
```

And append the dual-path rung tests to `tests/test_singular_mass.py`:

```python
ARK_SSP_G5 = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung1_dual_path(stepper):
    """Index 1, singular mass, exact solution. Both steppers must agree."""
    y, z = _rung1(stepper)
    assert y == pytest.approx(np.exp(-1.0), abs=1e-3)
    assert z == pytest.approx(-np.exp(-1.0), abs=1e-3)


def _rung2(stepper, dt=1e-3, tmax=1.0):
    """Index 2, the multiplier case: ydot = z - y, 0 = y - g(t), G = -y.

    g(t) = exp(-t), so z = gdot + g = 0 exactly and y = exp(-t). Returns
    (y, z, constraint_defect). The defect is what R3 stiff accuracy buys:
    near 1e-16 with b == A[s-1,:], near 1e-3 without.
    """
    mesh = UnitIntervalMesh(1)
    R = FunctionSpace(mesh, "R", 0)
    W = R * R
    w = Function(W)
    wdot = Function(W)
    y, z = split(w)
    ydot, zdot = split(wdot)
    vy, vz = TestFunctions(W)
    time = Constant(0.0)
    w.sub(0).assign(1.0)
    w.sub(1).assign(0.0)

    F = inner(ydot - z, vy) * dx + inner(y - exp(-time), vz) * dx
    G = -inner(y, vy) * dx

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), time=time, G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    y_val = float(w.sub(0).dat.data_ro[0])
    return y_val, float(w.sub(1).dat.data_ro[0]), abs(y_val - np.exp(-tmax))


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung2_constraint_defect_is_at_machine_precision(stepper):
    """Stiff accuracy makes the completion the last stage, so y satisfies
    the constraint exactly rather than to O(h^p)."""
    _, _, defect = _rung2(stepper, dt=1e-2)
    assert defect < 1e-10, f"constraint defect {defect:.3e}, expected ~1e-16"


def _rung3(stepper, dt=2e-3, tmax=0.1, n=8):
    """PDE scale: heat equation on V x R with a mean-value multiplier.

    u_t = laplacian(u) + lambda, with int u dx pinned. Real mixed space, real
    singular mass, index 2, fieldsplit-able -- and no momentum balance.
    """
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "P", 1)
    R = FunctionSpace(mesh, "R", 0)
    W = V * R
    w = Function(W)
    wdot = Function(W)
    u, lam = split(w)
    udot, lamdot = split(wdot)
    vu, vlam = TestFunctions(W)
    x, = SpatialCoordinate(mesh)
    w.sub(0).interpolate(1.0 + 0.5 * sin(2 * pi * x))

    mass_target = Constant(1.0)
    F = (
        inner(udot, vu) * dx
        + inner(grad(u), grad(vu)) * dx
        - inner(lam, vu) * dx
        + inner(u - mass_target, vlam) * dx
    )
    G = -0.1 * inner(u, vu) * dx  # an explicit reaction term

    problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, tmax), G=G)
    parameters = dict(RUNG_PARAMS, ts_time_step=dt, **stepper)
    firedrake_ts.DAESolver(
        problem, solver_parameters=parameters, options_prefix=""
    ).solve()
    return w


@pytest.mark.parametrize("stepper", [ARKIMEX, ARK_SSP_G5], ids=["arkimex", "arkssp"])
def test_rung3_pde_with_multiplier_runs(stepper):
    w = _rung3(stepper)
    assert np.all(np.isfinite(w.sub(0).dat.data_ro))
    assert np.all(np.isfinite(w.sub(1).dat.data_ro))
    assert norm(w.sub(0)) > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest --verbose tests/test_ark_ssp.py tests/test_singular_mass.py`
Expected: the three new `test_ark_ssp.py` tests fail on `AttributeError: 'ARKSSP' object has no attribute '_frozen_rows'`. The `arkssp` variants of the rung tests fail or give wrong answers.

- [ ] **Step 3: Write the implementation**

In `firedrake_ts/ark_ssp.py`, add the imports and the freeze.

```python
from firedrake import dmhooks
from firedrake_ts.solving_utils import explicitly_governed_fields, resolve_fields
```

Add to `__init__`: `self._frozen_rows = None`.

Extend `setUp` (after the work-vector allocation):

```python
        self._frozen_rows = self._find_frozen_rows(ts)
        if self._limiter is not None and self._frozen_rows is None:
            raise ValueError(
                "a stage limiter is registered, but no component of this "
                "problem is free of an implicit operator: every row of dF/du "
                "is structurally nonzero. Limiting a component that has an "
                "implicit operator is not sound -- the stage solve would undo "
                "the correction. Move the operator into G, or drop the limiter."
            )
```

And the helper plus the freeze logic:

```python
    def _find_frozen_rows(self, ts):
        """Global row indices of components with no implicit operator.

        Those rows' stage equation reduces to Y_i = Z_i, so their value comes
        entirely from the explicit Shu-Osher recursion. Pinning them during the
        implicit solve is what stops the solve from undoing the limiter.
        """
        ctx = dmhooks.get_appctx(ts.getDM())
        problem = ctx._problem
        V = problem.u_restrict.function_space()
        if len(V) <= 1:
            return None
        detected = explicitly_governed_fields(
            problem.F, problem.u_restrict, ctx._xdot, len(V)
        )
        fields = resolve_fields(
            "ts_explicitly_governed_fields", ts.getOptionsPrefix(), detected
        )
        if not fields:
            return None
        ises = problem.J.arguments()[0].function_space()._ises
        rows = np.concatenate([ises[i].getIndices() for i in fields])
        return rows.astype(PETSc.IntType)

    def _apply_freeze_residual(self, x, f):
        """Replace frozen rows of the residual with ``x - Y_i``."""
        if self._frozen_rows is None:
            return
        target = self._Y[self._stage]
        xa = x.getArray(readonly=True)
        ya = target.getArray(readonly=True)
        fa = f.getArray()
        lo, _ = x.getOwnershipRange()
        local = self._frozen_rows - lo
        fa[local] = xa[local] - ya[local]
```

Now wire it into the callbacks:

```python
    def formSNESFunction(self, args):
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
        if self._frozen_rows is not None:
            A.zeroRows(self._frozen_rows, diag=1.0)
            if B is not None and B.handle != A.handle:
                B.zeroRows(self._frozen_rows, diag=1.0)
```

Note `self._rhs` is reused as `xdot` scratch here; it is only live between `computeRHSFunction` and the copy into `_L[i]`, which never overlaps a stage solve. If that ever changes, allocate a dedicated vector.

**The trailing `True` on both `compute*` calls is load-bearing — do not change it to `False`.**
It is PETSc's `imex` flag. `True` means the stage residual is the implicit function alone.
`False` makes `TSComputeIFunction` additionally subtract the RHS function, which double-counts
the explicit part — once in `Z_i`, once in the residual — and the advance cancels out. Measured
on the completed Task 4 stepper: flipping these to `False` fails four tests, the first reporting
`solution never left its initial condition`. If you see that symptom, check this flag before
suspecting the freeze logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest --verbose tests/test_ark_ssp.py tests/test_singular_mass.py`
Expected: all pass, including both `arkimex` and `arkssp` variants of all three rungs.

If a rung passes under `arkimex` but fails under `arkssp`, the fault is in the stepper, not the projection — that separation is the entire point of running both.

Then the full suite: `uv run pytest --verbose tests/`

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/ark_ssp.py tests/test_ark_ssp.py tests/test_singular_mass.py
git commit -m "Freeze explicitly-governed components during the stage solve

The core of COOL-193. A component with no implicit operator has stage
equation Y_i = Z_i, so the solve would pull a limited value straight back
to the unlimited Z_i, and the correction would re-enter later stages
scaled by At_ji/At_ii -- measured at 1.879988 on esdirk_gamma5. Those
rows are now pinned: identity rows in the Jacobian and x - Y_i in the
residual.

Identity rows rather than a reduced index set, so the monolithic
Jacobian keeps its shape and a caller's fieldsplit over the remaining
block still applies unchanged.

Registering a limiter when no component is freezable is now a clear
error instead of a silently unsound answer.

Singular-mass rungs 1-3 (index-1 R-space, index-2 multiplier, PDE with a
mean-value multiplier) run against both arkimex and ARKSSP, so a
failure localises to the stepper rather than the projection."
```

---

### Task 9: Limiter, DG1 bounds test, and the M4 gate

**Files:**
- Modify: `firedrake_ts/ts_solver.py`
- Test: `tests/test_bounds.py` (create)

**Interfaces:**
- Consumes: `ARKSSP.set_stage_limiter` (Task 4), the freeze (Task 8).
- Produces: `DAESolver.set_stage_limiter(limiter)` forwarding to `self.ts.getPythonContext()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bounds.py`:

```python
"""The boundedness result that motivates COOL-193.

DG1 square-wave advection with a Zhang-Shu scaling limiter on every explicit
substage must hold 0 <= f <= 1 to machine precision. The reference
Butcher-form implementation gives -0.0099 / +1.0100, so the negative control
below must FAIL to bound -- otherwise this problem is not challenging and the
test proves nothing.
"""

import numpy as np
import pytest
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
    x, = SpatialCoordinate(mesh)
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
    x, = SpatialCoordinate(mesh)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_bounds.py`
Expected: `test_shu_osher_form_holds_bounds_with_no_post_step_clamp` fails with `AttributeError: 'DAESolver' object has no attribute 'set_stage_limiter'`.

- [ ] **Step 3: Add the forwarding method**

In `firedrake_ts/ts_solver.py`, add to `DAESolver`:

```python
    def set_stage_limiter(self, limiter):
        r"""Register a limiter fired on each explicit substage value.

        PETSc constructs the stepper itself from ``-ts_python_type``, so the
        caller never holds a reference to it; this forwards to that instance.

        :arg limiter: a callable taking the stage-value ``Vec`` and modifying
            it in place. It is called before anything consumes the value.
        """
        if self.ts.getType() != PETSc.TS.Type.PYTHON:
            raise ValueError(
                f"a stage limiter requires ts_type 'python' with "
                f"ts_python_type set to a Shu-Osher stepper; this TS is "
                f"'{self.ts.getType()}', whose stage values are not "
                f"available in a form a limiter can soundly act on"
            )
        context = self.ts.getPythonContext()
        if not hasattr(context, "set_stage_limiter"):
            raise ValueError(
                f"{type(context).__name__} does not support stage limiters"
            )
        context.set_stage_limiter(limiter)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest --verbose tests/test_bounds.py`

Then the whole suite: `uv run pytest --verbose tests/`

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/ts_solver.py tests/test_bounds.py
git commit -m "Add DAESolver.set_stage_limiter and the DG1 bounds test

PETSc constructs the stepper from -ts_python_type, so the caller never
holds a reference to it; DAESolver forwards to that instance and raises
a clear error when the TS type cannot support a sound limiter.

The bounds test asserts 0 <= f <= 1 to machine precision with NO
post-step clamp, so it tests the Shu-Osher induction itself. It is
paired with a negative control on the Butcher-form path: if the control
ever stops violating bounds, the problem is not challenging and the
bounds result is not evidence."
```

- [ ] **Step 6: ⛔ STOP. Report to the user.**

Do not begin Task 10. Report:

1. Measured `min` and `max` from `test_shu_osher_form_holds_bounds_with_no_post_step_clamp`.
2. Measured `min` and `max` from the negative control, and whether it failed to bound as expected (roughly `−0.0099 / +1.0100`).
3. Limiter call count per step (expect 4 for `esdirk_gamma5`).
4. Whether the frozen rows matched the limited predictor bit-for-bit after each implicit solve.

If the bounds test failed, work the four-item diagnostic list in the CHECKPOINT block at the top of this plan and report which one it was **before changing any code**.

---

### Task 10: `_petsc_shim.py` and adaptivity (M5)

**Files:**
- Create: `firedrake_ts/_petsc_shim.py`
- Modify: `firedrake_ts/ark_ssp.py`
- Test: `tests/test_adapt.py` (create)

**Interfaces:**
- Consumes: `ARKSSP` (Task 8).
- Produces:
  - `firedrake_ts._petsc_shim.ts_get_adapt(ts) -> ctypes.c_void_p`
  - `ts_adapt_candidates_clear(adapt)`, `ts_adapt_candidate_add(adapt, name, order, stage_order, ccfl, cost, inuse)`, `ts_adapt_choose(adapt, ts, h) -> tuple[int, float, bool, float, float, float]`
  - `ARKSSP.evaluatestep(ts, order, U)` — writes the order-`order` solution into `U`.

Recover the dlopen machinery rather than rewriting it: `git show 3766c20:firedrake_ts/_petsc_shim.py` contains `_dlopen_petsc()` (handles petsc4py loading libpetsc `RTLD_LOCAL`, which makes `CDLL(None)` blind to its symbols), `_load()`, `_check(ierr, what)`, and a working `TSGetAdapt` declaration with the opaque-handle signature already solved.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapt.py`:

```python
"""Error-controlled stepping via TSADAPTBASIC and the embedded pair."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts

EXACT = np.exp(-1.0)

ARK_SSP = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


def _decay(extra, dt=1e-2):
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 1.0), G=-inner(u, v) * dx
    )
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP, ts_time_step=dt, ts_exact_final_time="stepover", **extra
        ),
        options_prefix="",
    )
    solver.solve()
    return solver, float(u.dat.data_ro[0])


def test_shim_resolves_ts_adapt_symbols():
    from firedrake_ts._petsc_shim import ts_get_adapt

    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, 0.1), G=-inner(u, v) * dx
    )
    solver = firedrake_ts.DAESolver(problem, options_prefix="")
    assert ts_get_adapt(solver.ts) is not None


def test_evaluatestep_gives_the_embedded_solution():
    """Order p and order p-1 completions must differ, and both be sane."""
    solver, _ = _decay({"ts_adapt_type": "none"}, dt=0.1)
    ctx = solver.ts.getPythonContext()
    full = solver.ts.getSolution().duplicate()
    embedded = solver.ts.getSolution().duplicate()
    ctx.evaluatestep(solver.ts, ctx._tab.order, full)
    ctx.evaluatestep(solver.ts, ctx._tab.order - 1, embedded)
    assert full.norm() > 0.0
    assert embedded.norm() > 0.0
    diff = full.copy()
    diff.axpy(-1.0, embedded)
    assert 0.0 < diff.norm() < full.norm()


def test_adapt_basic_selects_steps():
    solver, value = _decay(
        {
            "ts_adapt_type": "basic",
            "ts_adapt_dt_min": 1e-6,
            "ts_adapt_dt_max": 0.2,
            "ts_rtol": 1e-6,
            "ts_atol": 1e-8,
        }
    )
    assert solver.ts.getStepNumber() > 0
    assert abs(value - EXACT) < 1e-4
    # The controller must actually have changed dt away from the initial guess.
    assert abs(solver.ts.getTimeStep() - 1e-2) > 1e-9


def test_design_order_two_without_the_limiter():
    """Order 2, unlimited. rk-method-spec.md reports 2.004 for this tableau."""
    _, coarse = _decay({"ts_adapt_type": "none"}, dt=4e-2)
    _, fine = _decay({"ts_adapt_type": "none"}, dt=2e-2)
    ratio = abs(coarse - EXACT) / abs(fine - EXACT)
    assert 3.4 < ratio < 4.6, f"observed order ratio {ratio}, expected ~4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_adapt.py`
Expected: `ModuleNotFoundError: No module named 'firedrake_ts._petsc_shim'`, and `AttributeError` for `evaluatestep`.

- [ ] **Step 3: Write the shim**

Create `firedrake_ts/_petsc_shim.py`. Start from `git show 3766c20:firedrake_ts/_petsc_shim.py > /tmp/old_shim.py` and keep `_dlopen_petsc`, `_load`, `_check`, and the `TSGetAdapt` declaration. Drop `repair_ts_snes_callbacks`, `set_stage_hook`, `stage_hook_propagates`, and `_CHECKSTAGE`. Add the three new declarations:

```python
"""ctypes access to the ``TSAdapt`` entry points petsc4py does not expose.

petsc4py 3.25 binds neither ``TSAdapt`` as a type nor ``TSGetAdapt``,
``TSAdaptCandidatesClear``, ``TSAdaptCandidateAdd`` or ``TSAdaptChoose``. A
``TSPYTHON`` stepper that owns its own ``step`` must call them itself to get
``-ts_adapt_type basic`` working, so the controller and its options
(``-ts_adapt_rtol``, ``-ts_adapt_clip``, ...) stay PETSc's rather than being
reimplemented in Python.

``_dlopen_petsc`` is recovered from commit 3766c20.
"""

import ctypes
import glob
import os

from firedrake.petsc import PETSc

__all__ = [
    "ts_adapt_candidate_add",
    "ts_adapt_candidates_clear",
    "ts_adapt_choose",
    "ts_get_adapt",
]

_lib = None


def _dlopen_petsc():
    """A ctypes handle whose symbol table includes libpetsc's TS symbols.

    petsc4py loads libpetsc ``RTLD_LOCAL``, so ``CDLL(None)`` cannot see it.
    Prefer the versioned libpetsc under ``PETSC_DIR``/``PETSC_ARCH``; fall back
    to re-``dlopen``-ing petsc4py's own extension module ``RTLD_GLOBAL``, which
    promotes its already-loaded dependency into the global namespace.
    """
    import petsc4py

    cfg = petsc4py.get_config()
    pattern = os.path.join(cfg["PETSC_DIR"], cfg["PETSC_ARCH"], "lib", "libpetsc.so*")
    candidates = [*sorted(glob.glob(pattern)), PETSc.__file__]
    for path in candidates:
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if hasattr(lib, "TSGetAdapt"):
            return lib
    raise RuntimeError(
        "could not resolve libpetsc's TS symbols; tried " + ", ".join(candidates)
    )


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = _dlopen_petsc()
    lib.TSGetAdapt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.TSGetAdapt.restype = ctypes.c_int
    lib.TSAdaptCandidatesClear.argtypes = [ctypes.c_void_p]
    lib.TSAdaptCandidatesClear.restype = ctypes.c_int
    lib.TSAdaptCandidateAdd.argtypes = [
        ctypes.c_void_p,      # TSAdapt
        ctypes.c_char_p,      # name
        ctypes.c_int,         # order
        ctypes.c_int,         # stageorder
        ctypes.c_double,      # ccfl
        ctypes.c_double,      # cost
        ctypes.c_int,         # inuse (PetscBool)
    ]
    lib.TSAdaptCandidateAdd.restype = ctypes.c_int
    lib.TSAdaptChoose.argtypes = [
        ctypes.c_void_p,                  # TSAdapt
        ctypes.c_void_p,                  # TS
        ctypes.c_double,                  # h
        ctypes.POINTER(ctypes.c_int),     # next_sc
        ctypes.POINTER(ctypes.c_double),  # next_h
        ctypes.POINTER(ctypes.c_int),     # accept (PetscBool)
        ctypes.POINTER(ctypes.c_double),  # wlte
        ctypes.POINTER(ctypes.c_double),  # wltea
        ctypes.POINTER(ctypes.c_double),  # wlter
    ]
    lib.TSAdaptChoose.restype = ctypes.c_int
    _lib = lib
    return _lib


def _check(ierr, what):
    if ierr:
        raise RuntimeError(f"{what} failed with ierr={ierr}")


def ts_get_adapt(ts):
    """The ``TSAdapt`` owned by ``ts``, as an opaque ctypes handle."""
    adapt = ctypes.c_void_p()
    _check(_load().TSGetAdapt(ts.handle, ctypes.byref(adapt)), "TSGetAdapt")
    return adapt


def ts_adapt_candidates_clear(adapt):
    _check(_load().TSAdaptCandidatesClear(adapt), "TSAdaptCandidatesClear")


def ts_adapt_candidate_add(
    adapt, name, order, stage_order, ccfl, cost, inuse
):
    _check(
        _load().TSAdaptCandidateAdd(
            adapt,
            name.encode() if name is not None else None,
            int(order),
            int(stage_order),
            float(ccfl),
            float(cost),
            1 if inuse else 0,
        ),
        "TSAdaptCandidateAdd",
    )


def ts_adapt_choose(adapt, ts, h):
    """Returns ``(next_scheme, next_h, accept, wlte, wltea, wlter)``."""
    next_sc = ctypes.c_int()
    next_h = ctypes.c_double()
    accept = ctypes.c_int()
    wlte = ctypes.c_double()
    wltea = ctypes.c_double()
    wlter = ctypes.c_double()
    _check(
        _load().TSAdaptChoose(
            adapt,
            ts.handle,
            float(h),
            ctypes.byref(next_sc),
            ctypes.byref(next_h),
            ctypes.byref(accept),
            ctypes.byref(wlte),
            ctypes.byref(wltea),
            ctypes.byref(wlter),
        ),
        "TSAdaptChoose",
    )
    return (
        next_sc.value,
        next_h.value,
        bool(accept.value),
        wlte.value,
        wltea.value,
        wlter.value,
    )
```

- [ ] **Step 4: Add `evaluatestep` and the adapt loop to `ARKSSP`**

First extract the stage loop from `step` into its own method so it can be retried. Replace the existing `for i in range(s):` block in `step` with a call to this:

```python
    def _take_stages(self, ts, tab, x, h, s):
        """Run all s stages from ``x`` with step ``h``. Populates Y, L, Ydot."""
        t = ts.getTime()
        for i in range(s):
            self._shu_osher_predictor(x, h, i)
            if self._limiter is not None:
                self._limiter(self._Y[i])
            self._build_offset(tab, x, h, i)
            if tab.At[i, i] > 0.0:
                self._solve_stage(ts, tab, h, i)
            ts.computeRHSFunction(t + tab.c[i] * h, self._Y[i], self._rhs)
            self._rhs.copy(self._L[i])
```

Then add to `firedrake_ts/ark_ssp.py`:

```python
    def evaluatestep(self, ts, order, U):
        """Write the order-``order`` completion into ``U``.

        ``TSADAPTBASIC`` gets its lower-order solution through
        ``TSEvaluateStep``, which routes here. The error estimate
        ``|h sum (b - bhat)_j L(Y_j)|`` reuses the ``L(Y_j)`` the step computed
        anyway, so error control costs no extra evaluations.
        """
        tab = self._tab
        h = self._last_h
        x = self._last_x
        weights_e = tab.b if order >= tab.order else tab.bhat
        weights_i = tab.bt if order >= tab.order else tab.bhat
        x.copy(U)
        for j in range(len(tab.b)):
            if weights_i[j] != 0.0:
                U.axpy(h * weights_i[j], self._Ydot[j])
            if weights_e[j] != 0.0:
                U.axpy(h * weights_e[j], self._L[j])
```

Add `self._last_h = None` and `self._last_x = None` to `__init__`, allocate `self._last_x = sol.duplicate()` in `setUp`, and record them at the top of the stage loop in `step`:

```python
        self._last_h = h
        x.copy(self._last_x)
```

Then wrap the stage loop in the reject/adapt loop. Replace the body of `step` after the recording with:

```python
        # petsc4py binds setMaxStepRejections but NOT a getter, so read the
        # option directly. PETSc's own default for ts->max_reject is 10.
        max_reject = PETSc.Options(ts.getOptionsPrefix() or "").getInt(
            "ts_max_reject", 10
        )
        adapt = ts_get_adapt(ts)
        for _ in range(max(1, max_reject + 1)):
            self._take_stages(ts, tab, self._last_x, h, s)
            ts_adapt_candidates_clear(adapt)
            ts_adapt_candidate_add(
                adapt, None, tab.order, tab.order, 1.0, float(s), True
            )
            _sc, next_h, accept, _wlte, _a, _r = ts_adapt_choose(adapt, ts, h)
            if accept:
                self._complete(tab, x, h)
                ts.setTime(t + h)
                ts.setTimeStep(next_h)
                return
            h = next_h
            ts.setTimeStep(h)
        ts.setConvergedReason(PETSc.TS.ConvergedReason.DIVERGED_STEP_REJECTED)
```

Move the existing `for i in range(s):` loop body into a `_take_stages(self, ts, tab, x, h, s)` method so it can be retried.

Import the shim helpers at the top of `ark_ssp.py`:

```python
from firedrake_ts._petsc_shim import (
    ts_adapt_candidate_add,
    ts_adapt_candidates_clear,
    ts_adapt_choose,
    ts_get_adapt,
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest --verbose tests/test_adapt.py`
Expected: all pass.

If `TSAdaptChoose` reports a nonsensical `next_h`, check that `evaluatestep` is being reached — `TSADAPTBASIC` calls `TSEvaluateStep(ts, order-1, ...)` and compares against `ts->vec_sol`, so `self._last_x` must hold the value at the *start* of the step, not the end.

Then the full suite: `uv run pytest --verbose tests/`

- [ ] **Step 6: Commit**

```bash
git add firedrake_ts/_petsc_shim.py firedrake_ts/ark_ssp.py tests/test_adapt.py
git commit -m "Add TSAdapt ctypes shim and error-controlled stepping

petsc4py 3.25 exposes neither TSAdapt as a type nor TSGetAdapt,
TSAdaptCandidatesClear, TSAdaptCandidateAdd or TSAdaptChoose, so a
TSPYTHON stepper owning its own step must call them via ctypes. Doing so
keeps the controller and all its options (-ts_adapt_rtol, -ts_adapt_clip,
...) PETSc's rather than reimplementing the PI controller in Python --
which is the single-configuration-path argument for TSPYTHON in the
first place.

_dlopen_petsc is recovered from 3766c20: petsc4py loads libpetsc
RTLD_LOCAL, so CDLL(None) is blind to its symbols and the versioned
library has to be found and re-dlopened RTLD_GLOBAL.

evaluatestep reuses the L(Y_j) the step already computed, so error
control adds no RHS evaluations."
```

---

### Task 11: Dense output (M6)

**Files:**
- Modify: `firedrake_ts/ark_ssp.py`
- Test: `tests/test_interpolate.py` (create)

**Interfaces:**
- Consumes: `ARKSSP` (Task 10), `ARKTableau.d` (Task 3).
- Produces: `ARKSSP.interpolate(ts, t, U)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_interpolate.py`:

```python
"""Dense output: X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i."""

import numpy as np
import pytest
from firedrake import *

import firedrake_ts

ARK_SSP = {
    "ts_type": "python",
    "ts_python_type": "firedrake_ts.ark_ssp.ARKSSP",
    "ts_ark_ssp_type": "esdirk_gamma5",
}


def _solver(dt=0.1, tmax=1.0):
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(1.0)
    problem = firedrake_ts.DAEProblem(
        inner(u_t, v) * dx, u, u_t, (0.0, tmax), G=-inner(u, v) * dx
    )
    return firedrake_ts.DAESolver(
        problem,
        solver_parameters=dict(
            ARK_SSP,
            ts_adapt_type="none",
            ts_time_step=dt,
            ts_exact_final_time="interpolate",
        ),
        options_prefix="",
    ), u


def test_endpoint_matches_the_accepted_step_exactly():
    """X(1) must equal y_{n+1}; a mismatch means the wrong weights."""
    solver, _ = _solver()
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    end = ts.getSolution().duplicate()
    ctx.interpolate(ts, ts.getTime(), end)
    diff = end.copy()
    diff.axpy(-1.0, ts.getSolution())
    assert diff.norm() < 1e-12, f"|X(1) - y_n+1| = {diff.norm()}"


def test_interpolation_is_at_least_first_order():
    """Midpoint interpolation must beat piecewise-constant."""
    solver, _ = _solver(dt=0.2, tmax=1.0)
    solver.solve()
    ts = solver.ts
    ctx = ts.getPythonContext()
    mid = ts.getSolution().duplicate()
    t_mid = ts.getTime() - 0.5 * ts.getTimeStep()
    ctx.interpolate(ts, t_mid, mid)
    exact = np.exp(-t_mid)
    assert abs(float(mid.getArray()[0]) - exact) < 1e-2


def test_exact_final_time_interpolate_lands_on_tmax():
    """Exercises TSInterpolate through the driver, not just directly."""
    solver, u = _solver(dt=0.03, tmax=1.0)
    solver.solve()
    assert solver.ts.getTime() == pytest.approx(1.0, abs=1e-12)
    assert float(u.dat.data_ro[0]) == pytest.approx(np.exp(-1.0), abs=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest --verbose tests/test_interpolate.py`
Expected: failures on `AttributeError`/`NotImplementedError` for `interpolate`, and `ts_exact_final_time: interpolate` erroring because the stepper provides no dense output.

- [ ] **Step 3: Write the implementation**

Add to `firedrake_ts/ark_ssp.py`:

```python
    def interpolate(self, ts, t, U):
        """Dense output at time ``t`` within the step just taken.

        X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i,
        with w = bt for the implicit part and w = b for the explicit part
        (rk-method-spec.md 5.1). The coefficients d are chosen so that
        d . g = 0 for the null direction g of the singular At, which is what
        keeps the stiff limit bounded -- the naive d = (1,0,0,0) diverges like
        z theta(theta-1) on stiff modes.
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest --verbose tests/test_interpolate.py`

If `test_endpoint_matches_the_accepted_step_exactly` fails, check `theta`. At the end of a step `ts.getTime()` is already advanced, so `theta = (t - (ts.getTime() - h))/h` gives `theta = 1` when `t == ts.getTime()`. Also confirm `sum(d_i·1 + (w_i − d_i)·1) == sum(w_i)`, i.e. the `theta = 1` weights collapse to the completion weights.

Then the full suite: `uv run pytest --verbose tests/`
Expected: everything passes.

- [ ] **Step 5: Commit**

```bash
git add firedrake_ts/ark_ssp.py tests/test_interpolate.py
git commit -m "Add dense output to ARKSSP

X(theta) = y_n + h sum_i [d_i theta + (w_i - d_i) theta^2] Ydot_i, with
w = bt implicit and w = b explicit. The d coefficients satisfy d.g = 0
for the null direction of the singular At, which keeps the stiff limit
contractive; the naive d = (1,0,0,0) diverges as z theta(theta-1) on
stiff modes -- the failure PETSc documents for its own a2/l2 tableaux.

Verified: |X(1) - y_n+1| = 0, first-order midpoint accuracy, and
ts_exact_final_time: interpolate landing on tmax through the driver."
```

---

## Final Verification

- [ ] Run the whole suite: `uv run pytest --verbose tests/`
- [ ] Run linting: `uv run --only-group dev ruff check .` and `uv run --only-group dev ruff format --check .`
- [ ] Confirm `firedrake_ts/tableaux.py` still imports with no Firedrake present (`test_module_does_not_import_firedrake`).
- [ ] Re-read `local/fill/rk-method-spec.md` §4 and confirm `acceptance_report` covers R3, R4, R5, R6, R8. R1 and R2 are structural (satisfied by construction), R7 (Higueras additive region) and R9 (dense output order) are not asserted numerically — note this to the user rather than silently leaving the impression all nine are checked.
- [ ] Report the M4 bounds numbers and the negative control numbers in the final summary. They are the result the issue exists to obtain.
