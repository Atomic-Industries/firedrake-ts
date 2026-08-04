# Design: Shu–Osher ARK-IMEX stepper as a `TSPYTHON` type

Date: 2026-08-04
Issue: [COOL-193](https://linear.app/atomic-industries/issue/COOL-193/tspython-stepper-for-ark-imex-in-shu-osher-form)
Related: COOL-189 (abandoned `TSSetPostStage` approach), `local/fill/rk-method-spec.md`

## 1. Goal

A custom time stepper registered as a `TSPYTHON` *type*, selected by
`-ts_type python -ts_python_type firedrake_ts.ark_ssp.ARKSSP`, running an additive
Runge–Kutta step with the **explicit** part in canonical Shu–Osher form, so that a
limiter applied to a stage value is mathematically sound.

COOL-193 carries the motivation and the measurements; this document does not restate them.
In one line: per-stage limiting is only sound as an induction over convex combinations of
stage **values**, and every PETSc stepper stores stage **derivatives**.

`firedrake_ts` gains no new public API beyond the type string and a limiter setter.

## 2. Scope

This is an **application-agnostic `firedrake_ts` feature**. Nothing in the stepper knows
about fill fractions, saddle points, or the `(f, u, p, T)` system. Problem structure is
either discovered from the user's forms (§6) or declared through the options database.

In scope: the stepper, the tableau algebra, singular-mass support in `_TSContext`, and the
`split()` fix that singular-mass verification depends on.

Out of scope: the adjoint (unreachable from `TSPYTHON` — `libpetsc4py.pyx` has no
`adjoint`/`getstages`; migrate to C if it becomes required), the coupled Stokes problem
itself, and the upstream `TSSSP` MR that COOL-193 suggests filing separately.

## 3. Environment findings

Verified against the installed stack (petsc4py 3.25.0, PETSc 3.25.0, `arch-firedrake-default`)
rather than assumed. These constrain the design.

**Available.** `ts.setPythonType` / `setPythonContext`. `TSPYTHON` dispatches `setUp`,
`reset`, `setFromOptions`, `view`, `step`, `rollback`, `interpolate`, `evaluatestep`,
`formSNESFunction`, `formSNESJacobian`, `solveStep`, `adaptStep`. `ts.computeIFunction`,
`computeIJacobian`, `computeRHSFunction`, `computeRHSJacobian` are all bound, so stage
residuals need no ctypes.

**Unavailable, hence the shim.** `TSAdapt` is not exposed as a petsc4py type at all.
`ts.getAdapt`, `ts.preStage`, `ts.postStage`, `ts.evaluateStep`, `ts.getStages` are unbound.

**Gotcha.** `libpetsc4py.pyx` calls `formSNESFunction(args)` with the 4-tuple as a *single*
positional argument, while `formSNESJacobian(*args)` is splatted. Asymmetric signatures.

**`TSStep_Python_default` is not reusable.** It is a single-stage driver: one
`VecCopy`, one `TSSolveStep_Python`, one `TSPostStage`, one `TSAdaptCheckStage`, then
`TSAdaptStep_Python`. Defining our own `step` means owning the reject loop outright; there
is no partial reuse.

**Existing convention.** `_TSContext._rhs_projection_solver` hands PETSc `M⁻¹G` in *state*
space (it solves `derivative(F, udot) · x = G`), not the raw dual residual. `tests/test_imex.py`
(7 tests) passes on this convention. It is exactly what a Shu–Osher substep needs: `Y + h·L(Y)`
becomes a plain `axpy` with no solve invented by the stepper.

**Pre-existing defect (blocker).** `_TSContext.split()` at `solving_utils.py:185` reads
`problem.u.subfunctions`, but `DAEProblem` defines only `u_restrict`. Any mixed problem with
`pc_type: fieldsplit` dies in `DMCreateFieldDecomposition` with an unhandled `AttributeError` —
reproduced on a mixed space, with and without `G`, singular and nonsingular mass. Firedrake's
own `_SNESContext.split()` (`firedrake/solving_utils.py:379`) uses `problem.u_restrict`;
`_TSContext.split()` is a stale copy predating the restricted-function-space rename, and
Firedrake's `NonlinearVariationalProblem` sets both names while `DAEProblem` sets one.
Occurrences to fix: lines 185, 230, 231, 234, 239.

**Tableau algebra, verified independently.** For the `rk-method-spec.md` §5.1 explicit part
(`A = [[0],[½,0],[½,½,0],[⅓,⅓,⅓,0]]`, `b = (⅓,⅓,⅓,0)`) at `r = 2`:

- `q = (1, 0, 0, ⅓, ⅓)`; `P` carries `1, 1, ⅔, ⅔` — the textbook form.
- Shu–Osher recursion vs. the Butcher map: max stage discrepancy **4.44e-16**.
- `R(A,b) = 2.0000000000002` by bisection; `r = 2.0001` is already inadmissible, so the
  nonnegativity gate in §7 has a sharp trigger.
- **Every row of `[P | q]` sums to 1 with nonnegative entries, including the completion row.**
  The completion row is *identical* to the last stage row, which is R3 stiff accuracy
  (`b = A[s−1,:]`) made visible. This is the certificate behind the §7 no-clamp decision.

## 4. Approach: own `step`

The alternative was implementing `solveStep` + `adaptStep` and riding
`TSStep_Python_default`, which needs zero ctypes and gets `TSPreStage` / `TSPostStage` /
`TSAdaptCheckStage` called for us. Rejected: PETSc's controller is bypassed, so we would
reimplement `TSAdaptChoose_Basic`'s PI controller in Python and hand-read
`-ts_adapt_rtol/atol/clip/safety`. That undercuts the single-configuration-path argument
that motivates `TSPYTHON`, and fails COOL-193's acceptance criterion as written.

So `step` is ours, mirroring `TSStep_ARKIMEX`. The ctypes surface is confined to four entry
points — `TSGetAdapt`, `TSAdaptCandidatesClear`, `TSAdaptCandidateAdd`, `TSAdaptChoose` —
added to the existing `_petsc_shim.py` dlopen/ierr machinery (`3766c20` is in-repo precedent).

The shim is **not needed before milestone 5**: every earlier milestone runs fixed-step at
`-ts_adapt_type none`. Deferring it keeps the riskiest unknown out of the critical path to
the boundedness result that motivates the issue.

## 5. Components

```
firedrake_ts/
  tableaux.py       NEW   pure numpy; no firedrake import
  ark_ssp.py        NEW   the ARKSSP TSPYTHON class
  _petsc_shim.py    EDIT  + 4 TSAdapt entry points        (milestone 5)
  solving_utils.py  EDIT  split() fix; structural predicates; restricted projection
```

**`tableaux.py`** — `ARKTableau` (`A, b, bhat, At, bt, c, ct, d, order, name`); a registry
(`imex_euler`, `ssprk2`, `esdirk_gamma5` = `rk-method-spec.md` §5.1, `ssp2_444_lsa` = its §5.2);
`shu_osher(A, b, r) → (P, q)` via `K = [[A,0],[bᵀ,0]]`, `M = I + rK`, `P = r M⁻¹K`,
`q = M⁻¹e`; `kraaijevanger_radius(A, b)`; `additive_region(A, At)`; and
`acceptance_report(tab)` returning the R1–R9 predicates of `rk-method-spec.md` §4.

Kept Firedrake-free deliberately: this is where the mathematical risk lives, it is testable
at machine precision with no PDE machinery, and it turns `rk-method-spec.md` §4 from prose
into an executable test.

Butcher form is the primary representation; Shu–Osher `(P, q)` is derived at `setUp`. That
is forced anyway by "`r` configurable, defaulting to `R(A,b)`", which requires the radius
computation. A unit test pins the derived `(P, q)` against the hand-derived textbook values.

**`ark_ssp.py`** — `ARKSSP` implementing `setUp`, `step`, `evaluatestep`, `interpolate`,
`rollback`, `reset`, `setFromOptions`, `view`, plus `set_stage_limiter(fn)`. The limiter is
an opaque callable taking the stage-value `Function` and mutating it in place; it receives
no tableau, index, or time.

**Reaching the instance.** PETSc constructs `ARKSSP` itself from the `-ts_python_type`
string, so the caller has no reference to it. `DAESolver.set_stage_limiter(fn)` therefore
forwards to `self.ts.getPythonContext().set_stage_limiter(fn)`, raising a clear error if the
TS type is not `python`. This keeps the user-facing call the flat
`solver.set_stage_limiter(zhang_shu)` while leaving the stepper independently constructible
in unit tests.

## 6. Structural predicates

Two UFL-structural predicates on the user's forms. Both are *discovered* properties of the
problem the caller handed us, not application knowledge.

**"Structurally zero" means UFL-empty, not numerically small.** A row is zero when the
corresponding sub-block of the derivative form contains no integrals — determined with
`ExtractSubBlock` (already imported in `solving_utils.py`) on `derivative(F, udot)` and
`derivative(F, u)`, per test-function component. It is *not* determined by assembling and
scanning for zero rows: a coefficient that happens to vanish at `t = 0` must not be mistaken
for an absent operator, and the predicates must be stable across the whole solve.

| predicate | meaning | consequence |
|---|---|---|
| structurally zero rows of `derivative(F, udot)` | algebraic components | projection skips them; assert `G` vanishes there |
| structurally zero rows of `derivative(F, u)` | no implicit operator acts on this component | **freeze after limiting** |

Each has an options-database override (default: detect), following PETSc's
structural-default-plus-runtime-override idiom. `-pc_fieldsplit_detect_saddle_point` is the
direct precedent: it detects the algebraic block from zero diagonals and uses it to *build a
preconditioner*. This design is strictly more conservative — the same class of structural
fact, used only to locate a projection and a freeze set, never to choose a solver.

**The two partitions are different, and conflating them would be the design error.** For the
motivating system, differential = `{f, T}` / algebraic = `{u, p}`, while implicit = `{u, p, T}`
/ explicit = `{f, T}`. `T` is in both. Neither predicate produces an IS, a split, or a
reordering for the implicit solve; problem-specific (e.g. Stokes) preconditioning is
configured through `solver_parameters` exactly as today, including nested fieldsplit over
`(u,p)`/`T` and custom `PCBase` reading `appctx`.

## 7. Data flow, one stage

Given `xⁿ`, `h`, tableau `(A, b, Ã, b̃)`, Shu–Osher `(P, q)` at radius `r`. For `i = 1..s`:

1. **Shu–Osher predictor** from already-limited earlier stages:
   `Y_i = q_i0·xⁿ + Σ_{j<i} P_ij·(Y_j + (h/r)·L_j)`, with `L_j = M⁻¹G(Y_j)` from the
   restricted projection.
2. **Fire the limiter on `Y_i`**, before anything consumes it.
3. **Implicit solve over the non-frozen components only**, with the limited frozen
   components entering as coefficients. `Z_i = xⁿ + h·Σ_{j<i}(ã_ij·Ẏ_j^I + a_ij·L_j)`,
   `Ẏ_i = (Y_i − Z_i)/(h·ã_ii)`, residual via `ts.computeIFunction`, Jacobian shift
   `σ = 1/(h·ã_ii)` via `ts.computeIJacobian`. Skipped when `ã_ii = 0` (stage is purely
   explicit; `c_1 = 0 ⇒ Y_1 = xⁿ`).
4. Record `L_i` and `Ẏ_i^I`.

Step 3 is where the entire issue lives. Left in the unknown set, `f`'s row
`M_f·(Y_f − Z_f)/(h·ã_ii) = 0` would drag `Y_f` straight back to the unlimited `Z_f` — the
`ã_ji/ã_ii` amplification COOL-193 measured at 1.879988. The freeze is implemented as
identity rows on the frozen components (the Dirichlet-BC mechanism), which preserves the
monolithic Jacobian *shape* so the caller's fieldsplit over the remaining block still
applies unchanged. A reduced-IS solve is the obvious later optimisation; correctness first.

**Completion.** `x^{n+1}` uses the same convex-combination row of `[P | q]`, so it is bounded
by the same induction as the stages. No post-step clamp — see the §3 certificate. This
deliberately contradicts `rk-method-spec.md` R8 ("a post-step clamp is required regardless of
tableau") and open item §6.6; §9 records how that gets resolved with evidence.

## 8. Error handling

Four loud failures in place of four silent unsoundnesses:

- `r > R(A,b)` ⇒ `min(P) < 0` or `min(q) < 0` ⇒ raise, naming both `r` and `R(A,b)`.
  Verified sharp: `r = 2.0001` already fails for the `rk-method-spec.md` §5.1 explicit part.
- `G` structurally nonzero on an algebraic row ⇒ raise; `M⁻¹` is undefined there.
- a limiter targeting a component whose `∂F/∂u` row is **nonzero** ⇒ raise. Limiting a
  component that has an implicit operator is exactly what the SSP argument forbids; this
  makes it an error rather than a plausible-looking wrong answer.
- SNES divergence ⇒ reject the step to the adapt loop; `TS_DIVERGED_STEP_REJECTED` once
  retries are exhausted.

ctypes calls reuse the existing `ierr`-checking pattern and assert `TSAdaptChoose`'s
`accept` output.

## 9. Verification

**Singular-mass ladder.** Three rungs, none Stokes-shaped, each with a closed-form solution.
Every rung runs against **both** `arkimex` and `ARKSSP`, so a rung that passes one and fails
the other localises the fault to the stepper rather than the projection.

1. *Index 1, `R`-space mixed.* `ẏ = z`, `0 = z + y`, `z` explicit. `M = diag(1,0)` genuinely
   singular, `G` vanishes exactly on the algebraic row, `y = e⁻ᵗ`. No spatial discretisation
   error, so observed order is the tableau's and nothing else.
2. *Index 2, the multiplier case.* `ẏ = z − y`, `0 = y − g(t)`, `G = −y` explicit, `z` the
   multiplier (`z = ġ + g`). This rung makes R3 falsifiable: with `b = A[s−1,:]` the
   constraint defect should sit near 1e-16, and without it the multiplier order-reduces to
   roughly 1e-3 — the discrepancy `rk-method-spec.md` §5.2 already reports against `2c`.
3. *PDE scale, still generic.* Heat or advection on `V × R` with a mean-value Lagrange
   multiplier: `u_t = Δu + λ + (explicit term)`, `∫u dx = m(t)`. Real mixed space, real
   singular mass, index 2, fieldsplit-able, and no momentum balance anywhere in it.

**Milestones.**

| M | deliverable | assertion |
|---|---|---|
| 0 | `split()` fix | mixed problem + `pc_type: fieldsplit` runs, with and without `G`; currently `AttributeError` |
| 1 | `tableaux.py` | Shu–Osher↔Butcher to ~1e-16; `q = (1,0,0,⅓,⅓)`, `P` carrying `1,1,⅔,⅔`; `R(A,b) = 2`; `r = 2.0001` rejected; R1–R9 report reproduces the `rk-method-spec.md` §5 table |
| 2a | IMEX Euler, fixed step | reproduces `tests/test_imex.py::_decay` under `-ts_type python` |
| 2b | explicit SSPRK2 in Shu–Osher form | order 2 on a smooth problem |
| 3 | combined ARK + restricted projection | rungs 1–3 dual-path; index-2 constraint defect ~1e-16 with R3, ~1e-3 without |
| 4 | limiter callback | DG1 square wave, **no clamp**: `min ≥ −1e-15`, `max ≤ 1+1e-15` |
| 5 | `evaluatestep` + TSAdapt shim | `-ts_adapt_type basic` selects steps; order 2 with the limiter inactive |
| 6 | `interpolate` | `\|X(1) − y_{n+1}\| = 0`; stiff limit `X(θ) → 1−θ`; dense-output order ≥ 1 |

Dependency chain: M0 and M1 are independent of each other and of everything else (a bug fix
and a Firedrake-free numpy module), so either can go first. M2a and M2b are independent of
each other but both depend on M1. M3 needs M0, M1, M2a and M2b. M4 needs M3. M5 and M6 are
additive on M4 and independent of each other. M0–M4 is the core deliverable; M5–M6 complete
COOL-193's acceptance list.

Milestones 2a and 2b are deliberately separate: 2a proves the stage solve (`Z_i`, the shift,
the SNES path) with a trivial tableau, 2b proves the Shu–Osher loop with no implicit part.
Neither can hide a bug in the other, and 3 combines two independently-green halves.

**Negative control for M4.** Pair the bounds test with a control that reproduces COOL-193's
reference `−0.0099 / +1.0100` under `arkimex` on the same problem. Without it, a passing
bounds test cannot distinguish "Shu–Osher works" from "this problem was never challenging."

**M4 also resolves `rk-method-spec.md` §6.6** either way: passing retires R8's clamp
requirement with evidence; failing localises to a negative Shu–Osher coefficient or an `r`
above `R(A,b)`, both of which §8 already gates.

## 10. Expectations and caveats

- Realized order falls toward 1 wherever the limiter is active — it is a first-order
  perturbation. Error control still functions; it measures the limited method.
- Depends on the IMEX residual fix in `3766c20` (`repair_ts_snes_callbacks`), without which
  the explicit part never reaches the state.
- `_TSContext.split()` builds sub-contexts without passing `options_prefix`, `project_rhs`,
  or `rhs_projection_parameters` (`solving_utils.py:266`). A split sub-context carrying `G`
  would therefore build a default-parameter projection solver with no prefix. Believed
  unreachable — the projection is only triggered from `form_rhs_function`, which split
  contexts do not serve — but it sits on the code path M0 touches, so confirm rather than
  assume while there.
- Performance is not a design constraint: per-stage cost is Firedrake assembly plus the
  implicit solve, so Python-level `axpy` orchestration is noise. The identity-row freeze and
  the un-skipped final explicit evaluation are both known, accepted inefficiencies.
