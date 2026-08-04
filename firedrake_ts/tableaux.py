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

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TABLEAUX",
    "ARKTableau",
    "ShuOsherError",
    "acceptance_report",
    "butcher_to_K",
    "kraaijevanger_radius",
    "shu_osher",
    "stability_function",
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
    return bool((Minv @ K >= -tol).all() and (Minv @ np.ones(n) >= -tol).all())


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
    rhs = np.linalg.solve(np.eye(n) - z * At, e)
    return 1.0 + z * (np.asarray(w, dtype=float) @ rhs)


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

TABLEAUX = {t.name: t for t in (_IMEX_EULER, _SSPRK2, _ESDIRK_GAMMA5, _SSP2_444_LSA)}
