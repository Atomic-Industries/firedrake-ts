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
