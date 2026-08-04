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


@pytest.mark.parametrize("A,b,r", [(SSPRK32_A, SSPRK32_B, 2.0), (HEUN_A, HEUN_B, 1.0)])
def test_shu_osher_reproduces_the_butcher_map(A, b, r):
    """The two representations must agree to machine precision."""
    P, q = shu_osher(A, b, r)
    for z in [-0.3, -1.0 + 0.4j, 0.7, -2.5, 1.5 - 2.0j]:
        np.testing.assert_allclose(
            _stages_shu_osher(z, P, q, r), _stages_butcher(z, A, b), atol=1e-14
        )


@pytest.mark.parametrize("A,b,r", [(SSPRK32_A, SSPRK32_B, 2.0), (HEUN_A, HEUN_B, 1.0)])
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
    with pytest.raises(ShuOsherError, match=r"2\.0"):
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
