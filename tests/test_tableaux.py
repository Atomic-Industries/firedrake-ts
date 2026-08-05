"""Tableau algebra. No Firedrake, no PETSc -- pure numpy at machine precision."""

import dataclasses

import numpy as np
import pytest

from firedrake_ts.tableaux import (
    ShuOsherError,
    butcher_to_K,
    kraaijevanger_radius,
    shu_osher,
)

# Ketcheson's optimal SSPRK(3,2) in stiffly accurate form. Four stages with
# SSP radius R(A,b) = 2. Used as the explicit part of production tableaux.
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

    This is what makes the accepted step bounded by a convex combination of
    previous stages, so no post-step clamping is needed.
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
    """Stability function matches its closed form.

    R(z) = -5(z^2 + 20z + 50) / (2(z-5)^3).
    """
    tab = TABLEAUX["esdirk_gamma5"]
    for z in [-1.0, -10.0, -100.0]:
        expected = -5 * (z**2 + 20 * z + 50) / (2 * (z - 5) ** 3)
        assert stability_function(tab.At, tab.bt, z) == pytest.approx(
            expected, rel=1e-10
        )


def test_acceptance_report_reproduces_the_spec_table():
    """Acceptance report reproduces published properties of esdirk_gamma5."""
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


def test_arktableau_uses_identity_equality():
    """ARKTableau uses identity-based equality and hash, not content equality.

    This is essential for registry lookups: numpy arrays are unhashable and
    direct equality is ambiguous. Regression here means eq=False was removed,
    breaking both hash() and == with array-valued fields.
    """
    # Test 1: hash does not raise. Creating a set exercises hashing on all
    # elements, so this verifies that hash() succeeds for all four tableaux.
    tab = TABLEAUX["imex_euler"]
    assert len({t for t in TABLEAUX.values()}) == 4

    # Test 2: two separately-constructed instances with different arrays
    # compare unequal without raising. This is the case that would raise
    # ValueError ("truth value of array is ambiguous") if eq=True.
    tab2 = dataclasses.replace(tab, b=np.array([2.0, -1.0]))
    assert tab != tab2  # Should not raise, should be False
    assert not (tab == tab2)  # Should not raise


@pytest.mark.parametrize(
    "name,null_direction",
    [
        ("esdirk_gamma5", [1.0, -1.5, 2.25, 0.0]),
        ("ssp2_444_lsa", [1.0, -2.0, 3.0, 0.0]),
    ],
)
def test_dense_output_coefficients_are_stiff_safe(name, null_direction):
    """d must satisfy sum(d) = 1, d.c = 0 and d.g = 0, with At.g = 0.

    The third constraint is what keeps dense output bounded on stiff modes.
    A naive d = (1, 0, 0, 0) has d.g != 0 and makes X(theta) diverge like
    z*theta*(theta - 1) as z -> -infinity. bt.g = 0 is the corresponding
    L-stability condition on the implicit completion weights.

    Nothing else in the suite reads d, so without this test a mistyped
    coefficient would go undetected until dense output is implemented.
    """
    tab = TABLEAUX[name]
    g = np.array(null_direction)
    # Confirm g really is a right null vector of At, so the constraints below
    # are checks on the tableau rather than assertions against magic numbers.
    np.testing.assert_allclose(tab.At @ g, np.zeros_like(g), atol=1e-14)
    assert tab.d.sum() == pytest.approx(1.0, abs=1e-14)
    assert tab.d @ tab.c == pytest.approx(0.0, abs=1e-14)
    assert tab.d @ g == pytest.approx(0.0, abs=1e-14)
    assert tab.bt @ g == pytest.approx(0.0, abs=1e-14)
