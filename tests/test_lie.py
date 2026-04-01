"""Regression tests for the Lie operad implementation.

Permutation inputs are passed as one-line notation lists, e.g. ``[2, 3, 1]``.
Using tuples may be interpreted by Sage in cycle-style constructors and can lead
to ambiguous behavior in tests.
"""

import itertools
import math
import time
from random import Random

import pytest
from sage.all import ZZ, QQ

from uconf import Lie


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


@pytest.mark.parametrize("n", range(0, 6))
def test_basis_it_size(n: int) -> None:
    basis = list(Lie(n, QQ).basis_iter(0))
    if n == 0:
        expected_size = 0
    else:
        expected_size = math.factorial(n - 1)
    assert len(basis) == expected_size, "Unexpected basis size in arity n."


def test_unit() -> None:
    u = Lie.unit(QQ)
    assert _as_dict(u) == {(): 1}, "Unit should be the arity-1 generator x1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_antisymmetry_via_permutation(sigma: list[int]) -> None:
    l2 = Lie(2, QQ)
    bracket = l2((1,))
    assert bracket.permute(sigma) == -bracket, "Swapping x1 and x2 must negate [x1,x2]."


def test_basic_composition() -> None:
    l2 = Lie(2, QQ)
    bracket = l2((1,))
    composed = Lie.compose(bracket, 1, bracket)
    target = composed.parent()
    expected = target((1, 2)) - target((2, 1))
    assert _as_dict(composed) == _as_dict(expected), "[x1,[x2,x3]] - [x2,[x1,x3]] expected."


def test_compose_requires_same_base_ring() -> None:
    x = Lie(2, QQ)((1,))
    y = Lie(2, base_ring=ZZ)((1,))
    with pytest.raises(TypeError, match="same base ring"):
        Lie.compose(x, 1, y)


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_operadic_unit_axioms(input_pos: int) -> None:
    l3 = Lie(3, QQ)
    x = l3((1, 2))
    unit = Lie.unit(QQ)

    left = Lie.compose(unit, 1, x)
    right = Lie.compose(x, input_pos, unit)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed: 1∘1 x = x."
    assert _as_dict(right) == x_dict, (
        f"Right unit axiom failed at input {input_pos}: x∘{input_pos} 1 = x."
    )


def test_jacobi_identity() -> None:
    l2 = Lie(2, QQ)
    bracket = l2((1,))

    jacobi_generator = Lie.compose(bracket, 1, bracket)
    jacobi = (
        jacobi_generator + jacobi_generator.permute([2, 3, 1]) + jacobi_generator.permute([3, 1, 2])
    )
    assert _as_dict(jacobi) == {}, "Jacobi identity failed in arity 4."


# ---------------------------------------------------------------------------
# Stress tests: correctness and timing at higher arities
# ---------------------------------------------------------------------------
# Each test checks an algebraic identity that requires Lie.compose to produce
# elements in arities 4–6, confirming both correctness and that the
# class-level caches keep per-call times well under practical limits.
# Time limits are intentionally generous (10 s / 30 s) so they remain green
# on slow CI machines while still catching catastrophic regressions.


def test_sequential_associativity_arity4() -> None:
    """Verify the operad sequential-associativity axiom at arity 4.

    For any μ ∈ Lie(2, QQ) the identity
    ``(μ ∘_1 μ) ∘_1 μ  =  μ ∘_1 (μ ∘_1 μ)``
    must hold in Lie(4, QQ).  Both sides require compositions through
    Lie(3, QQ) → Lie(4, QQ), exercising the full compose/project pipeline.
    """
    bracket = Lie(2, QQ)((1,))

    t0 = time.perf_counter()
    lhs = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)
    rhs = Lie.compose(bracket, 1, Lie.compose(bracket, 1, bracket))
    elapsed = time.perf_counter() - t0

    assert lhs == rhs, "Sequential associativity failed at arity 4."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 4 (limit: 10 s)."


def test_sequential_associativity_arity5() -> None:
    """Verify the operad sequential-associativity axiom at arity 5.

    Uses ``(μ ∘_1 (μ ∘_1 μ)) ∘_1 μ = μ ∘_1 ((μ ∘_1 μ) ∘_1 μ)`` in Lie(5, QQ),
    a non-trivial identity requiring four composition calls in Lie(3, QQ)–Lie(5, QQ).
    """
    bracket = Lie(2, QQ)((1,))
    mu3 = Lie.compose(bracket, 1, bracket)  # Lie(3, QQ)

    t0 = time.perf_counter()
    lhs = Lie.compose(Lie.compose(bracket, 1, mu3), 1, bracket)  # Lie(4, QQ)→Lie(5, QQ)
    rhs = Lie.compose(bracket, 1, Lie.compose(mu3, 1, bracket))  # Lie(4, QQ)→Lie(5, QQ)
    elapsed = time.perf_counter() - t0

    assert lhs == rhs, "Sequential associativity failed at arity 5."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 5 (limit: 10 s)."


def test_compose_full_basis_arity5() -> None:
    """Composition of full-basis-sum elements from Lie(3, QQ) lands in Lie(5, QQ).

    Builds the sum of all basis elements in Lie(3, QQ) and composes it with
    itself to produce an element of Lie(5, QQ).  The result must be non-zero
    and the call must complete quickly after caches are warm.
    """
    l3 = Lie(3, QQ)
    x = sum((l3(k) for k in l3._basis_keys()), l3.zero())  # sum all 2! basis elts
    assert x != l3.zero(), "Sum of Lie(3, QQ) basis should be non-zero."

    Lie.compose(x, 1, x)  # warm-up: build arity-5 caches

    t0 = time.perf_counter()
    result = Lie.compose(x, 1, x)
    elapsed = time.perf_counter() - t0

    assert result.parent().arity() == 5, "Compose of Lie(3, QQ)⊗Lie(3, QQ) must land in Lie(5, QQ)."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 5 warm (limit: 10 s)."


def test_compose_full_basis_arity6() -> None:
    """Composition of full-basis-sum elements from Lie(3, QQ) and Lie(4, QQ) lands in Lie(6, QQ).

    This exercises the arity-6 cache path (PBW matrix 720×120, left-inverse
    120×720).  The warm call should complete in well under 30 seconds.
    """
    l3 = Lie(3, QQ)
    l4 = Lie(4, QQ)
    x = sum((l3(k) for k in l3._basis_keys()), l3.zero())
    y = sum((l4(k) for k in l4._basis_keys()), l4.zero())

    Lie.compose(x, 1, y)  # warm-up: build arity-6 caches

    t0 = time.perf_counter()
    result = Lie.compose(x, 1, y)
    elapsed = time.perf_counter() - t0

    assert result.parent().arity() == 6, "Compose of Lie(3, QQ)⊗Lie(4, QQ) must land in Lie(6, QQ)."
    assert elapsed < 30, f"compose took {elapsed:.2f} s at arity 6 warm (limit: 30 s)."


# ===========================================================================
# Parallel associativity axiom:  (x∘_i y)∘_{k+n-1} z = (x∘_k z)∘_i y  for i < k
# ===========================================================================


@pytest.mark.parametrize("i,k", [(1, 2), (1, 3), (2, 3)])
def test_lie_parallel_associativity_arity4(i: int, k: int) -> None:
    """Parallel associativity for p ∈ Lie(3, QQ), q, r ∈ Lie(2, QQ) → Lie(5, QQ)."""
    mu3 = Lie.compose(Lie(2, QQ)((1,)), 1, Lie(2, QQ)((1,)))  # Lie(3, QQ)
    mu2 = Lie(2, QQ)((1,))
    n = 2
    lhs = Lie.compose(Lie.compose(mu3, i, mu2), k + n - 1, mu2)
    rhs = Lie.compose(Lie.compose(mu3, k, mu2), i, mu2)
    assert lhs == rhs, f"Lie parallel axiom failed for i={i}, k={k} at arity 4"


@pytest.mark.parametrize(
    "i,k",
    [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ],
)
def test_lie_parallel_associativity_arity5(i: int, k: int) -> None:
    """Parallel associativity for p ∈ Lie(4, QQ), q, r ∈ Lie(2, QQ) → Lie(6, QQ)."""
    bracket = Lie(2, QQ)((1,))
    mu4 = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)  # Lie(4, QQ)
    n = 2
    lhs = Lie.compose(Lie.compose(mu4, i, bracket), k + n - 1, bracket)
    rhs = Lie.compose(Lie.compose(mu4, k, bracket), i, bracket)
    assert lhs == rhs, f"Lie parallel axiom failed for i={i}, k={k} at arity 5"


# ===========================================================================
# Equivariance:  (σ·x)∘_i(τ·y) = (σ∘_iτ) · (x∘_{σ^{-1}(i)} y)
# ===========================================================================


def _block_permutation(sigma: list[int], i: int, tau: list[int]) -> list[int]:
    """Return the block permutation ``σ ∘_i τ`` in one-line notation."""
    shift = len(tau) - 1
    result = []
    for v in sigma:
        if v < i:
            result.append(v)
        elif v > i:
            result.append(v + shift)
        else:
            result.extend([t + i - 1 for t in tau])
    return result


def _inverse_one_line(sigma: list[int]) -> list[int]:
    """Return inverse of a one-line permutation."""
    inv = [0] * len(sigma)
    for pos, val in enumerate(sigma, start=1):
        inv[val - 1] = pos
    return inv


@pytest.mark.parametrize(
    "sigma,tau",
    [
        ([2, 1, 3], [2, 1]),  # σ ∈ S_3, τ ∈ S_2
        ([2, 3, 1], [2, 1]),  # σ = 3-cycle, τ = transposition
        ([3, 1, 2], [2, 1]),
        ([2, 1, 3], [1, 2]),  # τ = identity
    ],
)
def test_lie_equivariance_arity3_times_2(sigma: list[int], tau: list[int]) -> None:
    """(σ·x)∘_i(τ·y) = (σ∘_iτ)·(x∘_{σ^{-1}(i)} y) for Lie(3, QQ)×Lie(2, QQ)."""
    x = Lie.compose(Lie(2, QQ)((1,)), 1, Lie(2, QQ)((1,)))  # Lie(3, QQ)
    y = Lie(2, QQ)((1,))
    sigma_inv = _inverse_one_line(sigma)
    for i in range(1, 4):
        lhs = Lie.compose(x.permute(sigma), i, y.permute(tau))
        bp = _block_permutation(sigma, i, tau)
        rhs = Lie.compose(x, sigma_inv[i - 1], y).permute(bp)
        assert lhs == rhs, f"Lie equivariance failed for σ={sigma}, τ={tau}, i={i}"


@pytest.mark.parametrize(
    "sigma,tau",
    [
        ([2, 1, 3, 4], [2, 1]),  # σ ∈ S_4, τ ∈ S_2
        ([2, 3, 1, 4], [2, 1]),
        ([4, 1, 2, 3], [2, 1]),
        ([2, 1, 4, 3], [2, 1]),
    ],
)
def test_lie_equivariance_arity4_times_2(sigma: list[int], tau: list[int]) -> None:
    """(σ·x)∘_i(τ·y) = (σ∘_iτ)·(x∘_{σ^{-1}(i)} y) for Lie(4, QQ)×Lie(2, QQ)."""
    bracket = Lie(2, QQ)((1,))
    mu4 = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)  # Lie(4, QQ)
    y = Lie(2, QQ)((1,))
    sigma_inv = _inverse_one_line(sigma)
    for i in range(1, 5):
        lhs = Lie.compose(mu4.permute(sigma), i, y.permute(tau))
        bp = _block_permutation(sigma, i, tau)
        rhs = Lie.compose(mu4, sigma_inv[i - 1], y).permute(bp)
        assert lhs == rhs, f"Lie equivariance failed for σ={sigma}, τ={tau}, i={i}"


# ===========================================================================
# Square-zero differential:  d² = 0
# ===========================================================================


@pytest.mark.parametrize("n", range(0, 7))
def test_lie_differential_squared_zero(n: int) -> None:
    """d²(x) = 0 for every basis element of Lie(n, QQ) (boundary is identically 0)."""
    l = Lie(n, QQ)
    zero = l.zero()
    for elem in l.basis_iter(0):
        assert elem.boundary().boundary() == zero, f"d²({elem}) ≠ 0 in Lie({n}, QQ)"


# ===========================================================================
# Finite field support: Lie over GF(2) and GF(3)
# ===========================================================================


@pytest.mark.parametrize("n", range(2, 5))
def test_lie_pbw_left_inverse_gf2(n: int) -> None:
    """PBW left-inverse is correct over GF(2): L * M = I."""
    from sage.all import GF, identity_matrix

    R = GF(2)
    l = Lie(n, R)
    pbw = l._pbw_matrix()
    left_inv = l._pbw_left_inverse()
    assert left_inv * pbw == identity_matrix(R, pbw.ncols())


@pytest.mark.parametrize("n", range(2, 5))
def test_lie_permute_gf2(n: int) -> None:
    """Lie elements can be permuted over GF(2) without errors."""
    from sage.all import GF, SymmetricGroup

    R = GF(2)
    l = Lie(n, R)
    S = SymmetricGroup(n)
    sigma = S([2, 1] + list(range(3, n + 1)))
    for elem in l.basis_iter(0):
        perm = elem.permute(sigma)
        # Permuted element lives in same parent
        assert perm.parent() is l


def _random_coeff(rng: Random) -> int:
    coeff = 0
    while coeff == 0:
        coeff = rng.randint(-2, 2)
    return coeff


def _random_homogeneous_lie(rng: Random, arity: int, max_terms: int = 3) -> Lie.Element:
    parent = Lie(arity, QQ)
    basis = list(parent.basis_iter(0))
    assert basis, f"No basis for Lie({arity}, QQ)."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def test_stress_lie_linearity_and_unit() -> None:
    rng = Random(20260301)

    for _ in range(10):
        m = rng.randint(2, 3)
        n = rng.randint(2, 3)

        x = _random_homogeneous_lie(rng, m)
        y = _random_homogeneous_lie(rng, m)
        z = _random_homogeneous_lie(rng, n)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = Lie.compose(a * x + b * y, i, z)
        rhs = a * Lie.compose(x, i, z) + b * Lie.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = Lie.unit(QQ)
        assert _as_dict(Lie.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(Lie.compose(x, k, one)) == _as_dict(x)


@pytest.mark.parametrize("arity", [3, 4])
def test_stress_lie_jacobi_random_linear(arity: int) -> None:
    rng = Random(20260304 + arity)

    bracket = Lie(2, QQ)((1,))
    jacobi_generator = Lie.compose(bracket, 1, bracket)
    jacobi = [
        jacobi_generator,
        jacobi_generator.permute([2, 3, 1]),
        jacobi_generator.permute([3, 1, 2]),
    ]
    assert _as_dict(sum(jacobi)) == {}

    for _ in range(8):
        x = _random_homogeneous_lie(rng, arity)
        i = rng.randint(1, 3)
        inserted = sum(Lie.compose(y, i, x) for y in jacobi)
        assert _as_dict(inserted) == {}


@pytest.mark.parametrize("n", range(2, 4))
def test_lie_right_action(n: int) -> None:
    """(x·σ)·τ = x·(στ) for all σ, τ ∈ S(n)."""
    Lien = Lie(n, QQ)
    Sn = Lien._symmetric_group()
    for x in Lien.basis_iter(0):
        for sigma, tau in itertools.product(list(Sn), repeat=2):
            lhs = x.permute(sigma).permute(tau)
            rhs = x.permute(sigma * tau)
            assert lhs == rhs, f"Right action failed for x={x}, σ={sigma}, τ={tau}"
