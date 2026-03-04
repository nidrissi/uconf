"""Regression tests for the Lie operad implementation.

Permutation inputs are passed as one-line notation lists, e.g. ``[2, 3, 1]``.
Using tuples may be interpreted by Sage in cycle-style constructors and can lead
to ambiguous behavior in tests.
"""

import math
import time

import pytest
from sage.all import ZZ

from uconf import Lie


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


@pytest.mark.parametrize("n", range(0, 6))
def test_basis_it_size(n: int) -> None:
    basis = list(Lie(n).basis_it())
    if n == 0:
        expected_size = 0
    else:
        expected_size = math.factorial(n - 1)
    assert len(basis) == expected_size, "Unexpected basis size in arity n."


def test_unit() -> None:
    u = Lie.unit()
    assert _as_dict(u) == {(): 1}, "Unit should be the arity-1 generator x1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_antisymmetry_via_permutation(sigma: list[int]) -> None:
    l2 = Lie(2)
    bracket = l2((1,))
    assert bracket.permute(sigma) == -bracket, "Swapping x1 and x2 must negate [x1,x2]."


def test_basic_composition() -> None:
    l2 = Lie(2)
    bracket = l2((1,))
    composed = Lie.compose(bracket, 1, bracket)
    target = composed.parent()
    expected = target((1, 2)) - target((2, 1))
    assert _as_dict(composed) == _as_dict(expected), (
        "[x1,[x2,x3]] - [x2,[x1,x3]] expected."
    )


def test_compose_requires_same_base_ring() -> None:
    x = Lie(2)((1,))
    y = Lie(2, base_ring=ZZ)((1,))
    with pytest.raises(TypeError, match="same base ring"):
        Lie.compose(x, 1, y)


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_operadic_unit_axioms(input_pos: int) -> None:
    l3 = Lie(3)
    x = l3((1, 2))
    unit = Lie.unit()

    left = Lie.compose(unit, 1, x)
    right = Lie.compose(x, input_pos, unit)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed: 1∘1 x = x."
    assert _as_dict(right) == x_dict, (
        f"Right unit axiom failed at input {input_pos}: x∘{input_pos} 1 = x."
    )


def test_jacobi_identity() -> None:
    l2 = Lie(2)
    bracket = l2((1,))

    jacobi_generator = Lie.compose(bracket, 1, bracket)
    jacobi = (
        jacobi_generator
        + jacobi_generator.permute([2, 3, 1])
        + jacobi_generator.permute([3, 1, 2])
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

    For any μ ∈ Lie(2) the identity
    ``(μ ∘_1 μ) ∘_1 μ  =  μ ∘_1 (μ ∘_1 μ)``
    must hold in Lie(4).  Both sides require compositions through
    Lie(3) → Lie(4), exercising the full compose/project pipeline.
    """

    bracket = Lie(2)((1,))

    t0 = time.perf_counter()
    lhs = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)
    rhs = Lie.compose(bracket, 1, Lie.compose(bracket, 1, bracket))
    elapsed = time.perf_counter() - t0

    assert lhs == rhs, "Sequential associativity failed at arity 4."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 4 (limit: 10 s)."


def test_sequential_associativity_arity5() -> None:
    """Verify the operad sequential-associativity axiom at arity 5.

    Uses ``(μ ∘_1 (μ ∘_1 μ)) ∘_1 μ = μ ∘_1 ((μ ∘_1 μ) ∘_1 μ)`` in Lie(5),
    a non-trivial identity requiring four composition calls in Lie(3)–Lie(5).
    """

    bracket = Lie(2)((1,))
    mu3 = Lie.compose(bracket, 1, bracket)  # Lie(3)

    t0 = time.perf_counter()
    lhs = Lie.compose(Lie.compose(bracket, 1, mu3), 1, bracket)  # Lie(4)→Lie(5)
    rhs = Lie.compose(bracket, 1, Lie.compose(mu3, 1, bracket))  # Lie(4)→Lie(5)
    elapsed = time.perf_counter() - t0

    assert lhs == rhs, "Sequential associativity failed at arity 5."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 5 (limit: 10 s)."


def test_compose_full_basis_arity5() -> None:
    """Composition of full-basis-sum elements from Lie(3) lands in Lie(5).

    Builds the sum of all basis elements in Lie(3) and composes it with
    itself to produce an element of Lie(5).  The result must be non-zero
    and the call must complete quickly after caches are warm.
    """

    l3 = Lie(3)
    x = sum((l3(k) for k in l3._basis_keys()), l3.zero())  # sum all 2! basis elts
    assert x != l3.zero(), "Sum of Lie(3) basis should be non-zero."

    Lie.compose(x, 1, x)  # warm-up: build arity-5 caches

    t0 = time.perf_counter()
    result = Lie.compose(x, 1, x)
    elapsed = time.perf_counter() - t0

    assert result.parent().arity() == 5, "Compose of Lie(3)⊗Lie(3) must land in Lie(5)."
    assert elapsed < 10, f"compose took {elapsed:.2f} s at arity 5 warm (limit: 10 s)."


def test_compose_full_basis_arity6() -> None:
    """Composition of full-basis-sum elements from Lie(3) and Lie(4) lands in Lie(6).

    This exercises the arity-6 cache path (PBW matrix 720×120, left-inverse
    120×720).  The warm call should complete in well under 30 seconds.
    """

    l3 = Lie(3)
    l4 = Lie(4)
    x = sum((l3(k) for k in l3._basis_keys()), l3.zero())
    y = sum((l4(k) for k in l4._basis_keys()), l4.zero())

    Lie.compose(x, 1, y)  # warm-up: build arity-6 caches

    t0 = time.perf_counter()
    result = Lie.compose(x, 1, y)
    elapsed = time.perf_counter() - t0

    assert result.parent().arity() == 6, "Compose of Lie(3)⊗Lie(4) must land in Lie(6)."
    assert elapsed < 30, f"compose took {elapsed:.2f} s at arity 6 warm (limit: 30 s)."


# ===========================================================================
# Parallel associativity axiom:  (x∘_i y)∘_{k+n-1} z = (x∘_k z)∘_i y  for i < k
# ===========================================================================


@pytest.mark.parametrize("i,k", [(1, 2), (1, 3), (2, 3)])
def test_lie_parallel_associativity_arity4(i: int, k: int) -> None:
    """Parallel associativity for p ∈ Lie(3), q, r ∈ Lie(2) → Lie(5)."""
    mu3 = Lie.compose(Lie(2)((1,)), 1, Lie(2)((1,)))  # Lie(3)
    mu2 = Lie(2)((1,))
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
    """Parallel associativity for p ∈ Lie(4), q, r ∈ Lie(2) → Lie(6)."""
    bracket = Lie(2)((1,))
    mu4 = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)  # Lie(4)
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
    """(σ·x)∘_i(τ·y) = (σ∘_iτ)·(x∘_{σ^{-1}(i)} y) for Lie(3)×Lie(2)."""
    x = Lie.compose(Lie(2)((1,)), 1, Lie(2)((1,)))  # Lie(3)
    y = Lie(2)((1,))
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
    """(σ·x)∘_i(τ·y) = (σ∘_iτ)·(x∘_{σ^{-1}(i)} y) for Lie(4)×Lie(2)."""
    bracket = Lie(2)((1,))
    mu4 = Lie.compose(Lie.compose(bracket, 1, bracket), 1, bracket)  # Lie(4)
    y = Lie(2)((1,))
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
    """d²(x) = 0 for every basis element of Lie(n) (boundary is identically 0)."""
    l = Lie(n)
    zero = l.zero()
    for elem in l.basis_it():
        assert elem.boundary().boundary() == zero, f"d²({elem}) ≠ 0 in Lie({n})"
