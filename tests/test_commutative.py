"""Regression tests for the commutative operad implementation."""

import itertools

import pytest
from sage.all import ZZ, QQ

from uconf import Commutative


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


def _block_permutation(
    sigma: tuple[int, ...], i: int, tau: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the block permutation ``sigma ∘_i tau`` in one-line notation."""

    shift = len(tau) - 1
    result = []
    for value in sigma:
        if value < i:
            result.append(value)
        elif value > i:
            result.append(value + shift)
        else:
            result.extend([t + i - 1 for t in tau])
    return tuple(result)


def _inverse_one_line(sigma: tuple[int, ...]) -> tuple[int, ...]:
    """Return inverse permutation in one-line notation."""

    inv = [0] * len(sigma)
    for pos, value in enumerate(sigma, start=1):
        inv[value - 1] = pos
    return tuple(inv)


def test_commutative_unit() -> None:
    unit = Commutative.unit()
    assert _as_dict(unit) == {(): 1}


def test_commutative_basic_composition() -> None:
    x = Commutative(2, QQ)(())
    y = Commutative(3, QQ)(())
    result = Commutative.compose(x, 2, y)
    assert _as_dict(result) == {(): 1}
    assert result.parent().arity() == 4


@pytest.mark.parametrize("sigma", ([2, 1], [3, 1, 2]))
def test_commutative_trivial_permutation_action(sigma: list[int]) -> None:
    x = Commutative(len(sigma), QQ)(())
    assert x.permute(sigma) == x


def test_commutative_requires_same_base_ring() -> None:
    x = Commutative(2, QQ)(())
    y = Commutative(2, base_ring=ZZ)(())
    with pytest.raises(TypeError, match="same base ring"):
        Commutative.compose(x, 1, y)


@pytest.mark.parametrize("m,n,p", [(2, 3, 2), (3, 2, 3), (4, 2, 2)])
def test_commutative_sequential_associativity_axiom(m: int, n: int, p: int) -> None:
    """Check ``(x∘_i y)∘_{i+j-1} z = x∘_i(y∘_j z)`` for all valid ``i,j``."""

    x = Commutative(m, QQ)(())
    y = Commutative(n, QQ)(())
    z = Commutative(p, QQ)(())
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            lhs = Commutative.compose(Commutative.compose(x, i, y), i + j - 1, z)
            rhs = Commutative.compose(x, i, Commutative.compose(y, j, z))
            assert _as_dict(lhs) == _as_dict(rhs)


@pytest.mark.parametrize("m,n,p", [(4, 2, 3), (5, 2, 2)])
def test_commutative_parallel_associativity_axiom(m: int, n: int, p: int) -> None:
    """Check ``(x∘_i y)∘_{k+n-1} z = (x∘_k z)∘_i y`` for ``i < k``."""

    x = Commutative(m, QQ)(())
    y = Commutative(n, QQ)(())
    z = Commutative(p, QQ)(())
    for i in range(1, m):
        for k in range(i + 1, m + 1):
            lhs = Commutative.compose(Commutative.compose(x, i, y), k + n - 1, z)
            rhs = Commutative.compose(Commutative.compose(x, k, z), i, y)
            assert _as_dict(lhs) == _as_dict(rhs)


def test_commutative_axioms_on_linear_combinations() -> None:
    """Check associativity identities on non-basis elements."""

    x = 2 * Commutative(3, QQ)(())
    y = -3 * Commutative(2, QQ)(())
    z = 5 * Commutative(2, QQ)(())

    lhs = Commutative.compose(Commutative.compose(x, 2, y), 2, z)
    rhs = Commutative.compose(x, 2, Commutative.compose(y, 1, z))
    assert _as_dict(lhs) == _as_dict(rhs)

    lhs_parallel = Commutative.compose(Commutative.compose(x, 1, y), 3, z)
    rhs_parallel = Commutative.compose(Commutative.compose(x, 2, z), 1, y)
    assert _as_dict(lhs_parallel) == _as_dict(rhs_parallel)


def test_commutative_compose_equivariant_under_block_permutations() -> None:
    """Check ``(σx)∘_i(τy) = (x∘_{σ^{-1}(i)}y)^{σ∘_iτ}``."""

    m, n = 4, 3
    x = Commutative(m, QQ)(())
    y = Commutative(n, QQ)(())
    sigmas = list(itertools.permutations(range(1, m + 1), m))[:5]
    taus = list(itertools.permutations(range(1, n + 1), n))[:4]

    for sigma, tau in itertools.product(sigmas, taus):
        sigma_inv = _inverse_one_line(sigma)
        for i in range(1, m + 1):
            lhs = Commutative.compose(x.permute(list(sigma)), i, y.permute(list(tau)))
            rhs = Commutative.compose(x, sigma_inv[i - 1], y).permute(
                list(_block_permutation(sigma, i, tau))
            )
            assert _as_dict(lhs) == _as_dict(rhs)


def test_commutative_equivariance_on_linear_combinations() -> None:
    """Check the same equivariance identity on non-basis elements."""

    x = 7 * Commutative(3, QQ)(())
    y = -5 * Commutative(2, QQ)(())
    sigma = (3, 1, 2)
    tau = (2, 1)
    i = 1
    sigma_inv = _inverse_one_line(sigma)

    lhs = Commutative.compose(x.permute(list(sigma)), i, y.permute(list(tau)))
    rhs = Commutative.compose(x, sigma_inv[i - 1], y).permute(
        list(_block_permutation(sigma, i, tau))
    )
    assert _as_dict(lhs) == _as_dict(rhs)


@pytest.mark.parametrize("input_pos", range(1, 4))
def test_commutative_unit_axioms(input_pos: int) -> None:
    """1∘_1 x = x and x∘_i 1 = x for all valid i."""
    x = Commutative(3, QQ)(())
    one = Commutative.unit()
    assert _as_dict(Commutative.compose(one, 1, x)) == _as_dict(x), (
        "Left unit axiom failed for Commutative."
    )
    assert _as_dict(Commutative.compose(x, input_pos, one)) == _as_dict(x), (
        f"Right unit axiom failed at input {input_pos} for Commutative."
    )


@pytest.mark.parametrize("n", range(1, 6))
def test_commutative_differential_squared_zero(n: int) -> None:
    """d²(x) = 0 for every basis element of Com(n) (boundary is identically 0)."""
    zero = Commutative(n, QQ).zero()
    for elem in [Commutative(n, QQ)(())]:
        assert elem.boundary().boundary() == zero, f"d²({elem}) ≠ 0 in Com({n})"
