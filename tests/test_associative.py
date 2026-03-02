"""Regression tests for the associative operad implementation."""

import itertools

import pytest
from sage.all import ZZ

from uconf import Associative


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


def test_associative_unit() -> None:
    unit = Associative.unit()
    assert _as_dict(unit) == {(1,): 1}


def test_associative_basic_composition() -> None:
    x = Associative(2)((1, 2))
    y = Associative(2)((2, 1))
    result = Associative.compose(x, 1, y)
    assert _as_dict(result) == {(2, 1, 3): 1}


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_associative_unit_axioms(input_pos: int) -> None:
    x = Associative(3)((2, 1, 3))
    one = Associative.unit()
    assert _as_dict(Associative.compose(one, 1, x)) == _as_dict(x)
    assert _as_dict(Associative.compose(x, input_pos, one)) == _as_dict(x)


def test_associative_requires_same_base_ring() -> None:
    x = Associative(2)((1, 2))
    y = Associative(2, base_ring=ZZ)((1, 2))
    with pytest.raises(TypeError, match="same base ring"):
        Associative.compose(x, 1, y)


@pytest.mark.parametrize("m,n,p", [(3, 2, 2), (3, 2, 3)])
def test_associative_sequential_associativity_axiom(m: int, n: int, p: int) -> None:
    """Check ``(x∘_i y)∘_{i+j-1} z = x∘_i(y∘_j z)`` on a non-trivial basis sample."""

    xs = list(Associative(m).basis_it())
    ys = list(Associative(n).basis_it())
    zs = list(Associative(p).basis_it())

    sample_x = xs[: min(4, len(xs))]
    sample_y = ys[: min(3, len(ys))]
    sample_z = zs[: min(3, len(zs))]

    for x, y, z in itertools.product(sample_x, sample_y, sample_z):
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                lhs = Associative.compose(Associative.compose(x, i, y), i + j - 1, z)
                rhs = Associative.compose(x, i, Associative.compose(y, j, z))
                assert _as_dict(lhs) == _as_dict(rhs)


def test_associative_parallel_associativity_axiom() -> None:
    """Check ``(x∘_i y)∘_{k+n-1} z = (x∘_k z)∘_i y`` for ``i < k``."""

    m, n, p = 4, 2, 3
    xs = list(Associative(m).basis_it())[:3]
    ys = list(Associative(n).basis_it())[:2]
    zs = list(Associative(p).basis_it())[:2]

    for x, y, z in itertools.product(xs, ys, zs):
        for i in range(1, m):
            for k in range(i + 1, m + 1):
                lhs = Associative.compose(Associative.compose(x, i, y), k + n - 1, z)
                rhs = Associative.compose(Associative.compose(x, k, z), i, y)
                assert _as_dict(lhs) == _as_dict(rhs)


def test_associative_compose_equivariant_under_block_permutations() -> None:
    """Check ``(σx)∘_i(τy) = (x∘_{σ^{-1}(i)}y)^{σ∘_iτ}``."""

    m, n = 3, 2
    xs = list(Associative(m).basis_it())[:3]
    ys = list(Associative(n).basis_it())[:2]
    sigmas = list(itertools.permutations(range(1, m + 1), m))[:4]
    taus = list(itertools.permutations(range(1, n + 1), n))

    for x, y, sigma, tau in itertools.product(xs, ys, sigmas, taus):
        sigma_inv = _inverse_one_line(sigma)
        for i in range(1, m + 1):
            lhs = Associative.compose(x.permute(list(sigma)), i, y.permute(list(tau)))
            block = _block_permutation(sigma, i, tau)
            rhs = Associative.compose(x, sigma_inv[i - 1], y).permute(list(block))
            assert _as_dict(lhs) == _as_dict(rhs)


def test_associative_equivariance_on_linear_combinations() -> None:
    """Check the same equivariance identity on non-basis elements."""

    x = 2 * Associative(3)((1, 3, 2)) - Associative(3)((2, 1, 3))
    y = 3 * Associative(2)((2, 1)) + Associative(2)((1, 2))
    sigma = (2, 3, 1)
    tau = (2, 1)
    i = 2
    sigma_inv = _inverse_one_line(sigma)

    lhs = Associative.compose(x.permute(list(sigma)), i, y.permute(list(tau)))
    rhs = Associative.compose(x, sigma_inv[i - 1], y).permute(
        list(_block_permutation(sigma, i, tau))
    )
    assert _as_dict(lhs) == _as_dict(rhs)
