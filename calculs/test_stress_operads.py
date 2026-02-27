"""Randomized stress tests for core operads.

These tests are deterministic (fixed seeds) and check high-value algebraic
identities on random homogeneous elements.
"""

from __future__ import annotations

import random

import pytest

from uconf import BarrattEccles, Lie, Surjection


def _canonical_basis_key(basis):
    if not isinstance(basis, tuple):
        return basis
    out = []
    for item in basis:
        if hasattr(item, "tuple"):
            out.append(tuple(item.tuple()))
        else:
            out.append(item)
    return tuple(out)


def _as_dict(x):
    return {_canonical_basis_key(k): v for k, v in x if v != 0}


def _random_coeff(rng: random.Random) -> int:
    coeff = 0
    while coeff == 0:
        coeff = rng.randint(-2, 2)
    return coeff


def _random_homogeneous_surjection(
    rng: random.Random, arity: int, degree: int, max_terms: int = 3
):
    parent = Surjection(arity)
    basis = list(parent.basis_it(degree))
    assert basis, f"No basis for Surjection({arity}) in degree {degree}."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def _random_homogeneous_be(
    rng: random.Random, arity: int, degree: int, max_terms: int = 3
):
    parent = BarrattEccles(arity)
    basis = list(parent.basis_it(degree))
    assert basis, f"No basis for BarrattEccles({arity}) in degree {degree}."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def _random_homogeneous_lie(rng: random.Random, arity: int, max_terms: int = 3):
    parent = Lie(arity)
    basis = list(parent.basis_it())
    assert basis, f"No basis for Lie({arity})."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def _degree_of_homogeneous(element) -> int:
    parent = element.parent()
    supp = list(element.support())
    assert supp, "Expected nonzero homogeneous element"
    return parent.degree_on_basis(supp[0])


def test_stress_surjection_linearity_and_unit():
    rng = random.Random(20260227)

    for _ in range(12):
        m = rng.randint(2, 3)
        n = rng.randint(2, 3)
        dx = rng.randint(0, 1)
        dz = rng.randint(0, 1)

        x = _random_homogeneous_surjection(rng, m, dx)
        y = _random_homogeneous_surjection(rng, m, dx)
        z = _random_homogeneous_surjection(rng, n, dz)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = Surjection.compose(a * x + b * y, i, z)
        rhs = a * Surjection.compose(x, i, z) + b * Surjection.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = Surjection.unit()
        assert _as_dict(Surjection.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(Surjection.compose(x, k, one)) == _as_dict(x)


def test_stress_barratt_eccles_linearity_and_unit():
    rng = random.Random(20260228)

    for _ in range(10):
        m = rng.randint(2, 3)
        n = rng.randint(2, 3)
        # Keep BE stress moderate because basis grows quickly.
        dx = rng.randint(0, 1)
        dz = rng.randint(0, 1)

        x = _random_homogeneous_be(rng, m, dx)
        y = _random_homogeneous_be(rng, m, dx)
        z = _random_homogeneous_be(rng, n, dz)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = BarrattEccles.compose(a * x + b * y, i, z)
        rhs = a * BarrattEccles.compose(x, i, z) + b * BarrattEccles.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = BarrattEccles.unit()
        assert _as_dict(BarrattEccles.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(BarrattEccles.compose(x, k, one)) == _as_dict(x)


def test_stress_lie_linearity_and_unit():
    rng = random.Random(20260301)

    for _ in range(12):
        m = rng.randint(2, 4)
        n = rng.randint(2, 4)

        x = _random_homogeneous_lie(rng, m)
        y = _random_homogeneous_lie(rng, m)
        z = _random_homogeneous_lie(rng, n)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = Lie.compose(a * x + b * y, i, z)
        rhs = a * Lie.compose(x, i, z) + b * Lie.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = Lie.unit()
        assert _as_dict(Lie.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(Lie.compose(x, k, one)) == _as_dict(x)


def test_stress_surjection_boundary_squared_zero():
    rng = random.Random(20260302)

    for _ in range(20):
        r = rng.randint(2, 4)
        d = rng.randint(0, 2)
        x = _random_homogeneous_surjection(rng, r, d)
        assert _as_dict(x.boundary().boundary()) == {}


def test_stress_barratt_eccles_boundary_squared_zero():
    rng = random.Random(20260303)

    for _ in range(15):
        r = rng.randint(2, 4)
        d = rng.randint(0, 2)
        x = _random_homogeneous_be(rng, r, d)
        assert _as_dict(x.boundary().boundary()) == {}


@pytest.mark.parametrize("arity", [3, 4])
def test_stress_lie_jacobi_random_linear(arity: int):
    rng = random.Random(20260304 + arity)

    bracket = Lie(2)((1,))
    jacobi_generator = Lie.compose(bracket, 1, bracket)
    jacobi = [
        jacobi_generator,
        jacobi_generator.permute([2, 3, 1]),
        jacobi_generator.permute([3, 1, 2]),
    ]
    assert _as_dict(sum(jacobi)) == {}

    # random insertion stress: J ∘_i x should remain zero for random x
    for _ in range(8):
        x = _random_homogeneous_lie(rng, arity)
        i = rng.randint(1, 3)
        inserted = sum(Lie.compose(y, i, x) for y in jacobi)
        assert _as_dict(inserted) == {}
