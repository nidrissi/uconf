"""Tests for the Surjection action on reduced cochains of spheres."""

from itertools import permutations

import pytest
from sage.all import QQ

from uconf import (
    SimplicialCochains,
    Surjection,
    SurjectionSphereCochainAlgebra,
    surjection_cochain_action,
)


def _as_dict(x):
    return {k: v for k, v in x if v != 0}


def _sphere_coeff(x):
    return _as_dict(x).get((), 0)


def _top_coeff(x, top_key):
    return _as_dict(x).get(top_key, 0)


def test_sphere_surjection_unitality() -> None:
    alg = SurjectionSphereCochainAlgebra(d=3, base_ring=QQ)
    g = alg.module.generator()
    unit = Surjection.unit()

    assert alg.act(unit, [g]) == g
    assert alg.act(unit, [7 * g]) == 7 * g


@pytest.mark.parametrize("k1", [2, 3])
@pytest.mark.parametrize("k2", [2, 3])
@pytest.mark.parametrize("d1", [0, 1, 2])
@pytest.mark.parametrize("d2", [0, 1, 2])
def test_sphere_surjection_associativity_on_generator(
    k1: int, k2: int, d1: int, d2: int
) -> None:
    """Check μ_{p∘_i q}(g,..,g)=μ_p(g,..,μ_q(g,..,g),..,g) for d=2."""
    alg = SurjectionSphereCochainAlgebra(d=2, base_ring=QQ)
    g = alg.module.generator()

    basis1 = list(Surjection(k1, QQ).basis_it(d1))
    basis2 = list(Surjection(k2, QQ).basis_it(d2))

    for p in basis1:
        for q in basis2:
            for i in range(1, p.arity() + 1):
                left = alg.act(Surjection.compose(p, i, q), [g] * (k1 + k2 - 1))
                right_inner = alg.act(q, [g] * k2)
                right_list = [g] * k1
                right_list[i - 1] = right_inner
                right = alg.act(p, right_list)
                assert left == right


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("d", [0, 1, 2])
def test_sphere_surjection_equivariance(k: int, d: int) -> None:
    """Check μ_{u·σ}(a_1,a_2)=μ_u(a_{σ^{-1}(1)},a_{σ^{-1}(2)})."""
    alg = SurjectionSphereCochainAlgebra(d=2, base_ring=QQ)
    g = alg.module.generator()

    for u in Surjection(k, QQ).basis_it(d):
        for sigma in permutations(range(1, k + 1)):
            lhs = alg.act(u.permute(list(sigma)), [g] * k)
            rhs = alg.act(u, [g] * k)
            assert lhs == rhs


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("e", [0, 1, 2, 3, 4, 5])
def test_sphere_surjection_matches_top_cochain_action(d: int, k: int, e: int) -> None:
    """Compare with μ_u on the top cochain of Δ^d."""
    alg = SurjectionSphereCochainAlgebra(d=d, base_ring=QQ)
    g = alg.module.generator()

    C = SimplicialCochains(N=d, base_ring=QQ)
    top_key = tuple(range(d + 1))
    top = C(top_key)

    for u in Surjection(k, QQ).basis_it(e):
        sphere_val = alg.act(u, [g] * k)
        simplex_val = surjection_cochain_action(u, (top,) * k)
        assert _sphere_coeff(sphere_val) == _top_coeff(simplex_val, top_key)


def test_sphere_surjection_degree_mismatch_gives_zero() -> None:
    """A degree-mismatched operation acts by zero on the generator."""
    alg = SurjectionSphereCochainAlgebra(d=2, base_ring=QQ)
    g = alg.module.generator()
    u = Surjection(2, QQ)((1, 2, 1))  # degree 1, should be zero for d=2 and arity 2

    assert alg.act(u, [g, g]) == alg.module.zero()
