"""Tests for the Surjection action on reduced cochains of spheres."""

from itertools import permutations
from random import Random

import pytest
from sage.all import QQ

from uconf import (
    SimplicialCochains,
    Surjection,
    SurjectionSphereCochainAlgebra,
    surjection_cochain_action,
)
from uconf.core.signs import koszul_sign_of_permutation, sign_from_exponent


def _as_dict(x):
    return {k: v for k, v in x if v != 0}


def _sphere_coeff(x, d):
    return _as_dict(x).get(f"ɑ{d}", 0)


def _top_coeff(x, top_key):
    return _as_dict(x).get(top_key, 0)


def test_sphere_surjection_unitality():
    alg = SurjectionSphereCochainAlgebra(d=3, base_ring=QQ)
    g = alg.module.generator()
    unit = Surjection.unit(QQ)

    assert alg.act(unit, [g]) == g
    assert alg.act(unit, [7 * g]) == 7 * g


@pytest.mark.parametrize("e", [1, 2])
@pytest.mark.parametrize("k1", [2, 3])
@pytest.mark.parametrize("k2", [2, 3])
@pytest.mark.parametrize("d1", [0, 1, 2])
@pytest.mark.parametrize("d2", [0, 1, 2])
def test_sphere_surjection_associativity_on_generator(e: int, k1: int, k2: int, d1: int, d2: int):
    """Check μ_{p∘_i q}(g,..,g)=μ_p(g,..,μ_q(g,..,g),..,g) for d=2."""
    alg = SurjectionSphereCochainAlgebra(d=e, base_ring=QQ)
    g = alg.module.generator()

    basis1 = Surjection(k1, QQ).graded_basis(d1)
    basis2 = Surjection(k2, QQ).graded_basis(d2)

    for p in basis1:
        for q in basis2:
            for i in range(1, p.arity() + 1):
                left = alg.act(Surjection.compose(p, i, q), [g] * (k1 + k2 - 1))
                right_inner = alg.act(q, [g] * k2)
                right_list = [g] * k1
                right_list[i - 1] = right_inner
                right = alg.act(p, right_list)
                koszul_exponent = (q.degree() * (i - 1)) % 2
                koszul_sign = sign_from_exponent(koszul_exponent)
                right *= koszul_sign
                assert left == right


@pytest.mark.parametrize("e", [1, 2])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("d", [0, 1, 2])
def test_sphere_surjection_equivariance(e: int, k: int, d: int):
    """Check μ_{u·σ}(a_1,a_2)=μ_u(a_{σ^{-1}(1)},a_{σ^{-1}(2)})."""
    alg = SurjectionSphereCochainAlgebra(d=e, base_ring=QQ)
    g = alg.module.generator()

    for u in Surjection(k, QQ).graded_basis(d):
        for sigma in permutations(range(1, k + 1)):
            lhs = alg.act(u.permute(list(sigma)), [g] * k)
            rhs = alg.act(u, [g] * k)
            rhs *= koszul_sign_of_permutation([sigma[i] - 1 for i in range(k)], [e] * k)
            assert lhs == rhs


def test_sphere_surjection_matches_top_cochain_action():
    """Compare with μ_u on the top cochain of Δ^d."""
    rng = Random(20260317)

    count = 0
    for _ in range(50):
        d = rng.randint(1, 3)
        k = rng.randint(2, 4)
        e = rng.randint(0, 5)
        basis = Surjection(k, QQ).graded_basis(e)
        u = rng.choice(basis)

        alg = SurjectionSphereCochainAlgebra(d=d, base_ring=QQ)
        g = alg.module.generator()

        C = SimplicialCochains(N=d, base_ring=QQ)
        top_key = tuple(range(d + 1))
        top = C(top_key)

        sphere_val = alg.act(u, [g] * k)
        simplex_val = surjection_cochain_action(u, (top,) * k)
        coeff = _sphere_coeff(sphere_val, d)
        assert coeff == _top_coeff(simplex_val, top_key)
        if coeff:
            count += 1
    assert count, "All coefficients were zero"


def test_sphere_surjection_degree_mismatch_gives_zero():
    """A degree-mismatched operation acts by zero on the generator."""
    alg = SurjectionSphereCochainAlgebra(d=2, base_ring=QQ)
    g = alg.module.generator()
    u = Surjection(2, QQ)((1, 2, 1))  # degree 1, should be zero for d=2 and arity 2

    assert alg.act(u, [g, g]) == alg.module.zero()
