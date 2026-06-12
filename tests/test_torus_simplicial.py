"""Tests for the simplicial torus cochain model and its Surjection action.

The model is ``N*((Δ¹/∂Δ¹)²)`` (rank 6) with the Berger--Fresse
interval-cut action, defined by naturality and implemented in closed form
— see ``src/uconf/algebraic/torus_simplicial.py``.  The closed form is
compared term by term with the naturality reference (pullback to ``Δ^m``
+ ``surjection_cochain_action``), and the operad-algebra axioms
(chain map / Leibniz, equivariance, partial composition) are checked.
"""

import itertools
from random import Random

import pytest
from sage.all import GF, QQ

import uconf  # noqa: F401  — wires BarrattEccles.Element.table_reduction
from uconf.core.signs import koszul_sign_of_permutation
from uconf.models.surjection import Surjection
from uconf.algebraic.torus_simplicial import (
    SurjectionTorusSimplicialCochainAlgebra,
    TorusSimplicialCochains,
)

GENERATORS = ["v", "a", "b", "g", "t1", "t2"]


def _basis_keys(n, d, base_ring=QQ):
    return [next(iter(u.support())) for u in Surjection(n, base_ring).graded_basis(d)]


class TestTorusSimplicialCochains:
    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_basis_and_grading(self, base_ring):
        mod = TorusSimplicialCochains(base_ring)
        degrees = {"v": 0, "a": -1, "b": -1, "g": -1, "t1": -2, "t2": -2}
        for gen, deg in degrees.items():
            assert mod.degree_on_basis(gen) == deg
        assert mod.connectivity == -2
        assert len(list(mod.basis_iter(-1))) == 3
        assert len(list(mod.basis_iter(-2))) == 2

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_differential(self, base_ring):
        """δ[a] = δ[b] = [t1]+[t2], δ[g] = -[t1]-[t2], δ = 0 elsewhere."""
        mod = TorusSimplicialCochains(base_ring)
        t_sum = mod("t1") + mod("t2")
        assert mod.boundary(mod("a")) == t_sum
        assert mod.boundary(mod("b")) == t_sum
        assert mod.boundary(mod("g")) == -t_sum
        for gen in ["v", "t1", "t2"]:
            assert mod.boundary(mod(gen)) == mod.zero()
        for gen in GENERATORS:
            assert mod.boundary(mod.boundary(mod(gen))) == mod.zero()


class TestSurjectionTorusSimplicialAlgebra:
    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_unit_acts_as_identity(self, base_ring):
        alg = SurjectionTorusSimplicialCochainAlgebra(base_ring)
        unit = Surjection.unit(base_ring)
        for name in GENERATORS:
            z = alg.module.generator(name)
            assert alg.act(unit, [z]) == z

    def test_cup_products(self):
        """Pinned values of μ_{(1,2)}: only a-with-b products are nonzero."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        m = alg.module
        cup = Surjection(2, QQ)((1, 2))
        assert alg.act(cup, [m("a"), m("b")]) == -m("t1")
        assert alg.act(cup, [m("b"), m("a")]) == -m("t2")
        assert alg.act(cup, [m("v"), m("g")]) == m("g")
        for x, y in [("a", "a"), ("b", "b"), ("g", "g"), ("a", "g"), ("g", "b")]:
            assert alg.act(cup, [m(x), m(y)]) == m.zero(), (x, y)

    def test_cup_one_products(self):
        """μ_{(1,2,1)} pinned values: ∪₁-squares of the three edges."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        m = alg.module
        cup1 = Surjection(2, QQ)((1, 2, 1))
        for edge in ["a", "b", "g"]:
            assert alg.act(cup1, [m(edge), m(edge)]) == -m(edge)
        assert alg.act(cup1, [m("g"), m("t1")]) == -m("t1")
        assert alg.act(cup1, [m("t1"), m("a")]) == m("t1")

    @pytest.mark.parametrize(
        "n,degrees,exhaustive",
        [(1, range(3), True), (2, range(5), True), (3, range(3), True), (3, [3, 4], False)],
    )
    def test_closed_form_matches_reference(self, n, degrees, exhaustive):
        """The closed-form interval-cut enumeration equals the naturality
        reference (pullback to Δ^m + surjection_cochain_action)."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        rng = Random(20260612)
        for d in degrees:
            keys = _basis_keys(n, d)
            if not exhaustive:
                keys = rng.sample(keys, min(40, len(keys)))
            for key in keys:
                for labels in itertools.product(GENERATORS, repeat=n):
                    closed = dict(alg._single_action(key, labels))
                    reference = dict(alg._reference_single_action(key, labels))
                    assert closed == reference, (key, labels)

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3)])
    def test_closed_form_matches_reference_finite_fields(self, base_ring):
        alg = SurjectionTorusSimplicialCochainAlgebra(base_ring)
        rng = Random(20260613)
        for _ in range(150):
            n = rng.choice([2, 3])
            d = rng.choice([0, 1, 2, 3])
            key = rng.choice(_basis_keys(n, d, base_ring))
            labels = tuple(rng.choice(GENERATORS) for _ in range(n))
            assert dict(alg._single_action(key, labels)) == dict(
                alg._reference_single_action(key, labels)
            )

    @pytest.mark.parametrize("n,max_degree", [(1, 2), (2, 3), (3, 2)])
    def test_chain_map_identity(self, n, max_degree):
        """∂(θ_u(z̲)) = θ_{∂u}(z̲) + Σ_k (-1)^{|u|+Σ_{l<k}|z_l|} θ_u(…,∂z_k,…)."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        mod = alg.module
        gens = {name: mod.generator(name) for name in GENERATORS}
        X = Surjection(n, QQ)
        for d in range(max_degree + 1):
            for u in X.graded_basis(d):
                bu = u.boundary()
                for labels in itertools.product(GENERATORS, repeat=n):
                    zs = [gens[lab] for lab in labels]
                    lhs = mod.boundary(alg.act(u, zs))
                    rhs = alg.act(bu, zs) if bu else mod.zero()
                    cumulative = 0
                    for k in range(n):
                        sign = (-1) ** ((d + cumulative) % 2)
                        modified = list(zs)
                        modified[k] = mod.boundary(zs[k])
                        if modified[k]:
                            rhs = rhs + sign * alg.act(u, modified)
                        cumulative += mod.degree_on_basis(labels[k])
                    assert lhs == rhs, (list(u.support()), labels)

    @pytest.mark.parametrize("n,max_degree", [(2, 3), (3, 2)])
    def test_equivariance(self, n, max_degree):
        """μ_{u·σ}(z_1,…,z_n) = κ(σ; dims) · μ_u(z_{σ(1)},…,z_{σ(n)})."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        mod = alg.module
        gens = {name: mod.generator(name) for name in GENERATORS}
        rng = Random(20260614)
        X = Surjection(n, QQ)
        for d in range(max_degree + 1):
            basis = list(X.graded_basis(d))
            for sigma in itertools.permutations(range(1, n + 1)):
                for _ in range(8):
                    u = rng.choice(basis)
                    labels = [rng.choice(GENERATORS) for _ in range(n)]
                    zs = [gens[lab] for lab in labels]
                    dims = [-mod.degree_on_basis(lab) for lab in labels]
                    lhs = alg.act(u.permute(list(sigma)), zs)
                    rhs = alg.act(u, [zs[sigma[i] - 1] for i in range(n)])
                    kappa = koszul_sign_of_permutation([sigma[i] - 1 for i in range(n)], dims)
                    assert lhs == kappa * rhs, (list(u.support()), sigma, labels)

    def test_partial_composition_axiom(self):
        """μ_{p∘_i q} = (-1)^{|q|·Σ_{l<i}|z_l|} μ_p(…, μ_q(…), …)."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        mod = alg.module
        gens = {name: mod.generator(name) for name in GENERATORS}
        rng = Random(20260615)
        for _ in range(120):
            k1, k2 = rng.choice([(2, 2), (2, 3), (3, 2)])
            d1, d2 = rng.choice([0, 1, 2]), rng.choice([0, 1, 2])
            p = rng.choice(list(Surjection(k1, QQ).graded_basis(d1)))
            q = rng.choice(list(Surjection(k2, QQ).graded_basis(d2)))
            i = rng.randint(1, k1)
            labels = [rng.choice(GENERATORS) for _ in range(k1 + k2 - 1)]
            zs = [gens[lab] for lab in labels]
            lhs = alg.act(Surjection.compose(p, i, q), zs)
            inner = alg.act(q, zs[i - 1 : i - 1 + k2])
            rhs = alg.act(p, zs[: i - 1] + [inner] + zs[i - 1 + k2 :])
            exponent = d2 * sum(-mod.degree_on_basis(lab) for lab in labels[: i - 1])
            assert lhs == (-1) ** (exponent % 2) * rhs

    def test_unit_cochain_insertion_rule(self):
        """μ_u(…, [v] at slot i, …) is zero unless i occurs once in u, and
        then equals the action of u with the value i deleted, no sign."""
        alg = SurjectionTorusSimplicialCochainAlgebra(QQ)
        for n in [2, 3]:
            for d in range(3):
                for u in _basis_keys(n, d):
                    for labels in itertools.product(GENERATORS, repeat=n):
                        if "v" not in labels:
                            continue
                        i = labels.index("v") + 1
                        result = dict(alg._single_action(u, labels))
                        if u.count(i) > 1:
                            assert result == {}, (u, labels)
                            continue
                        reduced = tuple(w - 1 if w > i else w for w in u if w != i)
                        degenerate = not reduced or any(
                            x == y for x, y in zip(reduced, reduced[1:])
                        )
                        reduced_labels = labels[: i - 1] + labels[i:]
                        expected = (
                            {} if degenerate else dict(alg._single_action(reduced, reduced_labels))
                        )
                        assert result == expected, (u, labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
