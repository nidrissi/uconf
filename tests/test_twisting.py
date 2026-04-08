"""Tests for operadic twisting morphisms and bar/cobar constructions.

Covers:
- :class:`uconf.core.twisting.TwistingMorphism`
- :func:`uconf.morphisms.canonical_twisting.canonical_projection`
- :func:`uconf.morphisms.canonical_twisting.canonical_inclusion`
- :class:`uconf.constructions.bar_algebra.BarAlgebra`
- :class:`uconf.constructions.cobar_coalgebra.CobarCoalgebra`
"""

import itertools

import pytest
from sage.all import QQ
from sage.all import tensor as sage_tensor

from uconf import Associative, CoAssociative, Commutative
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_coalgebra import CobarCoalgebra
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.trees import RootedTree, children, decoration, is_leaf, vertex_arity
from uconf.core.twisting import TwistingMorphism
from uconf.morphisms.canonical_twisting import canonical_inclusion, canonical_projection


# ===========================================================================
# Helpers
# ===========================================================================


def _trivial_ass_algebra(base_ring=QQ):
    """Trivial Ass-algebra on 1-dim module k.  γ(p; a_1,...,a_n) = product."""
    module = Commutative(1, base_ring=base_ring)

    def _map(p_elem, a_list):
        result = module.zero()
        for _pk, pc in p_elem:
            coeff = pc
            for a in a_list:
                for _ak, ac in a:
                    coeff *= ac
            result += coeff * module(())
        return result

    return OperadAlgebra(module, Associative, _map)


def _trivial_coass_coalgebra(base_ring=QQ):
    """Trivial CoAss-coalgebra on 1-dim module k."""
    module = Commutative(1, base_ring=base_ring)

    def _comap(v_elem, n):
        left = CoAssociative(n, base_ring=base_ring)
        right = Commutative(1, base_ring=base_ring)
        if n == 1:
            target = sage_tensor([left, right])
        else:
            target = sage_tensor([left] + [right] * n)
        result = target.zero()
        for _vk, vc in v_elem:
            for sigma in itertools.permutations(range(1, n + 1)):
                l_elem = left(sigma)
                if n == 1:
                    r_elem = right(())
                    result += vc * l_elem.tensor(r_elem)
                else:
                    r_elem = right(())
                    for _ in range(n - 1):
                        r_elem = r_elem.tensor(right(()))
                    result += vc * l_elem.tensor(r_elem)
        return result

    return CooperadCoalgebra(module, CoAssociative, _comap)


def _trivial_omega_bar_ass_algebra(base_ring=QQ):
    """Trivial ΩB(Ass)-algebra: γ = 0 for all n ≥ 2."""
    bar_ass = BarConstruction(Associative)
    cobar_bar_ass = CobarConstruction(bar_ass)
    module = Commutative(1, base_ring=base_ring)

    def _map(p_elem, a_list):
        return module.zero()

    return OperadAlgebra(module, cobar_bar_ass, _map)


def _pullback_omega_bar_ass_algebra(base_ring=QQ):
    """ΩB(Ass)-algebra via ε-pullback: non-trivial action on single-vertex cobar trees."""
    bar_ass = BarConstruction(Associative)
    cobar_bar_ass = CobarConstruction(bar_ass)
    module = Commutative(1, base_ring=base_ring)

    def _map(p_elem, a_list):
        n = p_elem.arity()
        result = module.zero()
        for key, coeff in p_elem:
            if is_leaf(key):
                if n == 1:
                    a_coeff = coeff
                    for a in a_list:
                        for _ak, ac in a:
                            a_coeff *= ac
                    result += a_coeff * module(())
            else:
                va = vertex_arity(key)
                if va != n:
                    continue
                chs = children(key)
                if not all(is_leaf(c) for c in chs):
                    continue
                bar_dec = decoration(key)
                if is_leaf(bar_dec):
                    continue
                bva = vertex_arity(bar_dec)
                if bva != n:
                    continue
                bchs = children(bar_dec)
                if not all(is_leaf(c) for c in bchs):
                    continue
                a_coeff = coeff
                for a in a_list:
                    for _ak, ac in a:
                        a_coeff *= ac
                result += a_coeff * module(())
        return result

    return OperadAlgebra(module, cobar_bar_ass, _map)


# ===========================================================================
# TwistingMorphism tests
# ===========================================================================


class TestTwistingMorphism:
    """Tests for the TwistingMorphism class."""

    def test_construction(self):
        """TwistingMorphism can be constructed."""
        pi = canonical_projection(Associative)
        assert isinstance(pi, TwistingMorphism)

    def test_cooperad_operad_attributes(self):
        """TwistingMorphism stores cooperad and operad."""
        pi = canonical_projection(Associative)
        assert isinstance(pi.cooperad, BarConstruction)
        assert pi.operad is Associative


# ===========================================================================
# canonical_projection tests
# ===========================================================================


class TestCanonicalProjection:
    """Tests for the canonical projection π: B(P) → P."""

    def test_projects_corolla_to_element(self):
        """π maps a bar corolla to the P-element."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp2 = bar_ass(2, QQ)
        # Bar corolla: ((1,2), 1, 2) in B(Ass)(2)
        bar_elem = comp2(RootedTree((1, 2), 1, 2))
        result = pi(bar_elem)
        expected = Associative(2, QQ)((1, 2))
        assert result == expected

    def test_projects_two_vertex_tree_to_zero(self):
        """π maps a multi-vertex bar tree to zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp3 = bar_ass(3, QQ)
        tree = RootedTree((1, 2), RootedTree((1, 2), 1, 2), 3)
        bar_elem = comp3(tree)
        result = pi(bar_elem)
        assert result == Associative(3, QQ).zero()

    def test_projects_unit_to_zero(self):
        """π maps the single-leaf tree (arity 1) to zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp1 = bar_ass(1, QQ)
        bar_elem = comp1(1)
        result = pi(bar_elem)
        assert result == Associative(1, QQ).zero()

    def test_mc_holds_ass(self):
        """MC equation holds for π: B(Ass) → Ass."""
        pi = canonical_projection(Associative)
        assert pi.check_maurer_cartan(3, QQ)


# ===========================================================================
# canonical_inclusion tests
# ===========================================================================


class TestCanonicalInclusion:
    """Tests for the canonical inclusion ι: C → Ω(C)."""

    def test_includes_element_as_cobar_tree(self):
        """ι maps c ∈ C(n) to the single-vertex cobar tree (c, 1,...,n)."""
        iota = canonical_inclusion(CoAssociative)
        comp2 = CoAssociative(2, QQ)
        c_elem = comp2((1, 2))
        result = iota(c_elem)
        cobar_parent = CobarConstruction(CoAssociative)(2, QQ)
        expected = cobar_parent(RootedTree((1, 2), 1, 2))
        assert result == expected

    def test_arity_1_maps_to_zero(self):
        """ι maps arity-1 elements to zero (coaugmentation coideal)."""
        iota = canonical_inclusion(CoAssociative)
        comp1 = CoAssociative(1, QQ)
        c_elem = comp1((1,))
        result = iota(c_elem)
        cobar_parent = CobarConstruction(CoAssociative)(1, QQ)
        assert result == cobar_parent.zero()

    def test_mc_holds_coass(self):
        """MC equation holds for ι: CoAss → Ω(CoAss)."""
        iota = canonical_inclusion(CoAssociative)
        assert iota.check_maurer_cartan(3, QQ)

    def test_inclusion_of_bar(self):
        """ι: B(Ass) → Ω(B(Ass)) can be constructed and satisfies MC."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        assert iota.cooperad is bar_ass
        assert isinstance(iota.operad, CobarConstruction)
        assert iota.check_maurer_cartan(3, QQ)

    def test_includes_arity3_element(self):
        """ι maps c ∈ C(3) to a single-vertex cobar tree."""
        iota = canonical_inclusion(CoAssociative)
        comp3 = CoAssociative(3, QQ)
        c_elem = comp3((1, 3, 2))
        result = iota(c_elem)
        cobar_parent = CobarConstruction(CoAssociative)(3, QQ)
        expected = cobar_parent(RootedTree((1, 3, 2), 1, 2, 3))
        assert result == expected

    def test_inclusion_degree(self):
        """ι(c) has degree deg_C(c) - 1 in Ω(C)."""
        iota = canonical_inclusion(CoAssociative)
        comp2 = CoAssociative(2, QQ)
        c_elem = comp2((1, 2))
        result = iota(c_elem)
        cobar_parent = CobarConstruction(CoAssociative)(2, QQ)
        # CoAss elements have degree 0, so ι(c) has degree -1
        for key, _coeff in result:
            assert cobar_parent.degree_on_basis(key) == -1

    def test_inclusion_linear_combination(self):
        """ι is linear: ι(2c₁ + 3c₂) = 2·ι(c₁) + 3·ι(c₂)."""
        iota = canonical_inclusion(CoAssociative)
        comp2 = CoAssociative(2, QQ)
        c1 = comp2((1, 2))
        c2 = comp2((2, 1))
        result = iota(2 * c1 + 3 * c2)
        expected = 2 * iota(c1) + 3 * iota(c2)
        assert result == expected

    def test_inclusion_bar_arity3(self):
        """ι maps B(Ass)(3) corolla to single-vertex cobar tree in ΩB(Ass)(3)."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        comp3 = bar_ass(3, QQ)
        # B(Ass)(3) corolla: ((1,2,3), 1, 2, 3)
        corolla_key = RootedTree((1, 2, 3), 1, 2, 3)
        c_elem = comp3(corolla_key)
        result = iota(c_elem)
        # Result should be a single-vertex tree in ΩB(Ass)(3)
        cobar_parent = CobarConstruction(bar_ass)(3, QQ)
        expected = cobar_parent(RootedTree(corolla_key, 1, 2, 3))
        assert result == expected


# ===========================================================================
# BarAlgebra tests
# ===========================================================================


class TestBarAlgebra:
    """Tests for BarAlgebra B_α(A)."""

    def test_construction_with_canonical_projection(self):
        """B_π(A) can be constructed for a trivial Ass-algebra."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        bar = BarAlgebra(pi, alg)
        assert bar is not None

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        bar = BarAlgebra(pi, alg)
        B = bar.module
        C = bar.cooperad_cls
        elem = B((C.unit_key(), ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = 1."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        bar = BarAlgebra(pi, alg)
        B = bar.module
        C = bar.cooperad_cls
        R = QQ
        # Get a degree-1 c_key from C(2)
        c2 = C(2, R)
        c_keys = [k for e in c2.planar_basis_iter(1) for k in e.support()]
        assert len(c_keys) > 0
        elem = B((c_keys[0], ((), ())))
        assert elem.degree() == 1

    @pytest.mark.parametrize("w", [1, 2, 3])
    def test_d_squared_zero_projection(self, w):
        """d² = 0 on B_π(A) for π: B(Ass) → Ass at various weights."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        bar = BarAlgebra(pi, alg)
        B = bar.module
        for d in range(-1, 4):
            for elem in B.basis_weight_iter(d, w):
                assert B.boundary(B.boundary(elem)) == B.zero()


class TestBarAlgebraIota:
    """Tests for BarAlgebra B_ι(A) with ι: B(P) → Ω(B(P))."""

    def test_construction(self):
        """B_ι(A) can be constructed for ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        bar = BarAlgebra(iota, alg)
        assert bar is not None

    def test_single_leaf_degree(self):
        """Weight-0 element has degree 0."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        bar = BarAlgebra(iota, alg)
        B = bar.module
        C = bar.cooperad_cls
        elem = B((C.unit_key(), ((),)))
        assert elem.degree() == 0

    @pytest.mark.parametrize("w", [1, 2, 3])
    def test_d_squared_zero_trivial(self, w):
        """d² = 0 on B_ι(A) for the trivial ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        bar = BarAlgebra(iota, alg)
        B = bar.module
        for d in range(-1, 4):
            for elem in B.basis_weight_iter(d, w):
                assert B.boundary(B.boundary(elem)) == B.zero()

    @pytest.mark.parametrize("w", [1, 2, 3])
    def test_d_squared_zero_pullback(self, w):
        """d² = 0 on B_ι(A) for the pullback ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _pullback_omega_bar_ass_algebra()
        bar = BarAlgebra(iota, alg)
        B = bar.module
        for d in range(-1, 4):
            for elem in B.basis_weight_iter(d, w):
                assert B.boundary(B.boundary(elem)) == B.zero()


# ===========================================================================
# CobarCoalgebra tests
# ===========================================================================


class TestCobarCoalgebra:
    """Tests for CobarCoalgebra Ω_α(V)."""

    def test_construction_with_canonical_inclusion(self):
        """Ω_ι(V) can be constructed for a trivial CoAss-coalgebra."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        cobar = CobarCoalgebra(iota, coalg)
        assert cobar is not None

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_V(v)."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        cobar = CobarCoalgebra(iota, coalg)
        O = cobar.module
        P = cobar.operad_cls
        assert O.degree_on_basis((P.unit_key(), ((),))) == 0

    def test_dalpha_nonzero_on_leaf(self):
        """d_α on single leaf expands via coaction → non-zero."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        cobar = CobarCoalgebra(iota, coalg)
        O = cobar.module
        P = cobar.operad_cls
        elem = O((P.unit_key(), ((),)))
        result = O.d_alpha(elem)
        assert result != O.zero()


# ===========================================================================
# Pre-Lie product tests
# ===========================================================================


class TestPreLieProduct:
    """Tests for the pre-Lie convolution product α ⋆ β."""

    def test_star_on_corolla_bar_projection(self):
        """π ⋆ π on a bar corolla is zero (cocomposition of a corolla is zero)."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp2 = bar_ass(2, QQ)
        elem = comp2(RootedTree((1, 2), 1, 2))
        # A corolla has no internal edges → cocomposition gives zero
        result = pi.star(pi, elem)
        assert result == Associative(2, QQ).zero()

    def test_star_on_two_vertex_tree(self):
        """π ⋆ π on a two-vertex bar tree is non-zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp3 = bar_ass(3, QQ)
        # Two-vertex tree with contiguous leaves in child
        tree = RootedTree((1, 2), RootedTree((1, 2), 1, 2), 3)
        elem = comp3(tree)
        result = pi.star(pi, elem)
        # Should be non-zero: the cocomposition splits into two corollas
        assert result != Associative(3, QQ).zero()

    def test_partial_alpha_on_corolla(self):
        """∂π on a bar corolla is zero (Ass has zero boundary, P on corolla is zero)."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp2 = bar_ass(2, QQ)
        elem = comp2(RootedTree((1, 2), 1, 2))
        result = pi.partial_alpha(elem)
        # Both ∂_Ass and ∂_{B(Ass)} are zero for this element
        assert result == Associative(2, QQ).zero()
