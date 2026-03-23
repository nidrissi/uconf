"""Tests for operadic twisting morphisms and twisted complexes.

Covers:
- :class:`uconf.core.twisting.TwistingMorphism`
- :func:`uconf.morphisms.canonical_twisting.canonical_projection`
- :func:`uconf.morphisms.canonical_twisting.canonical_inclusion`
- :class:`uconf.constructions.twisted_complex.TwistedBarComplex`
- :class:`uconf.constructions.twisted_complex.TwistedCobarComplex`
"""

import itertools

import pytest
from sage.all import QQ
from sage.all import tensor as sage_tensor

from uconf import Associative, CoAssociative, Commutative
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.constructions.twisted_complex import TwistedBarComplex, TwistedCobarComplex
from uconf.core.twisting import TwistingMorphism
from uconf.core.trees import children, decoration, is_leaf, vertex_arity
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

    def test_repr(self):
        """TwistingMorphism has a readable repr."""
        pi = canonical_projection(Associative)
        assert "B(Ass)" in repr(pi)
        assert "Ass" in repr(pi)

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
        bar_elem = comp2.term(((1, 2), 1, 2))
        result = pi(bar_elem)
        expected = Associative(2, QQ)((1, 2))
        assert result == expected

    def test_projects_two_vertex_tree_to_zero(self):
        """π maps a multi-vertex bar tree to zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp3 = bar_ass(3, QQ)
        tree = ((1, 2), ((1, 2), 1, 2), 3)
        bar_elem = comp3.term(tree)
        result = pi(bar_elem)
        assert result == Associative(3, QQ).zero()

    def test_projects_unit_to_zero(self):
        """π maps the single-leaf tree (arity 1) to zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp1 = bar_ass(1, QQ)
        bar_elem = comp1.term(1)
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
        c_elem = comp2.term((1, 2))
        result = iota(c_elem)
        cobar_parent = CobarConstruction(CoAssociative)(2, QQ)
        expected = cobar_parent.term(((1, 2), 1, 2))
        assert result == expected

    def test_arity_1_maps_to_zero(self):
        """ι maps arity-1 elements to zero (coaugmentation coideal)."""
        iota = canonical_inclusion(CoAssociative)
        comp1 = CoAssociative(1, QQ)
        c_elem = comp1.term(1)
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


# ===========================================================================
# TwistedBarComplex tests
# ===========================================================================


class TestTwistedBarComplex:
    """Tests for TwistedBarComplex B_α(A)."""

    def test_construction_with_canonical_projection(self):
        """B_π(A) can be constructed for a trivial Ass-algebra."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)
        assert B is not None

    def test_element_repr_latex_and_svg(self):
        """Twisted-bar elements implement explicit LaTeX/SVG display methods."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)

        x = B((1, ((),)))
        ltx = x._repr_latex_()
        assert ltx

        svg = x._repr_svg_()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)
        elem = B((1, ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = 1."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)
        mu = (1, 2)
        elem = B(((mu, 1, 2), ((), ())))
        assert elem.degree() == 1

    def test_dalpha_matches_dact(self):
        """d_α with π: B(Ass) → Ass produces correct action on corolla."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)
        mu = (1, 2)
        elem = B(((mu, 1, 2), ((), ())))
        dalpha = elem.dalpha()
        # Should produce a single leaf element
        assert dalpha == B((1, ((),)))

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            ((_MU, 1, 2), ((), ())),
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_projection(self, tree, a_tuple):
        """d² = 0 on B_π(A) for π: B(Ass) → Ass."""
        pi = canonical_projection(Associative)
        alg = _trivial_ass_algebra()
        B = TwistedBarComplex(pi, alg)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_replaces_bar_complex_algebra(self):
        """B_π(A) with canonical_projection produces correct d_α on a corolla."""
        alg = _trivial_ass_algebra()
        pi = canonical_projection(Associative)
        B = TwistedBarComplex(pi, alg)

        mu = (1, 2)
        tree = (mu, 1, 2)
        a_tuple = ((), ())
        assert B.degree_on_basis((tree, a_tuple)) == 1
        elem = B((tree, a_tuple))
        # d_alpha should map corolla to single leaf
        assert elem.dalpha() == B((1, ((),)))


class TestTwistedBarComplexIota:
    """Tests for TwistedBarComplex B_ι(A) with ι: B(P) → Ω(B(P))."""

    def test_construction(self):
        """B_ι(A) can be constructed for ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        assert B is not None

    def test_single_leaf_degree(self):
        """Weight-0 element has degree 0."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        elem = B((1, ((),)))
        assert elem.degree() == 0

    def test_dtwist_trivial_algebra_zero(self):
        """d_α = 0 for the trivial ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        mu = (1, 2)
        elem = B(((mu, 1, 2), ((), ())))
        assert elem.dalpha() == B.zero()

    def test_dtwist_pullback_nonempty(self):
        """d_α is non-zero for the pullback ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        mu = (1, 2)
        elem = B(((mu, 1, 2), ((), ())))
        assert elem.dalpha() == B((1, ((),)))

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            ((_MU, 1, 2), ((), ())),
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_trivial(self, tree, a_tuple):
        """d² = 0 on B_ι(A) for the trivial ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _trivial_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            (((1, 2), 1, 2), ((), ())),
            (((1, 2), 1, ((1, 2), 2, 3)), ((), (), ())),
            (((1, 2), ((1, 2), 1, 2), 3), ((), (), ())),
            (((1, 2, 3), 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_pullback(self, tree, a_tuple):
        """d² = 0 on B_ι(A) for the pullback ΩB(Ass)-algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_linear_combination_d_squared_zero(self):
        """d² = 0 on a linear combination for the pullback algebra."""
        bar_ass = BarConstruction(Associative)
        iota = canonical_inclusion(bar_ass)
        alg = _pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(iota, alg)
        mu = (1, 2)
        t1 = B(((mu, 1, (mu, 2, 3)), ((), (), ())))
        t2 = B(((mu, (mu, 1, 2), 3), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert combo.boundary().boundary() == B.zero()


# ===========================================================================
# TwistedCobarComplex tests
# ===========================================================================


class TestTwistedCobarComplex:
    """Tests for TwistedCobarComplex Ω_α(V)."""

    def test_construction_with_canonical_inclusion(self):
        """Ω_ι(V) can be constructed for a trivial CoAss-coalgebra."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        OmC = TwistedCobarComplex(iota, coalg)
        assert OmC is not None

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_V(v)."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        OmC = TwistedCobarComplex(iota, coalg)
        assert OmC.degree_on_basis((1, ((),))) == 0

    def test_dalpha_nonzero_on_leaf(self):
        """d_α on single leaf expands via coaction → non-zero."""
        iota = canonical_inclusion(CoAssociative)
        coalg = _trivial_coass_coalgebra()
        OmC = TwistedCobarComplex(iota, coalg)
        elem = OmC((1, ((),)))
        result = elem.dalpha()
        assert result != OmC.zero()

    def test_replaces_cobar_complex_coalgebra(self):
        """Ω_ι(V) with canonical_inclusion produces correct degree."""
        coalg = _trivial_coass_coalgebra()
        iota = canonical_inclusion(CoAssociative)
        OmC = TwistedCobarComplex(iota, coalg)

        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        v_tuple = ((), ())
        assert OmC.degree_on_basis((tree, v_tuple)) == -1


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
        elem = comp2.term(((1, 2), 1, 2))
        # A corolla has no internal edges → cocomposition gives zero
        result = pi.star(pi, elem)
        assert result == Associative(2, QQ).zero()

    def test_star_on_two_vertex_tree(self):
        """π ⋆ π on a two-vertex bar tree is non-zero."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp3 = bar_ass(3, QQ)
        # Two-vertex tree with contiguous leaves in child
        tree = ((1, 2), ((1, 2), 1, 2), 3)
        elem = comp3.term(tree)
        result = pi.star(pi, elem)
        # Should be non-zero: the cocomposition splits into two corollas
        assert result != Associative(3, QQ).zero()

    def test_partial_alpha_on_corolla(self):
        """∂π on a bar corolla is zero (Ass has zero boundary, P on corolla is zero)."""
        pi = canonical_projection(Associative)
        bar_ass = BarConstruction(Associative)
        comp2 = bar_ass(2, QQ)
        elem = comp2.term(((1, 2), 1, 2))
        result = pi.partial_alpha(elem)
        # Both ∂_Ass and ∂_{B(Ass)} are zero for this element
        assert result == Associative(2, QQ).zero()
