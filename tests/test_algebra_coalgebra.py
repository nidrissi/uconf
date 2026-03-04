"""Tests for algebras over operads and coalgebras over cooperads.

Covers:
- :class:`uconf.algebra.OperadAlgebra`
- :class:`uconf.coalgebra.CooperadCoalgebra`
- :class:`uconf.algebra_bar.BarComplexAlgebra` (bar complex B_P(A))
- :class:`uconf.coalgebra_cobar.CobarComplexCoalgebra` (cobar complex Ω_C(V))
"""

import pytest
from sage.all import QQ

from uconf import (
    Associative,
    CoAssociative,
    Commutative,
    Lie,
)
from uconf.algebraic.algebra import OperadAlgebra
from uconf.constructions.algebra_bar import BarComplexAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.constructions.coalgebra_cobar import CobarComplexCoalgebra


# ===========================================================================
# Helpers: build simple algebra modules
# ===========================================================================


def _make_trivial_ass_algebra(base_ring=QQ):
    """Return an Ass-algebra structure on the 1-dimensional module k.

    The module k has a single basis element ``()`` in degree 0, product
    γ_n(σ; (),...,()) = ().
    """
    module = Commutative(1, base_ring=base_ring)

    def structure_map(p_elem, a_list):
        # p_elem ∈ Ass(n), a_list = [module_elem, ...]
        # All elements are scalar multiples of ()
        result = module.zero()
        for p_key, p_coeff in p_elem:
            # Coefficient from each a_i
            coeff = p_coeff
            for a_elem in a_list:
                for _a_key, a_coeff in a_elem:
                    coeff = coeff * a_coeff
            result += coeff * module.term(())
        return result

    return OperadAlgebra(module, Associative, structure_map)


def _make_trivial_coass_coalgebra(base_ring=QQ):
    """Return a CoAss-coalgebra structure on the 1-dimensional module k.

    The coaction δ_n: k → CoAss(n) ⊗ k^⊗n sends () to Σ_σ (σ ⊗ ()⊗...⊗()).
    Since CoAss(n) has a full basis {σ : σ ∈ S_n}, the coaction sums over all.
    """
    module = Commutative(1, base_ring=base_ring)
    coop = CoAssociative

    def costructure_map(v_elem, n):
        left_parent = coop(n, base_ring=base_ring)
        right_parent = Commutative(1, base_ring=base_ring)
        # Build V^⊗n as n-fold tensor
        import itertools

        right_factors = [right_parent] * n
        from sage.all import tensor as sage_tensor

        if n == 1:
            target = sage_tensor([left_parent, right_parent])
        else:
            right_tensor = sage_tensor(right_factors)
            target = sage_tensor([left_parent, right_tensor])

        result = target.zero()
        for _v_key, v_coeff in v_elem:
            for sigma in itertools.permutations(range(1, n + 1)):
                left_elem = left_parent.term(sigma)
                if n == 1:
                    right_elem = right_parent.term(())
                    result += v_coeff * left_elem.tensor(right_elem)
                else:
                    right_elem = right_parent.term(())
                    right_full = right_elem
                    for _ in range(n - 1):
                        right_full = right_full.tensor(right_parent.term(()))
                    result += v_coeff * left_elem.tensor(right_full)
        return result

    return CooperadCoalgebra(module, coop, costructure_map)


# ===========================================================================
# OperadAlgebra tests
# ===========================================================================


class TestOperadAlgebra:
    """Tests for OperadAlgebra wrapper."""

    def test_construction(self):
        alg = _make_trivial_ass_algebra()
        assert alg.operad_cls is Associative
        assert alg.module is not None

    def test_act_unary(self):
        """Unit axiom: γ_1(id; a) = a."""
        alg = _make_trivial_ass_algebra()
        module = alg.module
        a = module.term(())
        unit = Associative.unit()
        result = alg.act(unit, [a])
        assert result == a

    def test_act_binary(self):
        """Binary product γ_2(σ; a, a) = a (trivial algebra)."""
        alg = _make_trivial_ass_algebra()
        module = alg.module
        a = module.term(())
        p = Associative(2).term((1, 2))
        result = alg.act(p, [a, a])
        assert result == a

    def test_act_arity_mismatch(self):
        """act() raises ValueError when len(algebra_elements) != arity."""
        alg = _make_trivial_ass_algebra()
        module = alg.module
        a = module.term(())
        p = Associative(2).term((1, 2))
        with pytest.raises(ValueError, match="Expected 2"):
            alg.act(p, [a])

    def test_boundary_zero(self):
        """Boundary of algebra element is 0 (Commutative module has zero differential)."""
        alg = _make_trivial_ass_algebra()
        module = alg.module
        a = module.term(())
        assert alg.boundary(a) == module.zero()


# ===========================================================================
# CooperadCoalgebra tests
# ===========================================================================


class TestCooperadCoalgebra:
    """Tests for CooperadCoalgebra wrapper."""

    def test_construction(self):
        coalg = _make_trivial_coass_coalgebra()
        assert coalg.cooperad_cls is CoAssociative
        assert coalg.module is not None

    def test_boundary_zero(self):
        """Boundary of coalgebra element is 0."""
        coalg = _make_trivial_coass_coalgebra()
        module = coalg.module
        v = module.term(())
        assert coalg.boundary(v) == module.zero()


# ===========================================================================
# BarComplexAlgebra tests
# ===========================================================================


def _make_bar_complex(base_ring=QQ):
    """Build B_Ass(k) -- bar complex of the trivial 1-dim Ass-algebra."""
    alg = _make_trivial_ass_algebra(base_ring=base_ring)
    return BarComplexAlgebra(alg, base_ring=base_ring)


class TestBarComplexAlgebra:
    """Tests for BarComplexAlgebra."""

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        B = _make_bar_complex()
        # Basis key: (1, ((),))  -- single leaf with A-decoration ()
        elem = B((1, ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = 1 (deg_P(μ)+1 = 0+1 = 1)."""
        B = _make_bar_complex()
        # Tree: ((1,2); leaf1, leaf2) in Ass(2)
        mu = (1, 2)
        tree = (mu, 1, 2)
        a_tuple = ((), ())
        elem = B.term((tree, a_tuple))
        assert elem.degree() == 1

    def test_weight1_ternary_degree(self):
        """Weight-1 ternary tree has degree = 1."""
        B = _make_bar_complex()
        mu3 = (1, 2, 3)
        tree = (mu3, 1, 2, 3)
        a_tuple = ((), (), ())
        deg = B.degree_on_basis((tree, a_tuple))
        assert deg == 1

    def test_dact_weight1_binary_gives_weight0(self):
        """d_act on weight-1 binary tree produces weight-0 element."""
        B = _make_bar_complex()
        mu = (1, 2)
        tree = (mu, 1, 2)
        a_tuple = ((), ())
        elem = B.term((tree, a_tuple))
        result = elem.dact()
        # Should give +(1, ((),)) -- single leaf with the product value
        assert result == B.term((1, ((),)))

    def test_d_squared_zero_weight1_binary(self):
        """d² = 0 on a weight-1 binary tree element."""
        B = _make_bar_complex()
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B.term((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_d_squared_zero_weight2_tree1(self):
        """d² = 0 on weight-2 tree with right-nested structure."""
        B = _make_bar_complex()
        mu = (1, 2)
        # Tree: (μ; 1, (μ; 2, 3))
        tree = (mu, 1, (mu, 2, 3))
        elem = B.term((tree, ((), (), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_d_squared_zero_weight2_tree2(self):
        """d² = 0 on weight-2 tree with left-nested structure."""
        B = _make_bar_complex()
        mu = (1, 2)
        # Tree: (μ; (μ; 1, 2), 3)
        tree = (mu, (mu, 1, 2), 3)
        elem = B.term((tree, ((), (), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_d_squared_zero_weight1_ternary(self):
        """d² = 0 on weight-1 ternary tree."""
        B = _make_bar_complex()
        mu3 = (1, 2, 3)
        tree = (mu3, 1, 2, 3)
        elem = B.term((tree, ((), (), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_linear_combination_d_squared_zero(self):
        """d² = 0 on a linear combination of bar elements."""
        B = _make_bar_complex()
        mu = (1, 2)
        t1 = B.term(((mu, 1, (mu, 2, 3)), ((), (), ())))
        t2 = B.term(((mu, (mu, 1, 2), 3), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert combo.boundary().boundary() == B.zero()

    def test_d2_tree_connects_to_ternary(self):
        """d_2 on weight-2 binary tree produces weight-1 ternary term."""
        B = _make_bar_complex()
        mu = (1, 2)
        # Tree: (μ; 1, (μ; 2, 3)) -- d_2 contracts the internal edge
        tree = (mu, 1, (mu, 2, 3))
        elem = B.term((tree, ((), (), ())))
        d2_result = elem.d2()
        # The result should contain a weight-1 ternary tree
        has_ternary = False
        for (t, _a), _coeff in d2_result:
            if hasattr(t, "__len__") and len(t) == 4:  # (dec, leaf1, leaf2, leaf3)
                has_ternary = True
        assert has_ternary

    def test_commutative_algebra_bar(self):
        """Bar complex also works with the Commutative operad."""
        module = Commutative(1)

        def comm_structure_map(p_elem, a_list):
            result = module.zero()
            for _p_key, p_coeff in p_elem:
                coeff = p_coeff
                for a_elem in a_list:
                    for _a_key, a_coeff in a_elem:
                        coeff = coeff * a_coeff
                result += coeff * module.term(())
            return result

        alg = OperadAlgebra(module, Commutative, comm_structure_map)
        B = BarComplexAlgebra(alg)
        # Weight-1 arity-2 tree: Commutative(2) has basis {()}
        tree = ((), 1, 2)
        elem = B.term((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()


# ===========================================================================
# CobarComplexCoalgebra tests
# ===========================================================================


def _make_cobar_complex(base_ring=QQ):
    """Build Ω_CoAss(k) -- cobar complex of the trivial 1-dim CoAss-coalgebra."""
    coalg = _make_trivial_coass_coalgebra(base_ring=base_ring)
    return CobarComplexCoalgebra(coalg, base_ring=base_ring)


class TestCobarComplexCoalgebra:
    """Tests for CobarComplexCoalgebra."""

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_V(v)."""
        C = _make_cobar_complex()
        assert C.degree_on_basis((1, ((),))) == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree in Ω_C(V) has degree = -1 (deg_C(c)-1 = 0-1)."""
        C = _make_cobar_complex()
        # CoAssociative(2) basis key = (1,2) in degree 0
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        v_tuple = ((), ())
        deg = C.degree_on_basis((tree, v_tuple))
        assert deg == -1

    def test_d1_zero_for_trivial(self):
        """d_1 = 0 when the cooperad boundary is zero."""
        C = _make_cobar_complex()
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = C.term((tree, ((), ())))
        assert elem.d1() == C.zero()

    def test_dV_zero_for_trivial(self):
        """d_V = 0 when module boundary is zero."""
        C = _make_cobar_complex()
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = C.term((tree, ((), ())))
        assert elem.dV() == C.zero()


# ===========================================================================
# Lie operad -- bar complex with non-trivial operad
# ===========================================================================


class TestBarComplexLie:
    """Bar complex of a Lie-algebra (Chevalley-Eilenberg complex analog)."""

    def _make_trivial_lie_algebra(self, base_ring=QQ):
        """1-dimensional trivial Lie algebra with zero bracket."""
        module = Commutative(1, base_ring=base_ring)

        def structure_map(p_elem, a_list):
            # For the trivial Lie algebra, all brackets are 0
            return module.zero()

        return OperadAlgebra(module, Lie, structure_map)

    def test_d_squared_zero_weight1(self):
        """d² = 0 on a weight-1 Lie tree for the trivial Lie algebra."""
        alg = self._make_trivial_lie_algebra()
        B = BarComplexAlgebra(alg)
        # Lie(2) has basis key (1,) in degree 0
        lie_dec = (1,)
        tree = (lie_dec, 1, 2)
        elem = B.term((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()
