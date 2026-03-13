"""Tests for FreeOperadAlgebra and CofreeConilpotentCoalgebra.

Covers:
- :class:`uconf.free_algebra.FreeAlgebraModule`
- :class:`uconf.free_algebra.FreeOperadAlgebra`
- :class:`uconf.cofree_coalgebra.CofreeCoalgebraModule`
- :class:`uconf.cofree_coalgebra.CofreeConilpotentCoalgebra`
"""

import pytest
from sage.all import QQ

from uconf import (
    Associative,
    CoAssociative,
    Commutative,
    Lie,
)
from uconf.algebraic.free_algebra import FreeAlgebraModule, FreeOperadAlgebra
from uconf.algebraic.cofree_coalgebra import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


def _zero_diff_module(base_ring=QQ):
    """Return Commutative(1, QQ) as a 1-dim dg-module with zero differential."""
    return Commutative(1, base_ring=base_ring)


# ===========================================================================
# FreeAlgebraModule tests
# ===========================================================================


class TestFreeAlgebraModule:
    """Tests for the underlying dg-module of Free_P(M)."""

    def test_single_leaf_degree(self):
        """A single-leaf element has degree = deg_M(m)."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        key = (1, ((),))
        assert mod.degree_on_basis(key) == 0

    def test_weight1_binary_degree(self):
        """A weight-1 binary tree decorated by P has degree = deg_P(p) + deg_M × 2."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        # Ass(2) has basis key (1,2), degree 0
        key = ((1, 2), 1, 2)  # tree
        full_key = (key, ((), ()))
        assert mod.degree_on_basis(full_key) == 0

    def test_weight1_lie_degree(self):
        """Lie(2, QQ) has basis key (1,) in degree 0."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Lie, M, QQ)
        key = ((1,), 1, 2)  # tree with Lie decoration (1,) in arity 2
        full_key = (key, ((), ()))
        assert mod.degree_on_basis(full_key) == 0

    def test_zero_differential_on_leaf(self):
        """Differential is zero on single-leaf elements when M has zero differential."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        elem = mod((1, ((),)))
        assert elem.boundary() == mod.zero()

    def test_zero_differential_on_weight1(self):
        """d²=0 on weight-1 element (trivially, since d(vertex)=0 and d(M)=0)."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        tree = ((1, 2), 1, 2)
        elem = mod((tree, ((), ())))
        assert elem.boundary() == mod.zero()

    def test_element_constructor_dict(self):
        """Dict construction builds linear combinations."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        tree = ((1, 2), 1, 2)
        full_key = (tree, ((), ()))
        single_leaf = (1, ((),))
        elem = mod({full_key: 3, single_leaf: -1})
        d = {k: c for k, c in elem}
        assert d[full_key] == 3
        assert d[single_leaf] == -1

    def test_validate_rejects_bad_key(self):
        """Invalid keys are silently rejected (mapped to zero)."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        assert mod._validate_basis_key("bad") is None
        assert (
            mod._validate_basis_key((1, ((), ()))) is None
        )  # 2 m-keys for arity-1 leaf


# ===========================================================================
# FreeOperadAlgebra tests
# ===========================================================================


class TestFreeOperadAlgebra:
    """Tests for Free_P(M) as a P-algebra."""

    def test_construction_associative(self):
        """FreeOperadAlgebra stores operad_cls and exposes module."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        assert F.operad_cls is Associative
        assert isinstance(F.module, FreeAlgebraModule)

    def test_include(self):
        """η(m) = module((1, (m,)))."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m_key = ()
        included = F.include(m_key)
        assert _as_dict(included) == {(1, ((),)): 1}

    def test_act_unary_unit(self):
        """γ(id_1; η(m)) = η(m)  (unit axiom)."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m = F.module((1, ((),)))
        unit = Associative.unit()  # id in P(1)
        result = F.act(unit, [m])
        # Should give a weight-1 tree (id, leaf1) → (id_key, 1) with m-tuple ((),)
        assert result != F.module.zero()

    def test_act_binary_grafts_two_leaves(self):
        """γ(μ; η(m1), η(m2)) builds binary tree with two M-leaves."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m1 = F.module((1, ((),)))
        m2 = F.module((1, ((),)))
        mu = Associative(2, QQ)((1, 2))
        result = F.act(mu, [m1, m2])
        # Expected: single term ((1,2), 1, 2) with m-tuple ((), ())
        expected_tree = ((1, 2), 1, 2)
        assert _as_dict(result) == {(expected_tree, ((), ())): 1}

    def test_act_binary_comm_grafts(self):
        """FreeOperadAlgebra also works with Commutative operad."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Commutative, M, QQ)
        m1 = F.module((1, ((),)))
        m2 = F.module((1, ((),)))
        com = Commutative(2, QQ)(())
        result = F.act(com, [m1, m2])
        expected_tree = ((), 1, 2)
        assert _as_dict(result) == {(expected_tree, ((), ())): 1}

    def test_act_arity_mismatch_raises(self):
        """act() raises ValueError when wrong number of inputs supplied."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m = F.module((1, ((),)))
        mu = Associative(2, QQ)((1, 2))
        with pytest.raises(ValueError, match="Expected 2"):
            F.act(mu, [m])

    def test_act_nests_subtrees(self):
        """γ grafts sub-trees, not just leaves: weight-1 ⊗ weight-1 → weight-2."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        mu = Associative(2, QQ)((1, 2))
        # Build a weight-1 element first
        t1 = F.act(mu, [F.module((1, ((),))), F.module((1, ((),)))])
        # Now graft t1 as left child and a leaf as right child
        m3 = F.module((1, ((),)))
        result = F.act(mu, [t1, m3])
        # Should give a weight-2 tree with 3 M-leaves
        for key, coeff in result:
            tree, m_tuple = key
            assert len(m_tuple) == 3
            assert coeff == 1

    def test_boundary_zero_trivial(self):
        """Differential is 0 for trivial Ass operad (deg=0) and trivial M."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        tree = ((1, 2), 1, 2)
        elem = F.module((tree, ((), ())))
        assert F.boundary(elem) == F.module.zero()

    def test_act_lie(self):
        """FreeOperadAlgebra works with the Lie operad."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Lie, M, QQ)
        lie_dec = (1,)  # Lie(2, QQ) basis key
        bracket = Lie(2, QQ)(lie_dec)
        m1 = F.module((1, ((),)))
        m2 = F.module((1, ((),)))
        result = F.act(bracket, [m1, m2])
        expected_tree = (lie_dec, 1, 2)
        assert _as_dict(result) == {(expected_tree, ((), ())): 1}


# ===========================================================================
# CofreeCoalgebraModule tests
# ===========================================================================


class TestCofreeCoalgebraModule:
    """Tests for the underlying dg-module of T^c_C(M)."""

    def test_single_leaf_degree(self):
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        assert mod.degree_on_basis((1, ((),))) == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = deg_C((1,2)) = 0."""
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        tree = ((1, 2), 1, 2)
        assert mod.degree_on_basis((tree, ((), ()))) == 0

    def test_zero_differential_trivial(self):
        """Differential is zero when C and M both have zero differential."""
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        tree = ((1, 2), 1, 2)
        elem = mod((tree, ((), ())))
        assert elem.boundary() == mod.zero()


# ===========================================================================
# CofreeConilpotentCoalgebra tests
# ===========================================================================


class TestCofreeConilpotentCoalgebra:
    """Tests for T^c_C(M) as a C-coalgebra."""

    def test_construction(self):
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        assert T.cooperad_cls is CoAssociative
        assert isinstance(T.module, CofreeCoalgebraModule)

    def test_project_single_leaf(self):
        """π((1, (m,))) = m in M."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module((1, ((),)))
        projected = T.project(elem)
        assert _as_dict(projected) == {(): 1}

    def test_project_kills_weight1_tree(self):
        """π kills elements of weight ≥ 1 (internal vertex present)."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        tree = ((1, 2), 1, 2)
        elem = T.module((tree, ((), ())))
        projected = T.project(elem)
        assert projected == M.zero()

    def test_coact_single_leaf_gives_zero(self):
        """δ_k is zero on single-leaf elements (no root vertex to split)."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module((1, ((),)))
        result = T.coact(elem, 2)
        assert result.is_zero()

    def test_coact_binary_tree_arity2(self):
        """δ_2 on a binary tree (root arity 2) splits at root."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = T.module((tree, ((), ())))
        result = T.coact(elem, 2)
        # Should be non-zero
        assert not result.is_zero()
        # The result should contain terms with the root decoration (1,2)
        # and two single-leaf children
        terms = list(result)
        assert len(terms) >= 1
        # Each term key is (c_key, cofree_key_1, cofree_key_2)
        for (c_key, key1, key2), coeff in terms:
            assert key1 == (1, ((),))  # left child: single leaf with m-key ()
            assert key2 == (1, ((),))  # right child: single leaf with m-key ()

    def test_coact_wrong_arity_gives_zero(self):
        """δ_3 is zero on a binary tree (root arity 2 ≠ 3)."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = T.module((tree, ((), ())))
        result = T.coact(elem, 3)
        assert result.is_zero()

    def test_infinitesimal_cocompose_arity3_to_2x2(self):
        """Δ^{1;2,2} on arity-3 tree gives a tensor of two arity-2 trees."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        # But arity-3 tree has no weight-2 subtree unless it's nested
        # Use a weight-2 tree for a proper split
        c_dec2 = (1, 2)
        # Tree: (c_dec2; 1, (c_dec2; 2, 3)) -- root arity 2 with nested child
        nested_tree = (c_dec2, 1, (c_dec2, 2, 3))
        nested_elem = T.module((nested_tree, ((), (), ())))
        # Δ^{2;2,2}: split at the vertex covering leaves {2,3}
        result = T.infinitesimal_cocompose(nested_elem, 2, 2, 2)
        assert not result.is_zero()

    def test_infinitesimal_cocompose_wrong_arity_raises(self):
        """infinitesimal_cocompose raises ValueError for invalid i, m, n."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module.zero()
        with pytest.raises(ValueError, match="Arities must be positive"):
            T.infinitesimal_cocompose(elem, 1, 0, 2)
        with pytest.raises(ValueError, match="Index i must satisfy"):
            T.infinitesimal_cocompose(elem, 3, 2, 2)

    def test_boundary_zero(self):
        """Differential is zero for trivial CoAss and trivial M."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        tree = ((1, 2), 1, 2)
        elem = T.module((tree, ((), ())))
        assert T.boundary(elem) == T.module.zero()


# ===========================================================================
# Integration: Free algebra → Bar complex
# ===========================================================================


class TestFreeAlgebraBarComplex:
    """Bar complex of the free P-algebra."""

    def test_bar_of_free_lie_d_squared_zero(self):
        """d² = 0 on a weight-1 element of B(Lie; Free_Lie(M))."""
        from uconf.constructions.bar_algebra import BarComplexAlgebra

        M = _zero_diff_module()
        F = FreeOperadAlgebra(Lie, M, QQ)
        B = BarComplexAlgebra(F, QQ)

        lie_dec = (1,)
        # Outer tree: weight-1 Lie binary tree with two single-leaf inner trees
        outer_tree = (lie_dec, 1, 2)
        a_tuple = ((1, ((),)), (1, ((),)))  # two Free_Lie(M) basis keys (single leaves)
        elem = B((outer_tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_bar_of_free_ass_d_squared_zero(self):
        """d² = 0 on a weight-1 element of B(Ass; Free_Ass(M))."""
        from uconf.constructions.bar_algebra import BarComplexAlgebra

        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        B = BarComplexAlgebra(F, QQ)

        mu = (1, 2)
        outer_tree = (mu, 1, 2)
        a_tuple = ((1, ((),)), (1, ((),)))
        elem = B((outer_tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()
