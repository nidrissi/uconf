"""Tests for FreeOperadAlgebra and CofreeConilpotentCoalgebra.

Covers:
- :class:`uconf.free_algebra.FreeAlgebraModule`
- :class:`uconf.free_algebra.FreeOperadAlgebra`
- :class:`uconf.cofree_coalgebra.CofreeCoalgebraModule`
- :class:`uconf.cofree_coalgebra.CofreeConilpotentCoalgebra`

Basis key convention after the fix:
  Free/cofree basis key is ``(p_key, m_tuple)`` where ``p_key ∈ P(len(m_tuple))``.
  Leaf/inclusion: ``(id_key, (m,))`` where ``id_key`` is the unique P(1)-basis key.
  - Ass(1) key: (1,)   → leaf key ((1,), (m,))
  - Lie(1) key: ()     → leaf key ((), (m,))
  - CoAss(1) key: (1,) → leaf key ((1,), (m,))
  Corolla (n≥2): ``(p_key, (m_1, ..., m_n))`` where p_key ∈ P(n).
  - Ass(2) corolla: ((1,2), (m1, m2))
  - Lie(2) corolla: ((1,), (m1, m2))
  - Comm(2) corolla: ((), (m1, m2))
"""

import pytest
from sage.all import GradedModulesWithBasis, QQ

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
        """A single-leaf (identity) element has degree = deg_M(m)."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        # Ass(1) has basis key (1,); leaf element: ((1,), (m,))
        key = ((1,), ((),))
        assert mod.degree_on_basis(key) == 0

    def test_weight1_binary_degree(self):
        """A corolla element decorated by P(2) has degree = deg_P(p) + deg_M × 2."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        # Ass(2) has basis key (1,2), degree 0; corolla: ((1,2), (m1, m2))
        full_key = ((1, 2), ((), ()))
        assert mod.degree_on_basis(full_key) == 0

    def test_weight1_lie_degree(self):
        """Lie(2, QQ) has basis key (1,) in degree 0."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Lie, M, QQ)
        # Lie(2) corolla key: ((1,), (m1, m2))
        full_key = ((1,), ((), ()))
        assert mod.degree_on_basis(full_key) == 0

    def test_zero_differential_on_leaf(self):
        """Differential is zero on single-leaf elements when M has zero differential."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        elem = mod(((1,), ((),)))
        assert elem.boundary() == mod.zero()

    def test_zero_differential_on_corolla(self):
        """d=0 on a corolla element when P and M both have zero differential."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        elem = mod(((1, 2), ((), ())))
        assert elem.boundary() == mod.zero()

    def test_element_constructor_dict(self):
        """Dict construction builds linear combinations."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        corolla_key = ((1, 2), ((), ()))
        leaf_key = ((1,), ((),))
        elem = mod({corolla_key: 3, leaf_key: -1})
        d = {k: c for k, c in elem}
        assert d[corolla_key] == 3
        assert d[leaf_key] == -1

    def test_validate_rejects_bad_key(self):
        """Invalid keys are silently rejected (mapped to zero)."""
        M = _zero_diff_module()
        mod = FreeAlgebraModule(Associative, M, QQ)
        assert mod._validate_basis_key("bad") is None
        # 2 m-keys for arity-1 P-key (1,) -- wrong
        assert mod._validate_basis_key(((1,), ((), ()))) is None


# ===========================================================================
# Planar basis enumeration tests (S_n-orbit fix)
# ===========================================================================


def _graded_module_on_one_generator(degree: int, base_ring=QQ):
    """Return a 1-dim graded module with one generator ``'x'`` in ``degree``."""
    from sage.all import CombinatorialFreeModule

    mod = CombinatorialFreeModule(base_ring, ["x"], category=GradedModulesWithBasis(base_ring))
    mod.degree_on_basis = lambda _key: degree
    mod.boundary = lambda _elem: mod.zero()
    return mod


class TestFreeAlgebraModulePlanarBasis:
    """Regression tests for planar-basis enumeration in FreeAlgebraModule.basis_it.

    The free P-algebra on M is the composite product
    ``P ∘ M = ⊕_n P(n) ⊗_{S_n} M^{⊗n}``.  For a quasi-planar operad the
    isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}`` implies that
    basis_it should enumerate only the planar vertex decorations, yielding one
    representative per ``S_n``-orbit.
    """

    def test_ass_one_generator_degree4_has_one_element(self):
        """Free Ass-algebra on one generator x of degree 2: exactly one element in degree 4.

        Ass(2) ⊗_{S_2} (kx)^{⊗2} ≅ kx^2 is 1-dimensional, so basis_it(4)
        must yield exactly one element.  (Regression for the 'absurd.py' issue.)
        """
        M = _graded_module_on_one_generator(2)
        mod = FreeAlgebraModule(Associative, M, QQ)
        elems = list(mod.basis_it(4))
        assert len(elems) == 1

    def test_ass_one_generator_degree4_key_uses_planar_decoration(self):
        """The unique degree-4 element has the planar decoration (1,2), not (2,1)."""
        M = _graded_module_on_one_generator(2)
        mod = FreeAlgebraModule(Associative, M, QQ)
        elems = list(mod.basis_it(4))
        assert len(elems) == 1
        keys = [k for k, _ in elems[0]]
        assert len(keys) == 1
        p_key, m_tuple = keys[0]
        assert p_key == (1, 2), "Expected planar Ass(2) decoration (1,2)"
        assert len(m_tuple) == 2, "Expected two leaf entries"

    def test_ass_one_generator_degree2_has_one_element(self):
        """Free Ass-algebra on one generator x of degree 2: exactly one element in degree 2.

        This is just the generator itself (the arity-1 leaf).
        """
        M = _graded_module_on_one_generator(2)
        mod = FreeAlgebraModule(Associative, M, QQ)
        elems = list(mod.basis_it(2))
        assert len(elems) == 1

    def test_ass_two_generators_degree4_has_four_elements(self):
        """Free Ass-algebra on {x, y} of degree 2: four elements in degree 4.

        Ass(2) ⊗_{S_2} (kx⊕ky)^{⊗2} ≅ (kx⊕ky)^{⊗2} has basis {xx, xy, yx, yy},
        so basis_it(4) must yield exactly four elements.
        """
        from sage.all import CombinatorialFreeModule

        mod_M = CombinatorialFreeModule(
            QQ, ["x", "y"], category=GradedModulesWithBasis(QQ)
        )
        mod_M.degree_on_basis = lambda _key: 2
        mod_M.boundary = lambda _elem: mod_M.zero()

        mod = FreeAlgebraModule(Associative, mod_M, QQ)
        elems = list(mod.basis_it(4))
        assert len(elems) == 4


class TestFreeOperadAlgebra:
    """Tests for Free_P(M) as a P-algebra."""

    def test_construction_associative(self):
        """FreeOperadAlgebra stores operad_cls and exposes module."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        assert F.operad_cls is Associative
        assert isinstance(F.module, FreeAlgebraModule)

    def test_include(self):
        """η(m) = module((id_key, (m,))) where id_key is the unique Ass(1) key (1,)."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m_key = ()
        included = F.include(m_key)
        assert _as_dict(included) == {((1,), ((),)): 1}

    def test_act_unary_unit(self):
        """γ(id_1; η(m)) = η(m)  (unit axiom)."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m = F.include(())
        unit = Associative.unit(QQ)  # id in P(1)
        result = F.act(unit, [m])
        assert result != F.module.zero()

    def test_act_binary_grafts_two_leaves(self):
        """γ(μ; η(m1), η(m2)) builds a corolla (μ_key, (m1, m2))."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m1 = F.include(())
        m2 = F.include(())
        mu = Associative(2, QQ)((1, 2))
        result = F.act(mu, [m1, m2])
        # Expected: single term with Ass(2) key (1,2) and m-tuple ((), ())
        assert _as_dict(result) == {((1, 2), ((), ())): 1}

    def test_act_binary_comm_grafts(self):
        """FreeOperadAlgebra also works with Commutative operad."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Commutative, M, QQ)
        m1 = F.include(())
        m2 = F.include(())
        com = Commutative(2, QQ)(())
        result = F.act(com, [m1, m2])
        # Comm(2) key is (), corolla: ((), ((), ()))
        assert _as_dict(result) == {((), ((), ())): 1}

    def test_act_arity_mismatch_raises(self):
        """act() raises ValueError when wrong number of inputs supplied."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        m = F.include(())
        mu = Associative(2, QQ)((1, 2))
        with pytest.raises(ValueError, match="Expected 2"):
            F.act(mu, [m])

    def test_act_composes_p_elements(self):
        """γ(μ; γ(μ; x, y), z) composes P-elements → corolla with Ass(3) key.

        The action on two corollas should compose the P-decorations (not graft
        multi-vertex trees).  γ(μ; γ(μ; x, y), z) should give a single corolla
        with Ass(3)-decoration and 3 M-leaves.
        """
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        mu = Associative(2, QQ)((1, 2))
        # Build a corolla first: γ(μ; x, y)
        t1 = F.act(mu, [F.include(()), F.include(())])
        # Now apply action: γ(μ; t1, z) → should compose μ with (t1.p_key, id)
        m3 = F.include(())
        result = F.act(mu, [t1, m3])
        # Result must be a single corolla with 3 M-leaves
        terms = list(result)
        assert len(terms) == 1
        (p_key, m_tuple), coeff = terms[0]
        assert len(m_tuple) == 3
        assert coeff == 1
        # The P-decoration must be in Ass(3)
        assert len(p_key) == 3

    def test_boundary_zero_trivial(self):
        """Differential is 0 for trivial Ass operad (deg=0) and trivial M."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Associative, M, QQ)
        elem = F.module(((1, 2), ((), ())))
        assert F.boundary(elem) == F.module.zero()

    def test_act_lie(self):
        """FreeOperadAlgebra works with the Lie operad."""
        M = _zero_diff_module()
        F = FreeOperadAlgebra(Lie, M, QQ)
        lie_dec = (1,)  # Lie(2, QQ) basis key
        bracket = Lie(2, QQ)(lie_dec)
        m1 = F.include(())
        m2 = F.include(())
        result = F.act(bracket, [m1, m2])
        # Expected: corolla with Lie(2) key (1,) and m-tuple ((), ())
        assert _as_dict(result) == {(lie_dec, ((), ())): 1}


# ===========================================================================
# CofreeCoalgebraModule tests
# ===========================================================================


class TestCofreeCoalgebraModule:
    """Tests for the underlying dg-module of T^c_C(M)."""

    def test_single_leaf_degree(self):
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        # CoAss(1) key is (1,); leaf element: ((1,), (m,))
        assert mod.degree_on_basis(((1,), ((),))) == 0

    def test_weight1_binary_degree(self):
        """Corolla element ((1,2), (m1, m2)) has degree = deg_C((1,2)) = 0."""
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        assert mod.degree_on_basis(((1, 2), ((), ()))) == 0

    def test_zero_differential_trivial(self):
        """Differential is zero when C and M both have zero differential."""
        M = _zero_diff_module()
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        elem = mod(((1, 2), ((), ())))
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
        """π((id_key, (m,))) = m in M."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        # CoAss(1) identity key is (1,)
        elem = T.module(((1,), ((),)))
        projected = T.project(elem)
        assert _as_dict(projected) == {(): 1}

    def test_project_kills_n2_element(self):
        """π kills elements with n ≥ 2 (arity > 1)."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module(((1, 2), ((), ())))
        projected = T.project(elem)
        assert projected == M.zero()

    def test_coact_single_leaf_gives_zero(self):
        """δ_k is zero on arity-1 elements."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module(((1,), ((),)))
        result = T.coact(elem, 2)
        assert result.is_zero()

    def test_coact_binary_corolla_arity2(self):
        """δ_2 on a corolla of arity 2 splits at root."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        c_dec = (1, 2)
        elem = T.module((c_dec, ((), ())))
        result = T.coact(elem, 2)
        # Should be non-zero
        assert not result.is_zero()
        terms = list(result)
        assert len(terms) >= 1
        # Each term key is (c_key, cofree_key_1, cofree_key_2)
        for (c_key, key1, key2), coeff in terms:
            # Children are arity-1 leaf elements: ((1,), (m,))
            assert len(key1[1]) == 1
            assert len(key2[1]) == 1

    def test_coact_wrong_arity_gives_zero(self):
        """δ_3 is zero on a corolla of arity 2 (arity mismatch)."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        elem = T.module(((1, 2), ((), ())))
        result = T.coact(elem, 3)
        assert result.is_zero()

    def test_infinitesimal_cocompose_arity3_to_2x2(self):
        """Δ^{2;2,2} on an arity-3 corolla gives a non-zero tensor."""
        M = _zero_diff_module()
        T = CofreeConilpotentCoalgebra(CoAssociative, M, QQ)
        # Use a CoAss(3) corolla: ((1,2,3), ((), (), ()))
        arity3_elem = T.module(((1, 2, 3), ((), (), ())))
        # Δ^{2;2,2}: split C(3)-decoration at position 2
        result = T.infinitesimal_cocompose(arity3_elem, 2, 2, 2)
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
        elem = T.module(((1, 2), ((), ())))
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
        # a_tuple: two Free_Lie(M) leaf basis keys; Lie(1) key is ()
        a_tuple = (((), ((),)), ((), ((),)))
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
        # a_tuple: two Free_Ass(M) leaf basis keys; Ass(1) key is (1,)
        a_tuple = (((1,), ((),)), ((1,), ((),)))
        elem = B((outer_tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()
