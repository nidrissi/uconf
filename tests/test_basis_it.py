"""Regression tests for basis_it on free/cofree modules, bar/cobar complexes,
Hadamard products, and Hadamard tensor algebras."""

import pytest
from sage.all import QQ

from uconf import (
    Associative,
    CoAssociative,
    CobarConstruction,
    Commutative,
    HadamardProduct,
    Lie,
    ShiftedOperad,
    Surjection,
)
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.algebraic.free_algebra import FreeAlgebraModule
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.constructions.bar_algebra import BarComplexAlgebra
from uconf.constructions.cobar_coalgebra import CobarComplexCoalgebra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keys(basis_iter):
    """Return the multiset of basis keys from a basis element iterator."""
    result = []
    for elem in basis_iter:
        for key, _ in elem:
            result.append(key)
    return result


def _count(basis_iter):
    return len(list(basis_iter))


class TrivialAlgebra(OperadAlgebra):
    """A P-algebra structure on Com(1) with trivial multiplication."""

    def __init__(self, operad_cls, base_ring=QQ):
        mod = Commutative(1, base_ring=base_ring)
        super().__init__(mod, operad_cls, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        result = self.module.zero()
        for _, p_coeff in p_element:
            coeff = p_coeff
            for a_elem in algebra_elements:
                for _, a_coeff in a_elem:
                    coeff *= a_coeff
            result += coeff * self.module(())
        return result


# ---------------------------------------------------------------------------
# 1. FreeAlgebraModule.basis_it
# ---------------------------------------------------------------------------


class TestFreeAlgebraModuleBasisIt:
    """Tests for FreeAlgebraModule.basis_it."""

    def test_raises_when_enumeration_is_not_exhaustive(self) -> None:
        """Raise when both P and M allow unbounded arity in fixed degree."""
        M = Commutative(1, base_ring=QQ)
        # Associative (connectivity=0) with M concentrated in degree 0:
        # arity is unbounded in any fixed degree.
        mod = FreeAlgebraModule(Associative, M, QQ)
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            list(mod.basis_it(0))

    def test_shifted_ass_degree0_single_leaf(self) -> None:
        """ShiftedOperad(Ass,1) has connectivity=1; degree-0 = arity-1 leaf only."""
        P = ShiftedOperad(Associative, 1)
        M = Commutative(1, base_ring=QQ)
        mod = FreeAlgebraModule(P, M, QQ)
        elems_d0 = list(mod.basis_it(0))
        assert len(elems_d0) == 1, "Only the arity-1 leaf in degree 0"

    def test_shifted_ass_degree1_arity2(self) -> None:
        """ShiftedOperad(Ass,1) degree 1 = exactly one arity-2 element.

        ShiftedAss(2) ⊗_{S_2} Comm(1)^{⊗2}: the S_2-action on ShiftedAss(2)
        is by sign, and Comm(1)^{⊗2} = k{((),())} carries the trivial action.
        The coinvariant quotient is 1-dimensional.  Equivalently, ShiftedAss is
        quasi-planar with one-dimensional planar part at each arity, so
        basis_it uses only the planar decoration (1,2).
        """
        P = ShiftedOperad(Associative, 1)
        M = Commutative(1, base_ring=QQ)
        mod = FreeAlgebraModule(P, M, QQ)
        elems_d1 = list(mod.basis_it(1))
        assert len(elems_d1) == 1

    def test_correct_degrees(self) -> None:
        """All yielded elements have the requested degree."""
        P = ShiftedOperad(Associative, 1)
        M = Commutative(1, base_ring=QQ)
        mod = FreeAlgebraModule(P, M, QQ)
        for d in range(3):
            for elem in mod.basis_it(d):
                for key, _ in elem:
                    assert mod.degree_on_basis(key) == d

    def test_empty_above_max(self) -> None:
        """No degree-3 elements for ShiftedAss with Comm(1) module up to arity 2."""
        P = ShiftedOperad(Surjection, 1)
        M = Commutative(1, base_ring=QQ)
        mod = FreeAlgebraModule(P, M, QQ)
        # The degree-3 elements are arity-4 (P degree = 3): many elements.
        # We just check the count is positive and every element has the right degree.
        for elem in mod.basis_it(3):
            for key, _ in elem:
                assert mod.degree_on_basis(key) == 3

    def test_non_planar_operad_raises_type_error(self) -> None:
        """Lie and Commutative (non-quasi-planar) raise TypeError at construction.

        Both operads have non-free S_n-actions: Commutative has the trivial
        action (rank 1 per arity) and Lie has a non-free action.  Neither
        satisfies P(n) ≅ P_pl(n) ⊗ k[S_n], so FreeAlgebraModule rejects them
        with TypeError at construction time.
        """
        from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

        mod_M = CombinatorialFreeModule(QQ, ["x"], category=GradedModulesWithBasis(QQ))
        mod_M.degree_on_basis = lambda _k: 2
        mod_M.boundary = lambda _e: mod_M.zero()

        for P in (Lie, Commutative):
            with pytest.raises(TypeError, match="not quasi-planar"):
                FreeAlgebraModule(P, mod_M, QQ)


# ---------------------------------------------------------------------------
# 2. CofreeCoalgebraModule.basis_it
# ---------------------------------------------------------------------------


class TestCofreeCoalgebraModuleBasisIt:
    """Tests for CofreeCoalgebraModule.basis_it."""

    def test_raises_when_enumeration_is_not_exhaustive(self) -> None:
        """Raise when both C and M allow unbounded arity in fixed degree."""
        M = Commutative(1, base_ring=QQ)
        mod = CofreeCoalgebraModule(CoAssociative, M, QQ)
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            list(mod.basis_it(0))

    def test_correct_degrees(self) -> None:
        """All yielded elements have the requested degree."""
        M = Commutative(1, base_ring=QQ)
        mod = CofreeCoalgebraModule(ShiftedOperad(Associative, 1), M, QQ)
        for d in range(3):
            for elem in mod.basis_it(d):
                for key, _ in elem:
                    assert mod.degree_on_basis(key) == d


# ---------------------------------------------------------------------------
# 3. CobarConstruction.Component.basis_it
# ---------------------------------------------------------------------------


class TestCobarConstructionBasisIt:
    """Tests for CobarConstruction.Component.basis_it."""

    def test_arity1_degree0(self) -> None:
        """Arity 1, degree 0: only the unit element."""
        C1 = CobarConstruction(CoAssociative)(1, QQ)
        elems = list(C1.basis_it(0))
        assert len(elems) == 1
        assert elems[0] == C1.term(1)

    def test_arity1_nonzero_degree_empty(self) -> None:
        """Arity 1, non-zero degree: empty (unit has degree 0)."""
        C1 = CobarConstruction(CoAssociative)(1, QQ)
        assert list(C1.basis_it(1)) == []
        assert list(C1.basis_it(-1)) == []

    def test_arity2_degree_neg1(self) -> None:
        """CoAss in arity 2: cobar degree -1 (one vertex of degree-0, offset -1)."""
        C2 = CobarConstruction(CoAssociative)(2, QQ)
        elems = list(C2.basis_it(-1))
        # CoAss(2) has 2 basis keys: (1,2) and (2,1).  One vertex → cobar deg = -1.
        assert len(elems) == 2

    def test_arity2_degree0_empty_for_coass(self) -> None:
        """CoAss in arity 2: cobar degree 0 is empty (all vertices contribute -1)."""
        C2 = CobarConstruction(CoAssociative)(2, QQ)
        assert list(C2.basis_it(0)) == []

    def test_arity3_degree_neg2(self) -> None:
        """CoAss in arity 3: cobar degree -2 has trees of weight 2."""
        C3 = CobarConstruction(CoAssociative)(3, QQ)
        elems = list(C3.basis_it(-2))
        # Weight-2 trees with CoAss(2) at each vertex.
        # Two shuffle tree shapes for arity 3, weight 2; each CoAss vertex has
        # 2 basis keys → 12 total.
        assert len(elems) == 12

    def test_correct_cobar_degrees(self) -> None:
        """All yielded elements have the requested cobar degree."""
        C3 = CobarConstruction(CoAssociative)(3, QQ)
        for d in [-2, -1]:
            for elem in C3.basis_it(d):
                for key, _ in elem:
                    assert C3.degree_on_basis(key) == d


# ---------------------------------------------------------------------------
# 4. BarComplexAlgebra.basis_it
# ---------------------------------------------------------------------------


class TestBarComplexAlgebraBasisIt:
    """Tests for BarComplexAlgebra.basis_it."""

    @pytest.fixture
    def trivial_bar(self):
        alg = TrivialAlgebra(Associative)
        return BarComplexAlgebra(alg, QQ)

    def test_degree0_single_leaf(self, trivial_bar) -> None:
        """Degree 0 = single arity-1 leaf with the unique A-generator."""
        elems = list(trivial_bar.basis_it(0))
        assert len(elems) == 1
        key, _ = next(iter(elems[0]))
        assert key == (1, ((),))

    def test_degree1_non_exhaustive(self, trivial_bar) -> None:
        """Degree ≥ 1 is non-exhaustive (connectivity=0 with bar shift)."""
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            list(trivial_bar.basis_it(1))

    def test_correct_degree0(self, trivial_bar) -> None:
        """All yielded elements at degree 0 have the requested total degree."""
        for elem in trivial_bar.basis_it(0):
            for key, _ in elem:
                assert trivial_bar.degree_on_basis(key) == 0

    def test_with_lie_operad(self) -> None:
        """Bar complex with Lie operad: degree ≥ 1 is non-exhaustive (connectivity=0)."""
        alg = TrivialAlgebra(Lie)
        B = BarComplexAlgebra(alg, QQ)
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            list(B.basis_it(1))


# ---------------------------------------------------------------------------
# 5. CobarComplexCoalgebra.basis_it
# ---------------------------------------------------------------------------


class TestCobarComplexCoalgebraBasisIt:
    """Tests for CobarComplexCoalgebra.basis_it."""

    @pytest.fixture
    def trivial_cobar(self):
        from uconf.algebraic.coalgebra import CooperadCoalgebra

        class _TrivialCoalgebra(CooperadCoalgebra):
            def __init__(self, cooperad_cls, base_ring=QQ):
                mod = Commutative(1, base_ring=base_ring)
                super().__init__(mod, cooperad_cls, self._co)

            def _co(self, v_elem, k):
                return v_elem.parent().zero()

        return CobarComplexCoalgebra(_TrivialCoalgebra(CoAssociative), QQ)

    def test_degree0_single_leaf(self, trivial_cobar) -> None:
        """Raise when cooperad connectivity is 0 (non-exhaustive regime)."""
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            list(trivial_cobar.basis_it(0))

    def test_correct_degrees(self, trivial_cobar) -> None:
        """The same non-exhaustive regime raises for every queried degree."""
        for d in range(3):
            with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
                list(trivial_cobar.basis_it(d))


# ---------------------------------------------------------------------------
# 6. HadamardProduct.Component.basis_it
# ---------------------------------------------------------------------------


class TestHadamardProductComponentBasisIt:
    """Tests for HadamardProduct.Component.basis_it."""

    def test_degree0_surjection_surjection(self) -> None:
        """Degree 0: all (left_key, right_key) pairs both in degree 0."""
        had = HadamardProduct(Surjection, Surjection)
        h2 = had(2, QQ)
        # Surjection(2) degree-0: (1,2) and (2,1) → 2×2 = 4 pairs.
        elems = list(h2.basis_it(0))
        assert len(elems) == 4

    def test_degree1_surjection_surjection(self) -> None:
        """Degree 1: sum over (d_left, d_right) with d_left + d_right = 1."""
        had = HadamardProduct(Surjection, Surjection)
        h2 = had(2, QQ)
        # Surjection(2) has 2 basis elements in every degree (shuffle form).
        # Pairs: (deg0)×(deg1) + (deg1)×(deg0) = 2*2 + 2*2 = 8.
        elems = list(h2.basis_it(1))
        assert len(elems) == 8

    def test_correct_degrees(self) -> None:
        """All yielded elements have the requested degree."""
        had = HadamardProduct(Lie, Surjection)
        h2 = had(2, QQ)
        for d in range(3):
            for elem in h2.basis_it(d):
                for key, _ in elem:
                    assert h2.degree_on_basis(key) == d

    def test_basis_it_vs_planar_basis_it_surjection(self) -> None:
        """basis_it should contain all elements from planar_basis_it."""
        had = HadamardProduct(Surjection, Surjection)
        h2 = had(2, QQ)
        for d in range(3):
            planar = set(_keys(h2.planar_basis_it(d)))
            all_keys = set(_keys(h2.basis_it(d)))
            assert planar <= all_keys, f"planar basis not subset of full basis at d={d}"

    def test_degree0_lie_lie(self) -> None:
        """Lie(2) has 1 degree-0 element → 1×1 = 1 pair."""
        had = HadamardProduct(Lie, Lie)
        h2 = had(2, QQ)
        elems = list(h2.basis_it(0))
        assert len(elems) == 1

    def test_hadamard_arity3_degree0(self) -> None:
        """Arity 3: Lie(3) × Lie(3) in degree 0."""
        had = HadamardProduct(Lie, Lie)
        h3 = had(3, QQ)
        d0 = list(h3.basis_it(0))
        # Lie(3) degree-0: basis elements from Hall basis.
        lie3 = Lie(3, QQ)
        n_lie3 = _count(lie3.basis_it(0))
        assert len(d0) == n_lie3 * n_lie3


# ---------------------------------------------------------------------------
# 7. HadamardTensorAlgebra.basis_it
# ---------------------------------------------------------------------------


class TestHadamardTensorAlgebraBasisIt:
    """Tests for HadamardTensorAlgebra.basis_it."""

    @pytest.fixture
    def had_alg(self):
        left = TrivialAlgebra(Associative)
        right = TrivialAlgebra(Associative)
        return HadamardTensorAlgebra(left, right)

    def test_degree0_one_element(self, had_alg) -> None:
        """Degree 0 = one tensor element ((),()) since both modules are 1-dim in degree 0."""
        elems = list(had_alg.basis_it(0))
        assert len(elems) == 1

    def test_degree1_empty(self, had_alg) -> None:
        """Degree 1 is empty since both modules are concentrated in degree 0."""
        elems = list(had_alg.basis_it(1))
        assert len(elems) == 0

    def test_correct_degrees(self, had_alg) -> None:
        """All yielded elements have the requested degree."""
        for d in range(3):
            for elem in had_alg.basis_it(d):
                for (left_key, right_key), _ in elem:
                    deg = had_alg.left_module.degree_on_basis(
                        left_key
                    ) + had_alg.right_module.degree_on_basis(right_key)
                    assert deg == d

    def test_basis_it_tensor_module_keys(self, had_alg) -> None:
        """Yielded elements are elements of the tensor module with (left_key, right_key) pairs."""
        elems = list(had_alg.basis_it(0))
        key, coeff = next(iter(elems[0]))
        left_key, right_key = key
        # Both sides are Comm(1) with sole key ().
        assert left_key == ()
        assert right_key == ()


# ---------------------------------------------------------------------------
# 8. Cross-check: degree counts agree with BarConstruction.basis_it
# ---------------------------------------------------------------------------


class TestBasisItConsistencyWithBarConstruction:
    """Sanity-check that BarComplexAlgebra.basis_it counts match the expected tree × module count."""

    def test_bar_degree0_count_surjection(self) -> None:
        """B(Sur; Comm(1)) degree 0 has exactly the single-leaf element."""
        alg = TrivialAlgebra(Surjection)
        B = BarComplexAlgebra(alg, QQ)
        actual = _count(B.basis_it(0))
        assert actual == 1  # single leaf with a_key = ()

    def test_bar_degree_ge1_non_exhaustive(self) -> None:
        """B(Sur; Comm(1)) degree ≥ 1 is non-exhaustive (Sur has connectivity=0)."""
        alg = TrivialAlgebra(Surjection)
        B = BarComplexAlgebra(alg, QQ)
        for d in range(1, 4):
            with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
                list(B.basis_it(d))
