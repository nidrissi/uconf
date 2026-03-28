"""Tests for BarAlgebra and CobarCoalgebra.

Covers:
- :class:`uconf.constructions.bar_algebra.BarAlgebra`
- :class:`uconf.constructions.bar_algebra.BarAlgebraModule`
- :class:`uconf.constructions.cobar_coalgebra.CobarCoalgebra`
- :class:`uconf.constructions.cobar_coalgebra.CobarCoalgebraModule`
"""

import itertools

import pytest
from sage.all import GF, QQ, CombinatorialFreeModule, GradedModulesWithBasis, tensor

from uconf import Associative, CoAssociative, Commutative
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.constructions.cobar_coalgebra import CobarCoalgebra
from uconf.morphisms.canonical_twisting import canonical_inclusion, canonical_projection

# ===========================================================================
# Helpers
# ===========================================================================


class TrivialModule(CombinatorialFreeModule):
    def __init__(self, dimension: int, base_ring):
        super().__init__(base_ring, [()], category=GradedModulesWithBasis(base_ring))
        self._dimension = dimension
        self.boundary = lambda _: self.zero()
        self.connectivity = 0
        self.rename(f"K[{dimension}]")

    def degree_on_basis(self, key):
        return self._dimension

    def _repr_term(self, key):
        return "g"


class TrivialAssAlgebra(OperadAlgebra):
    """Trivial Ass-algebra on 1-dim module k: γ(μ; a_1,...,a_n) = ()."""

    def __init__(self, base_ring=QQ):
        self._R = base_ring
        module = TrivialModule(0, base_ring)
        super().__init__(module, Associative, self._structure_map)

    def _structure_map(self, p_elem, a_list):
        module = self.module
        result = module.zero()
        for _key, p_coeff in p_elem:
            coeff = p_coeff
            for a_elem in a_list:
                for _ak, a_coeff in a_elem:
                    coeff *= a_coeff
            result += coeff * module(())
        return result


class TrivialCoassCoalgebra(CooperadCoalgebra):
    """Trivial CoAss-coalgebra on 1-dim module k: δ_n(()) = Σ_σ σ ⊗ ()^n."""

    def __init__(self, base_ring=QQ):
        self._R = base_ring
        module = TrivialModule(0, base_ring)
        super().__init__(module, CoAssociative, self._coaction_map)

    def _coaction_map(self, v_element, n):
        R = self._R
        left_parent = CoAssociative(n, base_ring=R)
        right_parent = self.module
        if n == 1:
            target = tensor([left_parent, right_parent])
            result = target.zero()
            for _v_key, v_coeff in v_element:
                result += v_coeff * left_parent.term((1,)).tensor(right_parent.term(()))
            return result

        right_factors = [right_parent] * n
        right_tensor = tensor(right_factors)
        target = tensor([left_parent, right_tensor])
        result = target.zero()
        for _v_key, v_coeff in v_element:
            for sigma in itertools.permutations(range(1, n + 1)):
                left_elem = left_parent(sigma)
                right_elem = right_parent(())
                right_full = right_elem
                for _ in range(n - 1):
                    right_full = right_full.tensor(right_parent(()))
                result += v_coeff * left_elem.tensor(right_full)
        return result


# ===========================================================================
# BarAlgebra tests
# ===========================================================================


class TestBarAlgebra:
    """Tests for BarAlgebra (bar construction B_π(A))."""

    @pytest.fixture
    def bar(self):
        return BarAlgebra(canonical_projection(Associative), TrivialAssAlgebra())

    @pytest.fixture
    def bar_gf2(self):
        return BarAlgebra(canonical_projection(Associative), TrivialAssAlgebra(GF(2)))

    def test_construction(self, bar):
        assert bar.module is not None
        assert bar.cooperad_cls is not None

    def test_single_leaf_degree(self, bar):
        C = bar.cooperad_cls
        elem = bar.module((C.unit_key(), ((),)))
        assert elem.degree() == 0

    def test_single_leaf_boundary_zero(self, bar):
        C = bar.cooperad_cls
        elem = bar.module((C.unit_key(), ((),)))
        assert bar.module.boundary(elem) == bar.module.zero()

    def test_weight1_binary_d_alpha(self, bar):
        """d_α on binary corolla is zero (cogenerator has trivial cocomposition)."""
        C = bar.cooperad_cls
        comp2 = C(2, QQ)
        for p_elem in comp2.planar_basis_iter(1):
            for c_key in p_elem.support():
                elem = bar.module((c_key, ((), ())))
                d_alpha = bar.module.d_alpha(elem)
                # Single-vertex corollas are cogenerators of the cofree cooperad,
                # so their infinitesimal cocomposition is zero → d_α = 0.
                assert d_alpha == bar.module.zero()
                return

    def test_d_squared_zero_weight1(self, bar):
        C = bar.cooperad_cls
        comp2 = C(2, QQ)
        for p_elem in comp2.planar_basis_iter(1):
            for c_key in p_elem.support():
                elem = bar.module((c_key, ((), ())))
                dd = bar.module.boundary(bar.module.boundary(elem))
                assert dd == bar.module.zero()
                return

    def test_d_squared_zero_weight1_ternary(self, bar):
        C = bar.cooperad_cls
        comp3 = C(3, QQ)
        for p_elem in comp3.planar_basis_iter(1):
            for c_key in p_elem.support():
                elem = bar.module((c_key, ((), (), ())))
                dd = bar.module.boundary(bar.module.boundary(elem))
                assert dd == bar.module.zero()
                return

    def test_d_squared_zero_all_weight1_gf2(self, bar_gf2):
        bar = bar_gf2
        C = bar.cooperad_cls
        R = GF(2)
        for n in range(2, 5):
            comp_n = C(n, R)
            for d in range(4):
                for p_elem in comp_n.planar_basis_iter(d):
                    for c_key in p_elem.support():
                        m_tuple = tuple([()] * n)
                        elem = bar.module((c_key, m_tuple))
                        dd = bar.module.boundary(bar.module.boundary(elem))
                        assert dd == bar.module.zero(), (
                            f"d² ≠ 0 at arity {n}, degree {d}"
                        )

    def test_d_squared_zero_all_weight1_qq(self, bar):
        """d² = 0 for all weight-1 elements over QQ up to degree 4."""
        C = bar.cooperad_cls
        R = QQ
        for n in range(2, 6):
            comp_n = C(n, R)
            for d in range(5):
                for p_elem in comp_n.planar_basis_iter(d):
                    for c_key in p_elem.support():
                        m_tuple = tuple([()] * n)
                        elem = bar.module((c_key, m_tuple))
                        dd = bar.module.boundary(bar.module.boundary(elem))
                        assert dd == bar.module.zero(), (
                            f"d² ≠ 0 at arity {n}, degree {d}, c_key={c_key}"
                        )

    def test_commutative_bar(self):
        """Bar construction with Commutative operad requires quasi-planar cooperad."""
        R = QQ
        module = TrivialModule(0, R)

        def comm_structure_map(p_elem, a_list):
            result = module.zero()
            for _p_key, p_coeff in p_elem:
                coeff = p_coeff
                for a_elem in a_list:
                    for _a_key, a_coeff in a_elem:
                        coeff *= a_coeff
                result += coeff * module(())
            return result

        alg = OperadAlgebra(module, Commutative, comm_structure_map)
        pi = canonical_projection(Commutative)
        # B(Com) is not quasi-planar, so construction should raise TypeError
        with pytest.raises(TypeError, match="not quasi-planar"):
            BarAlgebra(pi, alg)

    def test_coalgebra_coaction(self, bar):
        C = bar.cooperad_cls
        elem = bar.module((C.unit_key(), ((),)))
        coaction = bar.coact(elem, 1)
        assert coaction != 0

    def test_project(self, bar):
        C = bar.cooperad_cls
        elem = bar.module((C.unit_key(), ((),)))
        proj = bar.project(elem)
        assert proj == bar._algebra.module.term(())


# ===========================================================================
# BarAlgebra with FreeOperadAlgebra
# ===========================================================================


class TestBarAlgebraFreeAlgebra:
    """Tests for BarAlgebra with FreeOperadAlgebra(Ass, k[1])."""

    @pytest.fixture
    def bar(self):
        M = TrivialModule(1, QQ)
        free_alg = FreeOperadAlgebra(Associative, M)
        return BarAlgebra(canonical_projection(Associative), free_alg)

    def test_construction(self, bar):
        assert bar.module is not None

    def test_single_leaf_degree(self, bar):
        C = bar.cooperad_cls
        P = bar._algebra.operad_cls
        id_key = P.unit_key()
        inner_key = (id_key, ((),))
        elem = bar.module((C.unit_key(), (inner_key,)))
        assert elem.degree() == 1

    def test_weight1_d_squared_zero(self, bar):
        C = bar.cooperad_cls
        P = bar._algebra.operad_cls
        id_key = P.unit_key()
        inner_key = (id_key, ((),))
        comp2 = C(2, QQ)
        for p_elem in comp2.planar_basis_iter(1):
            for c_key in p_elem.support():
                elem = bar.module((c_key, (inner_key, inner_key)))
                dd = bar.module.boundary(bar.module.boundary(elem))
                assert dd == bar.module.zero()
                return

    def test_weight4_d_squared_zero(self, bar):
        """d² = 0 for all arity-4 elements with generator leaves over QQ."""
        C = bar.cooperad_cls
        P = bar._algebra.operad_cls
        id_key = P.unit_key()
        inner_key = (id_key, ((),))
        comp4 = C(4, QQ)
        m_tuple = (inner_key,) * 4
        for d in range(5):
            for p_elem in comp4.planar_basis_iter(d):
                for c_key in p_elem.support():
                    elem = bar.module((c_key, m_tuple))
                    dd = bar.module.boundary(bar.module.boundary(elem))
                    assert dd == bar.module.zero(), (
                        f"d² ≠ 0 at degree {d}, c_key={c_key}"
                    )

    def test_weight5_d_squared_zero(self, bar):
        """d² = 0 for all arity-5 elements over QQ (up to degree 4)."""
        C = bar.cooperad_cls
        P = bar._algebra.operad_cls
        id_key = P.unit_key()
        inner_key = (id_key, ((),))
        comp5 = C(5, QQ)
        m_tuple = (inner_key,) * 5
        for d in range(5):
            for p_elem in comp5.planar_basis_iter(d):
                for c_key in p_elem.support():
                    elem = bar.module((c_key, m_tuple))
                    dd = bar.module.boundary(bar.module.boundary(elem))
                    assert dd == bar.module.zero(), (
                        f"d² ≠ 0 at degree {d}, c_key={c_key}"
                    )


# ===========================================================================
# CobarCoalgebra tests
# ===========================================================================


class TestCobarCoalgebra:
    """Tests for CobarCoalgebra (cobar construction Ω_ι(V))."""

    @pytest.fixture
    def cobar(self):
        """Cobar construction over QQ (default)."""
        return CobarCoalgebra(
            canonical_inclusion(CoAssociative), TrivialCoassCoalgebra(QQ)
        )

    @pytest.fixture
    def cobar_gf2(self):
        """Cobar construction over GF(2)."""
        return CobarCoalgebra(
            canonical_inclusion(CoAssociative), TrivialCoassCoalgebra(GF(2))
        )

    def test_construction(self, cobar):
        assert cobar.module is not None
        assert cobar.operad_cls is not None

    def test_single_leaf_degree(self, cobar):
        P = cobar.operad_cls
        elem = cobar.module((P.unit_key(), ((),)))
        assert elem.degree() == 0

    def test_d_alpha_on_leaf_gf2(self, cobar_gf2):
        """d_α on leaf vanishes over GF(2) (n! = 0 mod 2 for n ≥ 2)."""
        P = cobar_gf2.operad_cls
        elem = cobar_gf2.module((P.unit_key(), ((),)))
        d_alpha = cobar_gf2.module.d_alpha(elem)
        assert d_alpha == cobar_gf2.module.zero()

    def test_d_squared_zero_leaf_gf2(self, cobar_gf2):
        """d² = 0 on single leaf over GF(2)."""
        P = cobar_gf2.operad_cls
        elem = cobar_gf2.module((P.unit_key(), ((),)))
        dd = cobar_gf2.module.boundary(cobar_gf2.module.boundary(elem))
        assert dd == cobar_gf2.module.zero()

    def test_d_alpha_nonzero_on_leaf_qq(self, cobar):
        """d_α on leaf over QQ is nonzero (2·(arity-2 corolla))."""
        P = cobar.operad_cls
        elem = cobar.module((P.unit_key(), ((),)))
        d_alpha = cobar.module.d_alpha(elem)
        assert d_alpha != cobar.module.zero()

    def test_include(self, cobar):
        """The inclusion η: V → Ω_α(V) works."""
        elem = cobar.include(())
        P = cobar.operad_cls
        expected = cobar.module.term((P.unit_key(), ((),)))
        assert elem == expected

    def test_d_squared_zero_weight2_gf2(self, cobar_gf2):
        """d² = 0 on all arity-2 elements over GF(2)."""
        cobar = cobar_gf2
        P = cobar.operad_cls
        R = GF(2)
        for n in range(2, 4):
            comp_n = P(n, R)
            m_tuple = ((),) * n
            for d in range(4):
                for p_elem in comp_n.planar_basis_iter(d):
                    for p_key in p_elem.support():
                        elem = cobar.module.term((p_key, m_tuple))
                        dd = cobar.module.boundary(cobar.module.boundary(elem))
                        assert dd == cobar.module.zero(), (
                            f"d² ≠ 0 at arity {n}, degree {d}"
                        )

    def test_d_squared_zero_weight4_gf2(self, cobar_gf2):
        """d² = 0 on all arity-4 elements over GF(2) up to degree 3."""
        cobar = cobar_gf2
        P = cobar.operad_cls
        R = GF(2)
        comp4 = P(4, R)
        m_tuple = ((),) * 4
        for d in range(4):
            for p_elem in comp4.planar_basis_iter(d):
                for p_key in p_elem.support():
                    elem = cobar.module.term((p_key, m_tuple))
                    dd = cobar.module.boundary(cobar.module.boundary(elem))
                    assert dd == cobar.module.zero(), (
                        f"d² ≠ 0 at arity 4, degree {d}, p_key={p_key}"
                    )
