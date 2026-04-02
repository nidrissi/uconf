"""Tests for algebras over operads and coalgebras over cooperads.

Covers:
- :class:`uconf.algebraic.algebra.OperadAlgebra`
- :class:`uconf.algebraic.coalgebra.CooperadCoalgebra`
- :class:`uconf.constructions.bar_algebra.BarAlgebra` (bar construction B_α(A))
- :class:`uconf.constructions.cobar_coalgebra.CobarCoalgebra` (cobar construction Ω_α(V))
"""

import itertools

import pytest
from sage.all import GF, QQ, CombinatorialFreeModule, GradedModulesWithBasis, tensor

from uconf import Associative, CoAssociative, Commutative, Lie
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_coalgebra import CobarCoalgebra
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.trees import RootedTree
from uconf.morphisms.canonical_twisting import canonical_inclusion, canonical_projection

# ===========================================================================
# Helpers: build simple algebra modules
# ===========================================================================


@pytest.fixture(params=[QQ, GF(2)])
def R(request):
    return request.param


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
    """A trivial associative algebra structure on the 1-dimensional module k.

    The module k has a single basis element ``()`` in degree 0, and the product
    γ_n(σ; (),...,()) = () for all σ ∈ S_n.
    """

    def __init__(self, base_ring=QQ):
        self.module = TrivialModule(0, base_ring)
        super().__init__(self.module, Associative, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        # p_elem ∈ Ass(n), algebra_elements = [module_elem, ...]
        # All elements are scalar multiples of ()
        n = p_element.arity()
        if len(algebra_elements) != n:
            raise ValueError(f"Expected {n} algebra elements, got {len(algebra_elements)}.")

        result = self.module.zero()
        for p_key, p_coeff in p_element:
            # Coefficient from each a_i
            coeff = p_coeff
            for a_elem in algebra_elements:
                for _a_key, a_coeff in a_elem:
                    coeff = coeff * a_coeff
            result += coeff * self.module(())
        return result


@pytest.fixture
def trivial_ass_algebra(R):
    return TrivialAssAlgebra(base_ring=R)


class TrivialCoassCoalgebra(CooperadCoalgebra):
    """Return a CoAss-coalgebra structure on the 1-dimensional module k.

    The coaction δ_n: k → CoAss(n) ⊗ k^⊗n sends () to Σ_σ (σ ⊗ ()⊗...⊗()).
    Since CoAss(n) has a full basis {σ : σ ∈ S_n}, the coaction sums over all.
    """

    def __init__(self, base_ring=QQ):
        self.base_ring = base_ring
        self.module = TrivialModule(0, base_ring=base_ring)
        super().__init__(self.module, CoAssociative, self._coaction_map)

    def _coaction_map(self, v_element, n):
        left_parent = CoAssociative(n, base_ring=self.base_ring)
        right_parent = Commutative(1, base_ring=self.base_ring)
        right_factors = [right_parent] * n

        if n == 1:
            target = tensor([left_parent, right_parent])
        else:
            right_tensor = tensor(right_factors)
            target = tensor([left_parent, right_tensor])

        result = target.zero()
        for _v_key, v_coeff in v_element:
            for sigma in itertools.permutations(range(1, n + 1)):
                left_elem = left_parent(sigma)
                if n == 1:
                    right_elem = right_parent(())
                    result += v_coeff * left_elem.tensor(right_elem)
                else:
                    right_elem = right_parent(())
                    right_full = right_elem
                    for _ in range(n - 1):
                        right_full = right_full.tensor(right_parent(()))
                    result += v_coeff * left_elem.tensor(right_full)
        return result


@pytest.fixture
def trivial_coass_coalgebra(R):
    return TrivialCoassCoalgebra(base_ring=R)


# ===========================================================================
# OperadAlgebra tests
# ===========================================================================


class TestOperadAlgebra:
    """Tests for OperadAlgebra wrapper."""

    def test_construction(self, trivial_ass_algebra):
        assert trivial_ass_algebra.operad_cls is Associative
        assert trivial_ass_algebra.module is not None

    def test_act_unary(self, trivial_ass_algebra, R):
        """Unit axiom: γ_1(id; a) = a."""
        module = trivial_ass_algebra.module
        a = module(())
        unit = Associative.unit(R)
        result = trivial_ass_algebra.act(unit, [a])
        assert result == a

    def test_act_binary(self, trivial_ass_algebra, R):
        """Binary product γ_2(σ; a, a) = a (trivial algebra)."""
        module = trivial_ass_algebra.module
        a = module(())
        p = Associative(2, R)((1, 2))
        result = trivial_ass_algebra.act(p, [a, a])
        assert result == a

    def test_act_arity_mismatch(self, trivial_ass_algebra, R):
        """act() raises ValueError when len(algebra_elements) != arity."""
        module = trivial_ass_algebra.module
        a = module(())
        p = Associative(2, R)((1, 2))
        with pytest.raises(ValueError, match="Expected 2"):
            trivial_ass_algebra.act(p, [a])

    def test_boundary_zero(self, trivial_ass_algebra):
        """Boundary of algebra element is 0 (Commutative module has zero differential)."""
        module = trivial_ass_algebra.module
        a = module(())
        assert trivial_ass_algebra.boundary(a) == module.zero()


# ===========================================================================
# CooperadCoalgebra tests
# ===========================================================================


class TestCooperadCoalgebra:
    """Tests for CooperadCoalgebra wrapper."""

    def test_construction(self):
        coalg = TrivialCoassCoalgebra()
        assert coalg.cooperad_cls is CoAssociative
        assert coalg.module is not None

    def test_boundary_zero(self, trivial_coass_coalgebra):
        """Boundary of coalgebra element is 0."""
        coalg = trivial_coass_coalgebra
        module = coalg.module
        v = module(())
        assert coalg.boundary(v) == module.zero()


# ===========================================================================
# BarAlgebra tests
# ===========================================================================


class TestBarAlgebra:
    """Tests for BarAlgebra (bar construction B_π(A)) with TrivialAssAlgebra."""

    @pytest.fixture
    def bar(self):
        """Bar construction B_π(k) with trivial Ass-algebra over QQ."""
        return BarAlgebra(canonical_projection(Associative), TrivialAssAlgebra())

    def test_single_leaf_degree(self, bar):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        elem = bar.module((1, ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self, bar):
        """Weight-1 binary tree has degree = 1 (deg_P(μ)+1 = 0+1 = 1)."""
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        a_tuple = ((), ())
        elem = bar.module((tree, a_tuple))
        assert elem.degree() == 1

    def test_weight1_ternary_degree(self, bar):
        """Weight-1 ternary tree has degree = 1."""
        mu3 = (1, 2, 3)
        tree = RootedTree(mu3, 1, 2, 3)
        a_tuple = ((), (), ())
        deg = bar.module.degree_on_basis((tree, a_tuple))
        assert deg == 1

    def test_dact_weight1_binary_gives_zero(self, bar):
        """d_α on weight-1 binary corolla is zero (cogenerator has trivial cocomposition)."""
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        a_tuple = ((), ())
        elem = bar.module((tree, a_tuple))
        result = bar.module.d_alpha(elem)
        assert result == bar.module.zero()

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            (RootedTree(_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            (RootedTree(_MU, 1, RootedTree(_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            (RootedTree(_MU, RootedTree(_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            (RootedTree(_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero(self, bar, tree, a_tuple: tuple):
        """d² = 0 on tree elements of the bar construction."""
        elem = bar.module((tree, a_tuple))
        assert bar.module.boundary(bar.module.boundary(elem)) == bar.module.zero()

    def test_linear_combination_d_squared_zero(self, bar):
        """d² = 0 on a linear combination of bar elements."""
        mu = (1, 2)
        t1 = bar.module(((RootedTree(mu, 1, RootedTree(mu, 2, 3))), ((), (), ())))
        t2 = bar.module(((RootedTree(mu, RootedTree(mu, 1, 2), 3)), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert bar.module.boundary(bar.module.boundary(combo)) == bar.module.zero()

    def test_d_cofree_connects_to_ternary(self, bar):
        """d_cofree on weight-2 binary tree produces weight-1 ternary term."""
        mu = (1, 2)
        tree = RootedTree(mu, 1, RootedTree(mu, 2, 3))
        elem = bar.module((tree, ((), (), ())))
        d_cofree_result = bar.module.d_cofree(elem)
        # A ternary cooperad key has 4 elements: (decoration, leaf1, leaf2, leaf3)
        has_ternary = False
        for (c_key, _m_tuple), _coeff in d_cofree_result:
            if isinstance(c_key, RootedTree) and c_key._arity == 3:
                has_ternary = True
        assert has_ternary

    def test_commutative_bar_algebra(self):
        """Bar construction with Commutative operad requires quasi-planar cooperad."""
        module = Commutative(1, QQ)

        def comm_structure_map(p_elem, a_list):
            result = module.zero()
            for _p_key, p_coeff in p_elem:
                coeff = p_coeff
                for a_elem in a_list:
                    for _a_key, a_coeff in a_elem:
                        coeff = coeff * a_coeff
                result += coeff * module(())
            return result

        alg = OperadAlgebra(module, Commutative, comm_structure_map)
        with pytest.raises(TypeError, match="not quasi-planar"):
            BarAlgebra(canonical_projection(Commutative), alg)


# ===========================================================================
# BarAlgebra tests -- FreeOperadAlgebra algebra
# ===========================================================================


class TestBarAlgebraFreeAlgebra:
    """Tests for BarAlgebra with FreeOperadAlgebra(Ass, k[1]).

    The algebra module is Free_Ass(k[1]) = Ass ∘ k[1], whose basis elements
    are pairs (p_key, m_tuple) where p_key ∈ Ass(n) (planar) and m_tuple is an
    n-tuple of the single basis element () of k[1].

    Basis keys used below:
    - _GEN1 = ((1,), ((),))   → the arity-1 generator, degree 1
    - _GEN2 = ((1,2), ((),()))→ the arity-2 element, degree 2

    Bar construction basis key format: (c_key, m_tuple) where c_key is a cooperad
    basis key from B(Ass) and m_tuple is a tuple of FreeAlgebraModule basis keys.
    """

    # arity-1 generator of Free_Ass(k[1]): key = (Ass(1)-key, (inner-module-key,))
    _GEN1 = ((1,), ((),))
    # arity-2 element: γ(μ; gen1, gen1) = ((1,2), ((),()))
    _GEN2 = ((1, 2), ((), ()))
    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.fixture
    def bar(self):
        """Bar construction B_π(Free_Ass(k[1])) with free Ass-algebra over QQ."""
        return BarAlgebra(
            canonical_projection(Associative), FreeOperadAlgebra(Associative, TrivialModule(1, QQ))
        )

    def test_construction(self, bar):
        """BarAlgebra can be constructed with FreeOperadAlgebra."""
        assert bar.module is not None

    def test_single_leaf_degree(self, bar):
        """Single-leaf element has degree = deg_free(_GEN1) = 1."""
        gen1 = ((1,), ((),))
        elem = bar.module((1, (gen1,)))
        assert elem.degree() == 1

    def test_weight1_binary_degree(self, bar):
        """Weight-1 binary tree has degree = (deg_Ass(μ)+1) + 2*deg(_GEN1) = 3."""
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        elem = bar.module((tree, (gen1, gen1)))
        assert elem.degree() == 3

    def test_weight1_ternary_degree(self, bar):
        """Weight-1 ternary tree has degree = 1 + 3*1 = 4."""
        gen1 = ((1,), ((),))
        mu3 = (1, 2, 3)
        tree = RootedTree(mu3, 1, 2, 3)
        deg = bar.module.degree_on_basis((tree, (gen1, gen1, gen1)))
        assert deg == 4

    def test_weight2_binary_degree(self, bar):
        """Weight-2 right-nested binary tree has degree = 2 + 3*1 = 5."""
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = RootedTree(mu, 1, RootedTree(mu, 2, 3))
        deg = bar.module.degree_on_basis((tree, (gen1, gen1, gen1)))
        assert deg == 5

    def test_dact_weight1_binary_gives_zero(self, bar):
        """d_α on weight-1 binary corolla is zero (cogenerator has trivial cocomposition).

        In the cofree coalgebra, single-vertex corollas are cogenerators whose
        infinitesimal cocomposition is trivial, so d_α = 0.
        """
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        elem = bar.module((tree, (gen1, gen1)))
        result = bar.module.d_alpha(elem)
        assert result == bar.module.zero()

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            (RootedTree(_MU, 1, 2), (_GEN1, _GEN1)),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            (RootedTree(_MU, 1, RootedTree(_MU, 2, 3)), (_GEN1, _GEN1, _GEN1)),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            (RootedTree(_MU, RootedTree(_MU, 1, 2), 3), (_GEN1, _GEN1, _GEN1)),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            (RootedTree(_MU3, 1, 2, 3), (_GEN1, _GEN1, _GEN1)),
        ],
    )
    def test_d_squared_zero(self, bar, tree, a_tuple: tuple):
        """d² = 0 on tree elements of the bar construction with the free algebra."""
        elem = bar.module((tree, a_tuple))
        assert bar.module.boundary(bar.module.boundary(elem)) == bar.module.zero()

    def test_linear_combination_d_squared_zero(self, bar):
        """d² = 0 on a linear combination of bar elements with the free algebra."""
        gen1 = ((1,), ((),))
        mu = (1, 2)
        t1 = bar.module(((RootedTree(mu, 1, RootedTree(mu, 2, 3))), (gen1, gen1, gen1)))
        t2 = bar.module(((RootedTree(mu, RootedTree(mu, 1, 2), 3)), (gen1, gen1, gen1)))
        combo = 3 * t1 - 2 * t2
        assert bar.module.boundary(bar.module.boundary(combo)) == bar.module.zero()

    def test_d_cofree_connects_to_ternary(self, bar):
        """d_cofree on weight-2 binary tree produces a weight-1 ternary term."""
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = RootedTree(mu, 1, RootedTree(mu, 2, 3))
        elem = bar.module((tree, (gen1, gen1, gen1)))
        d_cofree_result = bar.module.d_cofree(elem)
        # A ternary cooperad key has 4 elements: (decoration, leaf1, leaf2, leaf3)
        has_ternary = False
        for (c_key, _m_tuple), _coeff in d_cofree_result:
            if isinstance(c_key, RootedTree) and c_key._arity == 3:
                has_ternary = True
        assert has_ternary


# ===========================================================================
# CobarCoalgebra tests
# ===========================================================================


class TestCobarCoalgebra:
    """Tests for CobarCoalgebra (cobar construction Ω_ι(V))."""

    @pytest.fixture
    def cobar(self):
        """Cobar construction Ω_ι(k) with trivial CoAss-coalgebra over QQ."""
        return CobarCoalgebra(canonical_inclusion(CoAssociative), TrivialCoassCoalgebra())

    @pytest.fixture
    def cobar_gf2(self):
        """Cobar construction Ω_ι(k) with trivial CoAss-coalgebra over GF(2)."""
        return CobarCoalgebra(canonical_inclusion(CoAssociative), TrivialCoassCoalgebra(GF(2)))

    def test_single_leaf_degree(self, cobar):
        """Weight-0 element (single leaf) has degree = deg_V(v)."""
        P = cobar.operad_cls
        assert cobar.module.degree_on_basis((P.unit_key(), ((),))) == 0

    def test_weight1_binary_degree(self, cobar):
        """Weight-1 binary tree in Ω_C(V) has degree = -1 (deg_C(c)-1 = 0-1)."""
        c_dec = (1, 2)
        tree = RootedTree(c_dec, 1, 2)
        v_tuple = ((), ())
        deg = cobar.module.degree_on_basis((tree, v_tuple))
        assert deg == -1

    def test_d_free_zero_for_trivial(self, cobar):
        """d_free = 0 when the operad and module differentials are both zero."""
        c_dec = (1, 2)
        tree = RootedTree(c_dec, 1, 2)
        elem = cobar.module((tree, ((), ())))
        assert cobar.module.d_free(elem) == cobar.module.zero()

    def test_dalpha_on_leaf_expands(self, cobar):
        """d_α on a single leaf produces weight-1 cobar trees."""
        P = cobar.operad_cls
        elem = cobar.module((P.unit_key(), ((),)))
        result = cobar.module.d_alpha(elem)
        assert result != cobar.module.zero()

    def test_d_free_on_weight1_zero(self, cobar):
        """d_free on a weight-1 (single-vertex) tree is zero (no edge to split)."""
        c_dec = (1, 2)
        tree = RootedTree(c_dec, 1, 2)
        elem = cobar.module((tree, ((), ())))
        assert cobar.module.d_free(elem) == cobar.module.zero()

    @pytest.mark.parametrize(
        "key",
        [
            # single leaf
            (1, ((),)),
            # weight-1 binary tree
            (RootedTree((1, 2), 1, 2), ((), ())),
        ],
    )
    def test_d_squared_zero(self, cobar_gf2, key):
        """d² = 0 on elements of the cobar construction Ω_ι(V)."""
        elem = cobar_gf2.module(key)
        assert cobar_gf2.module.boundary(cobar_gf2.module.boundary(elem)) == cobar_gf2.module.zero()

    def test_d_squared_zero_linear_combo(self, cobar_gf2):
        """d² = 0 on a linear combination of cobar elements."""
        P = cobar_gf2.operad_cls
        t1 = cobar_gf2.module((P.unit_key(), ((),)))
        c_dec = (1, 2)
        t2 = cobar_gf2.module((RootedTree(c_dec, 1, 2), ((), ())))
        combo = t1 + t2
        assert (
            cobar_gf2.module.boundary(cobar_gf2.module.boundary(combo)) == cobar_gf2.module.zero()
        )


# ===========================================================================
# Lie operad -- bar complex with non-trivial operad
# ===========================================================================


class TestBarComplexLie:
    """Bar construction of a Lie-algebra (Chevalley-Eilenberg complex analog)."""

    @pytest.fixture
    def trivial_lie_algebra(self):
        """1-dimensional trivial Lie algebra with zero bracket."""
        module = Commutative(1, base_ring=QQ)

        def structure_map(p_elem, a_list):
            # For the trivial Lie algebra, all brackets are 0
            return module.zero()

        return OperadAlgebra(module, Lie, structure_map)

    def test_d_squared_zero_weight1(self, trivial_lie_algebra):
        """Bar construction with Lie operad requires quasi-planar cooperad."""
        with pytest.raises(TypeError, match="not quasi-planar"):
            BarAlgebra(canonical_projection(Lie), trivial_lie_algebra)


# ===========================================================================
# BarAlgebra (with canonical inclusion ι) tests
# ===========================================================================


class TestBarAlgebraIota:
    """Tests for BarAlgebra with canonical inclusion ι (bar construction B_ι(A))."""

    @pytest.fixture
    def trivial_iota_bar(self):
        """B_ι(k) -- bar construction of the trivial 1-dim ΩB(Ass)-algebra over QQ."""
        bar_ass = BarConstruction(Associative)
        cobar_bar_ass = CobarConstruction(bar_ass)
        module = Commutative(1, base_ring=QQ)

        def trivial_structure_map(p_elem, a_list):
            return module.zero()

        alg = OperadAlgebra(module, cobar_bar_ass, trivial_structure_map)
        return BarAlgebra(canonical_inclusion(BarConstruction(Associative)), alg)

    @pytest.fixture
    def pullback_iota_bar(self):
        """B_ι(k) -- bar construction of the ε-pullback ΩB(Ass)-algebra over QQ."""
        bar_ass = BarConstruction(Associative)
        cobar_bar_ass = CobarConstruction(bar_ass)
        module = Commutative(1, base_ring=QQ)

        def pullback_structure_map(p_elem, a_list):
            n = p_elem.arity()
            result = module.zero()
            for key, coeff in p_elem:
                # Only single-vertex cobar trees with a bar-corolla decoration
                # contribute (projecting through the augmentation ε: ΩB(Ass) → Ass).
                if not isinstance(key, RootedTree):
                    if n == 1:
                        # Unit: ε(id) = id_Ass, act as identity
                        a_coeff = coeff
                        for a_elem in a_list:
                            for _ak, ac in a_elem:
                                a_coeff = a_coeff * ac
                        result += a_coeff * module(())
                else:
                    from uconf.core.trees import children, decoration, is_leaf, vertex_arity
                    v_arity = vertex_arity(key)
                    if v_arity != n:
                        continue
                    v_chs = children(key)
                    if not all(is_leaf(c) for c in v_chs):
                        continue
                    # Single-vertex cobar tree: decoration is a B(Ass)(n) key.
                    # A bar corolla key has the form RootedTree(ass_key, 1,...,n).
                    bar_dec = decoration(key)
                    if not isinstance(bar_dec, RootedTree):
                        continue
                    bar_v_arity = vertex_arity(bar_dec)
                    if bar_v_arity != n:
                        continue
                    bar_chs = children(bar_dec)
                    if not all(is_leaf(c) for c in bar_chs):
                        continue
                    # bar_dec = RootedTree(ass_key, 1,...,n) is a bar corolla; ε maps it to
                    # ass_key ∈ Ass(n).  Apply the trivial Ass-action.
                    a_coeff = coeff
                    for a_elem in a_list:
                        for _ak, ac in a_elem:
                            a_coeff = a_coeff * ac
                    result += a_coeff * module(())
            return result
            return result

        alg = OperadAlgebra(module, cobar_bar_ass, pullback_structure_map)
        return BarAlgebra(canonical_inclusion(BarConstruction(Associative)), alg)

    def test_construction(self, trivial_iota_bar):
        """BarAlgebra can be constructed from an ΩB(P)-algebra."""
        assert trivial_iota_bar is not None

    def test_single_leaf_degree(self, trivial_iota_bar):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        elem = trivial_iota_bar.module((1, ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self, trivial_iota_bar):
        """Weight-1 binary tree has degree = 1 (deg_P(μ)+1 = 0+1 = 1)."""
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        elem = trivial_iota_bar.module((tree, ((), ())))
        assert elem.degree() == 1

    def test_weight1_ternary_degree(self, trivial_iota_bar):
        """Weight-1 ternary tree has degree = 1."""
        mu3 = (1, 2, 3)
        tree = RootedTree(mu3, 1, 2, 3)
        deg = trivial_iota_bar.module.degree_on_basis((tree, ((), (), ())))
        assert deg == 1

    def test_dalpha_trivial_algebra_zero(self, trivial_iota_bar):
        """d_alpha = 0 for the trivial ΩB(Ass)-algebra (all weights)."""
        mu = (1, 2)
        tree = RootedTree(mu, 1, 2)
        elem = trivial_iota_bar.module((tree, ((), ())))
        assert trivial_iota_bar.module.d_alpha(elem) == trivial_iota_bar.module.zero()

    def test_dalpha_pullback_algebra_nonempty(self, pullback_iota_bar):
        """d_alpha is non-zero for the pullback ΩB(Ass)-algebra on weight-2 elements.

        Weight-1 corollas are cogenerators with trivial infinitesimal cocomposition,
        so d_alpha is always zero on them.  Test on a weight-2 element instead.
        """
        mu = (1, 2)
        tree = RootedTree(mu, 1, RootedTree(mu, 2, 3))
        elem = pullback_iota_bar.module((tree, ((), (), ())))
        twist_result = pullback_iota_bar.module.d_alpha(elem)
        assert twist_result != pullback_iota_bar.module.zero()

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            (RootedTree(_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            (RootedTree(_MU, 1, RootedTree(_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            (RootedTree(_MU, RootedTree(_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            (RootedTree(_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_trivial(self, trivial_iota_bar, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of B_ι(A) for the trivial ΩB(Ass)-algebra."""
        elem = trivial_iota_bar.module((tree, a_tuple))
        assert (
            trivial_iota_bar.module.boundary(trivial_iota_bar.module.boundary(elem))
            == trivial_iota_bar.module.zero()
        )

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            (RootedTree(_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            (RootedTree(_MU, 1, RootedTree(_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            (RootedTree(_MU, RootedTree(_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            (RootedTree(_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_pullback(self, pullback_iota_bar, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of B_ι(A) for the ε-pullback ΩB(Ass)-algebra."""
        elem = pullback_iota_bar.module((tree, a_tuple))
        assert (
            pullback_iota_bar.module.boundary(pullback_iota_bar.module.boundary(elem))
            == pullback_iota_bar.module.zero()
        )

    def test_linear_combination_d_squared_zero(self, pullback_iota_bar):
        """d² = 0 on a linear combination of elements for the pullback algebra."""
        mu = (1, 2)
        t1 = pullback_iota_bar.module(((RootedTree(mu, 1, RootedTree(mu, 2, 3))), ((), (), ())))
        t2 = pullback_iota_bar.module(((RootedTree(mu, RootedTree(mu, 1, 2), 3)), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert (
            pullback_iota_bar.module.boundary(pullback_iota_bar.module.boundary(combo))
            == pullback_iota_bar.module.zero()
        )
