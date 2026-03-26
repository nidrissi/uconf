"""Tests for algebras over operads and coalgebras over cooperads.

Covers:
- :class:`uconf.algebra.OperadAlgebra`
- :class:`uconf.coalgebra.CooperadCoalgebra`
- :class:`uconf.constructions.twisted_complex.TwistedBarComplex` (bar complex B_α(A))
- :class:`uconf.constructions.twisted_complex.TwistedCobarComplex` (cobar complex Ω_α(V))
"""

import itertools

import pytest
from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis
from sage.all import tensor as sage_tensor

from uconf import Associative, CoAssociative, Commutative, Lie
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.constructions.twisted_complex import TwistedBarComplex, TwistedCobarComplex
from uconf.morphisms.canonical_twisting import canonical_inclusion, canonical_projection
from uconf.core.trees import children, decoration, is_leaf, vertex_arity

# ===========================================================================
# Helpers: build simple algebra modules
# ===========================================================================


class TrivialAssAlgebra(OperadAlgebra):
    """A trivial associative algebra structure on the 1-dimensional module k.

    The module k has a single basis element ``()`` in degree 0, and the product
    γ_n(σ; (),...,()) = () for all σ ∈ S_n.
    """

    def __init__(self, base_ring=QQ):
        self.module = Commutative(1, base_ring=base_ring)
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


class TrivialModule(CombinatorialFreeModule):
    def __init__(self, dimension: int, base_ring):
        super().__init__(base_ring, [()], category=GradedModulesWithBasis(base_ring))
        self._dimension = dimension
        self.boundary = lambda _: self.zero()
        self.connectivity = 0
        self.rename(f"K[{dimension}]")

    def degree_on_basis(self, key):
        return self._dimension


class TrivialCoassCoalgebra(CooperadCoalgebra):
    """Return a CoAss-coalgebra structure on the 1-dimensional module k.

    The coaction δ_n: k → CoAss(n) ⊗ k^⊗n sends () to Σ_σ (σ ⊗ ()⊗...⊗()).
    Since CoAss(n) has a full basis {σ : σ ∈ S_n}, the coaction sums over all.
    """

    def __init__(self, base_ring=QQ):
        self.base_ring = base_ring
        self.module = Commutative(1, base_ring=base_ring)
        super().__init__(self.module, CoAssociative, self._coaction_map)

    def _coaction_map(self, v_element, n):
        left_parent = CoAssociative(n, base_ring=self.base_ring)
        right_parent = Commutative(1, base_ring=self.base_ring)
        right_factors = [right_parent] * n

        if n == 1:
            target = sage_tensor([left_parent, right_parent])
        else:
            right_tensor = sage_tensor(right_factors)
            target = sage_tensor([left_parent, right_tensor])

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


# ===========================================================================
# OperadAlgebra tests
# ===========================================================================


class TestOperadAlgebra:
    """Tests for OperadAlgebra wrapper."""

    def test_construction(self):
        alg = TrivialAssAlgebra()
        assert alg.operad_cls is Associative
        assert alg.module is not None

    def test_act_unary(self):
        """Unit axiom: γ_1(id; a) = a."""
        alg = TrivialAssAlgebra()
        module = alg.module
        a = module(())
        unit = Associative.unit(QQ)
        result = alg.act(unit, [a])
        assert result == a

    def test_act_binary(self):
        """Binary product γ_2(σ; a, a) = a (trivial algebra)."""
        alg = TrivialAssAlgebra()
        module = alg.module
        a = module(())
        p = Associative(2, QQ)((1, 2))
        result = alg.act(p, [a, a])
        assert result == a

    def test_act_arity_mismatch(self):
        """act() raises ValueError when len(algebra_elements) != arity."""
        alg = TrivialAssAlgebra()
        module = alg.module
        a = module(())
        p = Associative(2, QQ)((1, 2))
        with pytest.raises(ValueError, match="Expected 2"):
            alg.act(p, [a])

    def test_boundary_zero(self):
        """Boundary of algebra element is 0 (Commutative module has zero differential)."""
        alg = TrivialAssAlgebra()
        module = alg.module
        a = module(())
        assert alg.boundary(a) == module.zero()


# ===========================================================================
# CooperadCoalgebra tests
# ===========================================================================


class TestCooperadCoalgebra:
    """Tests for CooperadCoalgebra wrapper."""

    def test_construction(self):
        coalg = TrivialCoassCoalgebra()
        assert coalg.cooperad_cls is CoAssociative
        assert coalg.module is not None

    def test_boundary_zero(self):
        """Boundary of coalgebra element is 0."""
        coalg = TrivialCoassCoalgebra()
        module = coalg.module
        v = module(())
        assert coalg.boundary(v) == module.zero()


# ===========================================================================
# TwistedBarComplex tests
# ===========================================================================


def _make_bar_complex(base_ring=QQ, trivial_algebra=True):
    """Build B_Ass(k) -- bar complex of the trivial 1-dim Ass-algebra."""
    if trivial_algebra:
        alg = TrivialAssAlgebra(base_ring=base_ring)
    else:
        simple_module = TrivialModule(1, QQ)
        alg = FreeOperadAlgebra(Associative, simple_module)
    return TwistedBarComplex(canonical_projection(Associative), alg)


class TestTwistedBarComplex:
    """Tests for TwistedBarComplex (bar complex B_π(A)) with TrivialAssAlgebra."""

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
        elem = B((tree, a_tuple))
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
        elem = B((tree, a_tuple))
        result = elem.dalpha()
        # Should give +(1, ((),)) -- single leaf with the product value
        assert result == B((1, ((),)))

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            ((_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero(self, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of the bar complex."""
        B = _make_bar_complex()
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_linear_combination_d_squared_zero(self):
        """d² = 0 on a linear combination of bar elements."""
        B = _make_bar_complex()
        mu = (1, 2)
        t1 = B(((mu, 1, (mu, 2, 3)), ((), (), ())))
        t2 = B(((mu, (mu, 1, 2), 3), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert combo.boundary().boundary() == B.zero()

    def test_d2_tree_connects_to_ternary(self):
        """d_2 on weight-2 binary tree produces weight-1 ternary term."""
        B = _make_bar_complex()
        mu = (1, 2)
        # Tree: (μ; 1, (μ; 2, 3)) -- d_2 contracts the internal edge
        tree = (mu, 1, (mu, 2, 3))
        elem = B((tree, ((), (), ())))
        d2_result = elem.d2()
        # The result should contain a weight-1 ternary tree
        has_ternary = False
        for (t, _a), _coeff in d2_result:
            if hasattr(t, "__len__") and len(t) == 4:  # (dec, leaf1, leaf2, leaf3)
                has_ternary = True
        assert has_ternary

    def test_commutative_bar_algebra(self):
        """Bar complex also works with the Commutative operad."""
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
        B = TwistedBarComplex(canonical_projection(Commutative), alg)
        # Weight-1 arity-2 tree: Commutative(2, QQ) has basis {()}
        tree = ((), 1, 2)
        elem = B((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()


# ===========================================================================
# TwistedBarComplex tests -- FreeOperadAlgebra algebra
# ===========================================================================


class TestTwistedBarComplexFreeAlgebra:
    """Tests for TwistedBarComplex with FreeOperadAlgebra(Ass, k[1]).

    The algebra module is Free_Ass(k[1]) = Ass ∘ k[1], whose basis elements
    are pairs (p_key, m_tuple) where p_key ∈ Ass(n) (planar) and m_tuple is an
    n-tuple of the single basis element () of k[1].

    Basis keys used below:
    - _GEN1 = ((1,), ((),))   → the arity-1 generator, degree 1
    - _GEN2 = ((1,2), ((),()))→ the arity-2 element, degree 2

    Bar complex basis key format: (tree, a_tuple) where a_tuple is a tuple of
    FreeAlgebraModule basis keys.  Degrees:
    - Single leaf decorated by _GEN1: 0 + deg(_GEN1) = 1
    - Weight-1 binary tree with two _GEN1 leaves:
        (deg_Ass(μ) + 1) + 2*deg(_GEN1) = 1 + 2 = 3
    - Weight-1 ternary tree with three _GEN1 leaves: 1 + 3 = 4
    - Weight-2 binary tree (two internal vertices) with three _GEN1 leaves:
        2*1 + 3 = 5
    """

    # arity-1 generator of Free_Ass(k[1]): key = (Ass(1)-key, (inner-module-key,))
    _GEN1 = ((1,), ((),))
    # arity-2 element: γ(μ; gen1, gen1) = ((1,2), ((),()))
    _GEN2 = ((1, 2), ((), ()))
    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    def test_construction(self):
        """TwistedBarComplex can be constructed with FreeOperadAlgebra."""
        B = _make_bar_complex(trivial_algebra=False)
        assert B is not None

    def test_single_leaf_degree(self):
        """Single-leaf element has degree = deg_free(_GEN1) = 1."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        elem = B((1, (gen1,)))
        assert elem.degree() == 1

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = (deg_Ass(μ)+1) + 2*deg(_GEN1) = 3."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B((tree, (gen1, gen1)))
        assert elem.degree() == 3

    def test_weight1_ternary_degree(self):
        """Weight-1 ternary tree has degree = 1 + 3*1 = 4."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        mu3 = (1, 2, 3)
        tree = (mu3, 1, 2, 3)
        deg = B.degree_on_basis((tree, (gen1, gen1, gen1)))
        assert deg == 4

    def test_weight2_binary_degree(self):
        """Weight-2 right-nested binary tree has degree = 2 + 3*1 = 5."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = (mu, 1, (mu, 2, 3))
        deg = B.degree_on_basis((tree, (gen1, gen1, gen1)))
        assert deg == 5

    def test_dact_weight1_binary_gives_arity2_generator(self):
        """d_act on weight-1 binary tree produces a leaf decorated by the arity-2 generator.

        α(bar corolla at μ) = μ ∈ Ass(2), and γ(μ; gen1, gen1) = gen2 in Free_Ass(k[1]).
        """
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        gen2 = ((1, 2), ((), ()))
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B((tree, (gen1, gen1)))
        result = elem.dalpha()
        assert result == B((1, (gen2,)))

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            ((_MU, 1, 2), (_GEN1, _GEN1)),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            ((_MU, 1, (_MU, 2, 3)), (_GEN1, _GEN1, _GEN1)),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            ((_MU, (_MU, 1, 2), 3), (_GEN1, _GEN1, _GEN1)),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            ((_MU3, 1, 2, 3), (_GEN1, _GEN1, _GEN1)),
        ],
    )
    def test_d_squared_zero(self, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of the bar complex with the free algebra."""
        B = _make_bar_complex(trivial_algebra=False)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_linear_combination_d_squared_zero(self):
        """d² = 0 on a linear combination of bar elements with the free algebra."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        mu = (1, 2)
        t1 = B(((mu, 1, (mu, 2, 3)), (gen1, gen1, gen1)))
        t2 = B(((mu, (mu, 1, 2), 3), (gen1, gen1, gen1)))
        combo = 3 * t1 - 2 * t2
        assert combo.boundary().boundary() == B.zero()

    def test_d2_tree_connects_to_ternary(self):
        """d_2 on weight-2 binary tree produces a weight-1 ternary term."""
        B = _make_bar_complex(trivial_algebra=False)
        gen1 = ((1,), ((),))
        mu = (1, 2)
        tree = (mu, 1, (mu, 2, 3))
        elem = B((tree, (gen1, gen1, gen1)))
        d2_result = elem.d2()
        has_ternary = False
        for (t, _a), _coeff in d2_result:
            if hasattr(t, "__len__") and len(t) == 4:  # (dec, leaf1, leaf2, leaf3)
                has_ternary = True
        assert has_ternary


# ===========================================================================
# TwistedCobarComplex tests
# ===========================================================================


def _make_cobar_complex(base_ring=QQ):
    """Build Ω_CoAss(k) -- cobar complex of the trivial 1-dim CoAss-coalgebra."""
    coalg = TrivialCoassCoalgebra(base_ring=base_ring)
    return TwistedCobarComplex(canonical_inclusion(CoAssociative), coalg)


class TestTwistedCobarComplex:
    """Tests for TwistedCobarComplex (cobar complex Ω_ι(V))."""

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_V(v)."""
        C = _make_cobar_complex()
        assert C.degree_on_basis((1, ((),))) == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree in Ω_C(V) has degree = -1 (deg_C(c)-1 = 0-1)."""
        C = _make_cobar_complex()
        # CoAssociative(2, QQ) basis key = (1,2) in degree 0
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        v_tuple = ((), ())
        deg = C.degree_on_basis((tree, v_tuple))
        assert deg == -1

    def test_d1_zero_for_trivial(self):
        """d_internal = 0 when the cooperad and module boundaries are both zero."""
        C = _make_cobar_complex()
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = C((tree, ((), ())))
        assert elem.d_internal() == C.zero()

    def test_d_internal_zero_for_trivial(self):
        """d_internal = 0 when cooperad and module boundaries are both zero."""
        C = _make_cobar_complex()
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = C((tree, ((), ())))
        assert elem.d_internal() == C.zero()

    def test_dalpha_on_leaf_expands(self):
        """d_α on a single leaf produces weight-1 cobar trees."""
        C = _make_cobar_complex()
        elem = C((1, ((),)))
        result = elem.dalpha()
        assert result != C.zero()

    def test_d2_on_weight1_zero(self):
        """d_2 on a weight-1 (single-vertex) tree is zero (no edge to split)."""
        C = _make_cobar_complex()
        c_dec = (1, 2)
        tree = (c_dec, 1, 2)
        elem = C((tree, ((), ())))
        assert elem.d2() == C.zero()

    _CDEC2 = (1, 2)
    _CDEC3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "key",
        [
            # single leaf
            (1, ((),)),
            # weight-1 binary tree
            (((1, 2), 1, 2), ((), ())),
        ],
    )
    @pytest.mark.xfail(
        reason="Pre-existing sign issue in TwistedCobarComplex._dalpha_on_basis: "
        "d_α sign convention needs leaf-module-degree-dependent correction "
        "for the cobar complex (see interleaved DFS sign analysis).",
        strict=True,
    )
    def test_d_squared_zero(self, key):
        """d² = 0 on elements of the cobar complex Ω_ι(V)."""
        C = _make_cobar_complex()
        elem = C(key)
        assert elem.boundary().boundary() == C.zero()

    @pytest.mark.xfail(
        reason="Pre-existing sign issue in TwistedCobarComplex._dalpha_on_basis.",
        strict=True,
    )
    def test_d_squared_zero_linear_combo(self):
        """d² = 0 on a linear combination of cobar elements."""
        C = _make_cobar_complex()
        t1 = C((1, ((),)))
        c_dec = (1, 2)
        t2 = C(((c_dec, 1, 2), ((), ())))
        combo = 2 * t1 - 3 * t2
        assert combo.boundary().boundary() == C.zero()


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
        B = TwistedBarComplex(canonical_projection(Lie), alg)
        # Lie(2, QQ) has basis key (1,) in degree 0
        lie_dec = (1,)
        tree = (lie_dec, 1, 2)
        elem = B((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()


# ===========================================================================
# TwistedBarComplex (with canonical inclusion ι) tests
# ===========================================================================


def _make_trivial_omega_bar_ass_algebra(base_ring=QQ):
    """Build a trivial ΩB(Ass)-algebra on the 1-dim module k.

    The structure map γ: ΩB(Ass)(n) ⊗ k^⊗n → k sends every element to 0.
    This is the ΩB(Ass)-algebra with trivial (zero) action.
    """
    bar_ass = BarConstruction(Associative)
    cobar_bar_ass = CobarConstruction(bar_ass)
    module = Commutative(1, base_ring=base_ring)

    def trivial_structure_map(p_elem, a_list):
        return module.zero()

    return OperadAlgebra(module, cobar_bar_ass, trivial_structure_map)


def _make_pullback_omega_bar_ass_algebra(base_ring=QQ):
    """Build a pullback ΩB(Ass)-algebra via the augmentation ε: ΩB(Ass) → Ass.

    The structure map γ(p; a_1,...,a_n) applies the trivial Ass-algebra action
    to ε(p), where ε sends:
    - Single-vertex cobar trees (bar_key, 1,...,n) with bar_key = (σ, 1,...,n)
      to σ ∈ Ass(n) (the P-decoration at the bar corolla).
    - All multi-vertex cobar trees and the unit to 0.

    Since the trivial Ass-algebra maps everything to the unit () of k, the
    action sends any single-corolla input to () (the product of all a_i), and
    multi-vertex inputs to 0.
    """
    bar_ass = BarConstruction(Associative)
    cobar_bar_ass = CobarConstruction(bar_ass)
    module = Commutative(1, base_ring=base_ring)

    def pullback_structure_map(p_elem, a_list):
        n = p_elem.arity()
        result = module.zero()
        for key, coeff in p_elem:
            # Only single-vertex cobar trees with a bar-corolla decoration
            # contribute (projecting through the augmentation ε: ΩB(Ass) → Ass).
            if is_leaf(key):
                if n == 1:
                    # Unit: ε(id) = id_Ass, act as identity
                    a_coeff = coeff
                    for a_elem in a_list:
                        for _ak, ac in a_elem:
                            a_coeff = a_coeff * ac
                    result += a_coeff * module(())
            else:
                v_arity = vertex_arity(key)
                if v_arity != n:
                    continue
                v_chs = children(key)
                if not all(is_leaf(c) for c in v_chs):
                    continue
                # Single-vertex cobar tree: decoration is a B(Ass)(n) key.
                # A bar corolla key has the form (ass_key, 1,...,n).
                bar_dec = decoration(key)
                if is_leaf(bar_dec):
                    continue
                bar_v_arity = vertex_arity(bar_dec)
                if bar_v_arity != n:
                    continue
                bar_chs = children(bar_dec)
                if not all(is_leaf(c) for c in bar_chs):
                    continue
                # bar_dec = (ass_key, 1,...,n) is a bar corolla; ε maps it to
                # ass_key ∈ Ass(n).  Apply the trivial Ass-action.
                a_coeff = coeff
                for a_elem in a_list:
                    for _ak, ac in a_elem:
                        a_coeff = a_coeff * ac
                result += a_coeff * module(())
        return result

    return OperadAlgebra(module, cobar_bar_ass, pullback_structure_map)


def _make_twisted_bar(base_ring=QQ):
    """Build B_ι(k) -- twisted bar complex of the trivial 1-dim ΩB(Ass)-algebra."""
    alg = _make_trivial_omega_bar_ass_algebra(base_ring=base_ring)
    return TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)


class TestTwistedBarComplexIota:
    """Tests for TwistedBarComplex with canonical inclusion ι (twisted bar complex B_ι(A))."""

    def test_construction(self):
        """TwistedBarComplex can be constructed from an ΩB(P)-algebra."""
        alg = _make_trivial_omega_bar_ass_algebra()
        B = TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)
        assert B is not None

    def test_single_leaf_degree(self):
        """Weight-0 element (single leaf) has degree = deg_A(a)."""
        B = _make_twisted_bar()
        elem = B((1, ((),)))
        assert elem.degree() == 0

    def test_weight1_binary_degree(self):
        """Weight-1 binary tree has degree = 1 (deg_P(μ)+1 = 0+1 = 1)."""
        B = _make_twisted_bar()
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B((tree, ((), ())))
        assert elem.degree() == 1

    def test_weight1_ternary_degree(self):
        """Weight-1 ternary tree has degree = 1."""
        B = _make_twisted_bar()
        mu3 = (1, 2, 3)
        tree = (mu3, 1, 2, 3)
        deg = B.degree_on_basis((tree, ((), (), ())))
        assert deg == 1

    def test_dalpha_trivial_algebra_zero(self):
        """d_alpha = 0 for the trivial ΩB(Ass)-algebra."""
        B = _make_twisted_bar()
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B((tree, ((), ())))
        assert elem.dalpha() == B.zero()

    def test_dalpha_pullback_algebra_nonempty(self):
        """d_alpha is non-zero for the pullback ΩB(Ass)-algebra (ε-pullback)."""
        alg = _make_pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)
        # Weight-1 binary tree: (μ; 1, 2) with leaves decorated by () ∈ k
        mu = (1, 2)
        tree = (mu, 1, 2)
        elem = B((tree, ((), ())))
        # d_alpha applies γ(ι([μ]); (), ()) = () (non-zero via ε-pullback)
        twist_result = elem.dalpha()
        # Should produce a single-leaf element
        assert twist_result == B((1, ((),)))

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            ((_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_trivial(self, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of B_ι(A) for the trivial ΩB(Ass)-algebra."""
        B = _make_twisted_bar()
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            # weight-1 binary tree: (μ; 1, 2)
            ((_MU, 1, 2), ((), ())),
            # weight-2 right-nested: (μ; 1, (μ; 2, 3))
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            # weight-2 left-nested: (μ; (μ; 1, 2), 3)
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            # weight-1 ternary tree: (μ₃; 1, 2, 3)
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_pullback(self, tree: tuple, a_tuple: tuple):
        """d² = 0 on tree elements of B_ι(A) for the ε-pullback ΩB(Ass)-algebra."""
        alg = _make_pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    def test_linear_combination_d_squared_zero(self):
        """d² = 0 on a linear combination of elements for the pullback algebra."""
        alg = _make_pullback_omega_bar_ass_algebra()
        B = TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)
        mu = (1, 2)
        t1 = B(((mu, 1, (mu, 2, 3)), ((), (), ())))
        t2 = B(((mu, (mu, 1, 2), 3), ((), (), ())))
        combo = 3 * t1 - 2 * t2
        assert combo.boundary().boundary() == B.zero()
