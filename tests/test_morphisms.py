"""Tests for operad morphisms, pullback algebras, and the E-comodule morphism.

Covers:
- :class:`uconf.core.morphism.OperadMorphism`
- :class:`uconf.core.morphism.PullbackAlgebra`
- ``ass_to_com`` (augmentation Ass → Com)
- ``lie_to_ass`` (PBW inclusion Lie → Ass)
- ``make_e_comodule_morphism`` (Δ: Ω(C) → E ⊗ Ω(C))
"""

import pytest
from sage.all import QQ, SymmetricGroup

from uconf import (
    Associative,
    BarConstruction,
    BarrattEccles,
    CobarConstruction,
    Commutative,
    HadamardProduct,
    Lie,
    OperadAlgebra,
    OperadMorphism,
    ass_to_com,
    lie_to_ass,
    make_e_comodule_morphism,
)
from uconf.algebraic.pullback_algebra import PullbackAlgebra


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


# ===========================================================================
# ass_to_com tests
# ===========================================================================


class TestAssToComMorphism:
    """Tests for the augmentation morphism Ass → Com."""

    def test_unit_preservation(self):
        """f(id_Ass) = id_Com."""
        unit_ass = Associative.unit(QQ)
        result = ass_to_com(unit_ass)
        unit_com = Commutative.unit(QQ)
        assert _as_dict(result) == _as_dict(unit_com)

    def test_on_identity_permutation(self):
        """f((1,2,3)) = () in Com(3)."""
        x = Associative(3, QQ)((1, 2, 3))
        result = ass_to_com(x)
        expected = Commutative(3, QQ)(())
        assert _as_dict(result) == _as_dict(expected)

    def test_on_transposition(self):
        """f((2,1)) = () in Com(2)."""
        x = Associative(2, QQ)((2, 1))
        result = ass_to_com(x)
        expected = Commutative(2, QQ)(())
        assert _as_dict(result) == _as_dict(expected)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_equivariance(self, n):
        """f(x·σ) = f(x)·σ for Ass → Com (Com action is trivial)."""
        x = Associative(n, QQ)(tuple(range(1, n + 1)))
        S_n = SymmetricGroup(n)
        for sigma in S_n:
            perm_list = list(sigma.tuple())
            fx = ass_to_com(x.permute(perm_list))
            fperm = ass_to_com(x).permute(perm_list)
            assert _as_dict(fx) == _as_dict(fperm)

    @pytest.mark.parametrize("n", [2, 3])
    def test_compose_compatibility(self, n):
        """f(x ∘_i y) = f(x) ∘_i f(y) for all valid i."""
        x = Associative(n, QQ)(tuple(range(1, n + 1)))
        y = Associative(2, QQ)((1, 2))
        for i in range(1, n + 1):
            composed = Associative.compose(x, i, y)
            lhs = ass_to_com(composed)
            rhs = Commutative.compose(ass_to_com(x), i, ass_to_com(y))
            assert _as_dict(lhs) == _as_dict(rhs)

    def test_chain_map(self):
        """f(∂x) = ∂f(x) — trivial since both differentials are zero."""
        x = Associative(3, QQ)((2, 1, 3))
        assert _as_dict(ass_to_com(x.boundary())) == _as_dict(ass_to_com(x).boundary())

    def test_on_zero(self):
        """f(0) = 0."""
        zero = Associative(2, QQ).zero()
        result = ass_to_com(zero)
        assert not result  # zero element is falsy

    def test_on_linear_combination(self):
        """f is linear: f(a·x + b·y) = a·f(x) + b·f(y)."""
        A2 = Associative(2, QQ)
        x = A2((1, 2))
        y = A2((2, 1))
        combo = 3 * x + 5 * y
        result = ass_to_com(combo)
        expected = 8 * Commutative(2, QQ)(())
        assert _as_dict(result) == _as_dict(expected)


# ===========================================================================
# lie_to_ass tests
# ===========================================================================


class TestLieToAssMorphism:
    """Tests for the PBW inclusion Lie → Ass."""

    def test_unit_preservation(self):
        """f(id_Lie) = id_Ass."""
        unit_lie = Lie.unit(QQ)
        result = lie_to_ass(unit_lie)
        unit_ass = Associative.unit(QQ)
        assert _as_dict(result) == _as_dict(unit_ass)

    def test_bracket_to_commutator(self):
        """[x1, x2] maps to (1,2) - (2,1)."""
        bracket = Lie(2, QQ)((1,))
        result = lie_to_ass(bracket)
        A2 = Associative(2, QQ)
        expected = A2((1, 2)) - A2((2, 1))
        assert _as_dict(result) == _as_dict(expected)

    @pytest.mark.parametrize("n", [2, 3])
    def test_equivariance(self, n):
        """f(x·σ) = f(x)·σ for Lie → Ass."""
        L = Lie(n, QQ)
        basis = list(L.basis_iter(0))
        if not basis:
            return
        x = basis[0]
        S_n = SymmetricGroup(n)
        for sigma in S_n:
            perm_list = list(sigma.tuple())
            lhs = lie_to_ass(x.permute(perm_list))
            rhs = lie_to_ass(x).permute(perm_list)
            assert _as_dict(lhs) == _as_dict(rhs), f"Equivariance failed for sigma={perm_list}"

    @pytest.mark.parametrize("i", [1, 2])
    def test_compose_compatibility(self, i):
        """f(x ∘_i y) = f(x) ∘_i f(y) for the Lie bracket."""
        x = Lie(2, QQ)((1,))
        y = Lie(2, QQ)((1,))
        composed = Lie.compose(x, i, y)
        lhs = lie_to_ass(composed)
        rhs = Associative.compose(lie_to_ass(x), i, lie_to_ass(y))
        assert _as_dict(lhs) == _as_dict(rhs)

    def test_chain_map(self):
        """f(∂x) = ∂f(x) — both differentials are zero."""
        x = Lie(3, QQ)((1, 2))
        assert _as_dict(lie_to_ass(x.boundary())) == _as_dict(lie_to_ass(x).boundary())

    def test_on_zero(self):
        """f(0) = 0."""
        zero = Lie(2, QQ).zero()
        result = lie_to_ass(zero)
        assert not result

    def test_arity3_bracket(self):
        """[x1, [x2, x3]] maps to correct commutator expansion."""
        x = Lie(3, QQ)((1, 2))
        result = lie_to_ass(x)
        A3 = Associative(3, QQ)
        # [x1, [x2, x3]] = x1*x2*x3 - x1*x3*x2 - x2*x3*x1 + x3*x2*x1
        expected = A3((1, 2, 3)) - A3((1, 3, 2)) - A3((2, 3, 1)) + A3((3, 2, 1))
        assert _as_dict(result) == _as_dict(expected)


# ===========================================================================
# Composition of morphisms: Lie → Ass → Com
# ===========================================================================


class TestComposedMorphism:
    """Test composition of Lie → Ass → Com."""

    def test_composed_on_bracket(self):
        """Lie bracket maps to zero via Lie → Ass → Com."""
        bracket = Lie(2, QQ)((1,))
        # [x1,x2] -> (1,2)-(2,1) -> 1*()-1*() = 0
        result = ass_to_com(lie_to_ass(bracket))
        assert not result

    def test_composed_unit(self):
        """Unit is preserved through composition."""
        result = ass_to_com(lie_to_ass(Lie.unit(QQ)))
        assert _as_dict(result) == _as_dict(Commutative.unit(QQ))

    def test_composed_arity3(self):
        """All Lie basis elements in arity 3 map to zero in Com."""
        L3 = Lie(3, QQ)
        for elem in L3.basis_iter(0):
            result = ass_to_com(lie_to_ass(elem))
            assert not result


# ===========================================================================
# PullbackAlgebra tests
# ===========================================================================


class TestPullbackAlgebra:
    """Tests for the pullback algebra construction."""

    def _make_trivial_com_algebra(self):
        """Build a trivial Com-algebra on the 1-dim module k."""
        module = Commutative(1, QQ)

        def structure_map(p_element, algebra_elements):
            result = module.zero()
            for _, p_coeff in p_element:
                coeff = p_coeff
                for a in algebra_elements:
                    for _, a_coeff in a:
                        coeff *= a_coeff
                result += coeff * module(())
            return result

        return OperadAlgebra(module, Commutative, structure_map)

    def test_pullback_ass_to_com(self):
        """Pull back a Com-algebra along Ass → Com to get an Ass-algebra."""
        com_alg = self._make_trivial_com_algebra()
        ass_alg = PullbackAlgebra(ass_to_com, com_alg)
        assert ass_alg.operad_cls is Associative

        # Act with an Ass(2) element
        mu = Associative(2, QQ)((1, 2))
        a = com_alg.module(())
        result = ass_alg.act(mu, [a, a])
        assert _as_dict(result) == _as_dict(a)

    def test_pullback_lie_via_ass_to_com(self):
        """Pull back a Com-algebra along Lie → Ass → Com gives zero bracket."""
        com_alg = self._make_trivial_com_algebra()

        # Compose the morphisms: make a Lie-algebra from the Ass-algebra
        composed = OperadMorphism(
            Lie,
            Commutative,
            lambda elem: ass_to_com(lie_to_ass(elem)),
        )
        lie_alg = PullbackAlgebra(composed, com_alg)

        # The Lie bracket [a, a] should be zero
        bracket = Lie(2, QQ)((1,))
        a = com_alg.module(())
        result = lie_alg.act(bracket, [a, a])
        assert not result

    def test_pullback_unit_axiom(self):
        """Pullback preserves unit axiom: γ_P(id; a) = a."""
        com_alg = self._make_trivial_com_algebra()
        ass_alg = PullbackAlgebra(ass_to_com, com_alg)
        a = com_alg.module(())
        unit = Associative.unit(QQ)
        result = ass_alg.act(unit, [a])
        assert _as_dict(result) == _as_dict(a)

    def test_pullback_boundary(self):
        """Pullback boundary delegates to algebra boundary."""
        com_alg = self._make_trivial_com_algebra()
        ass_alg = PullbackAlgebra(ass_to_com, com_alg)
        a = com_alg.module(())
        result = ass_alg.boundary(a)
        assert not result


# ===========================================================================
# E-comodule morphism tests
# ===========================================================================


class TestEComoduleMorphism:
    """Tests for the operad morphism Δ: Ω(C) → E ⊗ Ω(C)."""

    def _setup(self, n=2):
        """Return (cobar_factory, target_factory, Delta, cobar_n, target_n)."""
        HLE = HadamardProduct(Lie, BarrattEccles)
        BH = BarConstruction(HLE)
        OBH = CobarConstruction(BH)
        Delta = make_e_comodule_morphism(BH)
        OBHn = OBH(n, QQ)
        target = HadamardProduct(BarrattEccles, OBH)
        target_n = target(n, QQ)
        return OBH, target, Delta, OBHn, target_n

    def test_unit_preservation(self):
        """Δ(unit) = unit of E ⊗ Ω(C)."""
        HLE = HadamardProduct(Lie, BarrattEccles)
        BH = BarConstruction(HLE)
        OBH = CobarConstruction(BH)
        Delta = make_e_comodule_morphism(BH)
        target = HadamardProduct(BarrattEccles, OBH)

        unit_cobar = OBH.unit(QQ)
        result = Delta(unit_cobar)
        unit_target = target.unit(QQ)
        assert _as_dict(result) == _as_dict(unit_target)

    def test_output_is_hadamard_element(self):
        """Δ(x) is an element of HadamardProduct(BE, Ω(C))(n)."""
        OBH, target, Delta, OBHn, target_n = self._setup(n=2)

        # Find a degree-1 generator
        BH = BarConstruction(HadamardProduct(Lie, BarrattEccles))
        BH2 = BH(2, QQ)
        for elem in BH2.planar_basis_it(1):
            tree_key = list(elem.support())[0]
            cobar_tree = (tree_key,) + tuple(range(1, 3))
            cobar_elem = OBHn(cobar_tree)
            result = Delta(cobar_elem)
            assert result.parent() == target_n or result.parent().arity() == 2
            break

    def test_on_zero(self):
        """Δ(0) = 0."""
        OBH, target, Delta, OBHn, target_n = self._setup(n=2)
        result = Delta(OBHn.zero())
        assert not result

    def test_generator_matches_e_comodule_on_generator(self):
        """On generators, Δ agrees with e_comodule_on_generator."""
        from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator

        HLE = HadamardProduct(Lie, BarrattEccles)
        BH = BarConstruction(HLE)
        OBH = CobarConstruction(BH)
        Delta = make_e_comodule_morphism(BH)

        BH2 = BH(2, QQ)
        OBH2 = OBH(2, QQ)
        planar_elems = list(BH2.planar_basis_it(1))
        assert len(planar_elems) > 0

        for elem in planar_elems[:2]:
            dec_key = list(elem.support())[0]
            # Build single-vertex cobar tree
            cobar_tree = (dec_key,) + tuple(range(1, 3))
            cobar_elem = OBH2(cobar_tree)

            # e_comodule_on_generator returns E(n) ⊗ C(n) (cooperad level).
            # Embed the C keys into Ω(C) as single-vertex cobar trees to compare
            # with the Hadamard product result from Delta.
            tensor_result = e_comodule_on_generator(elem)

            # Delta result (in HadamardProduct format)
            had_result = Delta(cobar_elem)

            # Compare: embed cooperad keys as cobar trees and convert to Hadamard keys
            tensor_dict = {}
            for tensor_basis, coeff in tensor_result:
                be_key, coop_key = tensor_basis
                # Embed cooperad key as single-vertex cobar tree
                cobar_key_raw = (coop_key,) + tuple(range(1, 3))
                # Normalize the cobar tree key
                cobar_normalized = OBH2(cobar_key_raw)
                for ck, cc in cobar_normalized:
                    had_key = (be_key, ck)
                    tensor_dict[had_key] = tensor_dict.get(had_key, QQ.zero()) + coeff * cc
            tensor_dict = {k: v for k, v in tensor_dict.items() if v != 0}

            had_dict = _as_dict(had_result)
            assert tensor_dict == had_dict, (
                f"Generator mismatch: tensor={tensor_dict}, had={had_dict}"
            )
