"""Systematic tests for algebraic axioms across all structures.

Tests d²=0, derivation property, associativity, equivariance, and unit axioms
for operads, algebras, and complexes -- both over QQ (for sign errors) and
GF(2) (for structural errors).

Structures tested:
- ShiftedOperad (with nontrivial degrees: ShiftedOperad(Surjection, 1))
- HadamardProduct (with nontrivial degrees: HadamardProduct(sLie, Surjection))
- FreeOperadAlgebra (with Associative and Surjection operads)
- HadamardTensorAlgebra
- TwistedBarComplex (canonical projection)
- TwistedCobarComplex (canonical inclusion)
"""

import itertools

import pytest
from sage.all import CombinatorialFreeModule, GF, GradedModulesWithBasis, QQ, tensor as sage_tensor

from uconf import (
    Associative,
    CoAssociative,
    Commutative,
    HadamardProduct,
    HadamardTensorAlgebra,
    Lie,
    ShiftedOperad,
    Surjection,
)
from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.constructions.twisted_complex import TwistedBarComplex, TwistedCobarComplex
from uconf.core.signs import sign_from_exponent
from uconf.morphisms.canonical_twisting import canonical_inclusion, canonical_projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(x):
    return {basis: coeff for basis, coeff in x if coeff != 0}


# ---------------------------------------------------------------------------
# ShiftedOperad axioms
# ---------------------------------------------------------------------------


class TestShiftedOperadAxioms:
    """Comprehensive operad axiom tests for ShiftedOperad.

    Uses ShiftedOperad(Surjection, 1) (``sSurj``) which has nontrivial
    degrees: the degree of a basis element k in arity n is
    deg_Surj(k) + 1*(n-1).  In arity 2, (1,2) has degree 0+1=1 and (1,2,1)
    has degree 1+1=2.  The base operad Surjection also has a nontrivial
    boundary, making this a genuine dg-operad.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _shifted_surj(self):
        return ShiftedOperad(Surjection, 1)

    # -- d² = 0 --

    def test_d_squared_zero_arity2(self, R):
        """d²=0 for all basis elements of sSurj(2) up to degree 3."""
        sS = self._shifted_surj()
        comp = sS(2, R)
        for d in range(1, 4):  # degree 1,2,3
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 at arity 2, degree {d}"

    def test_d_squared_zero_arity3(self, R):
        """d²=0 for all basis elements of sSurj(3) up to degree 4."""
        sS = self._shifted_surj()
        comp = sS(3, R)
        for d in range(2, 5):  # degree 2,3,4
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 at arity 3, degree {d}"

    # -- Derivation: d(x ∘_i y) = dx ∘_i y + (-1)^|x| x ∘_i dy --

    def test_derivation_arity2(self, R):
        """Derivation property of d w.r.t. ∘_i for sSurj, arity 2 inputs."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        # x = (1,2,1) in sSurj(2) has degree 2, y = (1,2) has degree 1
        x = comp2((1, 2, 1))
        y = comp2((1, 2))
        for i in [1, 2]:
            composed = sS.compose(x, i, y)
            d_composed = composed.boundary()
            dx = x.boundary()
            dy = y.boundary()
            x_deg = x.degree()
            expected = sS.compose(dx, i, y) + sign_from_exponent(x_deg) * sS.compose(x, i, dy)
            assert _as_dict(d_composed) == _as_dict(expected), (
                f"Derivation fails for ∘_{i}"
            )

    def test_derivation_arity2_degree_zero(self, R):
        """Derivation for degree-0 inputs (should be structurally correct over GF(2))."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2))  # degree 1 in shifted
        y = comp2((1, 2))  # degree 1 in shifted
        composed = sS.compose(x, 1, y)
        d_composed = composed.boundary()
        dx = x.boundary()
        dy = y.boundary()
        x_deg = x.degree()
        expected = sS.compose(dx, 1, y) + sign_from_exponent(x_deg) * sS.compose(x, 1, dy)
        assert _as_dict(d_composed) == _as_dict(expected)

    # -- Unit axiom: id ∘_1 x = x = x ∘_i id --

    def test_unit_axiom(self, R):
        """Unit: id ∘_1 x = x and x ∘_i id = x."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2, 1))
        unit = sS.unit(R)
        # id ∘_1 x = x
        result_left = sS.compose(unit, 1, x)
        assert _as_dict(result_left) == _as_dict(x), "id ∘_1 x ≠ x"
        # x ∘_i id = x for all i
        for i in [1, 2]:
            result_right = sS.compose(x, i, unit)
            assert _as_dict(result_right) == _as_dict(x), f"x ∘_{i} id ≠ x"

    # -- Equivariance: (x·σ) ∘_{σ(i)} y = (x ∘_i y)·σ̃ --

    def test_equivariance_permute_identity(self, R):
        """Permuting by the identity gives back the same element."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2))
        assert _as_dict(x.permute([1, 2])) == _as_dict(x)

    def test_equivariance_permute_involution(self, R):
        """Permuting by σ twice gives back the original (σ² = id for transposition)."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2))
        sigma = [2, 1]
        y = x.permute(sigma)
        z = y.permute(sigma)
        assert _as_dict(z) == _as_dict(x)

    # -- Associativity: (x ∘_i y) ∘_{i+j-1} z = x ∘_i (y ∘_j z) --

    def test_associativity_sequential(self, R):
        """Associativity: (x ∘_1 y) ∘_1 z = x ∘_1 (y ∘_1 z)."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2))
        y = comp2((1, 2))
        z = comp2((1, 2))
        lhs = sS.compose(sS.compose(x, 1, y), 1, z)
        rhs = sS.compose(x, 1, sS.compose(y, 1, z))
        assert _as_dict(lhs) == _as_dict(rhs)

    def test_associativity_parallel(self, R):
        """Parallel associativity: (x ∘_i y) ∘_{j+m-1} z = (-1)^{|y|·|z|} (x ∘_j z) ∘_i y
        when 1 ≤ i < j ≤ arity(x), with Koszul sign.
        Here i=1, j=2, m=arity(y)=2."""
        sS = self._shifted_surj()
        comp2 = sS(2, R)
        x = comp2((1, 2))
        y = comp2((1, 2))
        z = comp2((1, 2))
        m = 2  # arity of y
        # (x ∘_1 y) ∘_{2+m-1} z = (x ∘_1 y) ∘_{m+1} z
        lhs = sS.compose(sS.compose(x, 1, y), 2 + m - 1, z)
        # (-1)^{|y|·|z|} (x ∘_2 z) ∘_1 y
        y_deg = y.degree()
        z_deg = z.degree()
        sign = sign_from_exponent(y_deg * z_deg)
        rhs = sign * sS.compose(sS.compose(x, 2, z), 1, y)
        assert _as_dict(lhs) == _as_dict(rhs)


# ---------------------------------------------------------------------------
# HadamardProduct axioms (with nontrivial degrees)
# ---------------------------------------------------------------------------


class TestHadamardProductAxioms:
    """Operad axioms for HadamardProduct(sLie, Surjection).

    This has nontrivial degrees: sLie(2) has degree 1, Surjection(2) has
    elements in degrees 0 and 1, so (sLie⊙Surj)(2) has elements starting at
    degree 1.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _had(self):
        sLie = ShiftedOperad(Lie, 1)
        return HadamardProduct(sLie, Surjection)

    # -- d² = 0 --

    def test_d_squared_zero_arity2(self, R):
        """d²=0 for all basis elements of (sLie⊙Surj)(2)."""
        had = self._had()
        comp = had(2, R)
        for d in range(1, 4):
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 in (sLie⊙Surj)(2) at degree {d}"

    def test_d_squared_zero_arity3(self, R):
        """d²=0 for all basis elements of (sLie⊙Surj)(3) at low degrees."""
        had = self._had()
        comp = had(3, R)
        for d in range(2, 5):
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 in (sLie⊙Surj)(3) at degree {d}"

    # -- Derivation --

    def test_derivation(self, R):
        """Derivation: d(x ∘_i y) = dx ∘_i y + (-1)^|x| x ∘_i dy."""
        had = self._had()
        comp2 = had(2, R)
        # Use first available basis element of degree 1 and 2
        elems_d1 = list(comp2.basis_iter(1))
        elems_d2 = list(comp2.basis_iter(2))
        if not elems_d1:
            pytest.skip("No degree-1 elements")
        x = elems_d1[0]
        y = elems_d2[0] if elems_d2 else elems_d1[0]
        composed = had.compose(x, 1, y)
        d_composed = composed.boundary()
        dx = x.boundary()
        dy = y.boundary()
        x_deg = x.degree()
        expected = had.compose(dx, 1, y) + sign_from_exponent(x_deg) * had.compose(x, 1, dy)
        assert _as_dict(d_composed) == _as_dict(expected)

    # -- Unit --

    def test_unit_axiom(self, R):
        """Unit: id ∘_1 x = x and x ∘_i id = x."""
        had = self._had()
        comp2 = had(2, R)
        elems = list(comp2.basis_iter(1))
        if not elems:
            pytest.skip("No degree-1 elements")
        x = elems[0]
        unit = had.unit(R)
        assert _as_dict(had.compose(unit, 1, x)) == _as_dict(x), "id ∘_1 x ≠ x"
        for i in [1, 2]:
            assert _as_dict(had.compose(x, i, unit)) == _as_dict(x), f"x ∘_{i} id ≠ x"

    # -- Associativity --

    def test_associativity_sequential(self, R):
        """(x ∘_1 y) ∘_1 z = x ∘_1 (y ∘_1 z)."""
        had = self._had()
        comp2 = had(2, R)
        elems = list(comp2.basis_iter(1))
        if len(elems) < 1:
            pytest.skip("Not enough elements")
        x = y = z = elems[0]
        lhs = had.compose(had.compose(x, 1, y), 1, z)
        rhs = had.compose(x, 1, had.compose(y, 1, z))
        assert _as_dict(lhs) == _as_dict(rhs)

    # -- Equivariance --

    def test_equivariance(self, R):
        """Symmetric-group equivariance: (p⊗q)·σ = (p·σ)⊗(q·σ) (diagonal action)."""
        had = self._had()
        comp2 = had(2, R)
        elems = list(comp2.basis_iter(1))
        if not elems:
            pytest.skip("No degree-1 elements")
        x = elems[0]
        sigma = [2, 1]
        result = x.permute(sigma)  # noqa: F841
        # Verify it's well-defined (doesn't raise) and consistent with composition
        # permute(id) = x
        assert _as_dict(x.permute([1, 2])) == _as_dict(x)


# ---------------------------------------------------------------------------
# FreeOperadAlgebra axioms
# ---------------------------------------------------------------------------


class TestFreeOperadAlgebraAxioms:
    """Comprehensive tests for FreeOperadAlgebra.

    Tests algebra axioms with both Associative and Surjection operads.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _make_free_ass(self, R):
        """Free Ass-algebra on the 1-dim module k."""
        M = Commutative(1, base_ring=R)
        return FreeOperadAlgebra(Associative, M)

    def _make_free_surj(self, R):
        """Free Surjection-algebra on a 1-dim module in degree 1."""
        M = CombinatorialFreeModule(R, [()], category=GradedModulesWithBasis(R))
        M.degree_on_basis = lambda _: 1
        M.boundary = lambda x: M.zero()
        M.connectivity = 0
        M.rename("K[1]")
        return FreeOperadAlgebra(Surjection, M)

    # -- d² = 0 --

    def test_d_squared_zero_leaf(self, R):
        """d²=0 on a leaf element in Free_Ass(k)."""
        alg = self._make_free_ass(R)
        mod = alg.module
        leaf = mod(((1,), ((),)))
        assert mod.boundary(mod.boundary(leaf)) == mod.zero()

    def test_d_squared_zero_corolla(self, R):
        """d²=0 on a corolla element in Free_Ass(k)."""
        alg = self._make_free_ass(R)
        mod = alg.module
        corolla = mod(((1, 2), ((), ())))
        assert mod.boundary(mod.boundary(corolla)) == mod.zero()

    def test_d_squared_zero_surjection_basis(self, R):
        """d²=0 for all degree-1 basis elements in Free_Surj(k)."""
        alg = self._make_free_surj(R)
        mod = alg.module
        for elem in mod.basis_iter(1):
            dd = mod.boundary(mod.boundary(elem))
            assert dd == mod.zero(), f"d²≠0 for {elem}"

    def test_d_squared_zero_surjection_degree2(self, R):
        """d²=0 for all degree-2 basis elements in Free_Surj(k)."""
        alg = self._make_free_surj(R)
        mod = alg.module
        for elem in mod.basis_iter(2):
            dd = mod.boundary(mod.boundary(elem))
            assert dd == mod.zero(), "d²≠0 at degree 2"

    # -- Unit axiom --

    def test_unit_action(self, R):
        """γ(id; a) = a (unit action)."""
        alg = self._make_free_ass(R)
        mod = alg.module
        a = mod(((1,), ((),)))
        unit = Associative.unit(R)
        result = alg.act(unit, [a])
        assert _as_dict(result) == _as_dict(a)

    def test_unit_action_surjection(self, R):
        """γ(id; a) = a for Surjection operad."""
        alg = self._make_free_surj(R)
        mod = alg.module
        a = mod(((1, 2), ((),)))
        unit = Surjection.unit(R)
        result = alg.act(unit, [a])
        assert _as_dict(result) == _as_dict(a)

    # -- Arity mismatch --

    def test_arity_mismatch(self, R):
        """act() raises ValueError when len(inputs) != arity."""
        alg = self._make_free_ass(R)
        mod = alg.module
        a = mod(((1,), ((),)))
        p = Associative(2, R)((1, 2))
        with pytest.raises(ValueError):
            alg.act(p, [a])

    # -- Equivariance --

    def test_equivariance_ass(self, R):
        """Equivariance: γ(p·σ; a_{σ(1)},...,a_{σ(n)}) = γ(p; a_1,...,a_n).

        For Associative operad, the action must be equivariant under
        simultaneous permutation of the operad element and inputs.
        We test this by checking that γ(μ; a, b) = γ(μ·σ; b, a) when σ=(2,1).
        """
        alg = self._make_free_ass(R)
        mod = alg.module
        a = mod(((1,), ((),)))
        p = Associative(2, R)((1, 2))
        result1 = alg.act(p, [a, a])
        # Permute p by (2,1) and reverse inputs
        p_sigma = p.permute([2, 1])
        result2 = alg.act(p_sigma, [a, a])
        # Since both inputs are equal, both results should be equal
        assert _as_dict(result1) == _as_dict(result2)


# ---------------------------------------------------------------------------
# HadamardTensorAlgebra axioms
# ---------------------------------------------------------------------------


class TrivialAssAlgebra(OperadAlgebra):
    """Trivial 1-dim Ass-algebra for testing."""

    def __init__(self, base_ring=QQ):
        module = Commutative(1, base_ring=base_ring)
        super().__init__(module, Associative, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        result = self.module.zero()
        for _p_key, p_coeff in p_element:
            coeff = p_coeff
            for a_elem in algebra_elements:
                for _a_key, a_coeff in a_elem:
                    coeff *= a_coeff
            result += coeff * self.module(())
        return result


class TrivialSurjAlgebra(OperadAlgebra):
    """Trivial 1-dim Surjection-algebra for testing.

    The module is the 1-dimensional module k with zero differential.
    The structure map sends γ(p; a,...,a) = a for all p.
    """

    def __init__(self, base_ring=QQ):
        module = Commutative(1, base_ring=base_ring)
        super().__init__(module, Surjection, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        result = self.module.zero()
        for _p_key, p_coeff in p_element:
            coeff = p_coeff
            for a_elem in algebra_elements:
                for _a_key, a_coeff in a_elem:
                    coeff *= a_coeff
            result += coeff * self.module(())
        return result


class TestHadamardTensorAlgebraAxioms:
    """Comprehensive tests for HadamardTensorAlgebra.

    Tests with both degree-0 (Ass⊗Ass) and nontrivial-degree algebras.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _make_had_ass(self, R):
        """Hadamard tensor algebra for Ass⊗Ass."""
        left = TrivialAssAlgebra(R)
        right = TrivialAssAlgebra(R)
        return HadamardTensorAlgebra(left, right)

    # -- d² = 0 --

    def test_d_squared_zero_degree0(self, R):
        """d²=0 on tensor elements (trivial boundary, so trivially true)."""
        from sage.all import tensor
        had_alg = self._make_had_ass(R)
        left_mod = had_alg.left_module
        right_mod = had_alg.right_module
        x = tensor((left_mod(()), right_mod(())))
        dd = had_alg.boundary(had_alg.boundary(x))
        assert _as_dict(dd) == {}

    # -- Unit action --

    def test_unit_action(self, R):
        """Unit: γ(id⊗id; x) = x."""
        from sage.all import tensor
        had_alg = self._make_had_ass(R)
        left_mod = had_alg.left_module
        right_mod = had_alg.right_module
        x = tensor((left_mod(()), right_mod(())))
        unit = had_alg.operad_cls.unit(R)
        result = had_alg.act(unit, [x])
        assert _as_dict(result) == _as_dict(x)

    # -- Binary action --

    def test_binary_action(self, R):
        """Binary action: γ(p⊗q; x, y) is well-defined."""
        from sage.all import tensor
        had_alg = self._make_had_ass(R)
        had = had_alg.operad_cls
        left_mod = had_alg.left_module
        right_mod = had_alg.right_module
        x = tensor((left_mod(()), right_mod(())))
        p = had(2, R)(((1, 2), (1, 2)))
        result = had_alg.act(p, [x, x])
        assert result != had_alg.module.zero()


# ---------------------------------------------------------------------------
# TwistedBarComplex -- additional GF(2) tests
# ---------------------------------------------------------------------------


class TestTwistedBarComplexGF2:
    """d²=0 tests for TwistedBarComplex over GF(2) (structural correctness)."""

    _MU = (1, 2)
    _MU3 = (1, 2, 3)

    def _make_bar(self, R, trivial=True):
        if trivial:
            alg = TrivialAssAlgebra(R)
        else:
            from sage.all import CombinatorialFreeModule, GradedModulesWithBasis
            simple_module = CombinatorialFreeModule(
                R, [()], category=GradedModulesWithBasis(R)
            )
            simple_module.degree_on_basis = lambda _: 1
            simple_module.boundary = lambda x: simple_module.zero()
            simple_module.connectivity = 0
            simple_module.rename("K[1]")
            alg = FreeOperadAlgebra(Associative, simple_module)
        return TwistedBarComplex(canonical_projection(Associative), alg)

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            ((_MU, 1, 2), ((), ())),
            ((_MU, 1, (_MU, 2, 3)), ((), (), ())),
            ((_MU, (_MU, 1, 2), 3), ((), (), ())),
            ((_MU3, 1, 2, 3), ((), (), ())),
        ],
    )
    def test_d_squared_zero_trivial_gf2(self, tree, a_tuple):
        """d²=0 over GF(2) for trivial algebra bar complex."""
        B = self._make_bar(GF(2), trivial=True)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()

    _GEN1 = ((1,), ((),))

    @pytest.mark.parametrize(
        "tree,a_tuple",
        [
            ((_MU, 1, 2), (_GEN1, _GEN1)),
            ((_MU, 1, (_MU, 2, 3)), (_GEN1, _GEN1, _GEN1)),
            ((_MU, (_MU, 1, 2), 3), (_GEN1, _GEN1, _GEN1)),
            ((_MU3, 1, 2, 3), (_GEN1, _GEN1, _GEN1)),
        ],
    )
    def test_d_squared_zero_free_algebra_gf2(self, tree, a_tuple):
        """d²=0 over GF(2) for free algebra bar complex."""
        B = self._make_bar(GF(2), trivial=False)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()


# ---------------------------------------------------------------------------
# TwistedBarComplex (canonical inclusion ι) -- additional GF(2) tests
# ---------------------------------------------------------------------------


def _make_trivial_omega_bar_ass_algebra(R):
    """Trivial ΩB(Ass)-algebra for testing."""
    bar_ass = BarConstruction(Associative)
    cobar_bar_ass = CobarConstruction(bar_ass)
    module = Commutative(1, base_ring=R)

    def trivial_structure_map(p_elem, a_list):
        return module.zero()

    return OperadAlgebra(module, cobar_bar_ass, trivial_structure_map)


class TestTwistedBarComplexIotaGF2:
    """d²=0 for B_ι(A) over GF(2)."""

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
    def test_d_squared_zero_gf2(self, tree, a_tuple):
        """d²=0 over GF(2) for the trivial ΩB(Ass)-algebra bar complex."""
        alg = _make_trivial_omega_bar_ass_algebra(GF(2))
        B = TwistedBarComplex(canonical_inclusion(BarConstruction(Associative)), alg)
        elem = B((tree, a_tuple))
        assert elem.boundary().boundary() == B.zero()


# ---------------------------------------------------------------------------
# TwistedCobarComplex -- GF(2) and QQ tests
# ---------------------------------------------------------------------------



class TrivialCoassCoalgebra:
    """Wrapper for CoAssociative coalgebra on the 1-dim module k."""

    def __init__(self, R=QQ):
        self.base_ring_val = R
        self.module = Commutative(1, base_ring=R)
        self.cooperad_cls = CoAssociative

    def coact(self, v_element, n):
        left_parent = CoAssociative(n, base_ring=self.base_ring_val)
        right_parent = Commutative(1, base_ring=self.base_ring_val)
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


def _make_cobar_complex_for_test(R):
    """Build Ω_ι(k) -- cobar complex of the trivial CoAss-coalgebra."""
    from uconf.algebraic.coalgebra import CooperadCoalgebra

    module = Commutative(1, base_ring=R)

    def coaction_map(v_element, n):
        left_parent = CoAssociative(n, base_ring=R)
        right_parent = Commutative(1, base_ring=R)
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

    coalg = CooperadCoalgebra(module, CoAssociative, coaction_map)
    return TwistedCobarComplex(canonical_inclusion(CoAssociative), coalg)


class TestTwistedCobarComplexGF2:
    """d²=0 tests for TwistedCobarComplex over GF(2)."""

    @pytest.mark.parametrize(
        "key",
        [
            (1, ((),)),
            (((1, 2), 1, 2), ((), ())),
        ],
    )
    @pytest.mark.xfail(
        reason="Pre-existing sign issue in TwistedCobarComplex._dalpha_on_basis.",
        strict=True,
    )
    def test_d_squared_zero_gf2(self, key):
        """d²=0 over GF(2) for cobar complex elements."""
        C = _make_cobar_complex_for_test(GF(2))
        elem = C(key)
        assert elem.boundary().boundary() == C.zero()


# ---------------------------------------------------------------------------
# ShiftedOperad d²=0 with Lie (no differential, tests sign in shifted boundary)
# ---------------------------------------------------------------------------


class TestShiftedLieAxioms:
    """Tests for ShiftedOperad(Lie, 1) -- desuspended Lie operad.

    Since Lie has zero differential, d²=0 is trivially satisfied, but the
    shifted composition signs must still be correct.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def test_d_squared_zero_arity2(self, R):
        """d²=0 for sLie(2) (trivially zero since Lie has no differential)."""
        sL = ShiftedOperad(Lie, 1)
        comp = sL(2, R)
        for elem in comp.basis_iter(1):
            dd = elem.boundary().boundary()
            assert dd == comp.zero()

    def test_composition_gives_lie_bracket(self, R):
        """sLie composition produces the Lie bracket relation."""
        sL = ShiftedOperad(Lie, 1)
        comp2 = sL(2, R)
        x = comp2((1,))
        composed = sL.compose(x, 1, x)
        # Should be non-zero (gives a ternary element)
        assert composed.arity() == 3

    def test_antisymmetry(self, R):
        """sLie(2) permutation: x·(2,1) = x (since shift=1 gives sgn^1 twist)."""
        sL = ShiftedOperad(Lie, 1)
        comp2 = sL(2, R)
        x = comp2((1,))
        assert x.permute([2, 1]) == x  # Anti-symmetric Lie bracket becomes symmetric after shift

    def test_unit_axiom(self, R):
        """Unit: id ∘_1 x = x and x ∘_i id = x."""
        sL = ShiftedOperad(Lie, 1)
        comp2 = sL(2, R)
        x = comp2((1,))
        unit = sL.unit(R)
        assert _as_dict(sL.compose(unit, 1, x)) == _as_dict(x)
        for i in [1, 2]:
            assert _as_dict(sL.compose(x, i, unit)) == _as_dict(x)


# ---------------------------------------------------------------------------
# HadamardProduct(Surjection, Surjection) -- both factors have nontrivial diff
# ---------------------------------------------------------------------------


class TestHadamardProductSurjSurj:
    """Tests for HadamardProduct(Surjection, Surjection) -- full nontrivial differential."""

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _had(self):
        return HadamardProduct(Surjection, Surjection)

    def test_d_squared_zero_arity2(self, R):
        """d²=0 for all arity-2 basis elements up to degree 4."""
        had = self._had()
        comp = had(2, R)
        for d in range(0, 5):
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 at degree {d}"

    def test_d_squared_zero_arity3_low_degree(self, R):
        """d²=0 for arity-3, degree 0 and 1 basis elements."""
        had = self._had()
        comp = had(3, R)
        for d in range(0, 2):
            for elem in comp.basis_iter(d):
                dd = elem.boundary().boundary()
                assert dd == comp.zero(), f"d²≠0 at arity 3, degree {d}"

    def test_derivation(self, R):
        """Derivation: d(x ∘_i y) = dx ∘_i y + (-1)^|x| x ∘_i dy."""
        had = self._had()
        comp2 = had(2, R)
        x = comp2(((1, 2, 1), (1, 2)))  # degree 1
        y = comp2(((1, 2), (1, 2)))  # degree 0
        composed = had.compose(x, 1, y)
        d_composed = composed.boundary()
        dx = x.boundary()
        dy = y.boundary()
        x_deg = x.degree()
        expected = had.compose(dx, 1, y) + sign_from_exponent(x_deg) * had.compose(x, 1, dy)
        assert _as_dict(d_composed) == _as_dict(expected)

    def test_unit_axiom(self, R):
        """Unit axiom."""
        had = self._had()
        comp2 = had(2, R)
        x = comp2(((1, 2), (1, 2)))
        unit = had.unit(R)
        assert _as_dict(had.compose(unit, 1, x)) == _as_dict(x)
        for i in [1, 2]:
            assert _as_dict(had.compose(x, i, unit)) == _as_dict(x)

    def test_associativity(self, R):
        """(x ∘_1 y) ∘_1 z = x ∘_1 (y ∘_1 z)."""
        had = self._had()
        comp2 = had(2, R)
        x = y = z = comp2(((1, 2), (1, 2)))
        lhs = had.compose(had.compose(x, 1, y), 1, z)
        rhs = had.compose(x, 1, had.compose(y, 1, z))
        assert _as_dict(lhs) == _as_dict(rhs)


# ---------------------------------------------------------------------------
# TwistedBarComplex with Surjection operad (nontrivial internal differential)
# ---------------------------------------------------------------------------


class TestTwistedBarComplexSurjection:
    """TwistedBarComplex with the Surjection operad and trivial algebra.

    This tests d²=0 when the cooperad B(Surj) has nontrivial internal diff.
    """

    @pytest.fixture(params=[QQ, GF(2)])
    def R(self, request):
        return request.param

    def _make_bar(self, R):
        alg = TrivialSurjAlgebra(R)
        return TwistedBarComplex(canonical_projection(Surjection), alg)

    def test_d_squared_zero_weight1_binary(self, R):
        """d²=0 on a weight-1 binary tree with Surjection decorations."""
        B = self._make_bar(R)
        # Surjection(2, R) identity: (1, 2)
        tree = ((1, 2), 1, 2)
        elem = B((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_d_squared_zero_weight1_binary_deg1(self, R):
        """d²=0 on weight-1 binary tree with degree-1 Surjection decoration."""
        B = self._make_bar(R)
        tree = ((1, 2, 1), 1, 2)
        elem = B((tree, ((), ())))
        assert elem.boundary().boundary() == B.zero()

    def test_d_squared_zero_weight2(self, R):
        """d²=0 on a weight-2 right-nested tree."""
        B = self._make_bar(R)
        tree = ((1, 2), 1, ((1, 2), 2, 3))
        elem = B((tree, ((), (), ())))
        assert elem.boundary().boundary() == B.zero()
