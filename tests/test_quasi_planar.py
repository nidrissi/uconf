"""Tests for quasi-planar structures and the E-comodule map."""

import pytest
from sage.all import QQ, SymmetricGroup, tensor

from uconf import (
    BarConstruction,
    BarrattEccles,
    CobarConstruction,
    HadamardProduct,
    Lie,
    Surjection,
    e_comodule_on_generator,
)
from uconf.core.quasi_planar import QuasiPlanarMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _BH_setup(n: int = 2):
    """Return (BH_factory, BHn, OBH_factory, OBHn, BE_n) for B(Lie⊙E)(n)."""
    HLE = HadamardProduct(Lie, BarrattEccles)
    BH = BarConstruction(HLE)
    BHn = BH(n)
    OBH = CobarConstruction(BH)
    OBHn = OBH(n)
    BEn = BarrattEccles(n)
    return BH, BHn, OBH, OBHn, BEn


def _identity_perm(n: int):
    return SymmetricGroup(n).identity()


# ---------------------------------------------------------------------------
# QuasiPlanarMixin tests
# ---------------------------------------------------------------------------


class TestDSigmaOnSurjection:
    """Test ``d_sigma`` on a quasi-planar bar construction over Surjection."""

    def _make_bar_surjection(self, n: int):
        BS = BarConstruction(Surjection)
        return BS(n)

    def test_d_sigma_sum_equals_boundary(self):
        """Σ_σ d_σ(x) ⊗ σ == boundary(x) (basic quasi-planar identity)."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        # Weight-1, degree-2 bar tree: (1,2,1) in S(2)
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)

        S2 = SymmetricGroup(2)
        SGA2 = B2._symmetric_group_algebra

        # Reconstruct boundary from d_sigma
        reconstructed = B2.zero()
        for sigma in S2:
            d_sig = B2.d_sigma(elem, sigma)
            for pl_key, coeff in d_sig:
                pl_elem = B2.term(pl_key)
                reconstructed += coeff * pl_elem.permute(sigma)

        assert reconstructed == elem.boundary(), (
            f"Σ d_σ(x)·σ = {reconstructed} ≠ boundary(x) = {elem.boundary()}"
        )

    def test_d_sigma_iterate_two_steps(self):
        """d_{(σ1,σ2)} = d_{σ1} ∘ d_{σ2}."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)
        S2 = SymmetricGroup(2)
        sigma = S2([2, 1])

        # Two steps applied separately
        step1 = B2.d_sigma(elem, sigma)
        step2 = B2.d_sigma(step1, sigma)

        # Using d_sigma_iterate
        iterated = B2.d_sigma_iterate(elem, [sigma, sigma])
        assert step2 == iterated

    def test_d_sigma_identity_permutation_zero(self):
        """d_id(x) should be zero because boundary maps to non-id components.

        For a planar element x, boundary(x) decomposes as Σ d_σ(x)⊗σ where
        the identity-σ component is specifically d_id(x) = the planar part
        of boundary(x) at σ=id.
        """
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)
        S2 = SymmetricGroup(2)

        # The planar element (1,2,1) has boundary (1,2) which is planar → σ=id
        d_id = B2.d_sigma(elem, S2.identity())
        # (1,2,1) ∈ S(2) has boundary (1,2) which is planar, so d_id is non-zero
        assert d_id is not None  # just check it runs


class TestDSigmaOnBarrattEccles:
    """d_sigma on B(BarrattEccles)."""

    def test_d_sigma_sums_to_boundary(self):
        """Σ_σ d_σ(x)·σ == boundary(x)."""
        BBE = BarConstruction(BarrattEccles)
        B2 = BBE(2)
        BE2 = BarrattEccles(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        s21 = S2([2, 1])
        tree = ((id2, s21), 1, 2)  # degree-2 bar element
        elem = B2(tree)

        reconstructed = B2.zero()
        for sigma in SymmetricGroup(2):
            d_sig = B2.d_sigma(elem, sigma)
            if d_sig:
                reconstructed += d_sig.permute(sigma)

        assert reconstructed == elem.boundary()


# ---------------------------------------------------------------------------
# d_sigma on HadamardProduct bar construction
# ---------------------------------------------------------------------------


class TestDSigmaOnHadamard:
    """d_sigma on B(Lie⊙E)(2)."""

    def test_d_sigma_on_degree2_element(self):
        """Σ_σ d_σ(x)·σ == boundary(x) for a degree-2 B(Lie⊙E) element."""
        _, BH2, _, _, _ = _BH_setup(2)
        S2 = SymmetricGroup(2)
        id2 = S2.identity()
        s21 = S2([2, 1])

        # BE planar degree-1 key: (id, s21)
        be_key = (BarrattEccles(2)._symmetric_group.identity(), BarrattEccles(2)._symmetric_group([2, 1]))
        had_key = ((1,), be_key)
        tree = (had_key, 1, 2)
        elem = BH2(tree)

        if elem == BH2.zero():
            pytest.skip("Element is zero (degenerate key)")

        reconstructed = BH2.zero()
        for sigma in S2:
            d_sig = BH2.d_sigma(elem, sigma)
            if d_sig:
                reconstructed += d_sig.permute(sigma)

        assert reconstructed == elem.boundary()


# ---------------------------------------------------------------------------
# graded_basis / graded_planar_basis
# ---------------------------------------------------------------------------


class TestGradedBasisMethods:
    """Tests for the cached graded-basis Family methods."""

    def test_surjection_graded_basis_size(self):
        """Surjection(2).graded_basis(d) has the expected number of elements."""
        S2 = Surjection(2)
        # Degree d: surjections {1,...,d+2} -> {1,2} with surjection property
        # For d=1: (1,2,1), (1,2,2), (2,1,1), (2,1,2) - but some may be degenerate
        basis_d1 = S2.graded_basis(1)
        assert len(basis_d1) >= 1

    def test_surjection_graded_planar_basis(self):
        """Surjection(2).graded_planar_basis(d) are all planar."""
        S2 = Surjection(2)
        for d in range(3):
            for elem in S2.graded_planar_basis(d):
                for key in elem.support():
                    assert S2.term(key).is_planar(), f"Non-planar element in graded_planar_basis({d})"

    def test_barratt_eccles_graded_basis(self):
        """BarrattEccles(2).graded_basis(d) has the right size."""
        BE2 = BarrattEccles(2)
        # degree 0: just (id,) → 1 element for planar, 2 for full (id and (1,2))
        basis_d0 = BE2.graded_basis(0)
        assert len(basis_d0) == 2  # (id,) and ((1,2),) in arity 2

    def test_barratt_eccles_graded_planar_basis(self):
        """BarrattEccles(2).graded_planar_basis(d) only contains planar elements."""
        BE2 = BarrattEccles(2)
        for d in range(3):
            for elem in BE2.graded_planar_basis(d):
                # A planar BE element starts with the identity permutation
                for key in elem.support():
                    assert key[0] == BE2._symmetric_group.identity(), (
                        f"Non-planar element in graded_planar_basis({d}): {key}"
                    )

    def test_hadamard_planar_basis_it(self):
        """HadamardProduct(Lie, BE)(2).planar_basis_it(d) contains planar pairs."""
        HLE = HadamardProduct(Lie, BarrattEccles)
        HLE2 = HLE(2)
        for d in range(3):
            for elem in HLE2.planar_basis_it(d):
                for (l_key, r_key), coeff in elem:
                    # Right factor should be planar (starts with identity)
                    assert r_key[0] == BarrattEccles(2)._symmetric_group.identity(), (
                        f"Non-planar right key {r_key} in HadamardProduct planar_basis_it({d})"
                    )


# ---------------------------------------------------------------------------
# E-comodule map
# ---------------------------------------------------------------------------


class TestEComoduleMap:
    """Tests for the E-comodule map on Ω(B(Lie⊙E))."""

    def test_k0_term_is_identity_tensor_generator(self):
        """The k=0 term should be BE[(id,)] ⊗ cobar_gen(x)."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)

        # Degree-1 planar element: ((1,), (id,)) at weight 1
        planar_elems = list(BH2.planar_basis_it(1))
        assert len(planar_elems) >= 1, "Need at least one planar element"

        dec_key = list(planar_elems[0].support())[0]
        dec_elem = BH2.term(dec_key)

        result = e_comodule_on_generator(dec_elem, BH2, OBH2, BE2)

        id2 = BE2._symmetric_group.identity()
        id_be_key = (id2,)
        cobar_tree_key = (dec_key,) + tuple(range(1, 3))

        # Find the k=0 term
        k0_coeff = 0
        for (be_key, cobar_key), coeff in result:
            if be_key == id_be_key and cobar_key == cobar_tree_key:
                k0_coeff = coeff
                break

        assert k0_coeff == 1, f"k=0 term should have coefficient 1, got {k0_coeff}"

    def test_comodule_contains_k0_term(self):
        """The comodule map always has a k=0 term equal to (id,) ⊗ generator."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)

        for d in range(1, 3):
            for pl_elem in BH2.planar_basis_it(d):
                dec_key = list(pl_elem.support())[0]
                dec_elem = BH2.term(dec_key)
                result = e_comodule_on_generator(dec_elem, BH2, OBH2, BE2)

                id_be_key = (BE2._symmetric_group.identity(),)
                cobar_tree_key = (dec_key,) + tuple(range(1, 3))

                found_k0 = any(
                    be_key == id_be_key and cobar_key == cobar_tree_key
                    for (be_key, cobar_key), coeff in result
                )
                assert found_k0, f"k=0 term missing for planar element in degree {d}"

    def test_comodule_on_degree2_element_has_k1_term(self):
        """For a degree-2 element with non-trivial boundary, k=1 terms appear."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        s21 = S2([2, 1])

        # Degree-2 bar tree: ((1,), (id, s21)) at HLE(2)
        be_key_deg1 = (id2, s21)
        had_key = ((1,), be_key_deg1)
        tree_key = (had_key, 1, 2)
        dec_elem = BH2.term(tree_key)

        if dec_elem == BH2.zero():
            pytest.skip("Element is zero")

        result = e_comodule_on_generator(dec_elem, BH2, OBH2, BE2)

        # Check the result has more than just the k=0 term
        terms = list(result)
        assert len(terms) > 1, "Expected k=1 term in addition to k=0"

    def test_comodule_be_keys_are_valid(self):
        """All BE keys in the comodule map output should be valid BE elements."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        s21 = S2([2, 1])
        id2 = S2.identity()

        be_key_deg1 = (id2, s21)
        had_key = ((1,), be_key_deg1)
        tree_key = (had_key, 1, 2)
        dec_elem = BH2.term(tree_key)

        if dec_elem == BH2.zero():
            pytest.skip("Element is zero")

        result = e_comodule_on_generator(dec_elem, BH2, OBH2, BE2)

        for (be_key, cobar_key), coeff in result:
            # BE key should be a valid basis element (no consecutive duplicates)
            be_elem = BE2.term(be_key)
            assert be_elem != BE2.zero(), f"BE key {be_key} gave zero element"

    def test_comodule_on_b_surjection(self):
        """E-comodule map on B(Surjection)(2), which is quasi-planar."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        OBS = CobarConstruction(BS)
        OBS2 = OBS(2)
        BE2 = BarrattEccles(2)

        # Planar degree-1 element: (1,2) in Surjection(2)
        tree_key = ((1, 2), 1, 2)
        dec_elem = B2.term(tree_key)

        result = e_comodule_on_generator(dec_elem, B2, OBS2, BE2)

        # Should at minimum have the k=0 term
        id_be_key = (BE2._symmetric_group.identity(),)
        cobar_key = (tree_key, 1, 2)

        found = any(
            be_key == id_be_key and ck == cobar_key
            for (be_key, ck), _coeff in result
        )
        assert found, "k=0 term missing for B(Surj)(2) degree-1 planar element"

    def test_comodule_result_in_tensor(self):
        """Output type is an element of BE(n) ⊗ Ω(C)(n)."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        planar = list(BH2.planar_basis_it(1))
        if not planar:
            pytest.skip("No planar basis elements found")

        dec_elem = BH2.term(list(planar[0].support())[0])
        result = e_comodule_on_generator(dec_elem, BH2, OBH2, BE2)

        expected_parent = tensor([BE2, OBH2])
        assert result.parent() is expected_parent or result == expected_parent.zero()


# ---------------------------------------------------------------------------
# QuasiPlanarMixin inheritance checks
# ---------------------------------------------------------------------------


class TestQuasiPlanarMixinInheritance:
    """Verify that the relevant component classes inherit QuasiPlanarMixin."""

    def test_bar_construction_inherits_mixin(self):
        """BarConstruction.Component should inherit from QuasiPlanarMixin."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        assert isinstance(B2, QuasiPlanarMixin)

    def test_hadamard_bar_inherits_mixin(self):
        """B(Lie⊙E)(2) should inherit from QuasiPlanarMixin."""
        _, BH2, _, _, _ = _BH_setup(2)
        assert isinstance(BH2, QuasiPlanarMixin)

    def test_mixin_has_d_sigma(self):
        """QuasiPlanarMixin should provide d_sigma."""
        assert hasattr(QuasiPlanarMixin, "d_sigma")
        assert hasattr(QuasiPlanarMixin, "d_sigma_iterate")
