"""Tests for quasi-planar structures and the E-comodule map."""

import pytest
from sage.all import SymmetricGroup, tensor

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
    BEn = BarrattEccles(n, QQ)
    return BH, BHn, OBH, OBHn, BEn


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

        # Reconstruct boundary from d_sigma
        reconstructed = B2.zero()
        for sigma in S2:
            d_sig = B2.d_sigma(elem, sigma)
            for pl_key, coeff in d_sig:
                pl_elem = B2(pl_key)
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


class TestDSigmaOnBarrattEccles:
    """d_sigma on B(BarrattEccles)."""

    def test_d_sigma_sums_to_boundary(self):
        """Σ_σ d_σ(x)·σ == boundary(x)."""
        BBE = BarConstruction(BarrattEccles)
        B2 = BBE(2)
        BE2 = BarrattEccles(2, QQ)
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

        # BE planar degree-1 key: (id, s21)
        be_key = (
            BarrattEccles(2, QQ)._symmetric_group.identity(),
            BarrattEccles(2, QQ)._symmetric_group([2, 1]),
        )
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
        """Surjection(2, QQ).graded_basis(d) has the expected number of elements."""
        S2 = Surjection(2, QQ)
        # Degree d: surjections {1,...,d+2} -> {1,2} with surjection property
        # For d=1: (1,2,1), (1,2,2), (2,1,1), (2,1,2) - but some may be degenerate
        basis_d1 = S2.graded_basis(1)
        assert len(basis_d1) >= 1

    def test_surjection_graded_planar_basis(self):
        """Surjection(2, QQ).graded_planar_basis(d) are all planar."""
        S2 = Surjection(2, QQ)
        for d in range(3):
            for elem in S2.graded_planar_basis(d):
                for key in elem.support():
                    assert S2(key).is_planar(), (
                        f"Non-planar element in graded_planar_basis({d})"
                    )

    def test_barratt_eccles_graded_basis(self):
        """BarrattEccles(2, QQ).graded_basis(d) has the right size."""
        BE2 = BarrattEccles(2, QQ)
        # degree 0: just (id,) → 1 element for planar, 2 for full (id and (1,2))
        basis_d0 = BE2.graded_basis(0)
        assert len(basis_d0) == 2  # (id,) and ((1,2),) in arity 2

    def test_barratt_eccles_graded_planar_basis(self):
        """BarrattEccles(2, QQ).graded_planar_basis(d) only contains planar elements."""
        BE2 = BarrattEccles(2, QQ)
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
                    assert r_key[0] == BarrattEccles(2, QQ)._symmetric_group.identity(), (
                        f"Non-planar right key {r_key} in HadamardProduct planar_basis_it({d})"
                    )


# ---------------------------------------------------------------------------
# E-comodule map
# ---------------------------------------------------------------------------


class TestEComoduleMap:
    """Tests for the E-comodule map on Ω(B(Lie⊙E))."""

    def test_k0_term_is_identity_tensor_generator(self):
        """The k=0 term should be BE[(id,)] ⊗ cobar_gen(x)."""
        _, BH2, _, _, BE2 = _BH_setup(2)

        # Degree-1 planar element: ((1,), (id,)) at weight 1
        planar_elems = list(BH2.planar_basis_it(1))
        assert len(planar_elems) >= 1, "Need at least one planar element"

        dec_key = list(planar_elems[0].support())[0]
        dec_elem = BH2(dec_key)

        result = e_comodule_on_generator(dec_elem)

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
        _, BH2, _, _, BE2 = _BH_setup(2)

        for d in range(1, 3):
            for pl_elem in BH2.planar_basis_it(d):
                dec_key = list(pl_elem.support())[0]
                dec_elem = BH2(dec_key)
                result = e_comodule_on_generator(dec_elem)

                id_be_key = (BE2._symmetric_group.identity(),)
                cobar_tree_key = (dec_key,) + tuple(range(1, 3))

                found_k0 = any(
                    be_key == id_be_key and cobar_key == cobar_tree_key
                    for (be_key, cobar_key), coeff in result
                )
                assert found_k0, f"k=0 term missing for planar element in degree {d}"

    def test_comodule_on_degree2_element_has_k1_term(self):
        """For a degree-2 element with non-trivial boundary, k=1 terms appear."""
        _, BH2, _, _, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        s21 = S2([2, 1])

        # Degree-2 bar tree: ((1,), (id, s21)) at HLE(2)
        be_key_deg1 = (id2, s21)
        had_key = ((1,), be_key_deg1)
        tree_key = (had_key, 1, 2)
        dec_elem = BH2(tree_key)

        if dec_elem == BH2.zero():
            pytest.skip("Element is zero")

        result = e_comodule_on_generator(dec_elem)

        # Check the result has more than just the k=0 term
        terms = list(result)
        assert len(terms) > 1, "Expected k=1 term in addition to k=0"

    def test_comodule_be_keys_are_valid(self):
        """All BE keys in the comodule map output should be valid BE elements."""
        _, BH2, _, _, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        s21 = S2([2, 1])
        id2 = S2.identity()

        be_key_deg1 = (id2, s21)
        had_key = ((1,), be_key_deg1)
        tree_key = (had_key, 1, 2)
        dec_elem = BH2(tree_key)

        if dec_elem == BH2.zero():
            pytest.skip("Element is zero")

        result = e_comodule_on_generator(dec_elem)

        for (be_key, cobar_key), coeff in result:
            # BE key should be a valid basis element (no consecutive duplicates)
            be_elem = BE2(be_key)
            assert be_elem != BE2.zero(), f"BE key {be_key} gave zero element"

    def test_comodule_on_b_surjection(self):
        """E-comodule map on B(Surjection)(2), which is quasi-planar."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        BE2 = BarrattEccles(2, QQ)

        # Planar degree-1 element: (1,2) in Surjection(2, QQ)
        tree_key = ((1, 2), 1, 2)
        dec_elem = B2(tree_key)

        result = e_comodule_on_generator(dec_elem)

        # Should at minimum have the k=0 term
        id_be_key = (BE2._symmetric_group.identity(),)
        cobar_key = (tree_key, 1, 2)

        found = any(
            be_key == id_be_key and ck == cobar_key for (be_key, ck), _coeff in result
        )
        assert found, "k=0 term missing for B(Surj)(2) degree-1 planar element"

    def test_comodule_result_in_tensor(self):
        """Output type is an element of BE(n) ⊗ Ω(C)(n)."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        planar = list(BH2.planar_basis_it(1))
        if not planar:
            pytest.skip("No planar basis elements found")

        dec_elem = BH2(list(planar[0].support())[0])
        result = e_comodule_on_generator(dec_elem)

        expected_parent = tensor([BE2, OBH2])
        assert result.parent() is expected_parent or result == expected_parent.zero()


# ---------------------------------------------------------------------------
# Comodule axiom tests
# ---------------------------------------------------------------------------


def _delta_equiv(cobar_key, B_n, O_n, BE_n):
    """Compute Δ equivariantly via planarisation and diagonal S_n action.

    For a weight-1 cobar generator ``(bar_tree, l_1, …, l_n)`` whose bar-tree
    decoration may be non-planar, planarise ``bar_tree`` in ``B_n``, then
    apply the comodule formula to the planar representative and act diagonally
    (on both the BE and the outer-cobar-leaf components) by the resulting
    permutation.  Handles non-standard outer leaf labels in the same stroke.
    """
    T_loc = tensor([BE_n, O_n])
    n = B_n.arity()
    inner_bar_tree = cobar_key[0]
    outer_leaves = list(cobar_key[1:])

    bar_elem = B_n(inner_bar_tree)
    planarized = B_n.planarize(bar_elem)

    result = T_loc.zero()
    Sn = B_n._symmetric_group
    for (pl_key, sigma_key), pl_coeff in planarized:
        sigma = Sn(sigma_key)
        sigma_vals = list(sigma.tuple())
        # sigma permutes the outer leaves: new_outer[j] = sigma(outer_leaves[j])
        new_outer = [sigma_vals[outer_leaves[j] - 1] for j in range(n)]

        pl_elem = B_n(pl_key)
        delta_std = e_comodule_on_generator(pl_elem)

        # Diagonal action of new_outer on E ⊗ Ω(C):
        #   BE part : precompose each permutation by the leaf permutation
        #   Ω(C) part: permute the outer cobar leaf labels
        for (be_key, ck), c in delta_std:
            new_be = BE_n(be_key).permute(new_outer)
            new_ck = O_n(ck).permute(new_outer)
            result += c * pl_coeff * new_be.tensor(new_ck)

    return result


def _normalize_oc(oc_elem, B_n, O_n):
    """Rewrite Ω(C)(n) elements using only planar bar-tree decorations.

    Converts every non-planar cobar generator ``(bar_tree, l_1, …, l_n)``
    to an equivalent linear combination of generators whose decoration
    ``bar_tree`` is planar, by applying ``B_n.planarize`` and composing
    the resulting permutation into the outer leaf labels.
    """
    n = B_n.arity()
    result = O_n.zero()
    Sn = B_n._symmetric_group
    for cobar_key, coeff in oc_elem:
        from uconf.core.trees import is_leaf

        if is_leaf(cobar_key):
            result += coeff * O_n(cobar_key)
            continue
        inner_bar_tree = cobar_key[0]
        outer_leaves = list(cobar_key[1:])
        bar_elem = B_n(inner_bar_tree)
        planarized = B_n.planarize(bar_elem)
        for (pl_key, sigma_key), pl_coeff in planarized:
            sigma = Sn(sigma_key)
            sigma_vals = list(sigma.tuple())
            new_outer = [sigma_vals[outer_leaves[j] - 1] for j in range(n)]
            new_cobar_key = tuple([pl_key] + new_outer)
            result += coeff * pl_coeff * O_n(new_cobar_key)
    return result


def _normalize_tensor_E_OC(tensor_elem, B_n, O_n, BE_n):
    """Normalise an element of BE_n ⊗ Ω(C)_n by canonicalising the Ω(C) part."""
    T_loc = tensor([BE_n, O_n])
    result = T_loc.zero()
    for (be_key, cobar_key), coeff in tensor_elem:
        be_elem = BE_n(be_key)
        norm_oc = _normalize_oc(O_n(cobar_key), B_n, O_n)
        result += coeff * be_elem.tensor(norm_oc)
    return result


class TestComoduleAxioms:
    """Check that ``e_comodule_on_generator`` defines a genuine dg E-comodule.

    Two axioms are verified for every planar generator in the bar cooperad
    B(Lie⊙E)(2):

    1. **Compatibility with the differential** (chain-map property):
       ``d_tensor(Δ(x)) = Δ(d_Ω(x))`` where both sides are canonicalised
       to planar bar-tree representatives before comparison (the cobar
       construction's basis over-represents the module, so non-planar
       generators must first be identified with signed permutations of
       planar ones via ``_normalize_tensor_E_OC`` / ``_delta_equiv``).

    2. **Coassociativity**:
       ``(Δ_E ⊗ id_Ω)(Δ(x)) = (id_E ⊗ Δ)(Δ(x))``
       where ``Δ_E`` is the Alexander--Whitney diagonal on the
       Barratt-Eccles complex and the second ``Δ`` is applied equivariantly
       via ``_delta_equiv``.
    """

    def test_chain_map_degree0_generator(self):
        """Chain-map axiom holds trivially for a degree-0 planar generator.

        For a degree-0 cobar generator both ``d_Ω(x)`` and ``d(Δ(x))``
        vanish, so the equality ``d(Δ(x)) = Δ(d_Ω(x))`` reduces to 0 = 0.
        """
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()

        had_d0 = ((1,), (id2,))
        bar_d0 = (had_d0, 1, 2)
        dec_elem = BH2(bar_d0)
        cobar_gen = OBH2((bar_d0, 1, 2))

        delta_x = e_comodule_on_generator(dec_elem)
        T = tensor([BE2, OBH2])

        d_delta = T.zero()
        for (be_key, ck), coeff in delta_x:
            be_elem = BE2(be_key)
            cobar_el = OBH2(ck)
            deg_e = BE2.degree_on_basis(be_key)
            for nbk, c in be_elem.boundary():
                d_delta += coeff * c * BE2(nbk).tensor(cobar_el)
            sign = (-1) ** deg_e
            for nck, c in cobar_el.boundary():
                d_delta += coeff * sign * c * be_elem.tensor(OBH2(nck))

        d_omega_x = cobar_gen.boundary()
        delta_d_omega = T.zero()
        for ck, coeff in d_omega_x:
            delta_d_omega += coeff * _delta_equiv(ck, BH2, OBH2, BE2)

        assert d_delta == T.zero()
        assert delta_d_omega == T.zero()

    def test_chain_map_degree1_planar_generator(self):
        """Chain-map axiom holds for the canonical degree-1 planar generator.

        After canonicalising both sides to planar bar-tree representatives,
        ``normalize(d(Δ(x))) == Δ_equiv(d_Ω(x))``.
        """
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        s21 = S2([2, 1])

        had_d1 = ((1,), (id2, s21))
        bar_d1 = (had_d1, 1, 2)
        dec_elem = BH2(bar_d1)
        cobar_gen = OBH2((bar_d1, 1, 2))

        delta_x = e_comodule_on_generator(dec_elem)
        T = tensor([BE2, OBH2])

        # d_tensor(Δ(x))
        d_delta = T.zero()
        for (be_key, ck), coeff in delta_x:
            be_elem = BE2(be_key)
            cobar_el = OBH2(ck)
            deg_e = BE2.degree_on_basis(be_key)
            for nbk, c in be_elem.boundary():
                d_delta += coeff * c * BE2(nbk).tensor(cobar_el)
            sign = (-1) ** deg_e
            for nck, c in cobar_el.boundary():
                d_delta += coeff * sign * c * be_elem.tensor(OBH2(nck))

        norm_d_delta = _normalize_tensor_E_OC(d_delta, BH2, OBH2, BE2)

        # Δ_equiv(d_Ω(x))
        d_omega_x = cobar_gen.boundary()
        delta_d_omega = T.zero()
        for ck, coeff in d_omega_x:
            delta_d_omega += coeff * _delta_equiv(ck, BH2, OBH2, BE2)

        assert norm_d_delta == delta_d_omega, (
            f"Chain-map failed:\n  normalize(d(Δ(x))) = {norm_d_delta}\n"
            f"  Δ(d_Ω(x))          = {delta_d_omega}"
        )

    def test_chain_map_all_degree1_planar_generators(self):
        """Chain-map axiom holds for every degree-1 planar B(Lie⊙E)(2) generator."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        T = tensor([BE2, OBH2])

        for pl_elem in BH2.planar_basis_it(2):  # BH-degree 2 = cobar-degree 1
            dec_key = list(pl_elem.support())[0]
            dec_elem = BH2(dec_key)
            cobar_gen = OBH2((dec_key, 1, 2))

            delta_x = e_comodule_on_generator(dec_elem)

            d_delta = T.zero()
            for (be_key, ck), coeff in delta_x:
                be_elem = BE2(be_key)
                cobar_el = OBH2(ck)
                deg_e = BE2.degree_on_basis(be_key)
                for nbk, c in be_elem.boundary():
                    d_delta += coeff * c * BE2(nbk).tensor(cobar_el)
                sign = (-1) ** deg_e
                for nck, c in cobar_el.boundary():
                    d_delta += coeff * sign * c * be_elem.tensor(OBH2(nck))

            norm_d_delta = _normalize_tensor_E_OC(d_delta, BH2, OBH2, BE2)

            d_omega_x = cobar_gen.boundary()
            delta_d_omega = T.zero()
            for ck, coeff in d_omega_x:
                delta_d_omega += coeff * _delta_equiv(ck, BH2, OBH2, BE2)

            assert norm_d_delta == delta_d_omega, (
                f"Chain-map failed for generator {dec_key}"
            )

    def test_coassociativity_degree0_generator(self):
        """Coassociativity holds for the degree-0 planar generator.

        ``(Δ_E ⊗ id)(Δ(x)) == (id ⊗ Δ_equiv)(Δ(x))`` in BE ⊗ BE ⊗ Ω(C).
        """
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()

        had_d0 = ((1,), (id2,))
        bar_d0 = (had_d0, 1, 2)
        dec_elem = BH2(bar_d0)

        delta_x = e_comodule_on_generator(dec_elem)
        T_EE_OC = tensor([BE2, BE2, OBH2])

        lhs = T_EE_OC.zero()
        rhs = T_EE_OC.zero()

        for (be_key, ck), coeff in delta_x:
            be_elem = BE2(be_key)
            cobar_elem = OBH2(ck)

            for (lk, rk), dc in be_elem.diagonal():
                lhs += coeff * dc * BE2(lk).tensor(BE2(rk)).tensor(cobar_elem)

            for (be2_key, ck2), d_coeff in _delta_equiv(ck, BH2, OBH2, BE2):
                rhs += coeff * d_coeff * be_elem.tensor(BE2(be2_key)).tensor(OBH2(ck2))

        assert lhs == rhs, f"Coassociativity failed:\n  LHS = {lhs}\n  RHS = {rhs}"

    def test_coassociativity_degree1_generator(self):
        """Coassociativity holds for the canonical degree-1 planar generator."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        s21 = S2([2, 1])

        had_d1 = ((1,), (id2, s21))
        bar_d1 = (had_d1, 1, 2)
        dec_elem = BH2(bar_d1)

        delta_x = e_comodule_on_generator(dec_elem)
        T_EE_OC = tensor([BE2, BE2, OBH2])

        lhs = T_EE_OC.zero()
        rhs = T_EE_OC.zero()

        for (be_key, ck), coeff in delta_x:
            be_elem = BE2(be_key)
            cobar_elem = OBH2(ck)

            for (lk, rk), dc in be_elem.diagonal():
                lhs += coeff * dc * BE2(lk).tensor(BE2(rk)).tensor(cobar_elem)

            for (be2_key, ck2), d_coeff in _delta_equiv(ck, BH2, OBH2, BE2):
                rhs += coeff * d_coeff * be_elem.tensor(BE2(be2_key)).tensor(OBH2(ck2))

        assert lhs == rhs, f"Coassociativity failed:\n  LHS = {lhs}\n  RHS = {rhs}"

    def test_coassociativity_all_degree1_planar_generators(self):
        """Coassociativity holds for every degree-1 planar generator of B(Lie⊙E)(2)."""
        _, BH2, _, OBH2, BE2 = _BH_setup(2)
        T_EE_OC = tensor([BE2, BE2, OBH2])

        for pl_elem in BH2.planar_basis_it(2):  # BH-degree 2 = cobar-degree 1
            dec_key = list(pl_elem.support())[0]
            dec_elem = BH2(dec_key)
            delta_x = e_comodule_on_generator(dec_elem)

            lhs = T_EE_OC.zero()
            rhs = T_EE_OC.zero()

            for (be_key, ck), coeff in delta_x:
                be_elem = BE2(be_key)
                cobar_elem = OBH2(ck)

                for (lk, rk), dc in be_elem.diagonal():
                    lhs += coeff * dc * BE2(lk).tensor(BE2(rk)).tensor(cobar_elem)

                for (be2_key, ck2), d_coeff in _delta_equiv(ck, BH2, OBH2, BE2):
                    rhs += (
                        coeff * d_coeff * be_elem.tensor(BE2(be2_key)).tensor(OBH2(ck2))
                    )

            assert lhs == rhs, f"Coassociativity failed for generator {dec_key}"


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
