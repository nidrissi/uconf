"""Tests for chain complex construction and homology helpers."""

import pytest
from sage.all import GF, QQ

from uconf import (
    BarConstruction,
    BarrattEccles,
    CobarConstruction,
    HadamardProduct,
    Lie,
    ShiftedOperad,
    Surjection,
    chain_complex,
    euclidean_unordered_configuration_model,
    homology_basis,
)
from uconf.algebraic.conf import _make_surjection_comodule_morphism

# ---------------------------------------------------------------------------
# chain_complex
# ---------------------------------------------------------------------------


class TestChainComplex:
    """Tests for :func:`chain_complex`."""

    def test_surjection_arity2(self) -> None:
        """Surjection(2, QQ) has H_0=1 and H_d=0 for 1<=d<=4.

        H_{5} (one above the requested range) may be non-zero due to
        truncation; only Betti numbers for degrees 0-4 are checked.
        """
        S2 = Surjection(2, QQ)
        C = chain_complex(S2, degrees=range(5))
        for d in range(5):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_surjection_arity3(self) -> None:
        """Surjection(3, QQ) has H_0=1 and H_d=0 for d in 1..3."""
        S3 = Surjection(3, QQ)
        C = chain_complex(S3, degrees=range(4))
        for d in range(4):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_barratt_eccles_arity2(self) -> None:
        """BarrattEccles(2, QQ) has H_0=1 and H_d=0 for 1<=d<=4."""
        E2 = BarrattEccles(2, QQ)
        C = chain_complex(E2, degrees=range(5))
        for d in range(5):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_lie_arity2(self) -> None:
        """Lie is concentrated in degree 0; homology is the module itself."""
        L2 = Lie(2, QQ)
        C = chain_complex(L2, degrees=range(3))
        assert C.betti().get(0, 0) == 1
        assert C.betti().get(1, 0) == 0
        assert C.betti().get(2, 0) == 0

    def test_d_squared_zero(self) -> None:
        """The chain complex differential squares to zero (implicitly
        checked by ChainComplex, but verify explicitly)."""
        S2 = Surjection(2, QQ)
        C = chain_complex(S2, degrees=range(5))
        # Complex now spans degrees 0-5 (extended internally by 1)
        for d in range(1, 6):
            d_prev = C.differential(d - 1)
            d_curr = C.differential(d)
            assert (d_prev * d_curr).is_zero()

    def test_empty_degrees(self) -> None:
        """Empty degree range gives a trivial complex."""
        S2 = Surjection(2, QQ)
        C = chain_complex(S2, degrees=range(0))
        assert C.betti() == {}

    def test_weight_parameter_restricts_basis(self) -> None:
        """chain_complex with weight restricts basis to fixed-weight elements."""
        from uconf import Associative
        from uconf.algebraic.free_algebra import FreeAlgebraModule
        from uconf.models.commutative import Commutative

        # Associative (connectivity=0) + Commutative(1) (degree-0) → unbounded arity
        M = Commutative(1, QQ)
        mod = FreeAlgebraModule(Associative, M)

        # Without weight: raises (unbounded arity in degree 0)
        with pytest.raises(ValueError, match="Cannot exhaustively enumerate"):
            chain_complex(mod, degrees=range(2))

        # With weight=1: only arity-1 element; gives finite complex
        C = chain_complex(mod, degrees=range(2), weight=1)
        assert C is not None

    def test_weight_error_when_module_unsupported(self) -> None:
        """chain_complex raises ValueError when weight is used on unsupported module."""
        S2 = Surjection(2, QQ)
        with pytest.raises(ValueError, match="does not support the weight API"):
            chain_complex(S2, degrees=range(3), weight=1)

    def test_configuration_model_gf2(self) -> None:
        """chain_complex for euclidean configuration model over GF(2) succeeds.

        Regression test for ZeroDivisionError in Lie._pbw_left_inverse
        that occurred because (M^T M) is singular over GF(2).
        """

        model = euclidean_unordered_configuration_model(GF(2), 2)
        C = chain_complex(model, degrees=range(2), weight=1)
        # Just verify it computes without errors; the Betti numbers
        # are approximate due to weight truncation and d²≠0 at degree ≥3.
        assert C is not None

    def test_check_complex(self) -> None:
        """chain_complex over GF(2) with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(GF(2), 2)
        C = chain_complex(model, degrees=range(-2, 3), weight=3, check=True)
        assert C is not None

    def test_check_complex_QQ_weight2(self) -> None:
        """chain_complex over QQ at weight=2 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        complex = chain_complex(model, degrees=range(-2, 3), weight=2, check=True)
        assert complex is not None

    def test_check_complex_QQ_weight3(self) -> None:
        """chain_complex over QQ at weight=3 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        complex = chain_complex(model, degrees=range(-2, 3), weight=3, check=True)
        assert complex is not None

    def test_e_comodule_generator_chain_map_arity2(self) -> None:
        """e_comodule_on_generator satisfies the chain map property at arity 2.

        For c ∈ C(2), checks Δ_gen(∂c) = d_{E⊗Ω}(Δ_gen(c)).
        This is the core identity that ensures the E-comodule map
        on generators is compatible with the differential.
        """
        from sage.all import tensor

        from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator

        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        P = CobarConstruction(C)
        C2 = C(2, QQ)
        P2 = P(2, QQ)
        BE2 = BarrattEccles(2, QQ)

        for d in range(3):
            for elem in C2.basis_iter(d):
                for key in elem.support():
                    c = C2.term(key)
                    f_dc = e_comodule_on_generator(c.boundary())
                    d_fc = tensor([BE2, P2]).zero()
                    for (b, u), coeff in e_comodule_on_generator(c):
                        b_elem = BE2.term(b)
                        u_elem = P2.term(u)
                        d_fc += coeff * b_elem.boundary().tensor(u_elem)
                        d_fc += coeff * (-1) ** BE2.degree_on_basis(b) * b_elem.tensor(
                            P2.boundary(u_elem)
                        )
                    assert f_dc == d_fc, f"generator chain map failed at arity 2 deg {d}: {key}"

    def test_e_comodule_chain_map(self) -> None:
        """The composed morphism Ω(B(H)) → S⊙Ω(B(H)) should be a chain map.

        Checks φ(d(p)) = d(φ(p)) for all degree-0 generators at arity 2.
        """
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        P = CobarConstruction(C)
        phi = _make_surjection_comodule_morphism(C)

        P2 = P(2, QQ)
        for p_elem in P2.basis_iter(0):
            phi_dp = phi(P2.boundary(p_elem))
            phi_p = phi(p_elem)
            d_phi_p = phi_p.parent().boundary(phi_p)
            assert phi_dp == d_phi_p, f"chain map failed at arity 2 for {p_elem}"

    @pytest.mark.xfail(
        reason=(
            "Full morphism chain map at arity 3 involves _extend_tree for weight-2 "
            "cobar trees, that has a separate equivariance issue."
        ),
    )
    def test_e_comodule_chain_map_arity3(self) -> None:
        """Chain map property at arity 3 degree 0 (known open)."""
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        P = CobarConstruction(C)
        phi = _make_surjection_comodule_morphism(C)

        P3 = P(3, QQ)
        for p_elem in P3.basis_iter(0):
            phi_dp = phi(P3.boundary(p_elem))
            phi_p = phi(p_elem)
            d_phi_p = phi_p.parent().boundary(phi_p)
            assert phi_dp == d_phi_p, f"chain map failed at arity 3 for {p_elem}"


# ---------------------------------------------------------------------------
# homology_basis
# ---------------------------------------------------------------------------


class TestHomologyBasis:
    """Tests for :func:`homology_basis`."""

    def test_surjection_arity2_degree0(self) -> None:
        """Surjection(2, QQ) has a 1-dimensional H_0."""
        S2 = Surjection(2, QQ)
        gens = homology_basis(S2, 0, degrees=range(3))
        assert len(gens) == 1
        assert S2.boundary(gens[0]) == S2.zero()

    def test_surjection_arity2_degree1(self) -> None:
        """Surjection(2, QQ) has trivial H_1."""
        S2 = Surjection(2, QQ)
        gens = homology_basis(S2, 1, degrees=range(3))
        assert len(gens) == 0

    def test_barratt_eccles_arity2_degree0(self) -> None:
        """BarrattEccles(2, QQ) has a 1-dimensional H_0."""
        E2 = BarrattEccles(2, QQ)
        gens = homology_basis(E2, 0, degrees=range(3))
        assert len(gens) == 1
        assert E2.boundary(gens[0]) == E2.zero()

    def test_generators_are_cycles(self) -> None:
        """All returned generators must be cycles."""
        S3 = Surjection(3, QQ)
        for d in range(3):
            gens = homology_basis(S3, d, degrees=range(4))
            for g in gens:
                assert S3.boundary(g) == S3.zero(), f"Generator in H_{d} is not a cycle: {g}"

    def test_default_degrees(self) -> None:
        """Calling without explicit degrees uses a minimal range."""
        S2 = Surjection(2, QQ)
        gens = homology_basis(S2, 1)
        assert len(gens) == 0  # H_1 of Surjection(2) is 0

    def test_invalid_degree_range(self) -> None:
        """Requesting a degree not in the supplied range raises ValueError."""
        S2 = Surjection(2, QQ)
        with pytest.raises(ValueError, match="must be contained"):
            homology_basis(S2, 5, degrees=range(3))

    def test_lie_all_homology_in_degree0(self) -> None:
        """Lie(3, QQ) has all homology concentrated in degree 0."""
        L3 = Lie(3, QQ)
        gens_0 = homology_basis(L3, 0, degrees=range(3))
        assert len(gens_0) == 2  # Lie(3) has 2 basis elements in degree 0
        for d in range(1, 3):
            gens = homology_basis(L3, d, degrees=range(3))
            assert len(gens) == 0


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------


class TestConnectivity:
    """Tests for connectivity properties on dg-modules."""

    def test_simplicial_chains_connectivity(self) -> None:
        """SimplicialChains has connectivity 0 (vertices have degree 0)."""
        from uconf.models.simplicial import SimplicialChains

        SC = SimplicialChains(QQ)
        assert SC.connectivity == 0

    def test_simplicial_cochains_connectivity(self) -> None:
        """SimplicialCochains(N) has connectivity -N."""
        from uconf.models.simplicial import SimplicialCochains

        SC2 = SimplicialCochains(2, QQ)
        assert SC2.connectivity == -2
        SC5 = SimplicialCochains(5, QQ)
        assert SC5.connectivity == -5

    def test_simplicial_cochains_basis_it(self) -> None:
        """SimplicialCochains(2) enumerates basis correctly per degree."""
        from uconf.models.simplicial import SimplicialCochains

        SC = SimplicialCochains(2, QQ)
        # degree 0: 3 vertices (0), (1), (2)
        assert len(list(SC.basis_iter(0))) == 3
        # degree -1: 3 edges (0,1), (0,2), (1,2)
        assert len(list(SC.basis_iter(-1))) == 3
        # degree -2: 1 face (0,1,2)
        assert len(list(SC.basis_iter(-2))) == 1
        # degree -3: empty
        assert len(list(SC.basis_iter(-3))) == 0
        # degree 1: empty
        assert len(list(SC.basis_iter(1))) == 0

    def test_simplicial_cochains_boundary(self) -> None:
        """SimplicialCochains.boundary is an alias for coboundary."""
        from uconf.models.simplicial import SimplicialCochains

        SC = SimplicialCochains(2, QQ)
        vertex = SC.term((0,))
        assert SC.boundary(vertex) == SC.coboundary(vertex)

    def test_free_algebra_connectivity(self) -> None:
        """FreeAlgebraModule connectivity is that of the inner module."""
        from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

        from uconf.algebraic.free_algebra import FreeOperadAlgebra
        from uconf.models.surjection import Surjection

        M = CombinatorialFreeModule(QQ, ["a"], category=GradedModulesWithBasis(QQ))
        M.degree_on_basis = lambda _: 3
        M.connectivity = 3
        M.boundary = lambda _: M.zero()

        fa = FreeOperadAlgebra(Surjection, M)
        assert fa.module.connectivity == 3

    def test_tree_module_connectivity(self) -> None:
        """TreeModule connectivity is the min of leaf and tree contributions."""
        from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

        from uconf.algebraic.free_algebra import FreeOperadAlgebra
        from uconf.models.surjection import Surjection

        M = CombinatorialFreeModule(QQ, ["x"], category=GradedModulesWithBasis(QQ))
        M.degree_on_basis = lambda _: 2
        M.connectivity = 2
        M.boundary = lambda _: M.zero()
        fa = FreeOperadAlgebra(Surjection, M)
        # FreeAlgebraModule inherits connectivity from M
        assert fa.module.connectivity == 2
