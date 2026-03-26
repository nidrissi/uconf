from random import Random

import pytest

from uconf import (
    BarConstruction,
    BarrattEccles,
    CobarConstruction,
    HadamardProduct,
    Lie,
    ShiftedOperad,
    Surjection,
    compute_chain_complex,
    euclidean_unordered_configuration_model,
)
from uconf.algebraic.configuration import _make_surjection_comodule_morphism


from sage.all import GF, QQ


class TestConfigurationModelCore:
    """Tests for the configuration model and its homology."""

    def test_configuration_model_GF2(self) -> None:
        """chain_complex for euclidean configuration model over GF(2) succeeds."""

        model = euclidean_unordered_configuration_model(GF(2), 2)
        C = compute_chain_complex(model, degrees=range(3), weight=1)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_GF2_weight3(self, d: int) -> None:
        """chain_complex over GF(2) with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(GF(2), d)
        C = compute_chain_complex(model, degrees=range(-1, 3), weight=3, check=True)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_GF2_weight4(self, d: int) -> None:
        """chain_complex over GF(2) with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(GF(2), d)
        C = compute_chain_complex(model, degrees=range(-1, 2), weight=4, check=True)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_QQ_weight2(self, d: int) -> None:
        """chain_complex over QQ at weight=2 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, d)
        complex = compute_chain_complex(model, degrees=range(-1, 3), weight=2, check=True)
        assert complex is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_QQ_weight3(self, d: int) -> None:
        """chain_complex over QQ at weight=3 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, d)
        complex = compute_chain_complex(model, degrees=range(-1, 3), weight=3, check=True)
        assert complex is not None

    def test_check_complex_QQ_weight4(self) -> None:
        """chain_complex over QQ at weight=4 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        complex = compute_chain_complex(model, degrees=range(0, 2), weight=4, check=True)
        assert complex is not None


class TestConfigurationModelIntermediate:
    @pytest.mark.parametrize("n", [3, 4])
    @pytest.mark.parametrize("d", range(3))
    def test_check_bar_cobar_square_zero(self, n: int, d: int) -> None:
        """The bar-cobar construction on the shifted Lie–Surjection cooperad satisfies d²=0."""
        rng = Random(20260326)

        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        P = CobarConstruction(C)

        basis = P(n, QQ).graded_basis(d)
        for _ in range(20):
            p_elem = rng.choice(basis)
            dd = p_elem.boundary().boundary()
            assert dd == P(n, QQ).zero(), f"d²≠0 at arity {n} degree {d} for {p_elem}"


class TestConfigurationModelDSquaredMinimal:
    """Minimal d²=0 regression tests at weight 2.

    These verify that the inner-module normalization fix in
    TreeModule._boundary_on_basis correctly resolves the d²≠0 bug
    where non-planar free-algebra keys from d_internal failed to
    cancel with the planar keys produced by d_alpha.
    """

    def test_dsquared_weight2_d1(self) -> None:
        """d²=0 at weight=2 for d=1 (all degrees up to 4)."""
        model = euclidean_unordered_configuration_model(QQ, 1)
        for deg in range(-1, 5):
            for elem in model.graded_basis_by_weight(deg, 2):
                dd = model.boundary(model.boundary(elem))
                assert dd == model.zero(), f"d²≠0 at deg={deg} for {list(elem)[0][0]}"

    def test_dsquared_weight2_d2(self) -> None:
        """d²=0 at weight=2 for d=2 (all degrees up to 4)."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        for deg in range(-1, 5):
            for elem in model.graded_basis_by_weight(deg, 2):
                dd = model.boundary(model.boundary(elem))
                assert dd == model.zero(), f"d²≠0 at deg={deg} for {list(elem)[0][0]}"


class TestConfigurationModelBasis:
    """Tests for the graded basis on the configuration model."""

    @pytest.mark.parametrize("d", [1, 2])
    def test_homology_basis_weight1(self, d: int) -> None:
        """homology_basis for euclidean configuration model at weight 1 returns a non-empty basis."""
        model = euclidean_unordered_configuration_model(QQ, d)
        for k in range(-2, 3):
            for w in range(1, 4):
                basis = model.graded_basis_by_weight(k, w)
                assert len(basis) == len(set(basis)), (
                    f"Basis at degree {k} weight {w} contains duplicates"
                )


class TestConfigurationModelComodule:
    def test_e_comodule_generator_chain_map_arity3(self) -> None:
        """e_comodule_on_generator satisfies the cooperad-level chain map at arity 3.

        For c ∈ C(3), checks ν(∂_C c) = (d_E ⊗ 1 + 1 ⊗ ∂_C)(ν(c))
        where ν: C → E ⊗ C is the Berger–Fresse E-comodule structure map
        and ∂_C is the cooperad differential.
        """
        from sage.all import tensor

        from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator

        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        C3 = C(3, QQ)
        BE3 = BarrattEccles(3, QQ)

        for d in range(3):
            for elem in C3.basis_iter(d):
                nu_dc = e_comodule_on_generator(elem.boundary())
                d_nu_c = tensor([BE3, C3]).zero()
                for (b, u), coeff in e_comodule_on_generator(elem):
                    b_elem = BE3.term(b)
                    u_elem = C3.term(u)
                    d_nu_c += coeff * b_elem.boundary().tensor(u_elem)
                    d_nu_c += (
                        coeff * (-1) ** BE3.degree_on_basis(b) * b_elem.tensor(C3.boundary(u_elem))
                    )
                assert nu_dc == d_nu_c, f"generator chain map failed at arity 3 deg {d}: {elem}"

    def test_e_comodule_generator_chain_map_arity2(self) -> None:
        """e_comodule_on_generator satisfies the cooperad-level chain map at arity 2.

        For c ∈ C(2), checks ν(∂_C c) = (d_E ⊗ 1 + 1 ⊗ ∂_C)(ν(c))
        where ν: C → E ⊗ C is the Berger–Fresse E-comodule structure map
        and ∂_C is the cooperad differential.
        """
        from sage.all import tensor

        from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator

        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        C2 = C(2, QQ)
        BE2 = BarrattEccles(2, QQ)

        for d in range(3):
            for elem in C2.basis_iter(d):
                nu_dc = e_comodule_on_generator(elem.boundary())
                d_nu_c = tensor([BE2, C2]).zero()
                for (b, u), coeff in e_comodule_on_generator(elem):
                    b_elem = BE2.term(b)
                    u_elem = C2.term(u)
                    d_nu_c += coeff * b_elem.boundary().tensor(u_elem)
                    d_nu_c += (
                        coeff * (-1) ** BE2.degree_on_basis(b) * b_elem.tensor(C2.boundary(u_elem))
                    )
                assert nu_dc == d_nu_c, f"generator chain map failed at arity 2 deg {d}: {elem}"

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

    def test_e_comodule_chain_map_arity3(self) -> None:
        """Chain map property at arity 3 degree 0."""
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
