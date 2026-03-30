from random import Random

import pytest
from sage.all import GF, QQ, tensor

from uconf import (
    BarConstruction,
    BarrattEccles,
    CobarConstruction,
    HadamardProduct,
    Lie,
    ShiftedOperad,
    Surjection,
    compute_chain_complex,
    e_comodule_on_generator,
    euclidean_unordered_configuration_model,
)
from uconf.algebraic.configuration import (
    TrivialModule,
    _make_surjection_comodule_morphism,
)
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.spherical import SurjectionSphereCochainAlgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.morphisms.canonical_twisting import canonical_projection


# ---------------------------------------------------------------------------
# Helpers — build the individual layers of the configuration model
# ---------------------------------------------------------------------------


def _build_layers(base_ring, dimension: int):
    """Build every intermediate object in the configuration-model pipeline.

    Returns a dict mapping layer names to the constructed objects.
    """
    # Layer 1: manifold model — Surjection-algebra on sphere cochains
    manifold_model = SurjectionSphereCochainAlgebra(dimension, base_ring)

    # Layer 2: coefficient module — trivial module concentrated in degree d
    coefficients = TrivialModule(dimension, base_ring)

    # Layer 3: operadic layers
    sLie = ShiftedOperad(Lie, -1)
    XsLie = HadamardProduct(sLie, Surjection)
    BXsLie = BarConstruction(XsLie)
    OBXsLie = CobarConstruction(BXsLie)

    # Layer 4: free algebra
    free_alg = FreeOperadAlgebra(OBXsLie, coefficients)

    # Layer 5: Hadamard tensor algebra
    tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)

    # Layer 6: pullback via comodule morphism
    comodule_morphism = _make_surjection_comodule_morphism(BXsLie)
    pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)

    # Layer 7: final bar algebra
    pi = canonical_projection(pulled_back.operad_cls)
    bar = BarAlgebra(pi, pulled_back)

    return {
        "manifold_model": manifold_model,
        "coefficients": coefficients,
        "sLie": sLie,
        "XsLie": XsLie,
        "BXsLie": BXsLie,
        "OBXsLie": OBXsLie,
        "free_alg": free_alg,
        "tensor_alg": tensor_alg,
        "comodule_morphism": comodule_morphism,
        "pulled_back": pulled_back,
        "pi": pi,
        "bar": bar,
    }


# ============================================================================
# Layer 0: Hadamard product operad H = sLie ⊙ Surjection
# ============================================================================


class TestLayerHadamardOperadDSquared:
    """d²=0 for the Hadamard product operad H = s⁻¹Lie ⊙ Surjection.

    This is the innermost operad used in the configuration model.
    Elements have degree deg(sLie) + deg(Surjection).
    """

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("d", range(3))
    def test_d_squared_zero(self, n: int, d: int) -> None:
        """d²=0 for H(n) at degree d."""
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        Hn = H(n, QQ)
        for elem in Hn.basis_iter(d):
            dd = elem.boundary().boundary()
            assert dd == Hn.zero(), f"d²≠0 for H({n}) at degree {d}"


# ============================================================================
# Layer 1: Bar construction cooperad C = B(H)
# ============================================================================


class TestLayerBarCooperadDSquared:
    """d²=0 for the bar construction cooperad C = B(sLie ⊙ Surjection).

    The bar construction has cooperad connectivity -1 (degree -(n-1) at arity n).
    """

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("d", range(-1, 3))
    def test_d_squared_zero(self, n: int, d: int) -> None:
        """d²=0 for C(n) at degree d."""
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        Cn = C(n, QQ)
        for elem in Cn.basis_iter(d):
            dd = elem.boundary().boundary()
            assert dd == Cn.zero(), f"d²≠0 for B(H)({n}) at degree {d}"


# ============================================================================
# Layer 2: Cobar construction operad P = Ω(B(H))
# ============================================================================


class TestLayerCobarOperadDSquared:
    """d²=0 for the cobar construction operad P = Ω(B(sLie ⊙ Surjection)).

    This is the quasi-free resolution used in the configuration model.
    """

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("d", range(3))
    def test_d_squared_zero(self, n: int, d: int) -> None:
        """d²=0 for Ω(B(H))(n) at degree d (sampled)."""
        rng = Random(20260326)
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        C = BarConstruction(H)
        P = CobarConstruction(C)
        Pn = P(n, QQ)
        basis = Pn.graded_basis(d)
        if not basis:
            return
        for _ in range(min(20, len(basis))):
            p_elem = rng.choice(basis)
            dd = p_elem.boundary().boundary()
            assert dd == Pn.zero(), f"d²≠0 for Ω(B(H))({n}) at degree {d}"


# ============================================================================
# Layer 3: Free algebra T_P(k[d])
# ============================================================================


class TestLayerFreeAlgebra:
    """Tests for the free Ω(B(H))-algebra on the trivial module k[d].

    This is the algebra that captures the 'label' structure.
    """

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight1(self, d: int) -> None:
        """d²=0 at weight 1 for the free algebra."""
        layers = _build_layers(QQ, d)
        fa = layers["free_alg"]
        mod = fa.module
        for deg in range(-1, 4):
            for elem in mod.basis_weight_iter(deg, 1):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 1, degree {deg}"

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight2(self, d: int) -> None:
        """d²=0 at weight 2 for the free algebra."""
        layers = _build_layers(QQ, d)
        fa = layers["free_alg"]
        mod = fa.module
        for deg in range(-1, 3):
            for elem in mod.basis_weight_iter(deg, 2):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 2, degree {deg}"

    @pytest.mark.parametrize("d", [1, 2])
    def test_action_arity_2(self, d: int) -> None:
        """The algebra action γ(p; a₁, a₂) produces a valid element."""
        layers = _build_layers(QQ, d)
        fa = layers["free_alg"]
        mod = fa.module
        P = layers["OBXsLie"]
        # Get a degree-0 operad element at arity 2
        P2 = P(2, QQ)
        p_elems = list(P2.basis_iter(0))
        if not p_elems:
            return
        # Get two weight-1 algebra elements
        w1_elems = list(mod.basis_weight_iter(0, 1))
        if len(w1_elems) < 2:
            return
        p = p_elems[0]
        result = fa.act(p, [w1_elems[0], w1_elems[1]])
        assert result is not None


# ============================================================================
# Layer 4: HadamardTensorAlgebra (sphere cochains ⊗ free algebra)
# ============================================================================


class TestLayerHadamardTensorAlgebra:
    """Tests for the Hadamard tensor algebra of sphere cochains and free algebra.

    This algebra has operad (Surjection ⊙ Ω(B(H))) and module (cochains ⊗ free_alg).
    """

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight1(self, d: int) -> None:
        """d²=0 at weight 1 for the tensor algebra."""
        layers = _build_layers(QQ, d)
        ta = layers["tensor_alg"]
        for deg in range(-1, 3):
            for elem in ta.basis_weight_iter(deg, 1):
                dd = ta.boundary(ta.boundary(elem))
                assert dd == ta.module.zero(), f"d²≠0 at weight 1, degree {deg}"

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight2(self, d: int) -> None:
        """d²=0 at weight 2 for the tensor algebra."""
        layers = _build_layers(QQ, d)
        ta = layers["tensor_alg"]
        for deg in range(-1, 3):
            for elem in ta.basis_weight_iter(deg, 2):
                dd = ta.boundary(ta.boundary(elem))
                assert dd == ta.module.zero(), (
                    f"d²≠0 at weight 2, degree {deg} for {list(elem)[0][0]}"
                )


# ============================================================================
# Layer 5: PullbackAlgebra (via comodule morphism)
# ============================================================================


class TestLayerPullbackAlgebra:
    """Tests for the pullback algebra along the comodule morphism.

    The pullback converts the (Surjection ⊙ Ω(B(H)))-algebra into an
    Ω(B(H))-algebra by composing the action with the comodule morphism
    Ω(B(H)) → Surjection ⊙ Ω(B(H)).
    """

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight1(self, d: int) -> None:
        """d²=0 at weight 1 for the pullback algebra (same module as tensor algebra)."""
        layers = _build_layers(QQ, d)
        pb = layers["pulled_back"]
        mod = pb.module
        for deg in range(-1, 3):
            for elem in mod.basis_weight_iter(deg, 1):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 1, degree {deg}"


# ============================================================================
# Layer 6: BarAlgebra (final construction)
# ============================================================================


class TestLayerBarAlgebra:
    """Tests for the full bar algebra B_π(pulled_back).

    This is the final output of the configuration model.
    """

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight1(self, d: int) -> None:
        """d²=0 at weight 1."""
        layers = _build_layers(QQ, d)
        bar = layers["bar"]
        mod = bar.module
        for deg in range(-1, 4):
            for elem in mod.graded_basis_by_weight(deg, 1):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 1, degree {deg}"

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight2(self, d: int) -> None:
        """d²=0 at weight 2."""
        layers = _build_layers(QQ, d)
        bar = layers["bar"]
        mod = bar.module
        for deg in range(-1, 4):
            for elem in mod.graded_basis_by_weight(deg, 2):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 2, degree {deg}"

    @pytest.mark.parametrize("d", [1, 2])
    def test_d_squared_zero_weight3_gf2(self, d: int) -> None:
        """d²=0 at weight 3 over GF(2)."""
        layers = _build_layers(GF(2), d)
        bar = layers["bar"]
        mod = bar.module
        for deg in range(-1, 3):
            for elem in mod.graded_basis_by_weight(deg, 3):
                dd = mod.boundary(mod.boundary(elem))
                assert dd == mod.zero(), f"d²≠0 at weight 3, degree {deg} (GF(2))"


# ============================================================================
# Full model tests
# ============================================================================


class TestConfigurationModelCore:
    """Tests for the configuration model and its homology."""

    def test_configuration_model_GF2(self) -> None:
        """chain_complex for euclidean configuration model over GF(2) succeeds."""

        model = euclidean_unordered_configuration_model(GF(2), 2)
        C = compute_chain_complex(model.module, degrees=range(3), weight=1)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_GF2_weight3(self, d: int) -> None:
        """chain_complex over GF(2) with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(GF(2), d)
        C = compute_chain_complex(model.module, degrees=range(-1, 3), weight=3, check=True)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_GF2_weight4(self, d: int) -> None:
        """chain_complex over GF(2) with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(GF(2), d)
        C = compute_chain_complex(model.module, degrees=range(-1, 2), weight=4, check=True)
        assert C is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_QQ_weight2(self, d: int) -> None:
        """chain_complex over QQ at weight=2 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, d)
        complex = compute_chain_complex(model.module, degrees=range(-1, 3), weight=2, check=True)
        assert complex is not None

    @pytest.mark.parametrize("d", [1, 2])
    def test_check_complex_QQ_weight3(self, d: int) -> None:
        """chain_complex over QQ at weight=3 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, d)
        complex = compute_chain_complex(model.module, degrees=range(-1, 3), weight=3, check=True)
        assert complex is not None

    def test_check_complex_QQ_weight4(self) -> None:
        """chain_complex over QQ at weight=4 with check=True does not raise an error."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        complex = compute_chain_complex(model.module, degrees=range(0, 2), weight=4, check=True)
        assert complex is not None


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
            for elem in model.module.graded_basis_by_weight(deg, 2):
                dd = model.boundary(model.boundary(elem))
                assert dd == model.module.zero(), f"d²≠0 at deg={deg} for {list(elem)[0][0]}"

    def test_dsquared_weight2_d2(self) -> None:
        """d²=0 at weight=2 for d=2 (all degrees up to 4)."""
        model = euclidean_unordered_configuration_model(QQ, 2)
        for deg in range(-1, 5):
            for elem in model.module.graded_basis_by_weight(deg, 2):
                dd = model.boundary(model.boundary(elem))
                assert dd == model.module.zero(), f"d²≠0 at deg={deg} for {list(elem)[0][0]}"


class TestConfigurationModelBasis:
    """Tests for the graded basis on the configuration model."""

    @pytest.mark.parametrize("d", [1, 2])
    def test_homology_basis_weight1(self, d: int) -> None:
        """homology_basis for euclidean configuration model at weight 1 returns a non-empty basis."""
        model = euclidean_unordered_configuration_model(QQ, d)
        for k in range(-2, 3):
            for w in range(1, 4):
                basis = model.module.graded_basis_by_weight(k, w)
                assert len(basis) == len(set(basis)), (
                    f"Basis at degree {k} weight {w} contains duplicates"
                )


class TestConfigurationModelComodule:
    def test_e_comodule_generator_chain_map_arity3(self) -> None:
        """e_comodule_on_generator satisfies the cooperad-level chain map at arity 3.

        For c ∈ C(3), checks ν(∂_C c) = (d_E ⊗ 1 + 1 ⊗ ∂_C)(ν(c))
        where ν: C → E ⊗ C is the E-comodule structure map
        and ∂_C is the cooperad differential.
        """

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
        where ν: C → E ⊗ C is the E-comodule structure map
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
