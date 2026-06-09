"""Tests for the torus (Barratt--Eccles) cochain model and configuration model.

The torus cochains ``C*(S¹) ⊗ C*(S¹)`` carry an explicit *Barratt--Eccles*
algebra structure (the surjection operad is not Hopf, so a Surjection-algebra
structure is not available — see ``src/uconf/algebraic/torus.py``).

Status note: the resulting configuration-model chain complex satisfies
``d² = 0`` over ``GF(2)`` (the characteristic this project targets).  Over
``GF(3)``/``QQ`` the Case-4 (``[γ]``-involving) sign in the algebra structure
is not yet settled (flagged OQ1/OQ2/OQ3 in ``torus.py``); the ``d² = 0`` test
over those rings is therefore marked ``xfail``.
"""

import pytest
from sage.all import GF, QQ

from uconf.models.barratt_eccles import BarrattEccles
from uconf.algebraic.torus import BarrattEcclesTorusCochainAlgebra, ReducedTorusCochains
from uconf.algebraic.torus_configuration import (
    _build_torus_layers,
    unordered_torus_configuration_model,
)


# --------------------------------------------------------------------------- #
# The explicit Barratt--Eccles algebra on the torus cochains.
# --------------------------------------------------------------------------- #


class TestTorusCochainAlgebra:
    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_module_basis_and_grading(self, base_ring):
        mod = ReducedTorusCochains(base_ring)
        assert mod.degree_on_basis("0") == 0
        assert mod.degree_on_basis("α") == -1
        assert mod.degree_on_basis("β") == -1
        assert mod.degree_on_basis("γ") == -2
        assert mod.connectivity == -2

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_unit_acts_as_identity(self, base_ring):
        alg = BarrattEcclesTorusCochainAlgebra(base_ring)
        unit = BarrattEccles.unit(base_ring)
        for name in ["0", "α", "β", "γ"]:
            g = alg.module.generator(name)
            assert alg.act(unit, [g]) == g

    def test_alpha_square_is_zero(self):
        """The cup product μ_id([α],[α]) vanishes (single permutation, s≠a)."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        ga = alg.module.generator("α")
        e_id = BarrattEccles(2, QQ)([[1, 2]])  # degree 0
        assert alg.act(e_id, [ga, ga]) == alg.module.zero()

    def test_alpha_beta_graded_commutativity(self):
        """μ_id([α],[β]) = -μ_id([β],[α]) (both odd degree ⇒ anticommute)."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        ga, gb = alg.module.generator("α"), alg.module.generator("β")
        e_id = BarrattEccles(2, QQ)([[1, 2]])
        ab = alg.act(e_id, [ga, gb])
        ba = alg.act(e_id, [gb, ga])
        assert ab != alg.module.zero()
        assert ab == -ba

    def test_case2_alpha_product(self):
        """A degree-1 arity-2 element sends ([α],[α]) to ±[α]."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        ga = alg.module.generator("α")
        x = BarrattEccles(2, QQ)([[1, 2], [2, 1]])  # degree 1, s = a = 2
        out = alg.act(x, [ga, ga])
        assert out == -ga  # (-1)^{a(a-1)/2} ψ = (-1)^1 · (+1)


# --------------------------------------------------------------------------- #
# The configuration-model pipeline.
# --------------------------------------------------------------------------- #


class TestTorusConfigurationModel:
    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_model_builds(self, base_ring):
        model = unordered_torus_configuration_model(base_ring)
        assert model is not None
        assert model.module is not None

    def test_manifold_model_is_barratt_eccles_algebra(self):
        layers = _build_torus_layers(GF(2))
        assert layers.manifold_model.operad_cls == BarrattEccles

    def test_coefficients_dimension(self):
        layers = _build_torus_layers(GF(2))
        assert layers.coefficients._dimension == 2  # torus has dimension 2

    def test_all_layers_present(self):
        layers = _build_torus_layers(GF(2))
        for field in (
            "manifold_model",
            "coefficients",
            "sLie",
            "XsLie",
            "BXsLie",
            "OBXsLie",
            "free_alg",
            "tensor_alg",
            "comodule_morphism",
            "pulled_back",
            "pi",
            "bar",
        ):
            assert getattr(layers, field) is not None, f"layer {field} missing"

    def test_d_squared_is_zero_over_gf2(self):
        """d² = 0 over GF(2) — the characteristic this project targets."""
        mod = unordered_torus_configuration_model(GF(2)).module
        for w in range(1, 4):
            for d in range(-3, 6):
                for basis in mod.graded_basis_by_weight(d, w):
                    dd = mod.boundary(mod.boundary(mod(basis)))
                    assert dd == mod.zero(), f"d²≠0 at (w={w}, d={d}) on {basis}: {dd}"

    @pytest.mark.xfail(
        reason="Case-4 ([γ]-involving) sign over ℚ/GF(3) is unresolved "
        "(OQ1/OQ2/OQ3 in torus.py); d² has residual ±2[γ] terms.",
        strict=True,
    )
    @pytest.mark.parametrize("base_ring", [GF(3), QQ])
    def test_d_squared_is_zero_over_q(self, base_ring):
        mod = unordered_torus_configuration_model(base_ring).module
        for w in range(1, 4):
            for d in range(-3, 6):
                for basis in mod.graded_basis_by_weight(d, w):
                    dd = mod.boundary(mod.boundary(mod(basis)))
                    assert dd == mod.zero(), f"d²≠0 at (w={w}, d={d}) on {basis}: {dd}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
