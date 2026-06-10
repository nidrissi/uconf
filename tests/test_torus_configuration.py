"""Tests for the torus (Barratt--Eccles) cochain model and configuration model.

The torus cochains ``C*(S¹) ⊗ C*(S¹)`` carry an explicit *Barratt--Eccles*
algebra structure (the surjection operad is not Hopf, so a Surjection-algebra
structure is not available — see ``src/uconf/algebraic/torus.py``).

The closed-form action implemented in ``torus.py`` is the tensor square of
the circle action via the Alexander--Whitney diagonal of the Barratt--Eccles
operad; :class:`TestTorusActionAgainstFirstPrinciples` re-derives it from
table reduction + the Berger--Fresse interval-cut action on ``N*(Δ¹)`` and
compares term by term, and the chain-map identity ``μ_{∂σ̲} = 0`` (the axiom
the previous, incorrect Case-4 sign violated in odd characteristic) is
checked over ``QQ``.
"""

import itertools

import pytest
from sage.all import GF, QQ

import uconf  # noqa: F401  — wires BarrattEccles.Element.table_reduction
from uconf.algebraic.simplicial import surjection_cochain_action
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.simplicial import SimplicialCochains
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
        """μ_id([α],[β]) = [γ] = -μ_id([β],[α]) (cup product orientation)."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        ga, gb = alg.module.generator("α"), alg.module.generator("β")
        gg = alg.module.generator("γ")
        e_id = BarrattEccles(2, QQ)([[1, 2]])
        assert alg.act(e_id, [ga, gb]) == gg
        assert alg.act(e_id, [gb, ga]) == -gg

    def test_case2_alpha_product(self):
        """A degree-1 arity-2 element sends ([α],[α]) to ±[α]."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        ga = alg.module.generator("α")
        x = BarrattEccles(2, QQ)([[1, 2], [2, 1]])  # degree 1, s = a = 2
        out = alg.act(x, [ga, ga])
        assert out == -ga  # (-1)^{a(a-1)/2} ψ = (-1)^1 · (+1)

    def test_case4_gamma_products(self):
        """γ-involving products carry the bi-graded interleaving sign."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        m = alg.module
        ga, gb, gg = m.generator("α"), m.generator("β"), m.generator("γ")
        x = BarrattEccles(2, QQ)([[1, 2], [2, 1]])  # degree 1
        # Values pinned to (μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E; [γ] is even but behaves
        # bi-gradedly: μ_x([α],[γ]) and μ_x([γ],[α]) have opposite signs.
        assert alg.act(x, [ga, gg]) == -gg
        assert alg.act(x, [gg, ga]) == gg
        assert alg.act(x, [gb, gg]) == -gg
        assert alg.act(x, [gg, gb]) == gg

    @pytest.mark.parametrize("n,max_degree", [(2, 4), (3, 2)])
    def test_chain_map_identity_over_q(self, n, max_degree):
        """μ_{∂σ̲}(z̲) = 0: the action is a chain map (boundary on the module
        is zero).  This is the axiom the pre-2026-06 Case-4 sign violated."""
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        gens = [alg.module.generator(name) for name in ["0", "α", "β", "γ"]]
        E_n = BarrattEccles(n, QQ)
        zero = alg.module.zero()
        for d in range(1, max_degree + 1):
            for x in E_n.graded_basis(d):
                bx = x.boundary()
                if not bx:
                    continue
                for inputs in itertools.product(gens, repeat=n):
                    assert alg.act(bx, list(inputs)) == zero


class TestTorusActionAgainstFirstPrinciples:
    """Compare the closed form with ``(μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E`` computed from
    table reduction and the interval-cut action on ``N*(Δ¹)``."""

    LABELS = ["0", "α", "β", "γ"]
    # label -> (first circle factor, second circle factor)
    FACTORS = {"0": ("*", "*"), "α": ("a", "*"), "β": ("*", "a"), "γ": ("a", "a")}
    COMBINE = {("*", "*"): "0", ("a", "*"): "α", ("*", "a"): "β", ("a", "a"): "γ"}

    @pytest.fixture(scope="class")
    def cochains(self):
        C = SimplicialCochains(N=1, base_ring=QQ)
        # q*: N*(S¹) ↪ N*(Δ¹) for the quotient Δ¹ ↠ S¹ = Δ¹/∂Δ¹.
        return {"*": C((0,)) + C((1,)), "a": C((0, 1))}

    def _circle_action(self, n, key, circle_labels, cochains, cache):
        """{'*'/'a': coeff} for a BE basis key acting on circle generators."""
        cache_key = (key, circle_labels)
        if cache_key in cache:
            return cache[cache_key]
        x = BarrattEccles(n, QQ)([list(p) for p in key])
        result = {}
        u = x.table_reduction()
        if u:
            res = surjection_cochain_action(u, tuple(cochains[lab] for lab in circle_labels))
            d = {k: v for k, v in res if v}
            c0, c1, ca = d.pop((0,), 0), d.pop((1,), 0), d.pop((0, 1), 0)
            assert not d and c0 == c1  # lands in the S¹ subspace
            result = {lab: v for lab, v in [("*", c0), ("a", ca)] if v}
        cache[cache_key] = result
        return result

    def _reference_action(self, key, labels, cochains, cache):
        """(μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E on torus basis labels, as {label: coeff}."""
        n = len(labels)
        xs = tuple(self.FACTORS[z][0] for z in labels)
        ys = tuple(self.FACTORS[z][1] for z in labels)
        xdeg = [1 if t == "a" else 0 for t in xs]
        ydeg = [1 if t == "a" else 0 for t in ys]
        out = {}
        for i in range(len(key)):
            # AW term (σ_0,…,σ_i) ⊗ (σ_i,…,σ_{s-1}) with the Hadamard
            # Koszul sign |right|·Σ|x_p| + Σ_{p<q} |y_p||x_q|.
            exp = (len(key) - 1 - i) * sum(xdeg)
            exp += sum(ydeg[p] * xdeg[q] for p in range(n) for q in range(p + 1, n))
            sign = -1 if exp % 2 else 1
            lx = self._circle_action(n, key[: i + 1], xs, cochains, cache)
            ry = self._circle_action(n, key[i:], ys, cochains, cache)
            for (xl, xc), (yl, yc) in itertools.product(lx.items(), ry.items()):
                lab = self.COMBINE[(xl, yl)]
                out[lab] = out.get(lab, 0) + sign * xc * yc
        return {k: v for k, v in out.items() if v}

    @pytest.mark.parametrize("n,max_degree", [(1, 0), (2, 3), (3, 2)])
    def test_closed_form_matches_reference(self, n, max_degree, cochains):
        alg = BarrattEcclesTorusCochainAlgebra(QQ)
        cache = {}
        for d in range(max_degree + 1):
            keys = [
                tuple(tuple(p.tuple()) for p in next(iter(x.support())))
                for x in BarrattEccles(n, QQ).graded_basis(d)
            ]
            for labels in itertools.product(self.LABELS, repeat=n):
                for key in keys:
                    lab, eps = alg._single_action(key, labels)
                    closed = {lab: eps} if eps else {}
                    reference = self._reference_action(key, labels, cochains, cache)
                    assert closed == reference, (key, labels)


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

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_d_squared_is_zero(self, base_ring):
        mod = unordered_torus_configuration_model(base_ring).module
        for w in range(1, 4):
            for d in range(-3, 6):
                for basis in mod.graded_basis_by_weight(d, w):
                    dd = mod.boundary(mod.boundary(mod(basis)))
                    assert dd == mod.zero(), f"d²≠0 at (w={w}, d={d}) on {basis}: {dd}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
