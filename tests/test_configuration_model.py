"""Systematic operadic/cooperadic axiom tests for the configuration model.

Tests fundamental algebraic axioms at each intermediate layer:

Layer 0: H = s⁻¹Lie ⊙ Surjection (Hadamard product operad)
Layer 1: C = B(H) (Bar construction cooperad)
Layer 2: P = Ω(C) = Ω(B(H)) (Cobar construction operad)
Layer 3: Free_P(k[d]) (Free algebra)
Layer 4: HadamardTensorAlgebra (sphere cochains ⊗ free algebra)
Layer 4½: Comodule morphism φ: P → S ⊙ P (chain map, operad map, equivariance, comodule)
Layer 5: PullbackAlgebra
Layer 6: BarAlgebra (bar algebra)

Axioms per operad layer: d²=0, unit, sequential/parallel associativity,
Leibniz rule, equivariance, d-σ commutativity.

Axioms per cooperad layer: d²=0, counit, sequential/parallel coassociativity,
coderivation, equivariance, d-σ commutativity.

Axioms per algebra layer: d²=0, unit action, associativity, Leibniz.

Basis sizes (for reference):
  H(2): 2 per deg (deg ≥ -1)
  H(3): 12/-2, 36/-1, 84/0, 180/1
  B(H)(2): 2 per deg (deg ≥ 0)
  B(H)(3): 12/-1, 48/0, 108/1
  Ω(B(H))(2): 2 per deg (deg ≥ -1)
  Ω(B(H))(3): 24/-2, 72/-1, 144/0, 264/1
  FreeAlg dim=1: w1 d1:1 | w2 d1+:1 | w3 d1:4..d5:80
  FreeAlg dim=2: w1 d2:1 | w2 d3+:1 | w3 d4:4..d5:12
  TensorAlg dim=1: w1 d0:1 | w2 d0+:1 | w3 d0:4...
  TensorAlg dim=2: w1 d0:1 | w2 d1+:1 | w3 d2:4...
  Bar dim=1: w1 d0:1 | w2 d0+:2 | w3 d-1:4..d7:1628 | w4 d-2:21..d0:534
  Bar dim=2: w1 d0:1 | w2 d0:1,d1+:2 | w3 d-1:4..d7:1234 | w4 d-2:21..d0:462
"""

from random import Random

import pytest
from sage.all import GF, QQ, Partitions, SymmetricGroup

from uconf import (
    compute_chain_complex,
    euclidean_unordered_configuration_model,
)
from uconf.algebraic.configuration import (
    ConfigurationLayers,
    _build_layers,
)
from uconf.core.signs import koszul_sign_of_permutation, sign_from_exponent
from uconf.sampling import sample_basis, sample_algebra_pool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(x):
    """Convert a SageMath element to {basis_key: coeff}, dropping zeros."""
    return {basis: coeff for basis, coeff in x if coeff != 0}


def _sample(population, k, rng):
    """Sample min(k, len(population)) without replacement."""
    if len(population) <= k:
        return list(population)
    return rng.sample(list(population), k)


def _all_decompositions(total_arity):
    """Yield all (i, m, n) with m + n - 1 = total_arity, m ≥ 1, n ≥ 1."""
    for m in range(1, total_arity + 1):
        n = total_arity - m + 1
        if n < 1:
            continue
        for i in range(1, m + 1):
            yield (i, m, n)


def _build_algebra_pool(mod, weights=(1, 2), deg_range=range(-1, 6)):
    """Collect basis elements from *mod* across *weights* and *deg_range*."""
    pool = []
    for w in weights:
        for d in deg_range:
            pool.extend(mod.graded_basis_by_weight(d, w))
    return pool


def _elem_degree(mod, elem):
    """Return the degree of a (possibly tensor-product) basis element."""
    return mod.degree_on_basis(next(iter(elem.monomial_coefficients())))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEED = 20260330


@pytest.fixture
def rng():
    return Random(_SEED)


@pytest.fixture(params=[QQ, GF(2)], ids=["Q", "F2"])
def ring(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=["d1", "d2"])
def dim(request):
    return request.param


@pytest.fixture
def layers(dim: int, ring):
    """All configuration model layers at given dimension and ring."""
    return _build_layers(ring, dim)


# ===========================================================================
# Layer 0: H = s⁻¹Lie ⊙ Surjection
# ===========================================================================


class TestLayer0_H:
    """Operad axioms for H = s⁻¹Lie ⊙ Surjection."""

    class TestDSquared:
        @pytest.mark.parametrize(
            "n,d",
            [
                (2, -1),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, -2),
                (3, -1),
                (3, 0),
                (3, 1),
                (4, -3),
                (4, -2),
                (4, -1),
                (4, 0),
            ],
        )
        def test_d_squared(self, n: int, d: int, layers: ConfigurationLayers, ring):
            Hn = layers.XsLie(n, ring)
            tested = 0
            for elem in Hn.graded_basis(d):
                tested += 1
                assert elem.boundary().boundary() == Hn.zero()
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestUnit:
        @pytest.mark.parametrize("n", [2, 3])
        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_left(self, n: int, d: int, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            unit = H.unit(ring)
            for elem in H(n, ring).graded_basis(d):
                assert _as_dict(H.compose(unit, 1, elem)) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_right(self, d: int, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            unit = H.unit(ring)
            for elem in H(2, ring).graded_basis(d):
                for i in [1, 2]:
                    assert _as_dict(H.compose(elem, i, unit)) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-2, -1, 0])
        def test_right_arity3(self, d: int, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            unit = H.unit(ring)
            for elem in H(3, ring).graded_basis(d):
                for i in [1, 2, 3]:
                    assert _as_dict(H.compose(elem, i, unit)) == _as_dict(elem)

    class TestSeqAssociativity:
        """(x ∘_i y) ∘_{i+j-1} z = x ∘_i (y ∘_j z)."""

        @pytest.mark.parametrize("d_x", [-1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity2(self, d_x: int, d_y: int, d_z: int, layers: ConfigurationLayers, ring, rng):
            H = layers.XsLie
            H2 = H(2, ring)
            for x in sample_basis(H2, d_x, 12, rng):
                for y in sample_basis(H2, d_y, 12, rng):
                    for z in sample_basis(H2, d_z, 12, rng):
                        for i in [1, 2]:
                            for j in [1, 2]:
                                lhs = H.compose(H.compose(x, i, y), i + j - 1, z)
                                rhs = H.compose(x, i, H.compose(y, j, z))
                                assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity3_2(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring, rng):
            """x ∈ H(3), y,z ∈ H(2)."""
            H = layers.XsLie
            for x in sample_basis(H(3, ring), d_x, 12, rng):
                for y in sample_basis(H(2, ring), d_y, 12, rng):
                    for z in sample_basis(H(2, ring), d_z, 12, rng):
                        for i in [1, 2, 3]:
                            for j in [1, 2]:
                                lhs = H.compose(H.compose(x, i, y), i + j - 1, z)
                                rhs = H.compose(x, i, H.compose(y, j, z))
                                assert _as_dict(lhs) == _as_dict(rhs)

    class TestParAssociativity:
        """(x ∘_i y) ∘_{j+m-1} z = (-1)^{|y||z|} (x ∘_j z) ∘_i y, i < j."""

        @pytest.mark.parametrize("d_x", [-1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity2(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            H2 = H(2, ring)
            i, j, m = 1, 2, 2
            for x in H2.graded_basis(d_x):
                for y in H2.graded_basis(d_y):
                    for z in H2.graded_basis(d_z):
                        lhs = H.compose(H.compose(x, i, y), j + m - 1, z)
                        sign = sign_from_exponent(y.degree() * z.degree())
                        rhs = sign * H.compose(H.compose(x, j, z), i, y)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity3(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring, rng):
            """x ∈ H(3), y,z ∈ H(2): par-assoc for i < j ≤ arity(x)."""
            H = layers.XsLie
            m = 2  # arity of y
            for x in sample_basis(H(3, ring), d_x, 3, rng):
                for y in sample_basis(H(2, ring), d_y, 2, rng):
                    for z in sample_basis(H(2, ring), d_z, 2, rng):
                        for i in range(1, 4):
                            for j in range(i + 1, 4):
                                lhs = H.compose(H.compose(x, i, y), j + m - 1, z)
                                sign = sign_from_exponent(y.degree() * z.degree())
                                rhs = sign * H.compose(H.compose(x, j, z), i, y)
                                assert _as_dict(lhs) == _as_dict(rhs)

    class TestLeibniz:
        """d(x ∘_i y) = dx ∘_i y + (-1)^{|x|} x ∘_i dy."""

        @pytest.mark.parametrize("d_x", [-1, 0, 1])
        @pytest.mark.parametrize("d_y", [-1, 0, 1])
        def test_arity2(self, d_x, d_y, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            H2 = H(2, ring)
            for x in H2.graded_basis(d_x):
                for y in H2.graded_basis(d_y):
                    for i in [1, 2]:
                        lhs = H.compose(x, i, y).boundary()
                        rhs = H.compose(x.boundary(), i, y) + sign_from_exponent(
                            x.degree()
                        ) * H.compose(x, i, y.boundary())
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0, 1])
        def test_arity3_2(self, d_x, d_y, layers: ConfigurationLayers, ring, rng):
            """x ∈ H(3), y ∈ H(2)."""
            H = layers.XsLie
            for x in sample_basis(H(3, ring), d_x, 3, rng):
                for y in sample_basis(H(2, ring), d_y, 2, rng):
                    for i in [1, 2, 3]:
                        lhs = H.compose(x, i, y).boundary()
                        rhs = H.compose(x.boundary(), i, y) + sign_from_exponent(
                            x.degree()
                        ) * H.compose(x, i, y.boundary())
                        assert _as_dict(lhs) == _as_dict(rhs)

    class TestEquivariance:
        @pytest.mark.parametrize("n", [2, 3])
        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_d_commutes(self, n, d, layers: ConfigurationLayers, ring, rng):
            H = layers.XsLie
            Hn = H(n, ring)
            for elem in sample_basis(Hn, d, 10, rng):
                for sigma in SymmetricGroup(n):
                    sl = list(sigma.tuple())
                    assert _as_dict(elem.permute(sl).boundary()) == _as_dict(
                        elem.boundary().permute(sl)
                    )

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_identity(self, d, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            for elem in H(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([1, 2])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_involution_arity2(self, d, layers: ConfigurationLayers, ring):
            H = layers.XsLie
            for elem in H(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([2, 1]).permute([2, 1])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-2, -1, 0])
        def test_group_action_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            H = layers.XsLie
            perms = list(SymmetricGroup(3))
            for elem in sample_basis(H(3, ring), d, 5, rng):
                for sigma in _sample(perms, 3, rng):
                    for tau in _sample(perms, 3, rng):
                        sl, tl = list(sigma.tuple()), list(tau.tuple())
                        cl = list((sigma * tau).tuple())
                        assert _as_dict(elem.permute(sl).permute(tl)) == _as_dict(elem.permute(cl))


# ===========================================================================
# Layer 1: C = B(H) (cooperad)
# ===========================================================================


class TestLayer1_BH:
    """Cooperad axioms for B(H)."""

    class TestDSquared:
        @pytest.mark.parametrize(
            "n,d",
            [(2, 0), (2, 1), (2, 2), (2, 3), (3, -1), (3, 0), (3, 1), (4, -2), (4, -1), (4, 0)],
        )
        def test_d_squared(self, n, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            Cn = C(n, ring)
            tested = 0
            for elem in sample_basis(Cn, d, 30, rng):
                tested += 1
                assert elem.boundary().boundary() == Cn.zero()
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestCounit:
        def test_on_generator(self, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            assert C.counit(C.counit_element(ring)) == 1

        def test_zero_arity2(self, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            for elem in C(2, ring).graded_basis(0):
                assert C.counit(elem) == 0

        def test_reduced_kills_counit(self, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            C1 = C(1, ring)
            assert C.reduced(C.counit_element(ring)) == C1.zero()

        def test_Delta_1_1_n_is_eta_tensor_x(self, layers: ConfigurationLayers, ring):
            """Δ_{1;1,n}(x) = η ⊗ x (left counit identity)."""
            C = layers.BXsLie
            for n in [2, 3]:
                Cn = C(n, ring)
                C1 = C(1, ring)
                eta = C1.term(C.unit_key())
                for elem in Cn.graded_basis(0):
                    result = C.infinitesimal_cocompose(elem, 1, 1, n)
                    expected = eta.tensor(elem)
                    assert result == expected

    class TestSeqCoassociativity:
        """(id⊗Δ) ∘ Δ = (Δ⊗id) ∘ Δ on B(H)(4), m=n=p=2, i=j=1."""

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_arity4(self, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            C3 = C(3, ring)
            for x in sample_basis(C(4, ring), d, 10, rng):
                lhs = {}
                for (a, dk), c1 in C.infinitesimal_cocompose(x, 1, 2, 3):
                    for (b, c), c2 in C.infinitesimal_cocompose(C3(dk), 1, 2, 2):
                        k = (a, b, c)
                        lhs[k] = lhs.get(k, 0) + c1 * c2
                lhs = {k: v for k, v in lhs.items() if v}
                rhs = {}
                for (e, c), c1 in C.infinitesimal_cocompose(x, 1, 3, 2):
                    for (a, b), c2 in C.infinitesimal_cocompose(C3(e), 1, 2, 2):
                        k = (a, b, c)
                        rhs[k] = rhs.get(k, 0) + c1 * c2
                rhs = {k: v for k, v in rhs.items() if v}
                assert lhs == rhs

    class TestParCoassociativity:
        """Parallel coassociativity on B(H)(4): i=1, j=2, m=n=p=2."""

        @pytest.mark.parametrize("d", [-1, 0])
        def test_arity4(self, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            C3 = C(3, ring)
            C2 = C(2, ring)
            for x in sample_basis(C(4, ring), d, 10, rng):
                lhs = {}
                for (e, ck), c1 in C.infinitesimal_cocompose(x, 3, 3, 2):
                    for (a, b), c2 in C.infinitesimal_cocompose(C3(e), 1, 2, 2):
                        k = (a, b, ck)
                        lhs[k] = lhs.get(k, 0) + c1 * c2
                lhs = {k: v for k, v in lhs.items() if v}
                rhs = {}
                for (f, bk), c1 in C.infinitesimal_cocompose(x, 1, 3, 2):
                    for (a, ck), c2 in C.infinitesimal_cocompose(C3(f), 2, 2, 2):
                        sign = sign_from_exponent(C2.degree_on_basis(bk) * C2.degree_on_basis(ck))
                        k = (a, bk, ck)
                        rhs[k] = rhs.get(k, 0) + sign * c1 * c2
                rhs = {k: v for k, v in rhs.items() if v}
                assert lhs == rhs

    class TestCoderivation:
        """d is a coderivation: Δ(dx) = (d⊗1 + (-1)^|a| 1⊗d)(Δ(x))."""

        @pytest.mark.parametrize("n", [2, 3, 4])
        @pytest.mark.parametrize("d", [0, 1, 2])
        def test_coderivation(self, n, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            Cn = C(n, ring)
            for x in sample_basis(Cn, d, 15, rng):
                dx = x.boundary()
                for i, m, k in _all_decompositions(n):
                    Cm, Ck = C(m, ring), C(k, ring)
                    lhs = {}
                    for (l, r), c in C.infinitesimal_cocompose(dx, i, m, k):
                        lhs[(l, r)] = lhs.get((l, r), 0) + c
                    lhs = {k: v for k, v in lhs.items() if v}
                    rhs = {}
                    for (l, r), c in C.infinitesimal_cocompose(x, i, m, k):
                        l_deg = Cm.degree_on_basis(l)
                        for dl, dc in Cm(l).boundary():
                            rhs[(dl, r)] = rhs.get((dl, r), 0) + c * dc
                        sgn = sign_from_exponent(l_deg)
                        for dr, dc in Ck(r).boundary():
                            rhs[(l, dr)] = rhs.get((l, dr), 0) + sgn * c * dc
                    rhs = {k: v for k, v in rhs.items() if v}
                    assert lhs == rhs

    class TestEquivariance:
        @pytest.mark.parametrize("d", [0, 1, 2])
        def test_d_commutes_arity2(self, d, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            for elem in C(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([2, 1]).boundary()) == _as_dict(
                    elem.boundary().permute([2, 1])
                )

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_d_commutes_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            for elem in sample_basis(C(3, ring), d, 8, rng):
                for sigma in SymmetricGroup(3):
                    sl = list(sigma.tuple())
                    assert _as_dict(elem.permute(sl).boundary()) == _as_dict(
                        elem.boundary().permute(sl)
                    )

        @pytest.mark.parametrize("d", [0, 1, 2])
        def test_identity_arity2(self, d, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            for elem in C(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([1, 2])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [0, 1])
        def test_involution_arity2(self, d, layers: ConfigurationLayers, ring):
            C = layers.BXsLie
            for elem in C(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([2, 1]).permute([2, 1])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-1, 0])
        def test_group_action_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            C = layers.BXsLie
            perms = list(SymmetricGroup(3))
            for elem in sample_basis(C(3, QQ), d, 5, rng):
                for sigma in _sample(perms, 3, rng):
                    for tau in _sample(perms, 3, rng):
                        sl, tl = list(sigma.tuple()), list(tau.tuple())
                        cl = list((sigma * tau).tuple())
                        assert _as_dict(elem.permute(sl).permute(tl)) == _as_dict(elem.permute(cl))


# ===========================================================================
# Layer 2: P = Ω(B(H)) (cobar operad)
# ===========================================================================


class TestLayer2_ΩBH:
    """Operad axioms for Ω(B(H))."""

    class TestDSquared:
        @pytest.mark.parametrize(
            "n,d",
            [
                (2, -1),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, -2),
                (3, -1),
                (3, 0),
                (3, 1),
                (4, -3),
                (4, -2),
                (4, -1),
                (4, 0),
                (4, 1),
            ],
        )
        def test_d_squared(self, n, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            Pn = P(n, ring)
            for elem in Pn.graded_basis(d):
                assert elem.boundary().boundary() == Pn.zero()

    class TestUnit:
        @pytest.mark.parametrize("n", [2, 3])
        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_left(self, n, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            unit = P.unit(ring)
            for elem in sample_basis(P(n, ring), d, 20, rng):
                assert _as_dict(P.compose(unit, 1, elem)) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_right(self, d, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            unit = P.unit(ring)
            for elem in P(2, ring).graded_basis(d):
                for i in [1, 2]:
                    assert _as_dict(P.compose(elem, i, unit)) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-2, -1, 0])
        def test_right_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            unit = P.unit(ring)
            for elem in sample_basis(P(3, ring), d, 10, rng):
                for i in [1, 2, 3]:
                    assert _as_dict(P.compose(elem, i, unit)) == _as_dict(elem)

    class TestSeqAssociativity:
        """(x ∘_i y) ∘_{i+j-1} z = x ∘_i (y ∘_j z)."""

        @pytest.mark.parametrize("d_x", [-1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity2(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            P2 = P(2, ring)
            for x in P2.graded_basis(d_x):
                for y in P2.graded_basis(d_y):
                    for z in P2.graded_basis(d_z):
                        for i in [1, 2]:
                            for j in [1, 2]:
                                lhs = P.compose(P.compose(x, i, y), i + j - 1, z)
                                rhs = P.compose(x, i, P.compose(y, j, z))
                                assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity3_2(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring, rng):
            """x ∈ P(3), y,z ∈ P(2)."""
            P = layers.OBXsLie
            xs = P(3, ring).graded_basis(d_x)
            ys = P(2, ring).graded_basis(d_y)
            zs = P(2, ring).graded_basis(d_z)
            for x in xs:
                for y in ys:
                    for z in zs:
                        for i in [1, 2, 3]:
                            for j in [1, 2]:
                                lhs = P.compose(P.compose(x, i, y), i + j - 1, z)
                                rhs = P.compose(x, i, P.compose(y, j, z))
                                assert _as_dict(lhs) == _as_dict(rhs)

    class TestParAssociativity:
        """(x ∘_i y) ∘_{j+m-1} z = (-1)^{|y|·|z|} (x ∘_j z) ∘_i y."""

        @pytest.mark.parametrize("d_x", [-1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity2(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            P2 = P(2, ring)
            i, j, m = 1, 2, 2
            for x in P2.graded_basis(d_x):
                for y in P2.graded_basis(d_y):
                    for z in P2.graded_basis(d_z):
                        lhs = P.compose(P.compose(x, i, y), j + m - 1, z)
                        sign = sign_from_exponent(y.degree() * z.degree())
                        rhs = sign * P.compose(P.compose(x, j, z), i, y)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1])
        @pytest.mark.parametrize("d_y", [-1, 0])
        @pytest.mark.parametrize("d_z", [-1, 0])
        def test_arity3(self, d_x, d_y, d_z, layers: ConfigurationLayers, ring):
            """x ∈ P(3), y,z ∈ P(2): par-assoc for i < j ≤ arity(x)."""
            P = layers.OBXsLie
            m = 2  # arity of y
            xs = list(P(3, ring).graded_basis(d_x))
            ys = list(P(2, ring).graded_basis(d_y))
            zs = list(P(2, ring).graded_basis(d_z))
            for x in xs:
                for y in ys:
                    for z in zs:
                        for i in range(1, 4):
                            for j in range(i + 1, 4):
                                lhs = P.compose(P.compose(x, i, y), j + m - 1, z)
                                sign = sign_from_exponent(y.degree() * z.degree())
                                rhs = sign * P.compose(P.compose(x, j, z), i, y)
                                assert _as_dict(lhs) == _as_dict(rhs)

    class TestLeibniz:
        """d(x ∘_i y) = dx ∘_i y + (-1)^{|x|} x ∘_i dy."""

        @pytest.mark.parametrize("d_x", [-1, 0, 1])
        @pytest.mark.parametrize("d_y", [-1, 0, 1])
        def test_arity2(self, d_x, d_y, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            P2 = P(2, ring)
            for x in P2.graded_basis(d_x):
                for y in P2.graded_basis(d_y):
                    for i in [1, 2]:
                        lhs = P.compose(x, i, y).boundary()
                        rhs = P.compose(x.boundary(), i, y) + sign_from_exponent(
                            x.degree()
                        ) * P.compose(x, i, y.boundary())
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("d_x", [-2, -1, 0])
        @pytest.mark.parametrize("d_y", [-1, 0, 1])
        def test_arity3_2(self, d_x, d_y, layers: ConfigurationLayers, ring):
            """x ∈ P(3), y ∈ P(2)."""
            P = layers.OBXsLie
            xs = list(P(3, ring).graded_basis(d_x))
            ys = list(P(2, ring).graded_basis(d_y))
            for x in xs:
                for y in ys:
                    for i in [1, 2, 3]:
                        lhs = P.compose(x, i, y).boundary()
                        rhs = P.compose(x.boundary(), i, y) + sign_from_exponent(
                            x.degree()
                        ) * P.compose(x, i, y.boundary())
                        assert _as_dict(lhs) == _as_dict(rhs)

    class TestEquivariance:
        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_d_commutes_arity2(self, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            for elem in sample_basis(P(2, ring), d, 30, rng):
                assert _as_dict(elem.permute([2, 1]).boundary()) == _as_dict(
                    elem.boundary().permute([2, 1])
                )

        @pytest.mark.parametrize("d", [-2, -1, 0, 1])
        def test_d_commutes_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            for elem in sample_basis(P(3, ring), d, 30, rng):
                for sigma in SymmetricGroup(3):
                    sl = list(sigma.tuple())
                    assert _as_dict(elem.permute(sl).boundary()) == _as_dict(
                        elem.boundary().permute(sl)
                    )

        @pytest.mark.parametrize("d", [-2, -1, 0, 1])
        def test_d_commutes_arity4(self, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            for elem in sample_basis(P(4, ring), d, 30, rng):
                for sigma in SymmetricGroup(4):
                    sl = list(sigma.tuple())
                    assert _as_dict(elem.permute(sl).boundary()) == _as_dict(
                        elem.boundary().permute(sl)
                    )

        @pytest.mark.parametrize("d", [-1, 0, 1])
        def test_identity_arity2(self, d, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            for elem in P(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([1, 2])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-1, 0])
        def test_involution_arity2(self, d, layers: ConfigurationLayers, ring):
            P = layers.OBXsLie
            for elem in P(2, ring).graded_basis(d):
                assert _as_dict(elem.permute([2, 1]).permute([2, 1])) == _as_dict(elem)

        @pytest.mark.parametrize("d", [-2, -1, 0, 1])
        def test_group_action_arity3(self, d, layers: ConfigurationLayers, ring, rng):
            P = layers.OBXsLie
            perms = list(SymmetricGroup(3))
            for elem in sample_basis(P(3, ring), d, 10, rng):
                for sigma in perms:
                    for tau in perms:
                        sl, tl = list(sigma.tuple()), list(tau.tuple())
                        cl = list((sigma * tau).tuple())
                        assert _as_dict(elem.permute(sl).permute(tl)) == _as_dict(elem.permute(cl))


# ===========================================================================
# Layer 3: Free algebra T_P(k[d])
# ===========================================================================


class TestLayer3_ΩBH_Kd:
    """Free Ω(B(H))-algebra on the trivial module k[d]."""

    class TestDSquared:
        @pytest.mark.parametrize("weight", [1, 2, 3, 4])
        def test_d_squared(self, dim, ring, layers: ConfigurationLayers, weight, rng):
            mod = layers.free_alg.module
            tested = 0
            for deg in range(1, 7):
                basis = list(mod.graded_basis_by_weight(deg, weight))
                for elem in basis:
                    tested += 1
                    assert mod.boundary(mod.boundary(elem)) == mod.zero(), (
                        f"d²≠0 w={weight} deg={deg} dim={dim} ring={ring}"
                    )
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestUnitAction:
        @pytest.mark.parametrize("weight", [1, 2, 3, 4])
        def test_unit(self, layers: ConfigurationLayers, weight):
            fa = layers.free_alg
            mod = fa.module
            unit = layers.OBXsLie.unit(layers.bar.module.base_ring())
            for deg in range(1, 7):
                for elem in mod.graded_basis_by_weight(deg, weight):
                    assert _as_dict(fa.act(unit, [elem])) == _as_dict(elem)

    class TestAssociativityAction:
        """γ(p ∘_1 q; a, a, a) = γ(p; γ(q; a, a), a)."""

        @pytest.mark.parametrize("p_deg", [-1, 0])
        def test_compatible(self, p_deg, layers: ConfigurationLayers, rng):
            fa = layers.free_alg
            mod = fa.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            if not list(P2.graded_basis(p_deg)):
                pytest.skip("No P(2) elements at degree 0")
            # Weight-1 module has exactly 1 generator per dimension,
            # so use repeated copies which is algebraically valid.
            w1 = []
            for d_try in range(-1, 6):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a = w1[0]
            for p in sample_basis(P2, p_deg, 2, rng):
                for q in sample_basis(P2, p_deg, 2, rng):
                    lhs = fa.act(P.compose(p, 1, q), [a, a, a])
                    rhs = fa.act(p, [fa.act(q, [a, a]), a])
                    assert _as_dict(lhs) == _as_dict(rhs)

            # Also test with the second insertion position: p ∘_2 q
            # γ(p ∘_2 q; a,a,a) = (-1)^{|q|·|a|} γ(p; a, γ(q; a,a))
            for p in sample_basis(P2, p_deg, 2, rng):
                for q in sample_basis(P2, p_deg, 2, rng):
                    lhs = fa.act(P.compose(p, 2, q), [a, a, a])
                    rhs = fa.act(p, [a, fa.act(q, [a, a])])
                    rhs *= sign_from_exponent(q.degree() * a.degree())
                    assert _as_dict(lhs) == _as_dict(rhs)

    class TestLeibnizAction:
        """d(γ(p; a₁, a₂)) = γ(dp; a₁, a₂)
        + (-1)^|p| γ(p; da₁, a₂) + (-1)^{|p|+|a₁|} γ(p; a₁, da₂)."""

        @pytest.mark.parametrize("p_deg", [-1, 0])
        def test_leibniz(self, p_deg, layers: ConfigurationLayers, rng):
            fa = layers.free_alg
            mod = fa.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            if not list(P2.graded_basis(p_deg)):
                pytest.skip("No P(2) elements at degree 0")
            w1 = []
            for d_try in range(-1, 6):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a = w1[0]
            for p in sample_basis(P2, p_deg, 2, rng):
                lhs = mod.boundary(fa.act(p, [a, a]))
                rhs = fa.act(p.boundary(), [a, a])
                p_deg = p.degree()
                da = mod.boundary(a)
                rhs += sign_from_exponent(p_deg) * fa.act(p, [da, a])
                rhs += sign_from_exponent(p_deg + a.degree()) * fa.act(p, [a, da])
                assert _as_dict(lhs) == _as_dict(rhs)

    class TestEquivariance:
        """γ(p·σ; a₁,...,aₙ) = ε(σ⁻¹; a) · γ(p; a_{σ⁻¹(1)},...,a_{σ⁻¹(n)})."""

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2, 3])
        def test_equivariance_arity2(self, p_deg, layers: ConfigurationLayers, rng):
            fa = layers.free_alg
            mod = fa.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            if not list(P2.graded_basis(p_deg)):
                pytest.skip(f"No P(2) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng)
            if len(pool) < 2:
                pytest.skip("Not enough algebra elements")
            for _ in range(6):
                inputs = rng.choices(pool, k=2)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(2):
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 3)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 3)]
                    for p in sample_basis(P2, p_deg, 12, rng):
                        lhs = fa.act(p.permute(sl), inputs)
                        rhs = koszul * fa.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-2, -1, 0, 1, 2])
        def test_equivariance_arity3(self, p_deg, layers: ConfigurationLayers, rng):
            fa = layers.free_alg
            mod = fa.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P3 = P(3, R)
            if not list(P3.graded_basis(p_deg)):
                pytest.skip(f"No P(3) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng)
            if len(pool) < 3:
                pytest.skip("Not enough algebra elements")
            for _ in range(4):
                inputs = rng.choices(pool, k=3)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(3):
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 4)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 4)]
                    for p in sample_basis(P3, p_deg, 12, rng):
                        lhs = fa.act(p.permute(sl), inputs)
                        rhs = koszul * fa.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-3, -2, -1, 0, 1])
        def test_equivariance_arity4(self, p_deg, layers: ConfigurationLayers, rng):
            fa = layers.free_alg
            mod = fa.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P4 = P(4, R)
            # Use sample_basis as fast existence probe instead of materializing full P4 basis
            p_sample = sample_basis(P4, p_deg, 6, rng)
            if not p_sample:
                pytest.skip(f"No P(4) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng)
            if len(pool) < 4:
                pytest.skip("Not enough algebra elements")
            perms = _sample(list(SymmetricGroup(4)), 6, rng)
            for _ in range(3):
                inputs = rng.choices(pool, k=4)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in perms:
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 5)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 5)]
                    for p in p_sample:
                        lhs = fa.act(p.permute(sl), inputs)
                        rhs = koszul * fa.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)


# ===========================================================================
# Layer 4: HadamardTensorAlgebra
# ===========================================================================


class TestLayer4_S_ΩBH_Kd:
    """Hadamard tensor algebra (sphere cochains ⊗ free algebra)."""

    class TestDSquared:
        @pytest.mark.parametrize("weight", [1, 2, 3, 4])
        def test_d_squared(self, dim, ring, weight, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            tested = 0
            for deg in range(-1, 6):
                basis = ta.graded_basis_by_weight(deg, weight)
                for elem in basis:
                    tested += 1
                    assert ta.boundary(ta.boundary(elem)) == ta.module.zero(), (
                        f"d²≠0 w={weight} deg={deg} dim={dim} ring={ring}"
                    )
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestUnitAction:
        def test_unit(self, layers: ConfigurationLayers):
            ta = layers.tensor_alg
            unit = ta.operad_cls.unit(layers.bar.module.base_ring())
            for deg in range(-1, 4):
                for elem in ta.graded_basis_by_weight(deg, 1):
                    assert _as_dict(ta.act(unit, [elem])) == _as_dict(elem)

    class TestAssociativityAction:
        """γ(p ∘_1 q; a, a, a) = γ(p; γ(q; a, a), a)."""

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2])
        def test_algebra_associative(self, p_deg, dim, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            mod = ta.module
            R = layers.bar.module.base_ring()
            Q = ta.operad_cls
            Q2 = Q(2, R)
            if not list(Q2.graded_basis(p_deg)):
                pytest.skip(f"No Q(2) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(ta.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c = rng.choices(w1, k=3)
            for p in sample_basis(Q2, p_deg, 2, rng, sphere_nontrivial=True, sphere_dim=dim):
                for q in sample_basis(Q2, p_deg, 2, rng, sphere_nontrivial=True, sphere_dim=dim):
                    lhs = ta.act(Q.compose(p, 1, q), [a, b, c])
                    rhs = ta.act(p, [ta.act(q, [a, b]), c])
                    assert _as_dict(lhs) == _as_dict(rhs)

            for p in sample_basis(Q2, p_deg, 2, rng, sphere_nontrivial=True, sphere_dim=dim):
                for q in sample_basis(Q2, p_deg, 2, rng, sphere_nontrivial=True, sphere_dim=dim):
                    lhs = ta.act(Q.compose(p, 2, q), [a, b, c])
                    rhs = ta.act(p, [a, ta.act(q, [b, c])])
                    # a.degree() doesn't exist on CombinatorialFreeModule_Tensor
                    rhs_sign = sign_from_exponent(mod.degree_on_basis(a.support()[0]) * q.degree())
                    rhs *= rhs_sign
                    assert _as_dict(lhs) == _as_dict(rhs)

    class TestLeibnizAction:
        """d(γ(p; a₁, a₂)) = γ(dp; a₁, a₂)
        + (-1)^|p| γ(p; da₁, a₂) + (-1)^{|p|+|a₁|} γ(p; a₁, da₂)."""

        @pytest.mark.parametrize("p_deg", [-1, 0])
        def test_leibniz(self, p_deg, dim, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            R = layers.bar.module.base_ring()
            Q = ta.operad_cls
            Q2 = Q(2, R)
            if not list(Q2.graded_basis(p_deg)):
                pytest.skip(f"No Q(2) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(ta.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a = w1[0]
            for p in sample_basis(Q2, p_deg, 2, rng, sphere_nontrivial=True, sphere_dim=dim):
                lhs = ta.boundary(ta.act(p, [a, a]))
                rhs = ta.act(p.boundary(), [a, a])
                p_deg_val = p.degree()
                da = ta.boundary(a)
                a_deg = ta.module.degree_on_basis(next(iter(a.monomial_coefficients())))
                rhs += sign_from_exponent(p_deg_val) * ta.act(p, [da, a])
                rhs += sign_from_exponent(p_deg_val + a_deg) * ta.act(p, [a, da])
                assert _as_dict(lhs) == _as_dict(rhs)

    class TestEquivariance:
        """γ(p·σ; a₁,...,aₙ) = ε(σ⁻¹; a) · γ(p; a_{σ⁻¹(1)},...,a_{σ⁻¹(n)})."""

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2])
        def test_equivariance_arity2(self, p_deg, dim, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            mod = ta.module
            R = layers.bar.module.base_ring()
            Q = ta.operad_cls
            Q2 = Q(2, R)
            if not list(Q2.graded_basis(p_deg)):
                pytest.skip(f"No Q(2) elements at degree {p_deg}")
            pool = sample_algebra_pool(ta, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 2:
                pytest.skip("Not enough algebra elements")
            for _ in range(6):
                inputs = rng.choices(pool, k=2)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(2):
                    sigma_inv = sigma.inverse()
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma_inv(i) - 1 for i in range(1, 3)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma_inv(i) - 1] for i in range(1, 3)]
                    for p in sample_basis(Q2, p_deg, 3, rng, sphere_nontrivial=True, sphere_dim=dim):
                        lhs = ta.act(p.permute(sl), inputs)
                        rhs = koszul * ta.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-2, -1, 0])
        def test_equivariance_arity3(self, p_deg, dim, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            mod = ta.module
            R = layers.bar.module.base_ring()
            Q = ta.operad_cls
            Q3 = Q(3, R)
            if not list(Q3.graded_basis(p_deg)):
                pytest.skip(f"No Q(3) elements at degree {p_deg}")
            pool = sample_algebra_pool(ta, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 3:
                pytest.skip("Not enough algebra elements")
            for _ in range(4):
                inputs = rng.choices(pool, k=3)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(3):
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 4)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 4)]
                    for p in sample_basis(Q3, p_deg, 12, rng, sphere_nontrivial=True, sphere_dim=dim):
                        lhs = ta.act(p.permute(sl), inputs)
                        rhs = koszul * ta.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-3, -2, -1, 0])
        def test_equivariance_arity4(self, p_deg, dim, layers: ConfigurationLayers, rng):
            ta = layers.tensor_alg
            mod = ta.module
            R = layers.bar.module.base_ring()
            Q = ta.operad_cls
            Q4 = Q(4, R)
            # Use sample_basis as a fast existence probe instead of materializing
            # the full Q4 basis (which is expensive for Hadamard products of complex operads).
            p_sample = sample_basis(Q4, p_deg, 4, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not p_sample:
                pytest.skip(f"No sphere-admissible Q(4) elements at degree {p_deg}")
            pool = sample_algebra_pool(ta, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 4:
                pytest.skip("Not enough algebra elements")
            # Sample a subset of S4 permutations instead of all 24 to bound cost.
            all_perms = list(SymmetricGroup(4))
            perms = _sample(all_perms, min(6, len(all_perms)), rng)
            for _ in range(1):
                inputs = rng.choices(pool, k=4)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in perms:
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 5)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 5)]
                    for p in p_sample:
                        lhs = ta.act(p.permute(sl), inputs)
                        rhs = koszul * ta.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)


# ===========================================================================
# Layer 4½: Comodule morphism φ: P = Ω(C) → S ⊙ P
# ===========================================================================


class TestLayer4h_comodule_morphism:
    """Verify that the comodule morphism φ: Ω(C) → S ⊙ Ω(C) satisfies
    chain map, operad map, equivariance, and comodule map axioms.
    """

    class TestChainMap:
        """φ(dp) = d(φ(p)) — the morphism commutes with differentials."""

        @pytest.mark.parametrize("n", [2, 3])
        @pytest.mark.parametrize("deg", [-1, 0, 1, 2])
        def test_chain_map(self, n, deg, dim, layers: ConfigurationLayers, rng):
            P = layers.OBXsLie
            phi = layers.comodule_morphism
            R = layers.bar.module.base_ring()
            Pn = P(n, R)
            basis = sample_basis(Pn, deg, 10, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not basis:
                pytest.skip(f"No P({n}) elements at degree {deg}")
            tested = 0
            for p_elem in basis:
                phi_dp = phi(p_elem.boundary())
                d_phi_p = phi(p_elem).boundary()
                tested += 1
                assert phi_dp == d_phi_p, f"φ(dp) ≠ d(φ(p)) at n={n}, deg={deg} for p={p_elem}"
            assert tested > 0, "No basis elements tested (nontriviality check)"

    class TestOperadMap:
        """φ(p ∘ᵢ q) = φ(p) ∘ᵢ φ(q) — the morphism respects operad composition."""

        @pytest.mark.parametrize("p_deg", [-1, 0])
        @pytest.mark.parametrize("q_deg", [-1, 0])
        def test_operad_map(self, p_deg, q_deg, dim, layers: ConfigurationLayers, rng):
            P = layers.OBXsLie
            Q = layers.comodule_morphism.target  # S ⊙ P
            phi = layers.comodule_morphism
            R = layers.bar.module.base_ring()
            P2 = P(2, R)
            if not list(P2.graded_basis(p_deg)) or not list(P2.graded_basis(q_deg)):
                pytest.skip(f"No P(2) elements at deg {p_deg} or {q_deg}")
            tested = 0
            for p in sample_basis(P2, p_deg, 3, rng, sphere_nontrivial=True, sphere_dim=dim):
                for q in sample_basis(P2, q_deg, 3, rng, sphere_nontrivial=True, sphere_dim=dim):
                    for i in [1, 2]:
                        lhs = phi(P.compose(p, i, q))
                        rhs = Q.compose(phi(p), i, phi(q))
                        tested += 1
                        assert lhs == rhs, (
                            f"φ(p ∘_{i} q) ≠ φ(p) ∘_{i} φ(q) at p_deg={p_deg}, q_deg={q_deg}"
                        )
            assert tested > 0, "No compositions tested (nontriviality check)"

    class TestEquivariance:
        """φ(p · σ) = φ(p) · σ — the morphism respects symmetric group action."""

        @pytest.mark.parametrize("n", [2, 3])
        @pytest.mark.parametrize("deg", [-1, 0, 1])
        def test_equivariance(self, n, deg, dim, layers: ConfigurationLayers, rng):
            P = layers.OBXsLie
            phi = layers.comodule_morphism
            R = layers.bar.module.base_ring()
            Pn = P(n, R)
            basis = sample_basis(Pn, deg, 5, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not basis:
                pytest.skip(f"No P({n}) elements at degree {deg}")
            Sn = SymmetricGroup(n)
            tested = 0
            for p_elem in basis:
                for sigma in _sample(list(Sn), min(4, Sn.order()), rng):
                    p_sigma = p_elem.permute(sigma)
                    phi_p_sigma = phi(p_sigma)
                    phi_p = phi(p_elem)
                    phi_p_acted = phi_p.permute(sigma)
                    tested += 1
                    assert phi_p_sigma == phi_p_acted, (
                        f"φ(p·σ) ≠ φ(p)·σ at n={n}, deg={deg}, σ={sigma}"
                    )
            assert tested > 0, "No equivariance tests run (nontriviality check)"

    class TestComoduleMap:
        """Unit preservation and compatibility with E-comodule structure."""

        def test_unit_preservation(self, layers: ConfigurationLayers):
            """φ(id_P) = id_{S⊙P} — the unit is preserved."""
            P = layers.OBXsLie
            Q = layers.comodule_morphism.target
            phi = layers.comodule_morphism
            R = layers.bar.module.base_ring()
            unit_P = P.unit(R)
            result = phi(unit_P)
            unit_Q = Q.unit(R)
            assert _as_dict(result) == _as_dict(unit_Q), (
                f"φ(id) = {result} ≠ {unit_Q} = id_{'{S⊙P}'}"
            )

        @pytest.mark.parametrize("deg", [-1, 0, 1])
        def test_factorisation_through_e_comodule(self, deg, dim, layers: ConfigurationLayers, rng):
            """φ = (table_reduction ⊗ id) ∘ Δ — the morphism factorises correctly."""
            P = layers.OBXsLie
            phi = layers.comodule_morphism
            R = layers.bar.module.base_ring()
            from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism

            Delta = make_e_comodule_morphism(layers.BXsLie)
            Pn = P(2, R)
            Q = phi.target
            Qn = Q(2, R)
            basis = sample_basis(Pn, deg, 5, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not basis:
                pytest.skip(f"No P(2) elements at degree {deg}")
            tested = 0
            for p_elem in basis:
                # Apply E-comodule morphism, then table-reduce left factor
                delta_p = Delta(p_elem)
                delta_parent = delta_p.parent()
                be_parent = delta_parent.left_parent()
                result = Qn.zero()
                for (be_key, cobar_key), coeff in delta_p:
                    surj_elem = be_parent(be_key).table_reduction()
                    for surj_key, surj_coeff in surj_elem:
                        result += coeff * surj_coeff * Qn((surj_key, cobar_key))
                direct = phi(p_elem)
                tested += 1
                assert result == direct, (
                    f"Factorisation mismatch at deg={deg}: (TR⊗id)∘Δ ≠ φ for p={p_elem}"
                )
            assert tested > 0, "No factorisation tests run (nontriviality check)"


# ===========================================================================
# Layer 5: PullbackAlgebra
# ===========================================================================


class TestLayer5_pb_S_ΩBH_Kd:
    """Pullback algebra (same module as tensor algebra, different action)."""

    class TestDSquared:
        @pytest.mark.parametrize("weight", [1, 2, 3])
        def test_d_squared(self, dim, ring, weight, layers: ConfigurationLayers, rng):
            mod = layers.pulled_back.module
            tested = 0
            for deg in range(-1, 6):
                for elem in sample_basis(mod, deg, 30, rng, weight=weight):
                    tested += 1
                    assert mod.boundary(mod.boundary(elem)) == mod.zero(), (
                        f"d²≠0 w={weight} deg={deg} dim={dim} ring={ring}"
                    )
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestUnitAction:
        def test_unit(self, layers: ConfigurationLayers):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            unit = layers.OBXsLie.unit(R)
            for deg in range(-1, 4):
                for elem in mod.graded_basis_by_weight(deg, 1):
                    assert _as_dict(pb.act(unit, [elem])) == _as_dict(elem)

    class TestAssociativityAction:
        """Tests the associativity axioms for the pullback algebra action."""

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2, 3])
        def test_pullback_algebra_associative(self, p_deg, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            p_elems = list(P2.graded_basis(p_deg))
            if not p_elems:
                pytest.skip(f"No P(2) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c = rng.choices(w1, k=3)
            for p in p_elems:
                for q in p_elems:
                    lhs = pb.act(P.compose(p, 1, q), [a, b, c])
                    rhs = pb.act(p, [pb.act(q, [a, b]), c])
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity at i=1 for p={p}, q={q}, a={a}, b={b}, c={c}"
                    )

            for p in p_elems:
                for q in p_elems:
                    lhs = pb.act(P.compose(p, 2, q), [a, b, c])
                    rhs = pb.act(p, [a, pb.act(q, [b, c])])
                    # a.degree() doesn't exist on CombinatorialFreeModule_Tensor
                    rhs_sign = sign_from_exponent(mod.degree_on_basis(a.support()[0]) * q.degree())
                    rhs *= rhs_sign
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity at i=2 for p={p}, q={q}, a={a}, b={b}, c={c}"
                    )

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2])
        @pytest.mark.parametrize("q_deg", [-1, 0, 1, 2])
        def test_pullback_algebra_associative_weight4_arity3_into_2(
            self, p_deg, q_deg, layers: ConfigurationLayers, rng
        ):
            """γ(p ∘_i q; a,b,c,d) with p arity 3, q arity 2 (i=1,2,3): weight 4."""
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P3 = P(3, R)
            P2 = P(2, R)
            p_elems = P3.graded_basis(p_deg)
            q_elems = P2.graded_basis(q_deg)
            if not p_elems or not q_elems:
                pytest.skip(f"No elements at p_deg={p_deg} or q_deg={q_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c, d = rng.choices(w1, k=4)
            a_deg = mod.degree_on_basis(a.support()[0])
            b_deg = mod.degree_on_basis(b.support()[0])
            for p in p_elems:
                for q in q_elems:
                    q_deg_val = q.degree()
                    # i=1: γ(p ∘_1 q; a,b,c,d) = γ(p; γ(q; a,b), c, d)
                    lhs = pb.act(P.compose(p, 1, q), [a, b, c, d])
                    rhs = pb.act(p, [pb.act(q, [a, b]), c, d])
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity (arity3 ∘_1 arity2) for p={p}, q={q}"
                    )
                    # i=2: γ(p ∘_2 q; a,b,c,d) = (-1)^{|a||q|} γ(p; a, γ(q; b,c), d)
                    lhs = pb.act(P.compose(p, 2, q), [a, b, c, d])
                    rhs = sign_from_exponent(a_deg * q_deg_val) * pb.act(
                        p, [a, pb.act(q, [b, c]), d]
                    )
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity (arity3 ∘_2 arity2) for p={p}, q={q}"
                    )
                    # i=3: γ(p ∘_3 q; a,b,c,d) = (-1)^{(|a|+|b|)|q|} γ(p; a, b, γ(q; c,d))
                    lhs = pb.act(P.compose(p, 3, q), [a, b, c, d])
                    rhs = sign_from_exponent((a_deg + b_deg) * q_deg_val) * pb.act(
                        p, [a, b, pb.act(q, [c, d])]
                    )
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity (arity3 ∘_3 arity2) for p={p}, q={q}"
                    )

        @pytest.mark.parametrize("p_deg", [-1, 0, 1])
        @pytest.mark.parametrize("q_deg", [-1, 0, 1])
        def test_pullback_algebra_associative_weight4_arity2_into_3(
            self, p_deg, q_deg, layers: ConfigurationLayers, rng
        ):
            """γ(p ∘_i q; a,b,c,d) with p arity 2, q arity 3 (i=1,2): weight 4."""
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            P3 = P(3, R)
            p_elems = P2.graded_basis(p_deg)
            q_elems = P3.graded_basis(q_deg)
            if not p_elems or not q_elems:
                pytest.skip(f"No elements at p_deg={p_deg} or q_deg={q_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c, d = rng.choices(w1, k=4)
            a_deg = mod.degree_on_basis(a.support()[0])
            for p in p_elems:
                for q in q_elems:
                    q_deg_val = q.degree()
                    # i=1: γ(p ∘_1 q; a,b,c,d) = γ(p; γ(q; a,b,c), d)
                    lhs = pb.act(P.compose(p, 1, q), [a, b, c, d])
                    rhs = pb.act(p, [pb.act(q, [a, b, c]), d])
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity (arity2 ∘_1 arity3) for p={p}, q={q}"
                    )
                    # i=2: γ(p ∘_2 q; a,b,c,d) = (-1)^{|a||q|} γ(p; a, γ(q; b,c,d))
                    lhs = pb.act(P.compose(p, 2, q), [a, b, c, d])
                    rhs = sign_from_exponent(a_deg * q_deg_val) * pb.act(
                        p, [a, pb.act(q, [b, c, d])]
                    )
                    assert _as_dict(lhs) == _as_dict(rhs), (
                        f"Failed associativity (arity2 ∘_2 arity3) for p={p}, q={q}"
                    )

    class TestLeibnizAction:
        """d(γ(p; a₁, a₂)) = γ(dp; a₁, a₂)
        + (-1)^|p| γ(p; da₁, a₂) + (-1)^{|p|+|a₁|} γ(p; a₁, da₂)."""

        @pytest.mark.parametrize("p_deg", [-1, 0])
        def test_leibniz_arity2(self, p_deg, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            p_elems = list(P2.graded_basis(p_deg))
            if not p_elems:
                pytest.skip(f"No P(2) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a = w1[0]
            for p in p_elems:
                lhs = mod.boundary(pb.act(p, [a, a]))
                rhs = pb.act(p.boundary(), [a, a])
                p_deg_val = p.degree()
                da = mod.boundary(a)
                a_deg = mod.degree_on_basis(next(iter(a.monomial_coefficients())))
                rhs += sign_from_exponent(p_deg_val) * pb.act(p, [da, a])
                rhs += sign_from_exponent(p_deg_val + a_deg) * pb.act(p, [a, da])
                assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-2, -1, 0])
        def test_leibniz_arity3(self, p_deg, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P3 = P(3, R)
            p_elems = list(P3.graded_basis(p_deg))
            if not p_elems:
                pytest.skip(f"No P(3) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c = rng.choices(w1, k=3)
            a_deg = mod.degree_on_basis(a.support()[0])
            b_deg = mod.degree_on_basis(b.support()[0])
            da = mod.boundary(a)
            db = mod.boundary(b)
            dc = mod.boundary(c)
            for p in p_elems:
                lhs = mod.boundary(pb.act(p, [a, b, c]))
                p_deg_val = p.degree()
                rhs = pb.act(p.boundary(), [a, b, c])
                rhs += sign_from_exponent(p_deg_val) * pb.act(p, [da, b, c])
                rhs += sign_from_exponent(p_deg_val + a_deg) * pb.act(p, [a, db, c])
                rhs += sign_from_exponent(p_deg_val + a_deg + b_deg) * pb.act(p, [a, b, dc])
                assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-3, -2, -1])
        def test_leibniz_arity4(self, p_deg, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P4 = P(4, R)
            p_elems = list(P4.graded_basis(p_deg))
            if not p_elems:
                pytest.skip(f"No P(4) elements at degree {p_deg}")
            w1 = []
            for d_try in range(-1, 4):
                w1.extend(mod.graded_basis_by_weight(d_try, 1))
            if not w1:
                pytest.skip("No weight-1 elements")
            a, b, c, d = rng.choices(w1, k=4)
            a_deg = mod.degree_on_basis(a.support()[0])
            b_deg = mod.degree_on_basis(b.support()[0])
            c_deg = mod.degree_on_basis(c.support()[0])
            da = mod.boundary(a)
            db = mod.boundary(b)
            dc = mod.boundary(c)
            dd = mod.boundary(d)
            for p in p_elems:
                lhs = mod.boundary(pb.act(p, [a, b, c, d]))
                p_deg_val = p.degree()
                rhs = pb.act(p.boundary(), [a, b, c, d])
                rhs += sign_from_exponent(p_deg_val) * pb.act(p, [da, b, c, d])
                rhs += sign_from_exponent(p_deg_val + a_deg) * pb.act(p, [a, db, c, d])
                rhs += sign_from_exponent(p_deg_val + a_deg + b_deg) * pb.act(p, [a, b, dc, d])
                rhs += sign_from_exponent(p_deg_val + a_deg + b_deg + c_deg) * pb.act(
                    p, [a, b, c, dd]
                )
                assert _as_dict(lhs) == _as_dict(rhs)

    class TestEquivariance:
        """γ(p·σ; a₁,...,aₙ) = ε(σ⁻¹; a) · γ(p; a_{σ⁻¹(1)},...,a_{σ⁻¹(n)})."""

        @pytest.mark.parametrize("p_deg", [-1, 0, 1, 2, 3])
        def test_equivariance_arity2(self, p_deg, dim, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P2 = P(2, R)
            # Use sample_basis as fast existence probe
            p_sample = sample_basis(P2, p_deg, 3, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not p_sample:
                pytest.skip(f"No sphere-admissible P(2) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 2:
                pytest.skip("Not enough algebra elements")
            for _ in range(6):
                inputs = rng.choices(pool, k=2)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(2):
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 3)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 3)]
                    for p in p_sample:
                        lhs = pb.act(p.permute(sl), inputs)
                        rhs = koszul * pb.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-2, -1, 0, 1, 2])
        def test_equivariance_arity3(self, p_deg, dim, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P3 = P(3, R)
            # Use sample_basis as fast existence probe instead of materializing full P3 basis
            p_sample = sample_basis(P3, p_deg, 6, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not p_sample:
                pytest.skip(f"No sphere-admissible P(3) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 3:
                pytest.skip("Not enough algebra elements")
            for _ in range(4):
                inputs = rng.choices(pool, k=3)
                degrees = [_elem_degree(mod, e) for e in inputs]
                for sigma in SymmetricGroup(3):
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 4)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 4)]
                    for p in p_sample:
                        lhs = pb.act(p.permute(sl), inputs)
                        rhs = koszul * pb.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)

        @pytest.mark.parametrize("p_deg", [-2, -1, 0, 1, 2])
        def test_equivariance_arity4(self, p_deg, dim, layers: ConfigurationLayers, rng):
            pb = layers.pulled_back
            mod = pb.module
            R = layers.bar.module.base_ring()
            P = layers.OBXsLie
            P4 = P(4, R)
            # Use sample_basis as a fast existence probe instead of materializing
            # the full P4 basis (which is expensive for CobarConstruction at arity 4).
            p_sample = sample_basis(P4, p_deg, 8, rng, sphere_nontrivial=True, sphere_dim=dim)
            if not p_sample:
                pytest.skip(f"No sphere-admissible P(4) elements at degree {p_deg}")
            pool = sample_algebra_pool(mod, k_per_bucket=30, rng=rng, deg_range=range(-1, 4))
            if len(pool) < 4:
                pytest.skip("Not enough algebra elements")
            for _ in range(4):
                inputs = rng.choices(pool, k=4)
                degrees = [_elem_degree(mod, e) for e in inputs]
                perms = _sample(list(SymmetricGroup(4)), 6, rng)
                for sigma in perms:
                    sl = list(sigma.tuple())
                    sigma_0idx = [sigma(i) - 1 for i in range(1, 5)]
                    koszul = koszul_sign_of_permutation(sigma_0idx, degrees)
                    permuted = [inputs[sigma(i) - 1] for i in range(1, 5)]
                    for p in p_sample:
                        lhs = pb.act(p.permute(sl), inputs)
                        rhs = koszul * pb.act(p, permuted)
                        assert _as_dict(lhs) == _as_dict(rhs)


# ===========================================================================
# Layer 6: BarAlgebra (final construction)
# ===========================================================================


class TestLayer6_Bπ_pb_S_ΩBH_Kd:
    """Bar algebra B_π(pulled_back) — final output of the configuration model."""

    class TestDSquared:
        @pytest.mark.parametrize("weight", [1, 2])
        def test_d_squared(self, dim, ring, weight, layers: ConfigurationLayers):
            mod = layers.bar.module
            tested = 0
            for deg in range(-1, 6):
                for elem in mod.graded_basis_by_weight(deg, weight):
                    tested += 1
                    assert mod.boundary(mod.boundary(elem)) == mod.zero(), (
                        f"d²≠0 w={weight} deg={deg} dim={dim} ring={ring}"
                    )
            assert tested > 0, "No basis elements found (nontriviality check)"

    class TestDSquaredWeight3:
        """d²=0 at weight 3 — where the sign bug first manifests (QQ only)."""

        def test_d_squared(self, dim, ring, layers: ConfigurationLayers, rng):
            mod = layers.bar.module
            for deg in range(-1, 5):
                for elem in sample_basis(mod, deg, 30, rng, weight=3):
                    assert mod.boundary(mod.boundary(elem)) == mod.zero(), (
                        f"d²≠0 w=3 deg={deg} dim={dim} ring={ring}"
                    )

    class TestDSquaredWeight4:
        """d²=0 at weight 4."""

        def test_d_squared(self, dim, ring, layers: ConfigurationLayers, rng):
            mod = layers.bar.module
            for deg in range(-2, 4):
                basis = mod.graded_basis_by_weight(deg, 4)
                for elem in basis:
                    assert mod.boundary(mod.boundary(elem)) == mod.zero(), (
                        f"d²≠0 w=4 deg={deg} dim={dim} ring={ring}"
                    )

    class TestDecomposedDSquared:
        """Test the components of d separately: d_cofree and d_alpha.

        d = d_cofree + d_alpha,  so d² = d_cofree² + (d_alpha² + cross) = 0,
        where cross = d_cofree ∘ d_alpha + d_alpha ∘ d_cofree.

        This yields two independent identities:
        1. d_cofree² = 0  (the cofree coalgebra differential squares to zero)
        2. d_alpha² + cross = 0  (the Maurer-Cartan equation at the module level)

        Note: d_alpha² and cross are *not* individually zero in general; only
        their sum vanishes (as a consequence of the MC equation ∂α + α ⋆ α = 0).
        """

        @pytest.mark.parametrize("weight", [2, 3])
        def test_d_cofree_squared(self, dim, ring, weight, layers: ConfigurationLayers, rng):
            """d_cofree² = 0."""
            mod = layers.bar.module
            for deg in range(-1, 5):
                for elem in sample_basis(mod, deg, 20, rng, weight=weight):
                    dd = mod.d_cofree(mod.d_cofree(elem))
                    assert dd == mod.zero(), (
                        f"d_cofree²≠0 w={weight} deg={deg} dim={dim} ring={ring}"
                    )

        @pytest.mark.parametrize("weight", [2, 3])
        def test_mc_equation(self, dim, ring, weight, layers: ConfigurationLayers, rng):
            """d_alpha² + (d_cofree ∘ d_alpha + d_alpha ∘ d_cofree) = 0.

            This is the Maurer-Cartan equation at the module level: the combined
            contribution of d_alpha² and the cross term must vanish.  Neither
            term is individually zero in general.
            """
            mod = layers.bar.module
            tested = 0
            for deg in range(-1, 5):
                for elem in sample_basis(mod, deg, 20, rng, weight=weight):
                    tested += 1
                    dd = mod.d_alpha(mod.d_alpha(elem))
                    cross = mod.d_cofree(mod.d_alpha(elem)) + mod.d_alpha(mod.d_cofree(elem))
                    assert dd + cross == mod.zero(), (
                        f"d_α² + cross ≠ 0 for {elem} w={weight} deg={deg} dim={dim} ring={ring}"
                    )
            assert tested > 0, "No basis elements found (nontriviality check)"


# ===========================================================================
# Full model / chain complex tests
# ===========================================================================


class TestFullModel:
    class TestChainComplex:
        @pytest.mark.parametrize("w,dmax", [(2, 5), (3, 4), (4, 2)])
        def test_full_model(self, dim, ring, w: int, dmax: int):
            model = euclidean_unordered_configuration_model(ring, dim)
            C = compute_chain_complex(model.module, degrees=range(-1, dmax), weight=w, check=True)
            assert C is not None

    class TestBasis:
        def test_no_duplicates(self, dim):
            model = euclidean_unordered_configuration_model(QQ, dim)
            for k in range(-2, 6):
                for w in range(1, 4):
                    basis = model.module.graded_basis_by_weight(k, w)
                    assert len(basis) == len(set(basis))


@pytest.mark.skip(reason="Too slow at the moment!")
class TestExpectedDimension:
    @staticmethod
    def _count(d: int, k: int):
        """Count the expected dimension of the weight-k part of the model in degree d, based on the result of Fuks."""
        S = k - d
        if S < 0:
            return 0

        mersenne_parts = [2**i - 1 for i in range(1, d.bit_length() + 1)]
        P = Partitions(d, parts_in=mersenne_parts)

        return sum(1 for p in P if len(p) <= S)

    @pytest.mark.parametrize("w", [1, 2, 3, 4])
    def test_expected_dimension_gf2(self, w: int):
        model = euclidean_unordered_configuration_model(GF(2), 2)
        cc = compute_chain_complex(model.module, degrees=range(-1, 3), weight=w, check=True)
        print(
            "Found Betti numbers:",
            [cc.betti(deg) for deg in range(-1, 3)],
            f"Expected dimensions: {[self._count(deg, w) for deg in range(-1, 3)]}",
        )
        for deg in range(-1, 3):
            assert cc.betti(deg) == self._count(deg, w), (
                f"Betti number mismatch at deg={deg}, weight={w}"
            )

    @pytest.mark.parametrize("w", [1, 2, 3, 4])
    def test_expected_dimension_QQ(self, w: int):
        model = euclidean_unordered_configuration_model(QQ, 2)
        cc = compute_chain_complex(model.module, degrees=range(-1, 3), weight=w, check=True)
        print("Found Betti numbers:", [cc.betti(deg) for deg in range(-1, 3)])
        for deg in range(-1, 3):
            if deg == 0 or (w > 1 and deg == 1):
                assert cc.betti(deg) == 1, f"Betti number mismatch at deg={deg}, weight={w}"
            else:
                assert cc.betti(deg) == 0, f"Betti number mismatch at deg={deg}, weight={w}"
