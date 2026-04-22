"""Tests for the random sampling module."""

from random import Random

import pytest
from sage.all import QQ

from uconf.models.surjection import Surjection
from uconf.models.lie import Lie
from uconf.wrappers.shifted_operad import ShiftedOperad
from uconf.wrappers.hadamard_operad import HadamardProduct
from uconf.algebraic.spherical import (
    _extract_concatenated_permutations,
    _sphere_surjection_basis_sign,
)
from uconf.sampling import (
    random_surjection_key,
    random_surjection,
    random_planar_surjection_key,
    random_planar_surjection,
    random_sphere_admissible_surjection_key,
    random_sphere_admissible_surjection,
    random_lie_key,
    random_lie_element,
    random_barratt_eccles_key,
    random_barratt_eccles_element,
    random_hadamard_key,
    random_shuffle_tree,
    random_bar_element,
    random_cobar_element,
    random_free_algebra_element,
    random_cofree_coalgebra_element,
    random_tree_module_element,
    sample_basis,
    sample_operad_basis,
    sample_hadamard_basis,
    sample_algebra_pool,
    sphere_nontrivial_surjection_iter,
    sphere_nontrivial_operad_basis_iter,
)


@pytest.fixture
def rng():
    return Random(20260413)


# ---------------------------------------------------------------------------
# Surjection key generation
# ---------------------------------------------------------------------------


class TestRandomSurjectionKey:
    @pytest.mark.parametrize("n,degree", [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0)])
    def test_valid_surjection(self, n, degree, rng):
        """Generated keys must be valid surjections."""
        for _ in range(20):
            key = random_surjection_key(n, degree, rng)
            assert key is not None, f"Failed to generate surjection for n={n}, d={degree}"
            assert len(key) == n + degree
            assert set(key) == set(range(1, n + 1)), "Not surjective"
            for i in range(len(key) - 1):
                assert key[i] != key[i + 1], "Consecutive repeat"

    def test_arity_1(self, rng):
        assert random_surjection_key(1, 0, rng) == (1,)
        assert random_surjection_key(1, 1, rng) is None

    def test_invalid_inputs(self, rng):
        assert random_surjection_key(0, 0, rng) is None
        assert random_surjection_key(2, -1, rng) is None


class TestRandomSurjection:
    def test_produces_element(self, rng):
        elem = random_surjection(2, 1, QQ, rng)
        assert elem is not None
        assert elem.parent() == Surjection(2, QQ)

    def test_arity3(self, rng):
        elem = random_surjection(3, 1, QQ, rng)
        assert elem is not None
        assert elem.parent() == Surjection(3, QQ)


# ---------------------------------------------------------------------------
# Planar surjection
# ---------------------------------------------------------------------------


class TestRandomPlanarSurjection:
    @pytest.mark.parametrize("n,degree", [(2, 0), (2, 1), (3, 0), (3, 1)])
    def test_planarity(self, n, degree, rng):
        """Generated keys must be planar."""
        for _ in range(10):
            key = random_planar_surjection_key(n, degree, rng)
            assert key is not None
            # Check planarity: first occurrences in order
            first_occ = {}
            for i, v in enumerate(key):
                if v not in first_occ:
                    first_occ[v] = i
            positions = [first_occ[v] for v in range(1, n + 1)]
            assert positions == sorted(positions), f"Not planar: {key}"

    def test_element(self, rng):
        elem = random_planar_surjection(2, 1, QQ, rng)
        assert elem is not None
        assert elem.is_planar()


# ---------------------------------------------------------------------------
# Sphere-admissible surjection
# ---------------------------------------------------------------------------


class TestRandomSphereAdmissible:
    @pytest.mark.parametrize("n,dim", [(2, 1), (2, 2), (3, 1)])
    def test_admissibility(self, n, dim, rng):
        """Generated keys must be sphere-admissible."""
        for _ in range(10):
            key = random_sphere_admissible_surjection_key(n, dim, rng)
            assert key is not None, f"Failed for n={n}, dim={dim}"
            perms = _extract_concatenated_permutations(key, n, dim)
            assert perms is not None, f"Not sphere-admissible: {key}"
            assert len(perms) == dim + 1

    @pytest.mark.parametrize("n,dim", [(2, 1), (2, 2), (3, 1)])
    def test_nontrivial_sign(self, n, dim, rng):
        """Sphere-admissible surjections must have nonzero BF sign."""
        for _ in range(10):
            key = random_sphere_admissible_surjection_key(n, dim, rng)
            assert key is not None
            degree = dim * (n - 1)
            assert len(key) == n + degree
            sign = _sphere_surjection_basis_sign(key, n, dim)
            assert sign != 0, f"Zero sign for sphere-admissible: {key}"

    def test_element(self, rng):
        elem = random_sphere_admissible_surjection(2, 1, QQ, rng)
        assert elem is not None


# ---------------------------------------------------------------------------
# Lie key
# ---------------------------------------------------------------------------


class TestRandomLieKey:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_valid_key(self, n, rng):
        for _ in range(10):
            key = random_lie_key(n, rng)
            if n <= 1:
                assert key == ()
            else:
                assert len(key) == n - 1
                assert set(key) == set(range(1, n))

    def test_element(self, rng):
        elem = random_lie_element(3, QQ, rng)
        assert elem is not None
        assert elem.parent() == Lie(3, QQ)


# ---------------------------------------------------------------------------
# Hadamard product sampling
# ---------------------------------------------------------------------------


class TestRandomHadamardKey:
    def test_basic(self, rng):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        H2 = H(2, QQ)
        elem = random_hadamard_key(H2, 0, rng)
        assert elem is not None

    def test_sphere_nontrivial(self, rng):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        H2 = H(2, QQ)
        elem = random_hadamard_key(H2, 0, rng, sphere_nontrivial=True, sphere_dim=1)
        assert elem is not None
        # The surjection factor should act nontrivially
        # Degree at arity 2 for dim=1: surj_degree = 1*(2-1) = 1
        # Total degree 0 = left_deg + right_deg, left_deg = 0-1 = -1, right_deg = 1

    def test_sphere_nontrivial_requires_dim(self, rng):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        H2 = H(2, QQ)
        with pytest.raises(ValueError, match="sphere_dim"):
            random_hadamard_key(H2, 0, rng, sphere_nontrivial=True)


class TestSampleHadamardBasis:
    def test_sampling(self, rng):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        H2 = H(2, QQ)
        elems = sample_hadamard_basis(H2, 0, 5, rng)
        assert len(elems) > 0
        # Should get at most 2 distinct elements (basis size at deg=0 for H(2) is 2)
        assert len(elems) <= 2


# ---------------------------------------------------------------------------
# Generic sampling
# ---------------------------------------------------------------------------


class TestSampleBasis:
    def test_surjection(self, rng):
        S2 = Surjection(2, QQ)
        elems = sample_basis(S2, 1, 5, rng)
        # S(2) deg 1 has exactly 2 basis elements
        assert len(elems) == 2

    def test_lie(self, rng):
        L3 = Lie(3, QQ)
        elems = sample_basis(L3, 0, 5, rng)
        # Lie(3) deg 0 has exactly 2 basis elements
        assert len(elems) == 2

    def test_with_weight(self, rng):
        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(QQ, 1)
        mod = layers.free_alg.module
        elems = sample_basis(mod, 1, 10, rng, weight=1)
        assert len(elems) > 0

    def test_sample_fewer_than_k(self, rng):
        # Surjection(2) at degree 0 has exactly 2 elements: (1,2) and (2,1)
        S2 = Surjection(2, QQ)
        elems = sample_basis(S2, 0, 10, rng)
        assert len(elems) == 2


class TestSampleOperadBasis:
    def test_basic(self, rng):
        elems = sample_operad_basis(Surjection, 2, 1, 5, QQ, rng)
        assert len(elems) == 2  # exactly 2 basis elems at S(2) deg 1


class TestSampleBasisEmptyFastFail:
    """Inputs with provably no basis elements must return [] without retrying."""

    def test_lie_nonzero_degree(self, rng):
        import time

        L3 = Lie(3, QQ)
        t0 = time.time()
        elems = sample_basis(L3, 3, 10, rng)
        assert elems == []
        assert time.time() - t0 < 0.1

    def test_surjection_negative_degree(self, rng):
        import time

        S3 = Surjection(3, QQ)
        t0 = time.time()
        elems = sample_basis(S3, -2, 10, rng)
        assert elems == []
        assert time.time() - t0 < 0.1

    def test_sphere_nontrivial_wrong_degree(self, rng):
        import time
        from sage.all import GF

        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(GF(2), 2)
        P2 = layers.OBXsLie(2, layers.bar.module.base_ring())
        t0 = time.time()
        elems = sample_basis(P2, -1, 10, rng, sphere_nontrivial=True, sphere_dim=2)
        assert elems == []
        assert time.time() - t0 < 1.0, "sphere_nontrivial cobar at infeasible degree must fast-fail"


# ---------------------------------------------------------------------------
# Algebra pool sampling
# ---------------------------------------------------------------------------


class TestSampleAlgebraPool:
    def test_free_algebra(self, rng):
        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(QQ, 1)
        mod = layers.free_alg.module
        pool = sample_algebra_pool(mod, k_per_bucket=5, rng=rng)
        assert len(pool) > 0

    def test_tensor_algebra(self, rng):
        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(QQ, 1)
        pool = sample_algebra_pool(
            layers.tensor_alg,
            k_per_bucket=5,
            rng=rng,
            deg_range=range(-1, 3),
        )
        assert len(pool) > 0


# ---------------------------------------------------------------------------
# Sphere-nontrivial iterators
# ---------------------------------------------------------------------------


class TestSphereNontrivialIterators:
    @pytest.mark.parametrize("n,dim", [(2, 1), (2, 2), (3, 1)])
    def test_surjection_iter(self, n, dim):
        elems = list(sphere_nontrivial_surjection_iter(n, dim, QQ))
        assert len(elems) > 0
        for elem, sign in elems:
            assert sign in (-1, 1)

    def test_operad_basis_iter_hadamard(self):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        elems = list(sphere_nontrivial_operad_basis_iter(H, 2, 1, QQ))
        assert len(elems) == 2  # [x1,x2] ⊙ {1 2 1} and [x1,x2] ⊙ {2 1 2}

    def test_operad_basis_iter_arity3(self):
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        elems = list(sphere_nontrivial_operad_basis_iter(H, 3, 1, QQ))
        # 2 Lie elements × 12 nontrivial surjections = 24
        assert len(elems) == 24


# ---------------------------------------------------------------------------
# Barratt–Eccles sampling
# ---------------------------------------------------------------------------


class TestRandomBarrattEcclesKey:
    def test_valid_key(self, rng):
        key = random_barratt_eccles_key(2, 1, rng)
        assert key is not None
        assert len(key) == 2  # degree + 1 permutations
        # No consecutive equal perms
        assert key[0] != key[1]

    def test_degree_0(self, rng):
        key = random_barratt_eccles_key(3, 0, rng)
        assert key is not None
        assert len(key) == 1

    def test_element(self, rng):
        from uconf.models.barratt_eccles import BarrattEccles

        elem = random_barratt_eccles_element(3, 1, QQ, rng)
        assert elem is not None
        assert elem.parent() == BarrattEccles(3, QQ)


# ---------------------------------------------------------------------------
# Random shuffle tree
# ---------------------------------------------------------------------------


class TestRandomShuffleTree:
    def test_bar_tree(self, rng):
        """Generate a random bar construction tree."""
        tree = random_shuffle_tree(
            (1, 2, 3),
            2,
            Surjection,
            QQ,
            2,
            +1,
            rng,
        )
        assert tree is not None
        from uconf.core.trees import tree_arity, is_leaf

        assert not is_leaf(tree)
        assert tree_arity(tree) == 3

    def test_cobar_tree(self, rng):
        """Generate a random cobar construction tree."""
        from uconf.constructions.bar_construction import BarConstruction

        B = BarConstruction(Surjection)
        tree = random_shuffle_tree(
            (1, 2, 3),
            2,
            B,
            QQ,
            -2,
            -1,
            rng,
        )
        # Cobar trees with negative degree can be harder to generate
        # so we allow None
        if tree is not None:
            from uconf.core.trees import is_leaf

            assert not is_leaf(tree)


# ---------------------------------------------------------------------------
# Bar / Cobar construction sampling
# ---------------------------------------------------------------------------


class TestRandomBarElement:
    def test_arity2(self, rng):
        from uconf.constructions.bar_construction import BarConstruction

        B = BarConstruction(Surjection)
        parent = B(2, QQ)
        elem = random_bar_element(parent, 1, rng)
        assert elem is not None
        assert elem.parent() == parent

    def test_arity3(self, rng):
        from uconf.constructions.bar_construction import BarConstruction

        B = BarConstruction(Surjection)
        parent = B(3, QQ)
        elem = random_bar_element(parent, 1, rng)
        assert elem is not None

    def test_arity1(self, rng):
        from uconf.constructions.bar_construction import BarConstruction

        B = BarConstruction(Surjection)
        parent = B(1, QQ)
        elem = random_bar_element(parent, 0, rng)
        assert elem is not None


class TestRandomCobarElement:
    def test_arity2(self, rng):
        from uconf.constructions.bar_construction import BarConstruction
        from uconf.constructions.cobar_construction import CobarConstruction

        B = BarConstruction(Surjection)
        OmegaB = CobarConstruction(B)
        parent = OmegaB(2, QQ)
        elem = random_cobar_element(parent, 0, rng)
        assert elem is not None
        assert elem.parent() == parent


# ---------------------------------------------------------------------------
# Free/cofree algebra sampling
# ---------------------------------------------------------------------------


class TestRandomFreeAlgebraElement:
    @pytest.mark.parametrize("weight", [1, 2])
    def test_with_weight(self, weight, rng):
        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(QQ, 1)
        mod = layers.free_alg.module
        elem = random_free_algebra_element(mod, 1, rng, weight=weight)
        if elem is not None:
            assert elem.parent() == mod


class TestRandomCofreeCoalgebraElement:
    def test_basic(self, rng):
        from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
        from uconf.algebraic.configuration import TrivialModule

        M = TrivialModule(1, QQ)
        mod = CofreeCoalgebraModule(Surjection, M)
        elem = random_cofree_coalgebra_element(mod, 1, rng)
        if elem is not None:
            assert elem.parent() == mod


# ---------------------------------------------------------------------------
# sample_basis with construction-aware dispatch
# ---------------------------------------------------------------------------


class TestSampleBasisDispatch:
    """Test that sample_basis dispatches to construction-aware generators."""

    def test_surjection_dispatch(self, rng):
        """sample_basis on Surjection should use direct generation."""
        S3 = Surjection(3, QQ)
        elems = sample_basis(S3, 1, 5, rng)
        assert len(elems) > 0
        for e in elems:
            assert e.parent() == S3

    def test_bar_dispatch(self, rng):
        """sample_basis on BarConstruction should use random tree generation."""
        from uconf.constructions.bar_construction import BarConstruction

        B = BarConstruction(Surjection)
        parent = B(3, QQ)
        elems = sample_basis(parent, 1, 5, rng)
        assert len(elems) > 0
        for e in elems:
            assert e.parent() == parent

    def test_hadamard_dispatch(self, rng):
        """sample_basis on HadamardProduct should use factor-independent sampling."""
        sLie = ShiftedOperad(Lie, -1)
        H = HadamardProduct(sLie, Surjection)
        H2 = H(2, QQ)
        elems = sample_basis(H2, 0, 5, rng)
        assert len(elems) > 0

    def test_lie_dispatch(self, rng):
        """sample_basis on Lie should use direct key generation."""
        L3 = Lie(3, QQ)
        elems = sample_basis(L3, 0, 5, rng)
        assert len(elems) > 0

    def test_free_algebra_dispatch(self, rng):
        """sample_basis on FreeAlgebraModule should use construction-aware generation."""
        from uconf.algebraic.configuration import _build_layers

        layers = _build_layers(QQ, 1)
        mod = layers.free_alg.module
        elems = sample_basis(mod, 1, 5, rng, weight=1)
        # At least some elements should be generated
        assert isinstance(elems, list)


class TestRandomTreeModuleElement:
    def test_basic(self, rng):
        from uconf.algebraic.tree_module import TreeModule
        from uconf.algebraic.configuration import TrivialModule

        M = TrivialModule(1, QQ)
        tm = TreeModule(Surjection, M, name="TestTM")
        elem = random_tree_module_element(tm, 1, rng, weight=1)
        if elem is not None:
            assert elem.parent() == tm
