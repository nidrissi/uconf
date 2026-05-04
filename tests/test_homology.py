"""Tests for chain complex construction and homology helpers."""

import cProfile
import multiprocessing
import os
import pstats

import pytest
from sage.all import GF, QQ, cached_method

from uconf import (
    BarrattEccles,
    Lie,
    Surjection,
    compute_chain_complex,
    homology_basis,
)
from uconf.homology import _boundary_matrix, compute_homology_representatives

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
        C = compute_chain_complex(S2, degrees=range(5))
        for d in range(5):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_surjection_arity3(self) -> None:
        """Surjection(3, QQ) has H_0=1 and H_d=0 for d in 1..3."""
        S3 = Surjection(3, QQ)
        C = compute_chain_complex(S3, degrees=range(4))
        for d in range(4):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_barratt_eccles_arity2(self) -> None:
        """BarrattEccles(2, QQ) has H_0=1 and H_d=0 for 1<=d<=4."""
        E2 = BarrattEccles(2, QQ)
        C = compute_chain_complex(E2, degrees=range(5))
        for d in range(5):
            assert C.betti().get(d, 0) == (1 if d == 0 else 0)

    def test_lie_arity2(self) -> None:
        """Lie is concentrated in degree 0; homology is the module itself."""
        L2 = Lie(2, QQ)
        C = compute_chain_complex(L2, degrees=range(3))
        assert C.betti().get(0, 0) == 1
        assert C.betti().get(1, 0) == 0
        assert C.betti().get(2, 0) == 0

    def test_d_squared_zero(self) -> None:
        """The chain complex differential squares to zero (implicitly
        checked by ChainComplex, but verify explicitly)."""
        S2 = Surjection(2, QQ)
        C = compute_chain_complex(S2, degrees=range(5))
        # Complex now spans degrees 0-5 (extended internally by 1)
        for d in range(1, 6):
            d_prev = C.differential(d - 1)
            d_curr = C.differential(d)
            assert (d_prev * d_curr).is_zero()

    def test_empty_degrees(self) -> None:
        """Empty degree range gives a trivial complex."""
        S2 = Surjection(2, QQ)
        C = compute_chain_complex(S2, degrees=range(0))
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
            compute_chain_complex(mod, degrees=range(2))

        # With weight=1: only arity-1 element; gives finite complex
        C = compute_chain_complex(mod, degrees=range(2), weight=1)
        assert C is not None

    def test_weight_error_when_module_unsupported(self) -> None:
        """chain_complex raises ValueError when weight is used on unsupported module."""
        S2 = Surjection(2, QQ)
        with pytest.raises(ValueError, match="does not support the weight API"):
            compute_chain_complex(S2, degrees=range(3), weight=1)

    def test_boundary_matrix_fast_path_with_normalization(self) -> None:
        """Chain-complex assembly should use the on-basis fast path and key normalization."""

        class _FakeElement:
            def __init__(self, key):
                self._key = key

            def leading_support(self):
                return self._key

            def __repr__(self) -> str:
                return f"FakeElem({self._key!r})"

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

            def __call__(self, _elem):
                raise AssertionError("compute_chain_complex should use the on_basis fast path")

        class _FakeModule:
            def __init__(self):
                self._basis = {
                    0: [_FakeElement("a")],
                    1: [_FakeElement("b")],
                }
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return QQ

            def graded_basis(self, d):
                return self._basis.get(d, [])

            def _boundary_on_basis(self, key):
                if key == "b":
                    return [("raw_a", QQ.one())]
                return []

            def _normalize_key(self, key):
                if key == "raw_a":
                    return [(QQ.one(), "a")]
                return [(QQ.one(), key)]

        C = compute_chain_complex(_FakeModule(), degrees=range(2), check=True)
        assert C.differential(1)[0, 0] == 1

    def test_boundary_matrix_parallel_fast_path_matches_serial(self) -> None:
        """Parallel boundary assembly should agree with the serial on-basis path."""

        class _FakeElement:
            def __init__(self, key):
                self._key = key

            def leading_support(self):
                return self._key

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

        class _FakeModule:
            def __init__(self):
                self._basis = {
                    0: [_FakeElement("a"), _FakeElement("b")],
                    1: [_FakeElement("x"), _FakeElement("y")],
                }
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return GF(2)

            def graded_basis(self, d):
                return self._basis.get(d, [])

            def _boundary_on_basis(self, key):
                if key == "x":
                    return [("raw_a", GF(2).one())]
                if key == "y":
                    return [("raw_b", GF(2).one())]
                return []

            def _normalize_key(self, key):
                if key == "raw_a":
                    return [(GF(2).one(), "a")]
                if key == "raw_b":
                    return [(GF(2).one(), "b")]
                return [(GF(2).one(), key)]

        module = _FakeModule()
        serial = compute_chain_complex(module, degrees=range(2), check=True, n_jobs=1)
        parallel = compute_chain_complex(module, degrees=range(2), check=True, n_jobs=2)
        assert parallel.differential(1) == serial.differential(1)

    def test_boundary_matrix_profile_tracks_hot_path_costs(self) -> None:
        """Boundary-matrix profiling should track boundary, normalization, and merge costs."""

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

        class _FakeModule:
            def __init__(self):
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return QQ

            def _boundary_on_basis(self, key):
                if key == "x":
                    return [("raw_a", QQ.one())]
                if key == "y":
                    return [("a", QQ.one())]
                return []

            def _normalize_key(self, key):
                if key == "raw_a":
                    return [(QQ.one(), "a")]
                return [(QQ.one(), key)]

        profile: dict[str, float | int] = {}
        matrix = _boundary_matrix(
            _FakeModule(),
            basis_source_keys=["x", "y"],
            key_to_idx_target={"a": 0},
            n_target=1,
            sparse=True,
            profile=profile,
        )
        assert matrix[0, 0] == 1
        assert matrix[0, 1] == 1
        assert profile["boundary_on_basis_calls"] == 2
        assert profile["normalization_lookups"] == 1
        assert profile["normalized_matches"] == 1
        assert profile["boundary_on_basis_seconds"] >= 0.0
        assert profile["normalization_seconds"] >= 0.0
        assert profile["entry_merge_seconds"] >= 0.0

    def test_invalid_n_jobs_raises(self) -> None:
        """n_jobs must be positive."""
        S2 = Surjection(2, QQ)
        with pytest.raises(ValueError, match="n_jobs must be >= 1"):
            compute_chain_complex(S2, degrees=range(2), n_jobs=0)

    def test_parallel_worker_profiles_can_be_merged(self) -> None:
        """Parallel worker profiling should emit mergeable pstats files."""

        class _FakeElement:
            def __init__(self, key):
                self._key = key

            def leading_support(self):
                return self._key

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

        class _FakeModule:
            def __init__(self):
                self._basis = {
                    0: [_FakeElement("a"), _FakeElement("b")],
                    1: [_FakeElement("x"), _FakeElement("y"), _FakeElement("z")],
                }
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return QQ

            def graded_basis(self, d):
                return self._basis.get(d, [])

            def _boundary_on_basis(self, key):
                if key == "x":
                    return [("a", QQ.one())]
                if key == "y":
                    return [("b", QQ.one())]
                if key == "z":
                    return [("a", QQ.one()), ("b", QQ.one())]
                return []

        worker_profile_paths: list[str] = []
        profile = cProfile.Profile()
        profile.enable()
        compute_chain_complex(
            _FakeModule(),
            degrees=range(2),
            check=True,
            n_jobs=2,
            worker_profile_paths=worker_profile_paths,
            worker_profile_parent=profile,
        )
        profile.disable()

        try:
            assert worker_profile_paths
            merged = pstats.Stats(profile)
            merged.add(*worker_profile_paths)
        finally:
            for path in worker_profile_paths:
                if os.path.exists(path):
                    os.unlink(path)

        assert merged.total_calls > 0

    @pytest.mark.skipif(
        os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods(),
        reason="parallel cache prewarming requires POSIX fork workers",
    )
    def test_parallel_workers_inherit_prewarmed_cached_results(self, tmp_path) -> None:
        """Parent prewarming should avoid refilling cached_method entries in workers."""

        class _FakeElement:
            def __init__(self, key):
                self._key = key

            def leading_support(self):
                return self._key

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

        class _FakeModule:
            def __init__(self, log_path):
                self._basis = {
                    0: [_FakeElement("a"), _FakeElement("b")],
                    1: [_FakeElement("x"), _FakeElement("y"), _FakeElement("z")],
                }
                self._log_path = log_path
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return QQ

            def graded_basis(self, d):
                return self._basis.get(d, [])

            @cached_method
            def _expensive_boundary(self, key):
                with open(self._log_path, "a", encoding="utf-8") as handle:
                    handle.write(f"{key}\n")
                if key == "x":
                    return (("a", QQ.one()),)
                if key == "y":
                    return (("b", QQ.one()),)
                if key == "z":
                    return (("a", QQ.one()), ("b", QQ.one()))
                return ()

            def _boundary_on_basis(self, key):
                return self._expensive_boundary(key)

            def _prewarm_parallel_boundary_caches(self, source_keys):
                for key in source_keys:
                    self._expensive_boundary(key)

        log_path = tmp_path / "prewarm.log"
        compute_chain_complex(_FakeModule(os.fspath(log_path)), degrees=range(2), n_jobs=2)
        cache_misses = log_path.read_text(encoding="utf-8").splitlines()
        assert len(cache_misses) == 5
        assert set(cache_misses) == {"a", "b", "x", "y", "z"}

    def test_progress_reporting_writes_status(self, capsys) -> None:
        """Optional progress reporting should emit visible status updates."""

        class _FakeElement:
            def __init__(self, key):
                self._key = key

            def leading_support(self):
                return self._key

        class _Boundary:
            def __init__(self, on_basis):
                self._on_basis = on_basis

            def on_basis(self):
                return self._on_basis

        class _FakeModule:
            def __init__(self):
                self._basis = {
                    0: [_FakeElement("a"), _FakeElement("b")],
                    1: [_FakeElement("x"), _FakeElement("y")],
                }
                self.boundary = _Boundary(self._boundary_on_basis)

            def base_ring(self):
                return QQ

            def graded_basis(self, d):
                return self._basis.get(d, [])

            def _boundary_on_basis(self, key):
                if key == "x":
                    return [("a", QQ.one())]
                if key == "y":
                    return [("b", QQ.one())]
                return []

        compute_chain_complex(_FakeModule(), degrees=range(2), progress=True)
        captured = capsys.readouterr()
        assert "compute_chain_complex:" in captured.err
        assert "100%" in captured.err


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
        vertex = SC((0,))
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


# ---------------------------------------------------------------------------
# compute_homology_representatives — configuration model
# ---------------------------------------------------------------------------


class TestHomologyRepresentativesConfigModel:
    """Test compute_homology_representatives on bar algebra of the configuration model.

    Uses ``_build_euclidean_layers`` at low weight and degree to verify that
    the returned representatives are cycles and that their count matches the
    chain-complex Betti number.
    """

    @pytest.fixture(scope="class", params=[1, 2], ids=["dim1", "dim2"])
    def bar_module(self, request):
        from uconf.algebraic.configuration import _build_euclidean_layers

        layers = _build_euclidean_layers(GF(2), request.param)
        return layers.bar.module

    @pytest.mark.parametrize(
        "degree,weight",
        [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (0, 3)],
    )
    def test_representatives_are_cycles(self, bar_module, degree, weight) -> None:
        """Every returned representative must satisfy boundary = 0."""
        cc = compute_chain_complex(
            bar_module,
            degrees=range(degree - 1, degree + 2),
            weight=weight,
        )
        reps = compute_homology_representatives(bar_module, degree, weight, cc)
        for r in reps:
            assert bar_module.boundary(r) == bar_module.zero(), (
                f"Representative at degree={degree}, weight={weight} is not a cycle: {r}"
            )

    @pytest.mark.parametrize(
        "degree,weight",
        [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (0, 3)],
    )
    def test_representative_count_matches_betti(self, bar_module, degree, weight) -> None:
        """Number of representatives equals the Betti number at that degree."""
        cc = compute_chain_complex(
            bar_module,
            degrees=range(degree - 1, degree + 2),
            weight=weight,
        )
        reps = compute_homology_representatives(bar_module, degree, weight, cc)
        betti = cc.betti().get(degree, 0)
        assert len(reps) == betti, (
            f"Expected {betti} representative(s) at degree={degree}, weight={weight}, "
            f"got {len(reps)}"
        )
