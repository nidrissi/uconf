"""Tests for chain complex construction and homology helpers."""

import pytest
from sage.all import QQ

from uconf import BarrattEccles, Lie, Surjection
from uconf.homology import chain_complex, homology_basis


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
                assert S3.boundary(g) == S3.zero(), (
                    f"Generator in H_{d} is not a cycle: {g}"
                )

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
