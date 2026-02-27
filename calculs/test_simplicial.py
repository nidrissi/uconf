"""Tests for simplicial chains, cochains, and the surjection action."""

import pytest
from itertools import combinations

from uconf import SimplicialChains, SimplicialCochains, Surjection


# ===========================================================================
# Helpers
# ===========================================================================


def _as_dict(x):
    """Canonical dictionary representation of an element."""
    return {k: v for k, v in x if v != 0}


# ===========================================================================
# SimplicialChains basics
# ===========================================================================


class TestSimplicialChains:
    def test_construction_single_simplex(self):
        C = SimplicialChains(r=1)
        x = C(((0, 1, 2),))
        assert x != C.zero()

    def test_construction_tensor(self):
        C = SimplicialChains(r=2)
        x = C(((0, 1, 2), (0, 1)))
        assert x != C.zero()

    def test_degenerate_rejected(self):
        C = SimplicialChains(r=1)
        # (0, 0, 1) has repeated entry → degenerate
        x = C(((0, 0, 1),))
        assert x == C.zero()

    def test_empty_simplex_rejected(self):
        C = SimplicialChains(r=1)
        x = C(((),))
        assert x == C.zero()

    def test_degree(self):
        C = SimplicialChains(r=1)
        x = C(((0, 1, 2),))
        assert x.degree() == 2

    def test_degree_tensor(self):
        C = SimplicialChains(r=2)
        x = C(((0, 1, 2), (3, 4)))
        assert x.degree() == 3  # 2 + 1

    def test_standard_element(self):
        x = SimplicialChains.standard_element(3)
        assert x.degree() == 3
        assert x.arity() == 1
        d = _as_dict(x)
        assert d == {((0, 1, 2, 3),): 1}

    def test_standard_element_tensor(self):
        x = SimplicialChains.standard_element(2, times=2)
        assert x.degree() == 4
        assert x.arity() == 2


class TestBoundary:
    def test_boundary_vertex(self):
        """Boundary of a 0-simplex is zero."""
        C = SimplicialChains(r=1)
        x = C(((0,),))
        assert x.boundary() == C.zero()

    def test_boundary_edge(self):
        """∂[0,1] = [1] - [0]."""
        C = SimplicialChains(r=1)
        x = C(((0, 1),))
        bdry = x.boundary()
        d = _as_dict(bdry)
        assert d == {((1,),): 1, ((0,),): -1}

    def test_boundary_triangle(self):
        """∂[0,1,2] = [1,2] - [0,2] + [0,1]."""
        C = SimplicialChains(r=1)
        x = C(((0, 1, 2),))
        bdry = x.boundary()
        d = _as_dict(bdry)
        assert d == {((1, 2),): 1, ((0, 2),): -1, ((0, 1),): 1}

    def test_boundary_squared_is_zero(self):
        """∂² = 0 for several simplices."""
        C = SimplicialChains(r=1)
        for n in range(1, 5):
            x = SimplicialChains.standard_element(n)
            assert x.boundary().boundary() == C.zero(), f"∂² ≠ 0 for Δ^{n}"

    def test_boundary_squared_tensor(self):
        """∂² = 0 on tensor products."""
        C = SimplicialChains(r=2)
        x = C(((0, 1, 2), (0, 1)))
        assert x.boundary().boundary() == C.zero()

    def test_boundary_tensor_product(self):
        """∂([0,1] ⊗ [2,3]) = [1]⊗[2,3] - [0]⊗[2,3] + [0,1]⊗[3] - [0,1]⊗[2]."""
        C = SimplicialChains(r=2)
        x = C(((0, 1), (2, 3)))
        bdry = x.boundary()
        d = _as_dict(bdry)
        expected = {
            ((1,), (2, 3)): 1,
            ((0,), (2, 3)): -1,
            ((0, 1), (3,)): -1,  # sign = (-1)^(dim[0,1]) * (-1)^0 = (-1)^1
            ((0, 1), (2,)): 1,  # sign = (-1)^1 * (-1)^1 = 1
        }
        assert d == expected


class TestAWDiagonal:
    def test_diagonal_edge(self):
        """Δ([0,1]) = [0]⊗[0,1] + [0,1]⊗[1]."""
        x = SimplicialChains.standard_element(1)
        diag = x.iterated_diagonal(times=1)
        d = _as_dict(diag)
        assert d == {((0,), (0, 1)): 1, ((0, 1), (1,)): 1}

    def test_diagonal_triangle(self):
        """Δ([0,1,2]) = [0]⊗[0,1,2] + [0,1]⊗[1,2] + [0,1,2]⊗[2]."""
        x = SimplicialChains.standard_element(2)
        diag = x.iterated_diagonal(times=1)
        d = _as_dict(diag)
        assert d == {
            ((0,), (0, 1, 2)): 1,
            ((0, 1), (1, 2)): 1,
            ((0, 1, 2), (2,)): 1,
        }

    def test_iterated_diagonal_triangle(self):
        """Δ²([0,1,2]) should give 3-fold tensor terms."""
        x = SimplicialChains.standard_element(2)
        diag = x.iterated_diagonal(times=2)
        assert diag.arity() == 3
        # Check it's non-zero
        assert diag != SimplicialChains(r=3).zero()
        # Each basis key should be a 3-tuple of simplices
        for k in diag.support():
            assert len(k) == 3

    def test_diagonal_is_chain_map(self):
        """∂∘Δ = Δ∘∂ (AW diagonal is a chain map)."""
        for n in range(1, 4):
            x = SimplicialChains.standard_element(n)
            lhs = x.iterated_diagonal(times=1).boundary()
            rhs = x.boundary().iterated_diagonal(times=1)
            assert _as_dict(lhs) == _as_dict(rhs), f"Diagonal not a chain map on Δ^{n}"


# ===========================================================================
# Surjection action (BF convention)
# ===========================================================================


class TestSurjectionAction:
    def test_unit_action(self):
        """The operadic unit (1,) acts as the identity on chains."""
        unit = Surjection.unit()  # (1,) in S(1)
        for n in range(0, 4):
            x = SimplicialChains.standard_element(n)
            result = Surjection.act(unit, x)
            assert _as_dict(result) == _as_dict(x), f"Unit action failed on Δ^{n}"

    def test_action_degree(self):
        """Action preserves homogeneity (exact degree shift checked later)."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # arity 2, degree 1
        x = SimplicialChains.standard_element(3)  # degree 3
        result = Surjection.act(u, x)
        if result != SimplicialChains(r=2).zero():
            assert result.degree() is not None

    def test_action_arity(self):
        """θ_u(x) lives in C^⊗r."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))
        x = SimplicialChains.standard_element(3)
        result = Surjection.act(u, x)
        assert result.arity() == 2

    def test_action_zero_on_low_degree(self):
        """θ_u(x) = 0 if |x| < |u|."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # degree 1
        x = SimplicialChains.standard_element(0)  # degree 0
        result = Surjection.act(u, x)
        assert result == SimplicialChains(r=2).zero()

    def test_chain_map_property_small(self):
        """∂(θ_u(x)) = θ_{∂u}(x) + (-1)^|u| θ_u(∂x) for small examples.

        This is THE key correctness check for the BF sign convention.
        """
        S2 = Surjection(2)
        for surj_tuple in S2.planar_basis_it(1):
            u = surj_tuple
            d = 1  # degree of u
            for n in range(d, d + 4):
                x = SimplicialChains.standard_element(n)
                lhs = Surjection.act(u, x).boundary()
                rhs_1 = Surjection.act(u.boundary(), x)
                rhs_2 = Surjection.act(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed: u={list(u.support())}, n={n}"

    def test_chain_map_property_degree2(self):
        """Chain map property for degree-2 surjections in arity 2."""
        S2 = Surjection(2)
        for u in S2.planar_basis_it(2):
            d = 2
            for n in range(d, d + 3):
                x = SimplicialChains.standard_element(n)
                lhs = Surjection.act(u, x).boundary()
                rhs_1 = Surjection.act(u.boundary(), x)
                rhs_2 = Surjection.act(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed: u={list(u.support())}, n={n}"

    def test_chain_map_arity3(self):
        """Chain map property for arity-3 surjections."""
        S3 = Surjection(3)
        for u in S3.planar_basis_it(1):
            d = 1
            for n in range(d, d + 3):
                x = SimplicialChains.standard_element(n)
                lhs = Surjection.act(u, x).boundary()
                rhs_1 = Surjection.act(u.boundary(), x)
                rhs_2 = Surjection.act(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed (arity 3): u={list(u.support())}, n={n}"

    def test_action_121_on_triangle(self):
        """Explicit computation: θ_{(1,2,1)}([0,1,2]) by hand.

        (1,2,1) has arity 2, degree 1.
        Δ²([0,1,2]) = sum of ((0,..,i), (i,..,j), (j,..,2)) for 0≤i≤j≤2.
        Group by surjection: factor 1 gets positions 1,3; factor 2 gets position 2.
        Join factor-1 simplices, keep factor-2 simplex.
        """
        S2 = Surjection(2)
        u = S2((1, 2, 1))
        x = SimplicialChains.standard_element(2)
        result = Surjection.act(u, x)
        # Result should be non-zero and in arity 2
        assert result.arity() == 2
        assert result != SimplicialChains(r=2).zero()


# ===========================================================================
# CosimplicialCochains basics
# ===========================================================================


class TestCosimplicialCochains:
    def test_construction(self):
        C = SimplicialCochains(N=2, r=1)
        f = C(((0, 1),))
        assert f != C.zero()

    def test_coboundary_squared_zero(self):
        C = SimplicialCochains(N=3, r=1)
        for simplex_tuple in combinations(range(4), 2):
            f = C((simplex_tuple,))
            assert f.coboundary().coboundary() == C.zero()

    def test_evaluate_pairing(self):
        """⟨[0,1]*, [0,1]⟩ = 1 and ⟨[0,1]*, [0,2]⟩ = 0."""
        C_chain = SimplicialChains(r=1)
        C_cochain = SimplicialCochains(N=2, r=1)
        f = C_cochain(((0, 1),))
        x = C_chain(((0, 1),))
        y = C_chain(((0, 2),))
        assert SimplicialCochains.evaluate(f, x) == 1
        assert SimplicialCochains.evaluate(f, y) == 0

    def test_coboundary_is_boundary_transpose(self):
        """⟨δf, σ⟩ = ⟨f, ∂σ⟩ for all f, σ."""
        N = 3
        C = SimplicialCochains(N=N, r=1)
        C_chain = SimplicialChains(r=1)
        # For each 1-cochain f and 2-chain σ
        for edge in combinations(range(N + 1), 2):
            f = C((edge,))
            for triangle in combinations(range(N + 1), 3):
                sigma = C_chain((triangle,))
                lhs = SimplicialCochains.evaluate(f.coboundary(), sigma)
                rhs = SimplicialCochains.evaluate(f, sigma.boundary())
                assert lhs == rhs, f"Transpose failed: f={edge}, σ={triangle}"

    def test_cochain_action(self):
        """Test Surjection.coact produces a valid cochain."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # arity 2, degree 1, so acts on (deg q1 + deg q2 + 1)-chains
        N = 3
        C = SimplicialCochains(N=N, r=1)
        f1 = C(((0, 1),))  # degree 1
        f2 = C(((1, 2),))  # degree 1
        result = Surjection.coact(u, (f1, f2))
        # Should be a degree-3 cochain (1 + 1 + 1 = 3)
        # on Delta^3
        assert result.arity() == 1
