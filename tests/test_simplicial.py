"""Tests for simplicial chains, cochains, and the surjection action."""

from typing import Iterable

import pytest
from itertools import chain, combinations
from sage.all import tensor

from uconf import SimplicialChains, SimplicialCochains, Surjection
from uconf.algebraic.simplicial import SurjectionSimplicialCochainAlgebra


# ===========================================================================
# Helpers
# ===========================================================================


def _as_dict(x):
    """Canonical dictionary representation of an element."""
    return {k: v for k, v in x if v != 0}


def _evaluate_tensor_cochains_on_chain(cochains, chain):
    """Evaluate a pure-tensor cochain on a chain in matching arity.

    Parameters
    ----------
    cochains : tuple[SimplicialCochains.Element, ...]
        One arity-1 cochain per tensor factor.
    chain : element of tensor([SimplicialChains()]*r)
        A chain with r tensor factors, r = len(cochains).
    """
    r = len(cochains)
    SC = SimplicialChains()
    value = 0
    for basis_key, coeff in chain:
        # basis_key = (s1, ..., sr) for r >= 2, or a single simplex for r == 1
        if r == 1:
            factor_keys = (basis_key,)
        else:
            factor_keys = basis_key
        contribution = coeff
        for idx in range(r):
            contribution *= SimplicialCochains.evaluate(
                cochains[idx], SC.term(factor_keys[idx])
            )
            if contribution == 0:
                break
        value += contribution
    return value


def _surjection_cochain_action(u, cochains):
    """Evaluate the surjection action on cochains via the algebra wrapper."""
    parent: SimplicialCochains = cochains[0].parent()
    alg = SurjectionSimplicialCochainAlgebra(parent._N, base_ring=parent.base_ring())
    return alg.act(u, list(cochains))


# ===========================================================================
# SimplicialChains basics
# ===========================================================================


class TestSimplicialChains:
    def test_construction_single_simplex(self):
        C = SimplicialChains()
        x = C((0, 1, 2))
        assert x != C.zero()

    def test_construction_tensor(self):
        SC = SimplicialChains()
        T = tensor([SC, SC])
        x = T.term(((0, 1, 2), (0, 1)))
        assert x != T.zero()

    def test_degenerate_rejected(self):
        C = SimplicialChains()
        # (0, 0, 1) has repeated entry → degenerate
        x = C((0, 0, 1))
        assert x == C.zero()

    def test_empty_simplex_rejected(self):
        C = SimplicialChains()
        x = C(())
        assert x == C.zero()

    def test_negative_vertex_rejected(self):
        C = SimplicialChains()
        x = C((-1, 0, 1))
        assert x == C.zero()

    def test_non_integer_vertex_rejected(self):
        C = SimplicialChains()
        x = C((0, 1.5, 2))
        assert x == C.zero()

    def test_degree(self):
        C = SimplicialChains()
        x = C((0, 1, 2))
        assert x.degree() == 2

    def test_fundamental_chain(self):
        x = SimplicialChains.fundamental_chain(3)
        assert x.degree() == 3
        d = _as_dict(x)
        assert d == {(0, 1, 2, 3): 1}


class TestBoundary:
    def test_boundary_vertex(self):
        """Boundary of a 0-simplex is zero."""
        C = SimplicialChains()
        x = C((0,))
        assert x.boundary() == C.zero()

    def test_boundary_edge(self):
        """∂[0,1] = [1] - [0]."""
        C = SimplicialChains()
        x = C((0, 1))
        bdry = x.boundary()
        d = _as_dict(bdry)
        assert d == {(1,): 1, (0,): -1}

    def test_boundary_triangle(self):
        """∂[0,1,2] = [1,2] - [0,2] + [0,1]."""
        C = SimplicialChains()
        x = C((0, 1, 2))
        bdry = x.boundary()
        d = _as_dict(bdry)
        assert d == {(1, 2): 1, (0, 2): -1, (0, 1): 1}

    def test_boundary_squared_is_zero(self):
        """∂² = 0 for several simplices."""
        C = SimplicialChains()
        for n in range(1, 5):
            x = SimplicialChains.fundamental_chain(n)
            assert x.boundary().boundary() == C.zero(), f"∂² ≠ 0 for Δ^{n}"

    def test_boundary_squared_tensor(self):
        """∂² = 0 on tensor products using tensor_boundary."""
        SC = SimplicialChains()
        T = tensor([SC, SC])
        x = T.term(((0, 1, 2), (0, 1)))
        b = SimplicialChains.tensor_boundary
        assert b(b(x)) == T.zero()

    def test_boundary_tensor_product(self):
        """∂([0,1] ⊗ [2,3]) = [1]⊗[2,3] - [0]⊗[2,3] - [0,1]⊗[3] + [0,1]⊗[2]."""
        SC = SimplicialChains()
        T = tensor([SC, SC])
        x = T.term(((0, 1), (2, 3)))
        bdry = SimplicialChains.tensor_boundary(x)
        d = _as_dict(bdry)
        expected = {
            ((1,), (2, 3)): 1,
            ((0,), (2, 3)): -1,
            ((0, 1), (3,)): -1,
            ((0, 1), (2,)): 1,
        }
        assert d == expected


class TestAWDiagonal:
    def test_diagonal_edge(self):
        """Δ([0,1]) = [0]⊗[0,1] + [0,1]⊗[1]."""
        x = SimplicialChains.fundamental_chain(1)
        diag = x.iterated_diagonal(times=1)
        d = _as_dict(diag)
        assert d == {((0,), (0, 1)): 1, ((0, 1), (1,)): 1}

    def test_diagonal_triangle(self):
        """Δ([0,1,2]) = [0]⊗[0,1,2] + [0,1]⊗[1,2] + [0,1,2]⊗[2]."""
        x = SimplicialChains.fundamental_chain(2)
        diag = x.iterated_diagonal(times=1)
        d = _as_dict(diag)
        assert d == {
            ((0,), (0, 1, 2)): 1,
            ((0, 1), (1, 2)): 1,
            ((0, 1, 2), (2,)): 1,
        }

    def test_iterated_diagonal_triangle(self):
        """Δ²([0,1,2]) should give 3-fold tensor terms."""
        SC = SimplicialChains()
        x = SimplicialChains.fundamental_chain(2)
        diag = x.iterated_diagonal(times=2)
        T3 = tensor([SC, SC, SC])
        assert diag.parent() == T3
        assert diag != T3.zero()
        # Each basis key should be a 3-tuple of simplex tuples
        for k in diag.support():
            assert len(k) == 3

    def test_diagonal_is_chain_map(self):
        """∂∘Δ = Δ∘∂ (AW diagonal is a chain map)."""
        b = SimplicialChains.tensor_boundary
        for n in range(1, 4):
            x = SimplicialChains.fundamental_chain(n)
            lhs = b(x.iterated_diagonal(times=1))
            rhs = x.boundary().iterated_diagonal(times=1)
            assert _as_dict(lhs) == _as_dict(rhs), f"Diagonal not a chain map on Δ^{n}"


# ===========================================================================
# Surjection action
# ===========================================================================


class TestSurjectionAction:
    @pytest.mark.parametrize("n", range(0, 4))
    def test_unit_action(self, n: int):
        """The operadic unit (1,) acts as the identity on cochains."""
        unit = Surjection.unit()  # (1,) in S(1)
        x = SimplicialCochains.volume_form(n)
        result = _surjection_cochain_action(unit, [x])
        assert _as_dict(result) == _as_dict(x), f"Unit action failed on Δ^{n}"

    @pytest.mark.parametrize("r,d", [(2, 1), (2, 2), (3, 1)])
    def test_action_degree(self, r: int, d: int):
        """Cochain action: for nonzero θ_u(x, …, x), |θ_u(x, …, x)| = d - n*r.

        Here u ∈ S(r) has degree d, x is the volume form in degree n, and we
        test θ_u(x, …, x) with r copies of x.
        """
        for u in Surjection(r).planar_basis_it(d):
            for n in range(d, d + 4):
                x = SimplicialCochains.volume_form(n)
                l = [x] * r
                result = _surjection_cochain_action(u, l)
                if result != result.parent().zero():
                    assert result.degree() == d - n * r

    def test_action_zero_on_low_degree(self):
        """θ_u(x) = 0 if |x| < |u|."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # degree 1
        x = SimplicialCochains.volume_form(0)  # degree 0
        result = _surjection_cochain_action(u, [x] * 2)
        assert _as_dict(result) == {}

    @staticmethod
    def powerset_nonempty(iterable: Iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    @pytest.mark.parametrize("d", range(1, 4))
    def test_chain_map_property_arity2(self, d: int):
        """∂(θ_u(x)) = θ_{∂u}(x) + (-1)^|u| θ_u(∂x) for arity 2."""
        S2 = Surjection(2)
        for u in S2.planar_basis_it(d):
            for n in range(d, d + 2):
                possible_simplices = list(self.powerset_nonempty(range(n + 1)))
                for x in possible_simplices:
                    for y in possible_simplices:
                        x = SimplicialCochains(n)(x)
                        y = SimplicialCochains(n)(y)
                        lhs = _surjection_cochain_action(u, [x, y]).coboundary()
                        rhs_1 = _surjection_cochain_action(u.boundary(), [x, y])
                        rhs_2 = _surjection_cochain_action(u, [x.coboundary(), y])
                        rhs_3 = _surjection_cochain_action(u, [x, y.coboundary()])
                        rhs = (
                            rhs_1
                            + (-1) ** (u.degree()) * rhs_2
                            + (-1) ** (u.degree() + x.degree()) * rhs_3
                        )

                        assert _as_dict(lhs) == _as_dict(rhs), (
                            f"Chain map failed: u={list(u.support())}, x={x}, y={y}"
                        )

    @pytest.mark.parametrize("d", range(1, 4))
    def test_chain_map_property_arity3(self, d: int):
        """∂(θ_u(x)) = θ_{∂u}(x) + (-1)^|u| θ_u(∂x) for arity 3."""
        S3 = Surjection(3)
        for u in S3.planar_basis_it(d):
            for n in range(d, d + 2):
                possible_simplices = list(self.powerset_nonempty(range(n + 1)))
                for x in possible_simplices:
                    for y in possible_simplices:
                        for z in possible_simplices:
                            x = SimplicialCochains(n)(x)
                            y = SimplicialCochains(n)(y)
                            z = SimplicialCochains(n)(z)
                            lhs = _surjection_cochain_action(u, [x, y, z]).coboundary()
                            rhs_1 = _surjection_cochain_action(u.boundary(), [x, y, z])
                            rhs_2 = _surjection_cochain_action(
                                u, [x.coboundary(), y, z]
                            )
                            rhs_3 = _surjection_cochain_action(
                                u, [x, y.coboundary(), z]
                            )
                            rhs_4 = _surjection_cochain_action(
                                u, [x, y, z.coboundary()]
                            )
                            rhs = (
                                rhs_1
                                + (-1) ** (u.degree()) * rhs_2
                                + (-1) ** (u.degree() + x.degree()) * rhs_3
                                + (-1) ** (u.degree() + x.degree() + y.degree()) * rhs_4
                            )
                            assert _as_dict(lhs) == _as_dict(rhs), (
                                f"Chain map failed: u={list(u.support())}, n={n}"
                            )
