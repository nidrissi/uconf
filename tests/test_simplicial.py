"""Tests for simplicial chains, cochains, and the surjection action."""

import pytest
from itertools import combinations, product
from sage.all import ZZ, tensor

from uconf import SimplicialChains, SimplicialCochains, Surjection


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


def _surjection_chain_action(u, x):
    """Evaluate the surjection action on chains via the coalgebra wrapper."""
    coalg = SimplicialChains(base_ring=u.parent().base_ring()).as_surjection_coalgebra()
    return coalg.act(u, x)


def _surjection_cochain_action(u, cochains):
    """Evaluate the surjection action on cochains via the algebra wrapper."""
    alg = cochains[0].parent().as_surjection_algebra()
    return alg.act(u, list(cochains))


def _chain_arity(result):
    """Return number of tensor factors of a chain action result.

    Returns 1 if *result* is in a plain :class:`SimplicialChains`, otherwise
    returns the number of tensor factors.
    """
    parent = result.parent()
    if isinstance(parent, SimplicialChains):
        return 1
    return len(parent._sets)


def _chain_zero(r, base_ring=None):
    """Return the zero element of the appropriate chain space of arity r."""
    SC = (
        SimplicialChains()
        if base_ring is None
        else SimplicialChains(base_ring=base_ring)
    )
    if r == 1:
        return SC.zero()
    return tensor([SC] * r).zero()


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

    def test_standard_element(self):
        x = SimplicialChains.standard_element(3)
        assert x.degree() == 3
        d = _as_dict(x)
        assert d == {(0, 1, 2, 3): 1}

    def test_standard_element_tensor(self):
        SC = SimplicialChains()
        x = SimplicialChains.standard_element(2)
        T = tensor([SC, SC])
        y = x.iterated_diagonal(times=1)
        assert y.parent() == T


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
            x = SimplicialChains.standard_element(n)
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
        SC = SimplicialChains()
        x = SimplicialChains.standard_element(2)
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
            x = SimplicialChains.standard_element(n)
            lhs = b(x.iterated_diagonal(times=1))
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
            result = _surjection_chain_action(unit, x)
            assert _as_dict(result) == _as_dict(x), f"Unit action failed on Δ^{n}"

    def test_action_degree(self):
        """For nonzero terms, |θ_u(x)| = |x| + |u|."""
        for r, d in [(2, 1), (2, 2), (3, 1)]:
            S = Surjection(r)
            for u in S.planar_basis_it(d):
                for n in range(d, d + 4):
                    x = SimplicialChains.standard_element(n)
                    result = _surjection_chain_action(u, x)
                    if result != _chain_zero(r):
                        # Degree = sum of simplex dimensions across all tensor factors
                        first_key = next(iter(result.support()))
                        # first_key is an r-tuple of simplex tuples (r >= 2 here)
                        deg_sum = sum(len(simplex) - 1 for simplex in first_key)
                        assert deg_sum == n + d

    def test_action_arity(self):
        """θ_u(x) lives in SC^⊗r."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))
        x = SimplicialChains.standard_element(3)
        result = _surjection_chain_action(u, x)
        assert _chain_arity(result) == 2

    def test_action_zero_on_low_degree(self):
        """θ_u(x) = 0 if |x| < |u|."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # degree 1
        x = SimplicialChains.standard_element(0)  # degree 0
        result = _surjection_chain_action(u, x)
        assert _as_dict(result) == {}

    def test_chain_map_property_small(self):
        """∂(θ_u(x)) = θ_{∂u}(x) + (-1)^|u| θ_u(∂x) for small examples."""
        b = SimplicialChains.tensor_boundary
        S2 = Surjection(2)
        for u in S2.planar_basis_it(1):
            d = 1
            for n in range(d, d + 4):
                x = SimplicialChains.standard_element(n)
                lhs = b(_surjection_chain_action(u, x))
                rhs_1 = _surjection_chain_action(u.boundary(), x)
                rhs_2 = _surjection_chain_action(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed: u={list(u.support())}, n={n}"

    def test_chain_map_property_degree2(self):
        """Chain map property for degree-2 surjections in arity 2."""
        b = SimplicialChains.tensor_boundary
        S2 = Surjection(2)
        for u in S2.planar_basis_it(2):
            d = 2
            for n in range(d, d + 3):
                x = SimplicialChains.standard_element(n)
                lhs = b(_surjection_chain_action(u, x))
                rhs_1 = _surjection_chain_action(u.boundary(), x)
                rhs_2 = _surjection_chain_action(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed: u={list(u.support())}, n={n}"

    def test_chain_map_arity3(self):
        """Chain map property for arity-3 surjections."""
        b = SimplicialChains.tensor_boundary
        S3 = Surjection(3)
        for u in S3.planar_basis_it(1):
            d = 1
            for n in range(d, d + 3):
                x = SimplicialChains.standard_element(n)
                lhs = b(_surjection_chain_action(u, x))
                rhs_1 = _surjection_chain_action(u.boundary(), x)
                rhs_2 = _surjection_chain_action(u, x.boundary())
                rhs = rhs_1 + (-1) ** d * rhs_2
                assert _as_dict(lhs) == _as_dict(
                    rhs
                ), f"Chain map failed (arity 3): u={list(u.support())}, n={n}"

    def test_composed_surjection_chain_map(self):
        """Chain-map identity also holds for explicit composed surjections."""
        b = SimplicialChains.tensor_boundary
        S2 = Surjection(2)
        u = S2((1, 2, 1))
        v = S2((1, 2, 1))
        w = Surjection.compose(u, 1, v)
        d_w = next(iter({w.parent().degree_on_basis(k) for k in w.support()}))

        for n in range(2, 5):
            x = SimplicialChains.standard_element(n)
            lhs = b(_surjection_chain_action(w, x))
            rhs = _surjection_chain_action(w.boundary(), x) + (
                -1
            ) ** d_w * _surjection_chain_action(w, x.boundary())
            assert _as_dict(lhs) == _as_dict(rhs)

    def test_action_121_on_triangle(self):
        """Explicit computation: θ_{(1,2,1)}([0,1,2]) is non-zero and in SC⊗SC."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))
        x = SimplicialChains.standard_element(2)
        result = _surjection_chain_action(u, x)
        assert _chain_arity(result) == 2
        assert result != _chain_zero(2)


# ===========================================================================
# SimplicialCochains basics
# ===========================================================================


class TestCosimplicialCochains:
    def test_construction(self):
        C = SimplicialCochains(N=2)
        f = C((0, 1))
        assert f != C.zero()

    def test_degree_is_negative_simplex_dimension(self):
        C = SimplicialCochains(N=3)
        assert C((0,)).degree() == 0
        assert C((0, 1)).degree() == -1
        assert C((0, 1, 2)).degree() == -2

    def test_coboundary_squared_zero(self):
        C = SimplicialCochains(N=3)
        for simplex_tuple in combinations(range(4), 2):
            f = C(simplex_tuple)
            assert f.coboundary().coboundary() == C.zero()

    def test_evaluate_pairing(self):
        """⟨[0,1]*, [0,1]⟩ = 1 and ⟨[0,1]*, [0,2]⟩ = 0."""
        SC = SimplicialChains()
        C_cochain = SimplicialCochains(N=2)
        f = C_cochain((0, 1))
        x = SC((0, 1))
        y = SC((0, 2))
        assert SimplicialCochains.evaluate(f, x) == 1
        assert SimplicialCochains.evaluate(f, y) == 0

    def test_coboundary_is_boundary_transpose(self):
        """⟨δf, σ⟩ = ⟨f, ∂σ⟩ for all f, σ."""
        N = 3
        SC = SimplicialChains()
        C = SimplicialCochains(N=N)
        for edge in combinations(range(N + 1), 2):
            f = C(edge)
            for triangle in combinations(range(N + 1), 3):
                sigma = SC(triangle)
                lhs = SimplicialCochains.evaluate(f.coboundary(), sigma)
                rhs = SimplicialCochains.evaluate(f, sigma.boundary())
                assert lhs == rhs, f"Transpose failed: f={edge}, σ={triangle}"

    def test_cochain_action(self):
        """Surjection.coact produces a valid cochain in the same SimplicialCochains module."""
        S2 = Surjection(2)
        u = S2((1, 2, 1))  # arity 2, degree 1
        N = 3
        C = SimplicialCochains(N=N)
        f1 = C((0, 1))  # degree -1
        f2 = C((1, 2))  # degree -1
        result = _surjection_cochain_action(u, (f1, f2))
        # Result lives in SimplicialCochains(N=N) – always arity-1 by design.
        assert isinstance(result.parent(), SimplicialCochains)

    def test_coact_is_adjoint_to_chain_action_arity2(self):
        """⟨μ_u(f1⊗f2), x⟩ = ⟨f1⊗f2, θ_u(x)⟩ in small exhaustive arity-2 cases."""
        N = 3
        S2 = Surjection(2)
        Cco = SimplicialCochains(N=N)

        degree1_cochains = [Cco(simplex) for simplex in combinations(range(N + 1), 2)]

        for u in S2.planar_basis_it(1):
            d = 1
            for f1, f2 in product(degree1_cochains, repeat=2):
                mu = _surjection_cochain_action(u, (f1, f2))
                n = -f1.degree() - f2.degree() - d
                for simplex in combinations(range(N + 1), n + 1):
                    SC = SimplicialChains()
                    x = SC(simplex)
                    lhs = SimplicialCochains.evaluate(mu, x)
                    rhs = _evaluate_tensor_cochains_on_chain(
                        (f1, f2), _surjection_chain_action(u, x)
                    )
                    assert lhs == rhs

    def test_coact_is_adjoint_to_chain_action_arity3(self):
        """Adjointness pairing check in representative arity-3 degree-1 cases."""
        N = 4
        S3 = Surjection(3)
        Cco = SimplicialCochains(N=N)
        degree1_cochains = [Cco(simplex) for simplex in combinations(range(N + 1), 2)]

        sample = degree1_cochains[:3]
        for u in S3.planar_basis_it(1):
            d = 1
            for f1, f2, f3 in product(sample, repeat=3):
                mu = _surjection_cochain_action(u, (f1, f2, f3))
                n = -f1.degree() - f2.degree() - f3.degree() - d
                for simplex in combinations(range(N + 1), n + 1):
                    SC = SimplicialChains()
                    x = SC(simplex)
                    lhs = SimplicialCochains.evaluate(mu, x)
                    rhs = _evaluate_tensor_cochains_on_chain(
                        (f1, f2, f3), _surjection_chain_action(u, x)
                    )
                    assert lhs == rhs

    def test_act_requires_same_base_ring(self):
        u = Surjection(2)((1, 2))
        x = SimplicialChains(base_ring=ZZ)((0, 1))
        with pytest.raises(TypeError, match="same base ring"):
            _surjection_chain_action(u, x)

    def test_coact_requires_same_base_ring(self):
        u = Surjection(2)((1, 2))
        C = SimplicialCochains(N=2, base_ring=ZZ)
        f1 = C((0, 1))
        f2 = C((1, 2))
        with pytest.raises(TypeError, match="same base ring"):
            _surjection_cochain_action(u, (f1, f2))
