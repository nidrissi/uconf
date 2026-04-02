"""Tests for bar and cobar constructions."""

import itertools
import pytest
from sage.all import tensor, QQ

from uconf import (
    BarConstruction,
    BarrattEccles,
    HadamardProduct,
    CoAssociative,
    CoCommutative,
    CobarConstruction,
    Commutative,
    Lie,
    ShiftedOperad,
    Surjection,
    SurjectionDual,
)
from uconf.core.trees import (
    RootedTree,
    children,
    contract_edge,
    is_shuffle_tree,
    min_leaf,
    to_shuffle_tree_bar,
    decoration,
    expand_vertex,
    graft,
    internal_edges_dfs,
    is_internal,
    is_leaf,
    leaves,
    relabel_leaves,
    tree_arity,
    validate_tree,
    vertex_arity,
    vertices_dfs,
    weight,
)
from tests.planarize_helpers import planarize_round_trip_ok


class TestTrees:
    """Tests for tree utility functions."""

    def test_is_leaf(self):
        assert is_leaf(1)
        assert is_leaf(3)
        assert not is_leaf(((1, 2), 1, 2))

    def test_is_internal(self):
        tree = ((1, 2), 1, 2)  # Lie arity-3 tree
        assert is_internal(tree)
        assert not is_internal(1)

    def test_decoration(self):
        tree = ((1, 2), 1, 2)
        assert decoration(tree) == (1, 2)

    def test_children(self):
        tree = ((1, 2), 1, 2)
        assert children(tree) == (1, 2)

    def test_vertex_arity(self):
        tree = ((1, 2), 1, 2)
        assert vertex_arity(tree) == 2

        tree3 = ((1, 2), 1, 2, 3)
        assert vertex_arity(tree3) == 3

    def test_leaves(self):
        tree = ((1, 2), 1, 2)
        assert leaves(tree) == {1, 2}

        nested = ((2, 1), ((1,), 1, 2), 3)
        assert leaves(nested) == {1, 2, 3}

    def test_weight(self):
        assert weight(1) == 0
        tree = ((1, 2), 1, 2)
        assert weight(tree) == 1

        nested = ((2, 1), ((1,), 1, 2), 3)
        assert weight(nested) == 2

    def test_tree_arity(self):
        assert tree_arity(1) == 1
        tree = ((1, 2), 1, 2)
        assert tree_arity(tree) == 2

    def test_vertices_dfs(self):
        tree = ((1, 2), 1, 2)
        verts = vertices_dfs(tree)
        assert len(verts) == 1
        assert verts[0] is tree

        nested = ((2, 1), ((1,), 1, 2), 3)
        verts = vertices_dfs(nested)
        assert len(verts) == 2
        # DFS order: root first, then child
        assert verts[0] is nested
        assert decoration(verts[1]) == (1,)

    def test_internal_edges(self):
        tree = ((1, 2), 1, 2)
        assert internal_edges_dfs(tree) == []

        nested = ((2, 1), ((1,), 1, 2), 3)
        edges = internal_edges_dfs(nested)
        assert len(edges) == 1
        parent, pos, child = edges[0]
        assert parent is nested
        assert pos == 1
        assert decoration(child) == (1,)

    def test_relabel_leaves(self):
        tree = ((1, 2), 1, 2)
        relabeled = relabel_leaves(tree, {1: 3, 2: 4})
        assert leaves(relabeled) == {3, 4}
        assert decoration(relabeled) == (1, 2)

    def test_graft(self):
        # Graft a leaf onto another tree
        tree_top = ((1, 2), 1, 2)  # leaves {1, 2}
        tree_bot = ((1,), 1, 2)  # leaves {1, 2}

        # Graft tree_bot onto leaf 1 of tree_top
        grafted = graft(tree_top, 1, tree_bot)
        assert tree_arity(grafted) == 3  # Now has 3 leaves
        assert leaves(grafted) == {1, 2, 3}

    def test_validate_tree_lie(self):
        from sage.all import QQ

        # Valid Lie tree in arity 3
        tree = ((1, 2), 1, 2, 3)  # root has arity 3, decoration (1, 2)
        validated = validate_tree(tree, 3, Lie, QQ)
        assert validated is not None

        # Invalid: wrong leaves
        bad_tree = ((1, 2), 1, 2, 4)
        with pytest.raises(ValueError, match="Invalid leaves"):
            validate_tree(bad_tree, 3, Lie, QQ)


def test_bar_cobar_accept_nested_operad_providers() -> None:
    shifted_lie = ShiftedOperad(Lie, -1)
    surj_shifted_lie = HadamardProduct(Surjection, shifted_lie)

    bar = BarConstruction(surj_shifted_lie)
    b3 = bar(3, QQ)
    tree = (((1, 2, 3), (1, 2)), 1, 2, 3)

    assert b3(tree) != b3.zero()

    cobar = CobarConstruction(bar)
    assert cobar(2, QQ).arity() == 2


class TestShuffleTrees:
    """Tests for shuffle tree normalization."""

    def test_min_leaf(self):
        """Test min_leaf correctly finds minimum leaf in a tree."""
        assert min_leaf(1) == 1
        assert min_leaf(3) == 3
        assert min_leaf(((), 1, 2)) == 1
        assert min_leaf(((), 2, 1)) == 1
        # Nested tree: (((), 3, 4), 1, 2) has min leaf 1
        assert min_leaf((((), ((), 3, 4), 2), 1)) == 1

    def test_is_shuffle_tree(self):
        """Test is_shuffle_tree correctly identifies shuffle trees."""
        # Single leaf is always shuffle
        assert is_shuffle_tree(1)

        # Shuffle: children sorted by min leaf
        assert is_shuffle_tree(((), 1, 2))
        assert is_shuffle_tree(((), 1, 2, 3))

        # NOT shuffle: children not sorted
        assert not is_shuffle_tree(((), 2, 1))
        assert not is_shuffle_tree(((), 3, 1, 2))
        assert not is_shuffle_tree(((), 1, 3, 2))

        # Nested shuffle tree
        nested_shuffle = ((), ((), 1, 2), 3)
        assert is_shuffle_tree(nested_shuffle)

        # Nested non-shuffle: inner tree is not shuffle
        nested_non_shuffle = ((), ((), 2, 1), 3)
        assert not is_shuffle_tree(nested_non_shuffle)

    def test_to_shuffle_tree_bar_commutative(self):
        """Test that shuffle normalization works for Commutative operad."""
        # For Commutative, permuting children has no effect on decoration
        # ((), 2, 1) should normalize to ((), 1, 2) with sign

        tree = ((), 2, 1)
        terms = to_shuffle_tree_bar(tree, Commutative, QQ)

        # For Commutative, single term expected
        assert len(terms) == 1
        shuffle, sign = terms[0]

        # Should be sorted
        assert is_shuffle_tree(shuffle)
        assert shuffle == ((), 1, 2)

        # For Com with degree 0 decorations and weight-1 trees,
        # the Koszul sign is (-1)^{0*0} = 1
        # and the operad action sign is 1 (trivial action)
        assert sign == 1

    def test_to_shuffle_tree_bar_nested(self):
        """Test shuffle normalization on nested trees for Commutative."""
        # Nested tree: root with two children, child 1 is internal
        # ((), ((), 3, 4), ((), 1, 2))
        # Min leaves: child 1 has min 3, child 2 has min 1
        # Should swap children to get ((), ((), 1, 2), ((), 3, 4))

        tree = ((), ((), 3, 4), ((), 1, 2))
        terms = to_shuffle_tree_bar(tree, Commutative, QQ)

        # For Commutative, single term expected
        assert len(terms) == 1
        shuffle, sign = terms[0]

        assert is_shuffle_tree(shuffle)
        # Children should be swapped: first child has min 1, second has min 3
        assert min_leaf(shuffle[1]) == 1
        assert min_leaf(shuffle[2]) == 3

    def test_bar_commutative_shuffle_normalization(self):
        """Test that BarConstruction normalizes trees to shuffle form."""
        BCom = BarConstruction(Commutative)
        B2 = BCom(2, QQ)

        # These two trees should be equal after normalization
        tree1 = ((), 1, 2)
        tree2 = ((), 2, 1)

        elem1 = B2(tree1)
        elem2 = B2(tree2)

        # Both should be the same element (up to sign)
        # For Com, the sign is 1 (trivial action, degree 0)
        assert elem1 == elem2

    def test_bar_commutative_single_generator_arity2(self):
        """Test that B(Com)(2) has a single generator in the shuffle basis."""
        BCom = BarConstruction(Commutative)
        B2 = BCom(2, QQ)

        # The only shuffle tree in arity 2 with weight 1 is ((), 1, 2)
        shuffle_tree = ((), 1, 2)
        elem = B2(shuffle_tree)

        # Verify it's non-zero
        assert elem != B2.zero()

        # The non-shuffle version should give the same element
        non_shuffle_tree = ((), 2, 1)
        elem2 = B2(non_shuffle_tree)
        assert elem == elem2

    def test_bar_lie_shuffle_normalization(self):
        """Test shuffle normalization for BarConstruction(Lie)."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2, QQ)

        # Lie(2, QQ) has basis key (1,) which is skew-symmetric
        # Permuting by (12) gives a sign of -1
        tree1 = ((1,), 1, 2)
        tree2 = ((1,), 2, 1)

        elem1 = B2(tree1)
        elem2 = B2(tree2)

        # Lie is skew-symmetric, so elem2 = -elem1
        assert elem1 == -elem2


class TestBarConstructionLie:
    """Tests for bar construction of the Lie operad."""

    def test_bar_lie_creation(self):
        BLie = BarConstruction(Lie)
        B2 = BLie(2, QQ)
        assert B2.arity() == 2

    def test_bar_lie_counit_element(self):
        """Test the counit_element method returns the single-leaf tree."""
        BLie = BarConstruction(Lie)
        eta = BLie.counit_element(QQ)
        assert eta.arity() == 1
        # The single leaf has basis key = 1 (integer)
        assert len(list(eta)) == 1
        basis, coeff = next(iter(eta))
        assert basis == 1
        assert coeff == 1

    def test_bar_lie_counit_extracts_coefficient(self):
        """Test that counit extracts the coefficient of the single-leaf tree."""
        BLie = BarConstruction(Lie)
        B1 = BLie(1, QQ)

        # Create element: 3 * (single leaf)
        elem = 3 * B1(1)
        assert B1.counit(elem) == 3

        # Zero element
        assert B1.counit(B1.zero()) == 0

    def test_bar_lie_counit_zero_for_higher_arity(self):
        """Test that counit is zero for arity != 1."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2, QQ)
        tree = ((1,), 1, 2)
        elem = B2(tree)
        assert B2.counit(elem) == 0

    def test_bar_lie_reduced(self):
        """Test that reduced removes the counit component."""
        BLie = BarConstruction(Lie)
        B1 = BLie(1, QQ)

        # Element with only the single-leaf tree
        elem = 5 * B1(1)
        assert B1.reduced(elem) == B1.zero()

        # For arity 2, reduced is identity
        B2 = BLie(2, QQ)
        tree = ((1,), 1, 2)
        elem2 = B2(tree)
        assert B2.reduced(elem2) == elem2

    def test_bar_lie_degree(self):
        BLie = BarConstruction(Lie)
        B3 = BLie(3, QQ)
        tree = ((1, 2), 1, 2, 3)
        assert B3.degree_on_basis(tree) == 1

    def test_bar_lie_boundary_squared_zero_arity2(self):
        """d^2 = 0 for B(Lie)(2)."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2, QQ)

        # Arity 2 tree: single vertex with decoration (1,) and children 1, 2
        # Lie(2, QQ) has basis key (1,) - a permutation of (1,) representing [x_1, x_2]
        tree = ((1,), 1, 2)
        elem = B2(tree)

        # Lie has trivial differential, so d_1 = 0
        # d_2 contracts edges, but weight-1 tree has no internal edges to contract
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B2.zero(), f"d^2 != 0: {bdry2}"

    def test_bar_lie_boundary_squared_zero_arity3(self):
        """d^2 = 0 for B(Lie)(3)."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3, QQ)

        # Weight-1 tree in arity 3
        tree = ((1, 2), 1, 2, 3)
        elem = B3(tree)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B3.zero(), f"d^2 != 0: {bdry2}"

        # Another basis element
        tree2 = ((2, 1), 1, 2, 3)
        elem2 = B3(tree2)
        bdry = elem2.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B3.zero()

    def test_bar_lie_boundary_squared_zero_weight2(self):
        """d^2 = 0 for weight-2 trees in B(Lie)."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3, QQ)

        # Weight-2 tree: root with arity 2, one child is internal with arity 2
        # Root: (1,), children: internal node with (1,), and leaf 3
        # Child: (1,), children: 1, 2
        nested = ((1,), ((1,), 1, 2), 3)
        elem = B3(nested)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B3.zero(), f"d^2 != 0 for weight-2 tree: {bdry2}"

    def test_bar_lie_d2_edge_contraction(self):
        """Test that d_2 correctly contracts internal edges."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3, QQ)

        # Weight-2 tree
        nested = ((1,), ((1,), 1, 2), 3)
        elem = B3(nested)

        # d_2 should contract to a weight-1 tree
        d2 = elem.d2()
        assert d2 != B3.zero()

        # Result should be weight-1 trees (since we contract one edge)
        for tree, coeff in d2:
            assert weight(tree) == 1

    def test_bar_lie_boundary_squared_zero_arity4_weight2(self):
        """d^2 = 0 for a weight-2 tree in B(Lie)(4)."""
        BLie = BarConstruction(Lie)
        B4 = BLie(4, QQ)

        # Root arity 3 with one internal arity-2 child
        tree = ((1, 2), ((1,), 1, 2), 3, 4)
        elem = B4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B4.zero(), f"d^2 != 0 in arity 4: {bdry2}"

    def test_bar_lie_boundary_squared_zero_arity6_weight3(self):
        """d^2 = 0 for a weight-3 tree in B(Lie)(6)."""
        BLie = BarConstruction(Lie)
        B6 = BLie(6, QQ)

        tree = ((1, 2, 3), ((1,), 1, 2), ((1,), 3, 4), 5, 6)
        elem = B6(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B6.zero(), f"d^2 != 0 in arity 6: {bdry2}"

    def test_bar_lie_permute(self):
        """Test leaf permutation."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3, QQ)

        tree = ((1, 2), 1, 2, 3)
        elem = B3(tree)

        permuted = elem.permute([2, 1, 3])
        # Check that leaves are relabeled
        for new_tree, coeff in permuted:
            assert leaves(new_tree) == {1, 2, 3}


class TestBarConstructionSurjection:
    """Tests for bar construction of the Surjection operad."""

    def test_bar_surjection_creation(self):
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)
        assert B2.arity() == 2

    def test_bar_surjection_degree(self):
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)
        tree = ((1, 2), 1, 2)
        assert B2.degree_on_basis(tree) == 1

        # Surjection basis (1, 2, 1) has degree 1
        tree2 = ((1, 2, 1), 1, 2)
        assert B2.degree_on_basis(tree2) == 2

    def test_bar_surjection_d_squared_zero_weight1(self):
        """d^2 = 0 for weight-1 trees in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)

        # Weight-1 tree with (1, 2) decoration
        tree = ((1, 2), 1, 2)
        elem = B2(tree)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B2.zero(), f"d^2 != 0: {bdry2}"

    def test_bar_surjection_d_squared_zero_degree1(self):
        """d^2 = 0 for degree-1 basis elements."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)

        # Surjection basis (1, 2, 1) has degree 1
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B2.zero(), f"d^2 != 0: {bdry2}"

    def test_bar_surjection_internal_differential(self):
        """Test that d_1 applies Surjection.boundary correctly."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)

        # (1, 2, 1) has nontrivial boundary in Surjection
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)

        d1 = elem.d1()
        # Should produce trees with different decorations
        assert d1 != B2.zero()

    def test_bar_surjection_boundary_squared_zero_arity4_weight2(self):
        """d^2 = 0 for a higher-arity, weight-2 tree in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B4 = BS(4, QQ)

        tree = ((1, 2, 3, 1), ((1, 2, 1), 1, 2), 3, 4)
        elem = B4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B4.zero(), f"d^2 != 0 in arity 4: {bdry2}"

    def test_bar_surjection_boundary_squared_zero_arity5_weight3(self):
        """d^2 = 0 for a higher-arity, weight-2 tree in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B5 = BS(5, QQ)

        tree = ((1, 2, 3, 1), ((1, 2, 1), 1, 2), ((1, 2, 1), 3, 4), 5)
        elem = B5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B5.zero(), f"d^2 != 0 in arity 5: {bdry2}"


class TestCobarConstruction:
    """Tests for cobar construction of cooperads."""

    def test_cobar_creation(self):
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2, QQ)
        assert O2.arity() == 2

    def test_cobar_unit(self):
        OmegaS = CobarConstruction(SurjectionDual)
        unit = OmegaS.unit(QQ)
        assert unit.arity() == 1
        assert unit == OmegaS(1, QQ)(1)

    def test_cobar_degree(self):
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2, QQ)
        # SurjectionDual basis (1, 2) has degree 0
        # s^{-1}C degree = 0 - (2 - 1) = -1
        tree = ((1, 2), 1, 2)
        assert O2.degree_on_basis(tree) == -1

    def test_cobar_d_squared_zero_unit(self):
        """d^2 = 0 for the unit."""
        OmegaS = CobarConstruction(SurjectionDual)
        unit = OmegaS.unit(QQ)
        bdry = unit.boundary()
        assert bdry == OmegaS(1, QQ).zero()

    def test_cobar_compose(self):
        """Test free operad composition (grafting)."""
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2, QQ)

        tree1 = ((1, 2), 1, 2)
        elem1 = O2(tree1)

        # Compose with unit
        unit = OmegaS.unit(QQ)
        composed = OmegaS.compose(elem1, 1, unit)
        assert composed.arity() == 2
        # Composing with unit should give back the same element (essentially)
        assert composed == elem1

    def test_cobar_compose_two_trees(self):
        """Test composing two nontrivial trees."""
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2, QQ)

        tree1 = ((1, 2), 1, 2)
        tree2 = ((1, 2), 1, 2)
        elem1 = O2(tree1)
        elem2 = O2(tree2)

        # Graft tree2 onto leaf 1 of tree1
        composed = OmegaS.compose(elem1, 1, elem2)
        assert composed.arity() == 3
        # Result should be a weight-2 tree
        for tree, coeff in composed:
            assert weight(tree) == 2

    def test_cobar_compose_to_arity5(self):
        """Compose larger trees and check resulting arity/leaf normalization."""
        OmegaS = CobarConstruction(SurjectionDual)
        O3 = OmegaS(3, QQ)

        x = O3(((1, 2, 3), 1, 2, 3))
        y = O3(((1, 3, 2), 1, 2, 3))

        composed = OmegaS.compose(x, 2, y)
        assert composed.arity() == 5

        for tree, coeff in composed:
            assert leaves(tree) == {1, 2, 3, 4, 5}
            assert weight(tree) == 2

    def test_cobar_boundary_squared_zero_arity4_weight2(self):
        """d^2 = 0 for a higher-arity, weight-2 tree in Ω(S*)."""
        OmegaS = CobarConstruction(SurjectionDual)
        O4 = OmegaS(4, QQ)

        tree = ((1, 2, 3), ((1, 2), 1, 2), 3, 4)
        elem = O4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O4.zero(), f"d^2 != 0 in arity 4: {bdry2}"


class TestContractEdge:
    """Tests for edge contraction in trees."""

    def test_contract_simple(self):
        # Tree: root (1,) with children (internal (1,) with children 1, 2) and 3
        child_vertex = ((1,), 1, 2)
        root = ((1,), child_vertex, 3)
        # Contract root -> child edge at position 1
        # New vertex has arity 3, children 1, 2, 3
        # Result depends on new_decoration

        # Lie.compose(Lie(2, QQ)((1,)), 1, Lie(2, QQ)((1,))) computes the composition
        L2 = Lie(2, QQ)
        bracket = L2((1,))
        composed = Lie.compose(bracket, 1, bracket)
        # In Lie, id ∘_1 id = id in arity 3, which has basis keys permutations of (1, 2)
        # Lie(3, QQ) basis keys are permutations of (1, 2)
        for new_dec, coeff in composed:
            new_tree = contract_edge(root, root, 1, new_dec)
            assert weight(new_tree) == 1
            assert tree_arity(new_tree) == 3


class TestExpandVertex:
    """Tests for vertex expansion in trees."""

    def test_expand_simple(self):
        # Start with a single vertex tree
        tree = ((1, 2, 1), 1, 2, 3)  # Surjection arity 3

        # Expand vertex at position l=1, arities (2, 2)
        # Original has arity 3, expanded to arity-2 top + arity-2 bottom
        # Top gets children 1 (the bottom vertex), 3
        # Bottom gets children 1, 2
        expanded = expand_vertex(
            tree,
            tree,
            1,
            left_decoration=(1, 2),  # arity 2
            right_decoration=(1, 2),  # arity 2
            left_arity=2,
            right_arity=2,
        )
        assert weight(expanded) == 2
        assert tree_arity(expanded) == 3


class TestBarCoderivation:
    """Tests that the bar differential is a coderivation of the cooperad."""

    def _check_coderivation(self, factory, tree, i, m, n):
        """Check Δ(d(x)) == (d⊗id + (-1)^|a| id⊗d)(Δ(x)) for a single tree."""
        B = factory(m + n - 1, QQ)
        Bm = factory(m, QQ)
        Bn = factory(n, QQ)
        tgt = tensor([Bm, Bn])

        x = B(tree)
        dx = x.boundary()

        # LHS: Δ_{i;m,n}(d(x)), recast into tgt
        lhs = tgt.zero()
        for (ak, bk), coeff in B.infinitesimal_cocompose(dx, i, m, n):
            lhs += coeff * Bm(ak).tensor(Bn(bk))

        # RHS: (d⊗id + (-1)^|a| id⊗d)(Δ_{i;m,n}(x))
        rhs = tgt.zero()
        for (ak, bk), coeff in B.infinitesimal_cocompose(x, i, m, n):
            a_deg = Bm.degree_on_basis(ak)
            for dak, da_coeff in Bm(ak).boundary():
                rhs += coeff * da_coeff * Bm(dak).tensor(Bn(bk))
            sign_a = (-1) ** a_deg
            for dbk, db_coeff in Bn(bk).boundary():
                rhs += coeff * sign_a * db_coeff * Bm(ak).tensor(Bn(dbk))

        assert lhs == rhs, f"Coderivation property failed for tree {tree}"

    def test_bar_lie_coderivation_weight2_arity3(self):
        """d is a coderivation: Δ(d(x)) = (d⊗id + (-1)^|a| id⊗d)(Δ(x))."""
        BLie = BarConstruction(Lie)
        # weight-2 tree in B(Lie)(3): Δ_{1;2,2} and Δ_{2;2,2}
        tree = ((1,), ((1,), 1, 2), 3)
        self._check_coderivation(BLie, tree, 1, 2, 2)
        self._check_coderivation(BLie, tree, 2, 2, 2)

    def test_bar_lie_coderivation_weight2_arity4(self):
        """Coderivation property for B(Lie)(4) weight-2 tree."""
        BLie = BarConstruction(Lie)
        tree = ((1, 2), ((1,), 1, 2), 3, 4)
        self._check_coderivation(BLie, tree, 1, 2, 3)
        self._check_coderivation(BLie, tree, 2, 2, 3)
        self._check_coderivation(BLie, tree, 1, 3, 2)

    def test_bar_surjection_coderivation_weight1(self):
        """Coderivation property for weight-1 B(Surjection)(3) tree."""
        BS = BarConstruction(Surjection)
        # weight-1 tree: Δ gives 0 (no internal subtree to split off at arity 2)
        tree = ((1, 2, 3), 1, 2, 3)
        self._check_coderivation(BS, tree, 1, 2, 2)

    def test_bar_surjection_coderivation_weight2(self):
        """Coderivation property for weight-2 B(Surjection)(4) tree."""
        BS = BarConstruction(Surjection)
        tree = ((1, 2, 3), ((1, 2), 1, 2), 3, 4)
        self._check_coderivation(BS, tree, 1, 2, 3)
        self._check_coderivation(BS, tree, 2, 3, 2)

    def test_bar_surjection_coderivation_nontrivial_d1(self):
        """Coderivation holds when d_1 is nontrivial (Surjection has d1 ≠ 0)."""
        BS = BarConstruction(Surjection)
        # (1,2,1) has nontrivial boundary in Surjection(2, QQ)
        tree = ((1, 2, 1), 1, 2)
        self._check_coderivation(BS, tree, 1, 1, 2)


class TestCoAssociative:
    """Tests for the CoAssociative cooperad."""

    def test_coassociative_counit(self):
        """Counit evaluates correctly on arity-1 generator."""
        C1 = CoAssociative(1, QQ)
        assert CoAssociative.counit(C1((1,))) == 1
        C2 = CoAssociative(2, QQ)
        assert CoAssociative.counit(C2((1, 2))) == 0

    def test_coassociative_reduced(self):
        """Reduced kills the arity-1 counit component."""
        C1 = CoAssociative(1, QQ)
        elem = 3 * C1((1,))
        assert CoAssociative.reduced(elem) == C1.zero()
        C2 = CoAssociative(2, QQ)
        elem2 = C2((1, 2))
        assert CoAssociative.reduced(elem2) == elem2

    def test_coassociative_cocompose_arity3(self):
        """Cocomposition at arity 3: dual of Ass.compose."""
        from uconf import Associative

        C3 = CoAssociative(3, QQ)
        sigma = C3((1, 2, 3))  # identity permutation in S_3

        # Δ^{1;2,2}((1,2,3)): find (tau, rho) with Ass.compose(tau,1,rho)=(1,2,3)
        delta = CoAssociative.infinitesimal_cocompose(sigma, 1, 2, 2)
        assert delta != tensor([CoAssociative(2, QQ), CoAssociative(2, QQ)]).zero()

        # Verify: composing back should give original basis element
        for (left_key, right_key), coeff in delta:
            tau = CoAssociative(2, QQ)(left_key)
            rho = CoAssociative(2, QQ)(right_key)
            composed = Associative.compose(tau, 1, rho)
            assert dict(composed) == {(1, 2, 3): 1}

    def test_cobar_coassociative_d_squared_zero_arity3(self):
        """d^2 = 0 for Ω(CoAss)(3)."""
        OmegaCA = CobarConstruction(CoAssociative)
        O3 = OmegaCA(3, QQ)
        # Weight-1 tree with (1,2,3) decoration
        tree = ((1, 2, 3), 1, 2, 3)
        elem = O3(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O3.zero(), f"d^2 != 0 for CoAss: {bdry2}"

    def test_cobar_coassociative_d_squared_zero_arity4(self):
        """d^2 = 0 for Ω(CoAss)(4)."""
        OmegaCA = CobarConstruction(CoAssociative)
        O4 = OmegaCA(4, QQ)
        tree = ((1, 2, 3, 4), 1, 2, 3, 4)
        elem = O4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O4.zero(), f"d^2 != 0 for CoAss arity 4: {bdry2}"


class TestCoCommutative:
    """Tests for the CoCommutative cooperad."""

    def test_cocommutative_counit(self):
        """Counit evaluates correctly on arity-1 generator."""
        C1 = CoCommutative(1, QQ)
        assert CoCommutative.counit(C1(())) == 1
        C2 = CoCommutative(2, QQ)
        assert CoCommutative.counit(C2(())) == 0

    def test_cocommutative_reduced(self):
        """Reduced kills the arity-1 counit component."""
        C1 = CoCommutative(1, QQ)
        elem = 5 * C1(())
        assert CoCommutative.reduced(elem) == C1.zero()
        C2 = CoCommutative(2, QQ)
        elem2 = C2(())
        assert CoCommutative.reduced(elem2) == elem2

    def test_cocommutative_cocompose(self):
        """Cocomposition maps e_{m+n-1} to e_m ⊗ e_n."""
        C3 = CoCommutative(3, QQ)
        elem = C3(())

        # For any (i, m, n) with m+n-1=3 and i valid: Δ^{i;2,2}(e_3) = e_2 ⊗ e_2
        delta = CoCommutative.infinitesimal_cocompose(elem, 1, 2, 2)
        # delta should have exactly one term: e_2 ⊗ e_2 with coefficient 1
        assert dict(delta) == {((), ()): 1}

        # Different position i gives the same result (cocommutative)
        delta2 = CoCommutative.infinitesimal_cocompose(elem, 2, 2, 2)
        assert dict(delta2) == {((), ()): 1}

    def test_cobar_cocommutative_d_squared_zero_arity3(self):
        """d^2 = 0 for Ω(CoCom)(3)."""
        OmegaCC = CobarConstruction(CoCommutative)
        O3 = OmegaCC(3, QQ)
        # weight-1 tree with decoration () in arity 3
        tree = ((), 1, 2, 3)
        elem = O3(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O3.zero(), f"d^2 != 0 for CoCom: {bdry2}"


class TestCobarSignFix:
    """Tests verifying the corrected cumulative-sign in cobar d_2."""

    @pytest.mark.parametrize("arity", [3, 4, 5])
    def test_cobar_com_square_zero_arity5_weight1(self, arity: int):
        OmegaCom = CobarConstruction(CoCommutative)
        OC = OmegaCom(arity, QQ)
        tree = ((),) + tuple(range(1, arity + 1))
        elem = OC(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == OC.zero(), f"d^2 != 0 for CoCom arity {arity}: {bdry2}"

    def test_cobar_surjection_square_zero_arity4_weight2(self):
        """d^2 = 0 for a weight-2 tree in Ω(S*) where the sign fix matters."""
        OmegaS = CobarConstruction(SurjectionDual)
        O4 = OmegaS(4, QQ)
        # root arity 3 with one internal arity-2 child
        tree = ((1, 2, 3, 1), ((1, 2, 1), 1, 2), 3, 4)
        elem = O4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O4.zero(), f"d^2 != 0 in arity 4 weight 2: {bdry2}"

    def test_cobar_sign_arity5_weight2(self):
        """d^2 = 0 for a weight-2 tree where the sign fix matters.

        Root: arity 3, dec (1,2,3,1) in S*(3), s^{-1}C degree = -1 (odd).
        Inner: arity 3, dec (1,2,3) in S*(3), s^{-1}C degree = -2 (even).
        Without the cumulative_before sign, d^2 would be non-zero for this tree.
        """
        OmegaS = CobarConstruction(SurjectionDual)
        O5 = OmegaS(5, QQ)
        # root arity 3, inner arity 3 as first child, leaves 4 and 5 as other children
        tree = ((1, 2, 3, 1), ((1, 2, 3), 1, 2, 3), 4, 5)
        elem = O5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O5.zero(), f"d^2 != 0 in arity 5 weight 2: {bdry2}"

    def test_cobar_sign_arity5_weight2_inner_dec(self):
        """d^2 = 0 for a tree where the inner node has high-degree decoration."""
        OmegaS = CobarConstruction(SurjectionDual)
        O5 = OmegaS(5, QQ)
        tree = ((1, 2, 3, 1), ((1, 2, 1, 3), 1, 2, 3), 4, 5)
        elem = O5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O5.zero(), f"d^2 != 0 in arity 5 weight 2: {bdry2}"

    def test_cobar_sign_arity6_weight3(self):
        """d^2 = 0 for a weight-3 tree where the sign fix matters."""
        OmegaS = CobarConstruction(SurjectionDual)
        O6 = OmegaS(6, QQ)
        tree = ((1, 2, 3, 1, 4), ((1, 2, 1, 2), 1, 2), ((1, 2, 1), 3, 4), 5, 6)
        elem = O6(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O6.zero(), f"d^2 != 0 in arity 6 weight 3: {bdry2}"


class TestBarPlanarize:
    """Tests for the quasi-planar structure on B(P)."""

    # ------------------------------------------------------------------
    # Surjection
    # ------------------------------------------------------------------

    def test_planarize_surjection_planar_tree(self):
        """A planar tree planarizes to itself ⊗ identity."""
        from sage.all import SymmetricGroup

        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        tree = ((1, 2, 1, 3), 1, 2, 3)
        elem = B3(tree)
        result = elem.planarize()
        S3 = SymmetricGroup(3)
        items = list(result)
        assert len(items) == 1
        (pl_tree, sigma_key), coeff = items[0]
        assert coeff == 1
        assert pl_tree == ((1, 2, 1, 3), 1, 2, 3)
        assert S3(sigma_key) == S3.identity()

    def test_planarize_surjection_nonplanar_corolla(self):
        """(2,1,2,3) planarizes to (1,2,1,3) ⊗ (1,2) (the transposition)."""
        from sage.all import SymmetricGroup

        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        tree = ((2, 1, 2, 3), 1, 2, 3)
        elem = B3(tree)
        result = elem.planarize()
        items = list(result)
        assert len(items) == 1
        (pl_tree, sigma_key), coeff = items[0]
        assert coeff == 1
        assert pl_tree == ((1, 2, 1, 3), 1, 2, 3)
        S3 = SymmetricGroup(3)
        assert S3(sigma_key) == S3([2, 1, 3])

    def test_planarize_surjection_weight2_roundtrip(self):
        """Round-trip T_pl.permute(σ) == T for a weight-2 Surjection tree."""
        BS = BarConstruction(Surjection)
        B4 = BS(4, QQ)
        tree = ((2, 1, 2, 3), ((1, 2), 1, 2), 3, 4)
        elem = B4(tree)
        assert elem != B4.zero()
        assert planarize_round_trip_ok(elem)

    def test_planarize_surjection_roundtrip_weight1(self):
        """Round-trip holds for various weight-1 Surjection trees."""
        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        for tree in [
            ((1, 2, 3), 1, 2, 3),
            ((2, 1, 2, 3), 1, 2, 3),
            ((1, 3, 2, 1), 1, 2, 3),
        ]:
            elem = B3(tree)
            assert elem != B3.zero()
            assert planarize_round_trip_ok(elem), f"Round-trip failed for {tree}"

    def test_planarize_surjection_roundtrip_higher_weights(self):
        """Round-trip holds on higher-weight Surjection bar trees."""
        BS = BarConstruction(Surjection)

        B5 = BS(5, QQ)
        weight3_tree = ((2, 1, 2, 3), ((1, 2, 1), 1, 2), ((2, 1), 3, 4), 5)
        elem = B5(weight3_tree)
        assert elem != B5.zero()
        assert planarize_round_trip_ok(elem)

        B6 = BS(6, QQ)
        weight3_tree_arity6 = ((2, 1, 3, 4, 2), ((1, 2), 1, 2), ((1, 2, 1), 3, 4), 5, 6)
        elem2 = B6(weight3_tree_arity6)
        assert elem2 != B6.zero()
        assert planarize_round_trip_ok(elem2)

    # ------------------------------------------------------------------
    # Barratt–Eccles
    # ------------------------------------------------------------------

    def test_planarize_barratt_eccles_nonplanar(self):
        """A non-planar BE decoration planarizes with the correct permutation."""
        from sage.all import SymmetricGroup

        BBE = BarConstruction(BarrattEccles)
        B2 = BBE(2, QQ)
        BE2 = BarrattEccles(2, QQ)
        S2 = BE2._symmetric_group
        perm21 = S2([2, 1])
        id2 = S2.identity()
        tree = ((perm21, id2), 1, 2)
        elem = B2(tree)
        result = elem.planarize()
        items = list(result)
        assert len(items) == 1
        (pl_tree, sigma_key), coeff = items[0]
        assert coeff == 1
        # The planar decoration must start with identity
        assert pl_tree[0][0] == id2
        # The global permutation must be (1,2) (the transposition)
        S2_global = SymmetricGroup(2)
        assert S2_global(sigma_key) == S2_global([2, 1])

    def test_planarize_barratt_eccles_roundtrip(self):
        """Round-trip holds for B(BE)(2) elements."""
        BBE = BarConstruction(BarrattEccles)
        B2 = BBE(2, QQ)
        BE2 = BarrattEccles(2, QQ)
        S2 = BE2._symmetric_group
        perm21 = S2([2, 1])
        id2 = S2.identity()
        for tree in [
            ((perm21, id2), 1, 2),
            ((id2, perm21), 1, 2),
        ]:
            elem = B2(tree)
            assert elem != B2.zero()
            assert planarize_round_trip_ok(elem), f"Round-trip failed for {tree}"

    def test_planarize_barratt_eccles_roundtrip_weight2(self):
        """Round-trip holds for a weight-2 tree in B(BE)."""
        BBE = BarConstruction(BarrattEccles)
        B3 = BBE(3, QQ)
        BE2 = BarrattEccles(2, QQ)
        S2 = BE2._symmetric_group
        id2 = S2.identity()
        perm21 = S2([2, 1])
        tree = ((perm21, id2), ((id2, perm21), 1, 2), 3)
        elem = B3(tree)
        assert elem != B3.zero()
        assert planarize_round_trip_ok(elem)

    # ------------------------------------------------------------------
    # planar_basis_iter
    # ------------------------------------------------------------------

    def test_planar_basis_it_surjection_degree1_arity2(self):
        """B(Surj)(2) degree-1 planar basis is exactly {(1,2)}."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)
        basis = list(B2.planar_basis_iter(1))
        assert len(basis) == 1
        tree_key = list(basis[0].support())[0]
        assert tree_key == ((1, 2), 1, 2)

    def test_planar_basis_it_surjection_degree1_arity3(self):
        """B(Surj)(3) degree-1 planar basis is exactly {(1,2,3)}."""
        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        basis = list(B3.planar_basis_iter(1))
        assert len(basis) == 1
        tree_key = list(basis[0].support())[0]
        assert tree_key == ((1, 2, 3), 1, 2, 3)

    def test_planar_basis_it_surjection_degree2_arity2(self):
        """B(Surj)(2) degree-2 planar basis is exactly {(1,2,1)}."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)
        basis = list(B2.planar_basis_iter(2))
        assert len(basis) == 1
        tree_key = list(basis[0].support())[0]
        assert tree_key == ((1, 2, 1), 1, 2)

    def test_planar_basis_it_all_planar(self):
        """Every element yielded by planar_basis_iter is planar (σ = id)."""
        from sage.all import SymmetricGroup

        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        S3 = SymmetricGroup(3)
        for d in range(3):
            for elem in B3.planar_basis_iter(d):
                result = elem.planarize()
                for (_, sigma_key), _ in result:
                    assert S3(sigma_key) == S3.identity(), (
                        f"Non-identity sigma for planar element in degree {d}"
                    )

    def test_planar_basis_it_barratt_eccles(self):
        """B(BE)(2) degree-1 planar basis contains the identity-starting tuple."""
        BBE = BarConstruction(BarrattEccles)
        B2 = BBE(2, QQ)
        basis = list(B2.planar_basis_iter(1))
        assert len(basis) == 1
        tree_key = list(basis[0].support())[0]
        dec = tree_key[0]
        # Planar means first permutation is the identity
        BE2 = BarrattEccles(2, QQ)
        assert dec[0] == BE2._symmetric_group.identity()

    # ------------------------------------------------------------------
    # Element.planarize() convenience method
    # ------------------------------------------------------------------

    def test_element_planarize_method(self):
        """Element.planarize() delegates to Component.planarize()."""
        BS = BarConstruction(Surjection)
        B3 = BS(3, QQ)
        tree = ((2, 1, 2, 3), 1, 2, 3)
        elem = B3(tree)
        # Both call paths should give the same result
        result_component = B3.planarize(elem)
        result_element = elem.planarize()
        assert result_component == result_element

    # ------------------------------------------------------------------
    # HadamardProduct
    # ------------------------------------------------------------------

    def test_planarize_hadamard_roundtrip(self):
        """Round-trip holds for B(S⊙BE)(2)."""
        HBE = HadamardProduct(Surjection, BarrattEccles)
        BH = BarConstruction(HBE)
        B2 = BH(2, QQ)
        BE2 = BarrattEccles(2, QQ)
        S2 = BE2._symmetric_group
        perm21 = S2([2, 1])
        id2 = S2.identity()
        had_tree = (((1, 2), (perm21, id2)), 1, 2)
        elem = B2(had_tree)
        assert elem != B2.zero()
        assert planarize_round_trip_ok(elem)


class TestBasisIter:
    """Tests for BarConstruction.Component.basis_iter and planar_basis_iter raising."""

    def test_planar_basis_it_raises_for_commutative(self):
        """planar_basis_iter should raise NotImplementedError for Com (no planarize)."""
        BCom = BarConstruction(Commutative)
        B2 = BCom(2, QQ)
        with pytest.raises(NotImplementedError):
            list(B2.planar_basis_iter(1))

    def test_planar_basis_it_raises_for_lie(self):
        """planar_basis_iter should raise NotImplementedError for Lie (no planarize)."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2, QQ)
        with pytest.raises(NotImplementedError):
            list(B2.planar_basis_iter(1))

    def test_basis_it_arity1_degree0(self):
        """B(Com)(1) in degree 0 is the single leaf."""
        B1 = BarConstruction(Commutative)(1, QQ)
        basis = list(B1.basis_iter(0))
        assert len(basis) == 1
        key, coeff = next(iter(basis[0]))
        assert key == 1 and coeff == 1

    def test_basis_it_arity1_positive_degree(self):
        """B(Com)(1) in degree > 0 is empty."""
        B1 = BarConstruction(Commutative)(1, QQ)
        assert list(B1.basis_iter(1)) == []

    def test_basis_it_commutative_arity2_degree1(self):
        """B(Com)(2) in degree 1 has exactly one shuffle tree."""
        B2 = BarConstruction(Commutative)(2, QQ)
        basis = list(B2.basis_iter(1))
        assert len(basis) == 1
        (key, coeff) = next(iter(basis[0]))
        assert key == ((), 1, 2)
        assert coeff == 1

    def test_basis_it_commutative_arity3_degree1(self):
        """B(Com)(3) in degree 1 has one tree (single arity-3 vertex)."""
        B3 = BarConstruction(Commutative)(3, QQ)
        basis = list(B3.basis_iter(1))
        assert len(basis) == 1
        (key, _) = next(iter(basis[0]))
        assert key == ((), 1, 2, 3)

    def test_basis_it_commutative_arity3_degree2(self):
        """B(Com)(3) in degree 2 has exactly three shuffle trees (weight 2)."""
        B3 = BarConstruction(Commutative)(3, QQ)
        basis = list(B3.basis_iter(2))
        keys = [next(iter(e))[0] for e in basis]
        assert len(keys) == 3
        # All three weight-2 shuffle trees for arity 3:
        assert ((), 1, ((), 2, 3)) in keys  # root(leaf1, internal{2,3})
        assert ((), ((), 1, 2), 3) in keys  # root(internal{1,2}, leaf3)
        assert ((), ((), 1, 3), 2) in keys  # root(internal{1,3}, leaf2)  — non-planar

    def test_basis_it_all_shuffle_trees(self):
        """Every element from basis_iter is a valid shuffle tree."""
        from uconf.core.trees import is_shuffle_tree

        B3 = BarConstruction(Commutative)(3, QQ)
        for d in range(1, 4):
            for elem in B3.basis_iter(d):
                for key, _ in elem:
                    assert is_shuffle_tree(key), f"Not a shuffle tree: {key} in degree {d}"

    def test_basis_it_lie_consistent_with_planar_for_surjection(self):
        """For Surjection, basis_iter and planar_basis_iter correspond element-by-element."""
        BS = BarConstruction(Surjection)
        B2 = BS(2, QQ)
        for d in range(1, 4):
            planar = {next(iter(e))[0] for e in B2.planar_basis_iter(d)}
            full = {next(iter(e))[0] for e in B2.basis_iter(d)}
            # Every planar tree is also a shuffle tree.
            assert planar <= full


class TestBarConstructionNegativeDegree:
    """Regression tests for BarConstruction with negative-degree operads (issue #21)."""

    def test_basis_it_shifted_lie_arity2_degree0(self):
        """B(sLie)(2) in degree 0 should be non-empty (issue #21 example)."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        BsLie2 = BsLie(2, QQ)
        basis = list(BsLie2.basis_iter(0))
        assert basis != []

    def test_basis_it_shifted_lie_arity2_degree0_exact_key(self):
        """B(sLie)(2) in degree 0 has exactly one corolla tree."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        BsLie2 = BsLie(2, QQ)
        basis = list(BsLie2.basis_iter(0))
        assert len(basis) == 1
        (key, coeff) = next(iter(basis[0]))
        # Corolla with root decorated by sLie(2) basis element (1,) and leaves 1, 2
        assert key == ((1,), 1, 2)
        assert coeff == 1

    def test_basis_it_shifted_lie_arity3_degree_minus1(self):
        """B(sLie)(3) in degree -1 has two corolla trees (one per Lie(3) basis element)."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        BsLie3 = BsLie(3, QQ)
        basis = list(BsLie3.basis_iter(-1))
        assert len(basis) == 2  # Two Lie(3) basis elements

    def test_basis_it_shifted_lie_arity3_degree0(self):
        """B(sLie)(3) in degree 0 has multi-vertex shuffle trees."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        BsLie3 = BsLie(3, QQ)
        basis = list(BsLie3.basis_iter(0))
        assert basis != []

    def test_basis_it_shifted_lie_negative_degree_only_valid(self):
        """B(sLie)(2) in degree -1 should be empty (below minimum)."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        BsLie2 = BsLie(2, QQ)
        # sLie(2) has degree -1; minimum bar degree of corolla = -1+1 = 0.
        # No tree can have bar degree < 0 for arity 2.
        assert list(BsLie2.basis_iter(-1)) == []

    def test_basis_it_shifted_lie_all_shuffle_trees(self):
        """Every element from basis_iter on B(sLie) is a valid shuffle tree."""
        from uconf.core.trees import is_shuffle_tree

        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        B3 = BsLie(3, QQ)
        for d in range(-1, 2):
            for elem in B3.basis_iter(d):
                for key, _ in elem:
                    assert is_shuffle_tree(key), f"Not a shuffle tree: {key} in degree {d}"

    def test_boundary_squared_zero_shifted_lie(self):
        """Boundary squares to zero on B(sLie) basis elements."""
        sLie = ShiftedOperad(Lie, -1)
        BsLie = BarConstruction(sLie)
        for n in (2, 3):
            B = BsLie(n, QQ)
            for d in range(-1, 2):
                for elem in B.basis_iter(d):
                    d1 = B.boundary(elem)
                    d2 = B.boundary(d1)
                    assert d2 == B.zero(), f"boundary^2 != 0 on {elem} in B(sLie)({n}) degree {d}"


# ===========================================================================
# Right action:  (x·σ)·τ = x·(στ) for all σ, τ ∈ S(n)
# ===========================================================================


@pytest.mark.parametrize("n", range(2, 4))
@pytest.mark.parametrize("d", range(-1, 2))
def test_bar_construction_right_action(n: int, d: int) -> None:
    """(x·σ)·τ = x·(στ) for all σ, τ ∈ S(n)."""
    Bar = BarConstruction(Surjection)
    Barn = Bar(n, QQ)
    Sn = Barn._symmetric_group
    for x in Barn.basis_iter(d):
        for sigma, tau in itertools.product(list(Sn), repeat=2):
            lhs = x.permute(sigma).permute(tau)
            rhs = x.permute(sigma * tau)
            assert lhs == rhs, f"Right action failed for x={x}, σ={sigma}, τ={tau}"


@pytest.mark.parametrize("n", range(2, 4))
@pytest.mark.parametrize("d", range(-1, 2))
def test_cobar_construction_right_action(n: int, d: int) -> None:
    """(x·σ)·τ = x·(στ) for all σ, τ ∈ S(n)."""
    Cobar = CobarConstruction(BarConstruction(Surjection))
    Cobarn = Cobar(n, QQ)
    Sn = Cobarn._symmetric_group
    for x in Cobarn.basis_iter(d):
        for sigma, tau in itertools.product(list(Sn), repeat=2):
            lhs = x.permute(sigma).permute(tau)
            rhs = x.permute(sigma * tau)
            assert lhs == rhs, f"Right action failed for x={x}, σ={sigma}, τ={tau}"
