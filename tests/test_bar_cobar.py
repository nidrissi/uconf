"""Tests for bar and cobar constructions."""

import pytest
from sage.all import tensor, QQ

from uconf import (
    BarConstruction,
    CoAssociative,
    CoCommutative,
    CobarConstruction,
    Commutative,
    Lie,
    Surjection,
    SurjectionDual,
)
from uconf.core.trees import (
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
        assert validate_tree(bad_tree, 3, Lie, QQ) is None


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
        shuffle, sign = to_shuffle_tree_bar(tree, Commutative, QQ)

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
        shuffle, sign = to_shuffle_tree_bar(tree, Commutative, QQ)

        assert is_shuffle_tree(shuffle)
        # Children should be swapped: first child has min 1, second has min 3
        assert min_leaf(shuffle[1]) == 1
        assert min_leaf(shuffle[2]) == 3

    def test_bar_commutative_shuffle_normalization(self):
        """Test that BarConstruction normalizes trees to shuffle form."""
        BCom = BarConstruction(Commutative)
        B2 = BCom(2)

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
        B2 = BCom(2)

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
        B2 = BLie(2)

        # Lie(2) has basis key (1,) which is skew-symmetric
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
        B2 = BLie(2)
        assert B2.arity() == 2

    def test_bar_lie_counit_element(self):
        """Test the counit_element method returns the single-leaf tree."""
        BLie = BarConstruction(Lie)
        eta = BLie.counit_element()
        assert eta.arity() == 1
        # The single leaf has basis key = 1 (integer)
        assert len(list(eta)) == 1
        basis, coeff = next(iter(eta))
        assert basis == 1
        assert coeff == 1

    def test_bar_lie_counit_extracts_coefficient(self):
        """Test that counit extracts the coefficient of the single-leaf tree."""
        BLie = BarConstruction(Lie)
        B1 = BLie(1)

        # Create element: 3 * (single leaf)
        elem = 3 * B1.term(1)
        assert B1.counit(elem) == 3

        # Zero element
        assert B1.counit(B1.zero()) == 0

    def test_bar_lie_counit_zero_for_higher_arity(self):
        """Test that counit is zero for arity != 1."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2)
        tree = ((1,), 1, 2)
        elem = B2(tree)
        assert B2.counit(elem) == 0

    def test_bar_lie_reduced(self):
        """Test that reduced removes the counit component."""
        BLie = BarConstruction(Lie)
        B1 = BLie(1)

        # Element with only the single-leaf tree
        elem = 5 * B1.term(1)
        assert B1.reduced(elem) == B1.zero()

        # For arity 2, reduced is identity
        B2 = BLie(2)
        tree = ((1,), 1, 2)
        elem2 = B2(tree)
        assert B2.reduced(elem2) == elem2

    def test_bar_lie_degree(self):
        BLie = BarConstruction(Lie)
        B3 = BLie(3)
        tree = ((1, 2), 1, 2, 3)
        assert B3.degree_on_basis(tree) == 1

    def test_bar_lie_boundary_squared_zero_arity2(self):
        """d^2 = 0 for B(Lie)(2)."""
        BLie = BarConstruction(Lie)
        B2 = BLie(2)

        # Arity 2 tree: single vertex with decoration (1,) and children 1, 2
        # Lie(2) has basis key (1,) - a permutation of (1,) representing [x_1, x_2]
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
        B3 = BLie(3)

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
        B3 = BLie(3)

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
        B3 = BLie(3)

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
        B4 = BLie(4)

        # Root arity 3 with one internal arity-2 child
        tree = ((1, 2), ((1,), 1, 2), 3, 4)
        elem = B4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B4.zero(), f"d^2 != 0 in arity 4: {bdry2}"

    def test_bar_lie_boundary_squared_zero_arity6_weight3(self):
        """d^2 = 0 for a weight-3 tree in B(Lie)(6)."""
        BLie = BarConstruction(Lie)
        B6 = BLie(6)

        tree = ((1, 2, 3), ((1,), 1, 2), ((1,), 3, 4), 5, 6)
        elem = B6(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B6.zero(), f"d^2 != 0 in arity 6: {bdry2}"

    def test_bar_lie_permute(self):
        """Test leaf permutation."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3)

        tree = ((1, 2), 1, 2, 3)
        elem = B3(tree)

        permuted = elem.permute([2, 1, 3])
        # Check that leaves are relabeled
        for new_tree, coeff in permuted:
            assert leaves(new_tree) == {1, 2, 3}

    def test_bar_pretty_print(self):
        """Bar terms should display in tree form with operad labels."""
        BLie = BarConstruction(Lie)
        B3 = BLie(3)
        tree = ((1, 2), 1, 2, 3)
        rep = B3._repr_term(tree)
        latex = B3._latex_term(tree)
        assert rep.startswith("(Lie")
        assert "; 1, 2, 3)" in rep
        assert "\\operatorname{Lie}" in latex

    def test_bar_element_pretty_print(self):
        """Bar elements should support direct pretty-printing."""
        from sage.all import latex

        BLie = BarConstruction(Lie)
        B3 = BLie(3)
        elem = B3(((1, 2), 1, 2, 3)) + 2 * B3(((2, 1), 1, 2, 3))

        pretty = elem.pretty()
        assert "(Lie(1, 2); 1, 2, 3)" in pretty
        assert "2*(Lie(2, 1); 1, 2, 3)" in pretty
        assert repr(elem) == pretty

        latex_repr = elem.pretty_latex()
        assert "\\operatorname{Lie}" in latex_repr
        assert latex(elem) == latex_repr


class TestBarConstructionSurjection:
    """Tests for bar construction of the Surjection operad."""

    def test_bar_surjection_creation(self):
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        assert B2.arity() == 2

    def test_bar_surjection_degree(self):
        BS = BarConstruction(Surjection)
        B2 = BS(2)
        tree = ((1, 2), 1, 2)
        assert B2.degree_on_basis(tree) == 1

        # Surjection basis (1, 2, 1) has degree 1
        tree2 = ((1, 2, 1), 1, 2)
        assert B2.degree_on_basis(tree2) == 2

    def test_bar_surjection_d_squared_zero_weight1(self):
        """d^2 = 0 for weight-1 trees in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)

        # Weight-1 tree with (1, 2) decoration
        tree = ((1, 2), 1, 2)
        elem = B2(tree)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B2.zero(), f"d^2 != 0: {bdry2}"

    def test_bar_surjection_d_squared_zero_degree1(self):
        """d^2 = 0 for degree-1 basis elements."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)

        # Surjection basis (1, 2, 1) has degree 1
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)
        bdry = elem.boundary()
        bdry2 = bdry.boundary()
        assert bdry2 == B2.zero(), f"d^2 != 0: {bdry2}"

    def test_bar_surjection_internal_differential(self):
        """Test that d_1 applies Surjection.boundary correctly."""
        BS = BarConstruction(Surjection)
        B2 = BS(2)

        # (1, 2, 1) has nontrivial boundary in Surjection
        tree = ((1, 2, 1), 1, 2)
        elem = B2(tree)

        d1 = elem.d1()
        # Should produce trees with different decorations
        assert d1 != B2.zero()

    def test_bar_surjection_boundary_squared_zero_arity4_weight2(self):
        """d^2 = 0 for a higher-arity, weight-2 tree in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B4 = BS(4)

        tree = ((1, 2, 3, 1), ((1, 2, 1), 1, 2), 3, 4)
        elem = B4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B4.zero(), f"d^2 != 0 in arity 4: {bdry2}"

    def test_bar_surjection_boundary_squared_zero_arity5_weight3(self):
        """d^2 = 0 for a higher-arity, weight-2 tree in B(Surjection)."""
        BS = BarConstruction(Surjection)
        B5 = BS(5)

        tree = ((1, 2, 3, 1), ((1, 2, 1), 1, 2), ((1, 2, 1), 3, 4), 5)
        elem = B5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == B5.zero(), f"d^2 != 0 in arity 5: {bdry2}"


class TestCobarConstruction:
    """Tests for cobar construction of cooperads."""

    def test_cobar_creation(self):
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2)
        assert O2.arity() == 2

    def test_cobar_unit(self):
        OmegaS = CobarConstruction(SurjectionDual)
        unit = OmegaS.unit()
        assert unit.arity() == 1
        assert unit == OmegaS(1).term(1)

    def test_cobar_degree(self):
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2)
        # SurjectionDual basis (1, 2) has degree 0
        # s^{-1}C degree = 0 - (2 - 1) = -1
        tree = ((1, 2), 1, 2)
        assert O2.degree_on_basis(tree) == -1

    def test_cobar_d_squared_zero_unit(self):
        """d^2 = 0 for the unit."""
        OmegaS = CobarConstruction(SurjectionDual)
        unit = OmegaS.unit()
        bdry = unit.boundary()
        assert bdry == OmegaS(1).zero()

    def test_cobar_compose(self):
        """Test free operad composition (grafting)."""
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2)

        tree1 = ((1, 2), 1, 2)
        elem1 = O2(tree1)

        # Compose with unit
        unit = OmegaS.unit()
        composed = OmegaS.compose(elem1, 1, unit)
        assert composed.arity() == 2
        # Composing with unit should give back the same element (essentially)
        assert composed == elem1

    def test_cobar_compose_two_trees(self):
        """Test composing two nontrivial trees."""
        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2)

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
        O3 = OmegaS(3)

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
        O4 = OmegaS(4)

        tree = ((1, 2, 3), ((1, 2), 1, 2), 3, 4)
        elem = O4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O4.zero(), f"d^2 != 0 in arity 4: {bdry2}"

    def test_cobar_pretty_print(self):
        """Cobar terms should display trees and unit as id."""
        OmegaS = CobarConstruction(SurjectionDual)
        O1 = OmegaS(1)
        O2 = OmegaS(2)

        assert O1._repr_term(1) == "id"
        assert O1._latex_term(1) == "\\mathrm{id}"

        tree = ((1, 2), 1, 2)
        rep = O2._repr_term(tree)
        assert rep.startswith("(S*")

    def test_cobar_element_pretty_print(self):
        """Cobar elements should support direct pretty-printing."""
        from sage.all import latex

        OmegaS = CobarConstruction(SurjectionDual)
        O2 = OmegaS(2)
        elem = O2(((1, 2), 1, 2)) + 3 * O2(((1, 2, 1), 1, 2))

        pretty = elem.pretty()
        assert "(S*(1, 2); 1, 2)" in pretty
        assert "3*(S*(1, 2, 1); 1, 2)" in pretty
        assert repr(elem) == pretty

        latex_repr = elem.pretty_latex()
        assert "\\operatorname{S*}" in latex_repr
        assert latex(elem) == latex_repr


class TestContractEdge:
    """Tests for edge contraction in trees."""

    def test_contract_simple(self):
        # Tree: root (1,) with children (internal (1,) with children 1, 2) and 3
        child_vertex = ((1,), 1, 2)
        root = ((1,), child_vertex, 3)
        # Contract root -> child edge at position 1
        # New vertex has arity 3, children 1, 2, 3
        # Result depends on new_decoration

        # Lie.compose(Lie(2).term((1,)), 1, Lie(2).term((1,))) computes the composition
        L2 = Lie(2)
        bracket = L2.term((1,))
        composed = Lie.compose(bracket, 1, bracket)
        # In Lie, id ∘_1 id = id in arity 3, which has basis keys permutations of (1, 2)
        # Lie(3) basis keys are permutations of (1, 2)
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
        B = factory(m + n - 1)
        Bm = factory(m)
        Bn = factory(n)
        tgt = tensor([Bm, Bn])

        x = B(tree)
        dx = x.boundary()

        # LHS: Δ_{i;m,n}(d(x)), recast into tgt
        lhs = tgt.zero()
        for (ak, bk), coeff in B.infinitesimal_cocompose(dx, i, m, n):
            lhs += coeff * Bm.term(ak).tensor(Bn.term(bk))

        # RHS: (d⊗id + (-1)^|a| id⊗d)(Δ_{i;m,n}(x))
        rhs = tgt.zero()
        for (ak, bk), coeff in B.infinitesimal_cocompose(x, i, m, n):
            a_deg = Bm.degree_on_basis(ak)
            for dak, da_coeff in Bm.term(ak).boundary():
                rhs += coeff * da_coeff * Bm.term(dak).tensor(Bn.term(bk))
            sign_a = (-1) ** a_deg
            for dbk, db_coeff in Bn.term(bk).boundary():
                rhs += coeff * sign_a * db_coeff * Bm.term(ak).tensor(Bn.term(dbk))

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
        # (1,2,1) has nontrivial boundary in Surjection(2)
        tree = ((1, 2, 1), 1, 2)
        self._check_coderivation(BS, tree, 1, 1, 2)


class TestCoAssociative:
    """Tests for the CoAssociative cooperad."""

    def test_coassociative_counit(self):
        """Counit evaluates correctly on arity-1 generator."""
        C1 = CoAssociative(1)
        assert CoAssociative.counit(C1((1,))) == 1
        C2 = CoAssociative(2)
        assert CoAssociative.counit(C2((1, 2))) == 0

    def test_coassociative_reduced(self):
        """Reduced kills the arity-1 counit component."""
        C1 = CoAssociative(1)
        elem = 3 * C1.term((1,))
        assert CoAssociative.reduced(elem) == C1.zero()
        C2 = CoAssociative(2)
        elem2 = C2((1, 2))
        assert CoAssociative.reduced(elem2) == elem2

    def test_coassociative_cocompose_arity3(self):
        """Cocomposition at arity 3: dual of Ass.compose."""
        from uconf import Associative

        C3 = CoAssociative(3)
        sigma = C3((1, 2, 3))  # identity permutation in S_3

        # Δ^{1;2,2}((1,2,3)): find (tau, rho) with Ass.compose(tau,1,rho)=(1,2,3)
        delta = CoAssociative.infinitesimal_cocompose(sigma, 1, 2, 2)
        assert delta != tensor([CoAssociative(2), CoAssociative(2)]).zero()

        # Verify: composing back should give original basis element
        for (left_key, right_key), coeff in delta:
            tau = CoAssociative(2).term(left_key)
            rho = CoAssociative(2).term(right_key)
            composed = Associative.compose(tau, 1, rho)
            assert dict(composed) == {(1, 2, 3): 1}

    def test_cobar_coassociative_d_squared_zero_arity3(self):
        """d^2 = 0 for Ω(CoAss)(3)."""
        OmegaCA = CobarConstruction(CoAssociative)
        O3 = OmegaCA(3)
        # Weight-1 tree with (1,2,3) decoration
        tree = ((1, 2, 3), 1, 2, 3)
        elem = O3(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O3.zero(), f"d^2 != 0 for CoAss: {bdry2}"

    def test_cobar_coassociative_d_squared_zero_arity4(self):
        """d^2 = 0 for Ω(CoAss)(4)."""
        OmegaCA = CobarConstruction(CoAssociative)
        O4 = OmegaCA(4)
        tree = ((1, 2, 3, 4), 1, 2, 3, 4)
        elem = O4(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O4.zero(), f"d^2 != 0 for CoAss arity 4: {bdry2}"


class TestCoCommutative:
    """Tests for the CoCommutative cooperad."""

    def test_cocommutative_counit(self):
        """Counit evaluates correctly on arity-1 generator."""
        C1 = CoCommutative(1)
        assert CoCommutative.counit(C1(())) == 1
        C2 = CoCommutative(2)
        assert CoCommutative.counit(C2(())) == 0

    def test_cocommutative_reduced(self):
        """Reduced kills the arity-1 counit component."""
        C1 = CoCommutative(1)
        elem = 5 * C1.term(())
        assert CoCommutative.reduced(elem) == C1.zero()
        C2 = CoCommutative(2)
        elem2 = C2(())
        assert CoCommutative.reduced(elem2) == elem2

    def test_cocommutative_cocompose(self):
        """Cocomposition maps e_{m+n-1} to e_m ⊗ e_n."""
        C3 = CoCommutative(3)
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
        O3 = OmegaCC(3)
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
        OC = OmegaCom(arity)
        tree = ((),) + tuple(range(1, arity + 1))
        elem = OC(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == OC.zero(), f"d^2 != 0 for CoCom arity {arity}: {bdry2}"

    def test_cobar_surjection_square_zero_arity4_weight2(self):
        """d^2 = 0 for a weight-2 tree in Ω(S*) where the sign fix matters."""
        OmegaS = CobarConstruction(SurjectionDual)
        O4 = OmegaS(4)
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
        O5 = OmegaS(5)
        # root arity 3, inner arity 3 as first child, leaves 4 and 5 as other children
        tree = ((1, 2, 3, 1), ((1, 2, 3), 1, 2, 3), 4, 5)
        elem = O5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O5.zero(), f"d^2 != 0 in arity 5 weight 2: {bdry2}"

    def test_cobar_sign_arity5_weight2_inner_dec(self):
        """d^2 = 0 for a tree where the inner node has high-degree decoration."""
        OmegaS = CobarConstruction(SurjectionDual)
        O5 = OmegaS(5)
        tree = ((1, 2, 3, 1), ((1, 2, 1, 3), 1, 2, 3), 4, 5)
        elem = O5(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O5.zero(), f"d^2 != 0 in arity 5 weight 2: {bdry2}"

    def test_cobar_sign_arity6_weight3(self):
        """d^2 = 0 for a weight-3 tree where the sign fix matters."""
        OmegaS = CobarConstruction(SurjectionDual)
        O6 = OmegaS(6)
        tree = ((1, 2, 3, 1, 4), ((1, 2, 1, 2), 1, 2), ((1, 2, 1), 3, 4), 5, 6)
        elem = O6(tree)
        bdry2 = elem.boundary().boundary()
        assert bdry2 == O6.zero(), f"d^2 != 0 in arity 6 weight 3: {bdry2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
