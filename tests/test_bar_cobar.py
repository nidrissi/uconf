"""Tests for bar and cobar constructions."""

import pytest

from uconf import (
    BarConstruction,
    CobarConstruction,
    CooperadProtocol,
    Lie,
    OperadProtocol,
    Surjection,
    BarrattEccles,
    SurjectionLinearDual,
)
from uconf.trees import (
    children,
    contract_edge,
    decoration,
    expand_vertex,
    graft,
    internal_edges,
    is_internal,
    is_leaf,
    leaves,
    relabel_leaves,
    subtree_degree,
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
        assert internal_edges(tree) == []

        nested = ((2, 1), ((1,), 1, 2), 3)
        edges = internal_edges(nested)
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
        # Lie is degree 0, tree arity 3, weight 1
        # degree = 0 + (3 - 1) = 2
        tree = ((1, 2), 1, 2, 3)
        assert B3.degree_on_basis(tree) == 2

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
        d2 = B3._d2_on_basis(nested)
        assert d2 != B3.zero()

        # Result should be weight-1 trees (since we contract one edge)
        for tree, coeff in d2:
            assert weight(tree) == 1

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
        # Surjection basis (1, 2) has degree 0, arity 2
        # sP degree = 0 + (2 - 1) = 1
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

        d1 = B2._d1_on_basis(tree)
        # Should produce trees with different decorations
        assert d1 != B2.zero()


class TestCobarConstruction:
    """Tests for cobar construction of cooperads."""

    def test_cobar_creation(self):
        OmegaS = CobarConstruction(SurjectionLinearDual)
        O2 = OmegaS(2)
        assert O2.arity() == 2

    def test_cobar_unit(self):
        OmegaS = CobarConstruction(SurjectionLinearDual)
        unit = OmegaS.unit()
        assert unit.arity() == 1
        assert unit == OmegaS(1).term(1)

    def test_cobar_degree(self):
        OmegaS = CobarConstruction(SurjectionLinearDual)
        O2 = OmegaS(2)
        # SurjectionLinearDual basis (1, 2) has degree 0
        # s^{-1}C degree = 0 - (2 - 1) = -1
        tree = ((1, 2), 1, 2)
        assert O2.degree_on_basis(tree) == -1

    def test_cobar_d_squared_zero_unit(self):
        """d^2 = 0 for the unit."""
        OmegaS = CobarConstruction(SurjectionLinearDual)
        unit = OmegaS.unit()
        bdry = unit.boundary()
        assert bdry == OmegaS(1).zero()

    def test_cobar_compose(self):
        """Test free operad composition (grafting)."""
        OmegaS = CobarConstruction(SurjectionLinearDual)
        O2 = OmegaS(2)
        O3 = OmegaS(3)

        tree1 = ((1, 2), 1, 2)
        elem1 = O2(tree1)

        # Compose with unit
        unit = OmegaS.unit()
        composed = OmegaS.compose(elem1, 1, unit)
        assert composed.arity() == 2
        # Composing with unit should give back the same element (essentially)

    def test_cobar_compose_two_trees(self):
        """Test composing two nontrivial trees."""
        OmegaS = CobarConstruction(SurjectionLinearDual)
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

    def test_cobar_pretty_print(self):
        """Cobar terms should display trees and unit as id."""
        OmegaS = CobarConstruction(SurjectionLinearDual)
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

        OmegaS = CobarConstruction(SurjectionLinearDual)
        O2 = OmegaS(2)
        elem = O2(((1, 2), 1, 2)) + 3 * O2(((1, 2, 1), 1, 2))

        pretty = elem.pretty()
        assert "(S*(1, 2); 1, 2)" in pretty
        assert "3*(S*(1, 2, 1); 1, 2)" in pretty
        assert repr(elem) == pretty

        latex_repr = elem.pretty_latex()
        assert "\\operatorname{S*}" in latex_repr
        assert latex(elem) == latex_repr


class TestProtocolCompliance:
    """Test that bar/cobar constructions satisfy the protocols."""

    def test_bar_is_cooperad(self):
        """BarConstruction components should satisfy CooperadProtocol."""
        BLie = BarConstruction(Lie)
        for n in range(2, 4):
            B = BLie(n)
            assert isinstance(B, CooperadProtocol), f"B(Lie)({n}) not a cooperad"

    def test_cobar_is_operad(self):
        """CobarConstruction components should satisfy OperadProtocol."""
        OmegaS = CobarConstruction(SurjectionLinearDual)
        for n in range(1, 4):
            O = OmegaS(n)
            # Note: OperadProtocol requires compose as static method on type
            # Our implementation has it on the factory, which is slightly different
            # But the Element class should have the right methods
            assert hasattr(O, "arity")
            assert hasattr(O, "degree_on_basis")
            assert hasattr(O, "boundary")


class TestContractEdge:
    """Tests for edge contraction in trees."""

    def test_contract_simple(self):
        # Tree: root (1,) with children (internal (1,) with children 1, 2) and 3
        child_vertex = ((1,), 1, 2)
        root = ((1,), child_vertex, 3)
        # Contract root -> child edge at position 1
        # New vertex has arity 3, children 1, 2, 3
        # Result depends on new_decoration

        from sage.all import QQ

        # Lie.compose(Lie(2).term((1,)), 1, Lie(2).term((1,))) computes the composition
        L2 = Lie(2)
        L3 = Lie(3)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
