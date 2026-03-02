"""Rooted tree utilities for bar-cobar constructions.

Trees are encoded as nested tuples:
- A leaf is an ``int`` in ``{1, ..., n}`` (the arity).
- An internal vertex is ``(decoration, child_1, ..., child_k)`` where
  ``decoration`` is an operad/cooperad basis key (a tuple) and each
  ``child_j`` is a leaf or subtree.  For connected operads, ``k >= 2``.

Example (arity 3, weight 2):
    ``((1, 2), ((1,), 1, 2), 3)``
represents a tree with root decorated by the Lie basis key ``(1, 2)``
(a binary bracket), whose first child is another internal vertex
decorated by ``(1,)`` with leaves 1 and 2, and whose second child is leaf 3.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal


def is_leaf(node) -> bool:
    """Return True if ``node`` is a leaf (an integer)."""
    return isinstance(node, int)


def is_internal(node) -> bool:
    """Return True if ``node`` is an internal vertex (a tuple)."""
    return isinstance(node, tuple) and len(node) >= 1 and isinstance(node[0], tuple)


def decoration(vertex: tuple) -> tuple:
    """Return the decoration (operad basis key) of an internal vertex."""
    return vertex[0]


def children(vertex: tuple) -> tuple:
    """Return the children of an internal vertex as a tuple."""
    return vertex[1:]


def vertex_arity(vertex: tuple) -> int:
    """Return the arity (number of children) of an internal vertex."""
    return len(vertex) - 1


def leaves(tree) -> set[int]:
    """Return the set of all leaf labels in the tree."""
    if is_leaf(tree):
        return {tree}
    result: set[int] = set()
    for child in children(tree):
        result |= leaves(child)
    return result


def weight(tree) -> int:
    """Return the weight (number of internal vertices) of the tree."""
    if is_leaf(tree):
        return 0
    return 1 + sum(weight(child) for child in children(tree))


def tree_arity(tree) -> int:
    """Return the arity of the tree (number of leaves)."""
    return len(leaves(tree))


def vertices_dfs(tree) -> list[tuple]:
    """Return all internal vertices in depth-first (pre-order) traversal.

    Each entry is the internal vertex tuple itself.
    """
    if is_leaf(tree):
        return []
    result = [tree]
    for child in children(tree):
        result.extend(vertices_dfs(child))
    return result


def subtree_degree(tree, operad_cls, base_ring) -> int:
    """Compute the total s*P-bar degree of all decorations in a subtree.

    For the bar construction, each vertex contributes
    ``deg_P(decoration) + (vertex_arity - 1)``.
    """
    if is_leaf(tree):
        return 0
    parent = operad_cls(vertex_arity(tree), base_ring)
    dec = decoration(tree)
    vertex_deg = parent.degree_on_basis(dec) + (vertex_arity(tree) - 1)
    child_deg = sum(subtree_degree(c, operad_cls, base_ring) for c in children(tree))
    return vertex_deg + child_deg


def subtree_degree_cobar(tree, cooperad_cls, base_ring) -> int:
    """Compute the total s^{-1}*C-bar degree for the cobar construction.

    Each vertex contributes ``deg_C(decoration) - (vertex_arity - 1)``.
    """
    if is_leaf(tree):
        return 0
    parent = cooperad_cls(vertex_arity(tree), base_ring)
    dec = decoration(tree)
    vertex_deg = parent.degree_on_basis(dec) - (vertex_arity(tree) - 1)
    child_deg = sum(
        subtree_degree_cobar(c, cooperad_cls, base_ring) for c in children(tree)
    )
    return vertex_deg + child_deg


def internal_edges(tree) -> list[tuple[tuple, int, tuple]]:
    """Enumerate all internal edges (parent-child pairs between internal vertices).

    Returns a list of ``(parent_vertex, child_position, child_vertex)`` tuples
    where ``child_position`` is 1-indexed (the slot in the operad composition).
    """
    if is_leaf(tree):
        return []
    result = []
    for i, child in enumerate(children(tree), start=1):
        if is_internal(child):
            result.append((tree, i, child))
            result.extend(internal_edges(child))
    return result


def contract_edge(
    tree: tuple, parent_vertex: tuple, child_pos: int, new_decoration: tuple
) -> tuple:
    """Contract one internal edge, merging parent and child vertices.

    The ``child_pos``-th child of ``parent_vertex`` is an internal vertex.
    Replace both with a single vertex having ``new_decoration`` and the combined
    children (children of parent before child_pos, children of child, children
    of parent after child_pos).

    This function recursively finds and contracts the specified edge in the tree.
    """
    if is_leaf(tree):
        return tree

    if tree is parent_vertex:
        # This is the parent vertex to contract
        child_vertex = children(tree)[child_pos - 1]
        assert is_internal(child_vertex), "Child must be internal for contraction"

        # Build new children list
        new_children = []
        for i, c in enumerate(children(tree), start=1):
            if i < child_pos:
                new_children.append(c)
            elif i == child_pos:
                # Insert children of the child vertex here
                new_children.extend(children(child_vertex))
            else:
                new_children.append(c)

        return (new_decoration,) + tuple(new_children)

    # Recurse into children
    new_children = []
    for child in children(tree):
        new_children.append(
            contract_edge(child, parent_vertex, child_pos, new_decoration)
        )
    return (decoration(tree),) + tuple(new_children)


def graft(tree_top, i: int, tree_bot, relabel_bot: dict | None = None) -> tuple:
    """Graft ``tree_bot`` onto leaf ``i`` of ``tree_top``.

    If ``relabel_bot`` is provided, it maps each leaf of ``tree_bot`` to its
    new label in the grafted tree.  If not provided, leaves of ``tree_bot``
    are relabeled to ``{i, i+1, ..., i+n_bot-1}`` and leaves > i in ``tree_top``
    are shifted by ``n_bot - 1``.

    Returns the grafted tree.
    """
    n_bot = tree_arity(tree_bot)

    if relabel_bot is None:
        # Default relabeling: tree_bot leaves become i, i+1, ..., i+n_bot-1
        bot_leaves_sorted = sorted(leaves(tree_bot))
        relabel_bot = {old: i + idx for idx, old in enumerate(bot_leaves_sorted)}

    def relabel_top(leaf: int) -> int:
        if leaf < i:
            return leaf
        elif leaf > i:
            return leaf + n_bot - 1
        else:
            raise ValueError("Leaf i should be replaced, not relabeled")

    def graft_rec(node):
        if is_leaf(node):
            if node == i:
                # Replace with tree_bot (relabeled)
                return relabel_leaves(tree_bot, relabel_bot)
            else:
                return relabel_top(node)
        # Internal vertex: recurse
        new_children = tuple(graft_rec(c) for c in children(node))
        return (decoration(node),) + new_children

    return graft_rec(tree_top)


def relabel_leaves(tree, mapping: dict):
    """Apply a leaf relabeling according to ``mapping``."""
    if is_leaf(tree):
        return mapping.get(tree, tree)
    new_children = tuple(relabel_leaves(c, mapping) for c in children(tree))
    return (decoration(tree),) + new_children


def split_at_vertex(
    tree: tuple, target_vertex: tuple
) -> tuple[tuple | int, int, tuple] | None:
    """Split the tree at a given internal vertex.

    Returns ``(tree_top, position, tree_bot)`` where:
    - ``tree_bot`` is the subtree rooted at ``target_vertex``
    - ``tree_top`` is the tree with ``target_vertex`` replaced by a single leaf
    - ``position`` is the position of that leaf in the grafting sense

    Returns None if ``target_vertex`` is the root (can't split at root).
    """
    if tree is target_vertex:
        return None  # Can't split at root

    def find_and_replace(node, replacement_leaf: int):
        """Replace target_vertex with replacement_leaf, return new node."""
        if is_leaf(node):
            return node
        if node is target_vertex:
            return replacement_leaf
        new_children = tuple(
            find_and_replace(c, replacement_leaf) for c in children(node)
        )
        return (decoration(node),) + new_children

    # Determine what leaf label to use as placeholder
    # Use min leaf of target_vertex's subtree
    bot_leaves = leaves(target_vertex)
    placeholder = min(bot_leaves)

    tree_top = find_and_replace(tree, placeholder)
    tree_bot = target_vertex

    return (tree_top, placeholder, tree_bot)


def validate_tree(tree, arity: int, operad_cls, base_ring) -> tuple | Literal[1] | None:
    """Validate a tree for use in bar/cobar constructions.

    Checks:
    - Leaves are exactly {1, ..., arity}
    - All internal vertices have arity >= 2 (connected assumption)
    - All decorations are valid for the given operad/cooperad

    Returns the validated tree (possibly normalized) or None if invalid.
    """
    # Check that tree_arity matches
    if is_leaf(tree):
        # A single leaf is only valid for arity 1
        if arity == 1 and tree == 1:
            return tree
        return None

    # Check leaves
    tree_leaves = leaves(tree)
    if tree_leaves != set(range(1, arity + 1)):
        return None

    # Validate all internal vertices
    def validate_vertex(node):
        if is_leaf(node):
            return node

        v_arity = vertex_arity(node)
        if v_arity < 2:
            return None  # Connected assumption

        dec = decoration(node)
        parent = operad_cls(v_arity, base_ring)

        if hasattr(parent, "_validate_basis_key"):
            clean_dec = parent._validate_basis_key(dec)
            if clean_dec is None:
                return None
        else:
            clean_dec = dec

        # Validate children
        new_children = []
        for child in children(node):
            validated = validate_vertex(child)
            if validated is None:
                return None
            new_children.append(validated)

        return (clean_dec,) + tuple(new_children)

    return validate_vertex(tree)


def enumerate_trees_by_weight(
    arity: int,
    weight_bound: int,
    operad_cls,
    base_ring,
) -> Iterator[tuple]:
    """Enumerate all valid trees in a given arity up to weight bound.

    Yields tree basis keys with leaves {1, ..., arity} and weight in [1, weight_bound].
    Uses the operad's ``basis_it`` method if available.
    """
    if arity < 2:
        return  # No nontrivial trees for arity < 2 (connected assumption)

    # Weight 1 trees: single internal vertex with arity leaves
    if weight_bound >= 1:
        parent = operad_cls(arity, base_ring)
        if hasattr(parent, "basis_it"):
            # Iterate over all degrees
            for deg in range(20):  # reasonable upper bound
                try:
                    for elem in parent.basis_it(deg):
                        dec = next(iter(elem.support()))
                        tree = (dec,) + tuple(range(1, arity + 1))
                        yield tree
                except (StopIteration, ValueError):
                    break
        else:
            # Fallback: just use degree 0
            try:
                for dec in parent.basis():
                    tree = (dec,) + tuple(range(1, arity + 1))
                    yield tree
            except (AttributeError, NotImplementedError):
                pass

    # Weight >= 2: recursively combine smaller trees
    if weight_bound >= 2:
        # For each way to partition arity into vertex_arity children
        for v_arity in range(2, arity):  # vertex_arity of root
            # For each decoration of the root
            root_parent = operad_cls(v_arity, base_ring)
            root_decorations = []
            if hasattr(root_parent, "basis_it"):
                for deg in range(20):
                    try:
                        for elem in root_parent.basis_it(deg):
                            root_decorations.append(next(iter(elem.support())))
                    except (StopIteration, ValueError):
                        break
            else:
                try:
                    root_decorations = list(root_parent.basis())
                except (AttributeError, NotImplementedError):
                    pass

            for root_dec in root_decorations:
                # For each partition of {1,...,arity} into v_arity nonempty subsets
                # and for each assignment of subtrees to these subsets
                yield from _enumerate_with_root(
                    arity, v_arity, root_dec, weight_bound - 1, operad_cls, base_ring
                )


def _enumerate_with_root(
    arity: int,
    v_arity: int,
    root_dec: tuple,
    remaining_weight: int,
    operad_cls,
    base_ring,
) -> Iterator[tuple]:
    """Helper to enumerate trees with a fixed root vertex."""
    from itertools import combinations

    # Enumerate partitions of {1,...,arity} into v_arity nonempty ordered parts
    # For simplicity, we use ordered partitions where part[i] contains min(part[i])
    # in increasing order across parts

    if v_arity == 2:
        # Partition into two parts
        leaves_set = set(range(1, arity + 1))
        for size1 in range(1, arity):
            size2 = arity - size1
            for part1 in combinations(range(1, arity + 1), size1):
                part1_set = set(part1)
                part2 = tuple(sorted(leaves_set - part1_set))
                # Child 1 gets part1, child 2 gets part2
                # Enumerate subtrees for each part
                for child1 in _subtrees_for_leaves(
                    part1, remaining_weight, operad_cls, base_ring
                ):
                    for child2 in _subtrees_for_leaves(
                        part2, remaining_weight - weight(child1), operad_cls, base_ring
                    ):
                        if weight(child1) + weight(child2) <= remaining_weight:
                            yield (root_dec, child1, child2)


def _subtrees_for_leaves(
    leaf_set: tuple,
    max_weight: int,
    operad_cls,
    base_ring,
) -> Iterator:
    """Enumerate subtrees with exactly the given leaves (as a tuple)."""
    n = len(leaf_set)
    if n == 0:
        return
    if n == 1:
        yield leaf_set[0]  # Single leaf
        return

    # For n >= 2, we need internal vertices
    if max_weight < 1:
        return

    # Create a mapping from {1,...,n} to leaf_set
    mapping = {i + 1: leaf_set[i] for i in range(n)}

    # Enumerate trees with arity n and relabel
    for tree in enumerate_trees_by_weight(n, max_weight, operad_cls, base_ring):
        yield relabel_leaves(tree, mapping)


def tree_to_string(tree, operad_name: str = "P") -> str:
    """Return a human-readable string representation of a tree."""
    if is_leaf(tree):
        return str(tree)
    dec_str = f"{operad_name}{decoration(tree)}"
    children_str = ", ".join(tree_to_string(c, operad_name) for c in children(tree))
    return f"({dec_str}; {children_str})"


def tree_to_latex(tree, operad_name: str = "P") -> str:
    """Return a LaTeX representation of a decorated rooted tree."""
    if is_leaf(tree):
        return str(tree)

    dec = ",".join(str(i) for i in decoration(tree))
    dec_str = f"\\operatorname{{{operad_name}}}_{{({dec})}}"
    children_str = ", ".join(tree_to_latex(c, operad_name) for c in children(tree))
    return f"\\left({dec_str}; {children_str}\\right)"


def copy_tree_structure(old_tree, new_decorations: list[tuple]) -> tuple:
    """Create a tree with the same structure but new decorations.

    ``new_decorations`` should be in DFS order matching ``vertices_dfs(old_tree)``.
    """
    if is_leaf(old_tree):
        return old_tree

    dec_iter = iter(new_decorations)

    def rebuild(node):
        if is_leaf(node):
            return node
        new_dec = next(dec_iter)
        new_children = tuple(rebuild(c) for c in children(node))
        return (new_dec,) + new_children

    return rebuild(old_tree)


def replace_vertex_decoration(
    tree: tuple, target: tuple, new_decoration: tuple
) -> tuple:
    """Replace the decoration of a specific vertex in the tree."""
    if is_leaf(tree):
        return tree

    if tree is target:
        return (new_decoration,) + children(tree)

    new_children = tuple(
        replace_vertex_decoration(c, target, new_decoration) if is_internal(c) else c
        for c in children(tree)
    )
    return (decoration(tree),) + new_children


def expand_vertex(
    tree: tuple,
    target_vertex: tuple,
    child_pos: int,
    left_decoration: tuple,
    right_decoration: tuple,
    left_arity: int,
    right_arity: int,
) -> tuple:
    """Expand a vertex into two vertices connected by an edge.

    For the cobar differential d_2: replaces ``target_vertex`` with two vertices.
    The original children are reassigned: children 1..child_pos-1+left_arity-1
    go to left_decoration, the rest to right_decoration, with right_decoration
    becoming child_pos of left_decoration.

    Actually, this is more nuanced. The infinitesimal cocomposition
    Delta^{l; a, b}(c) splits an arity-k vertex (k = a + b - 1) into:
    - Top vertex with arity a, decoration c_L
    - Bottom vertex with arity b, decoration c_R
    - They connect at position l of the top vertex
    - Original children 1..l-1 go to top positions 1..l-1
    - Original children l..l+b-1 go to bottom positions 1..b
    - Original children l+b..k go to top positions l+1..a
    """
    if is_leaf(tree):
        return tree

    if tree is target_vertex:
        orig_children = children(tree)
        k = len(orig_children)
        a, b = left_arity, right_arity
        l = child_pos
        assert k == a + b - 1, f"Arity mismatch: {k} != {a} + {b} - 1"

        # Build bottom vertex children: original children l, l+1, ..., l+b-1
        bottom_children = orig_children[l - 1 : l - 1 + b]
        bottom_vertex = (right_decoration,) + bottom_children

        # Build top vertex children
        top_children = (
            orig_children[: l - 1]  # positions 1..l-1
            + (bottom_vertex,)  # position l
            + orig_children[l - 1 + b :]  # remaining
        )
        return (left_decoration,) + top_children

    # Recurse
    new_children = tuple(
        (
            expand_vertex(
                c,
                target_vertex,
                child_pos,
                left_decoration,
                right_decoration,
                left_arity,
                right_arity,
            )
            if is_internal(c)
            else c
        )
        for c in children(tree)
    )
    return (decoration(tree),) + new_children
