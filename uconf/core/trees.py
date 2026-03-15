"""Rooted tree utilities for bar/cobar constructions.

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
    """Compute the total shifted bar degree of a subtree.

    In the conventions used by ``BarConstruction``, each internal vertex
    contributes ``deg_P(decoration) + 1``.
    """
    if is_leaf(tree):
        return 0
    parent = operad_cls(vertex_arity(tree), base_ring)
    dec = decoration(tree)
    vertex_deg = parent.degree_on_basis(dec) + 1
    child_deg = sum(subtree_degree(c, operad_cls, base_ring) for c in children(tree))
    return vertex_deg + child_deg


def subtree_degree_cobar(tree, cooperad_cls, base_ring) -> int:
    """Compute the total shifted cobar degree of a subtree.

    In the conventions used by ``CobarConstruction``, each internal vertex
    contributes ``deg_C(decoration) - 1``.
    """
    if is_leaf(tree):
        return 0
    parent = cooperad_cls(vertex_arity(tree), base_ring)
    dec = decoration(tree)
    vertex_deg = parent.degree_on_basis(dec) - 1
    child_deg = sum(
        subtree_degree_cobar(c, cooperad_cls, base_ring) for c in children(tree)
    )
    return vertex_deg + child_deg


def internal_edges_dfs(tree) -> list[tuple[tuple, int, tuple]]:
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
            result.extend(internal_edges_dfs(child))
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

    Returns a validated tree with cleaned decorations, or ``None`` if invalid.
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
            # TODO #21 This "reasonable upper bound" is hacky. We can use the connectivity assumption to get a better bound.
            for deg in range(20):
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
        for size in range(1, arity):
            for part1 in combinations(range(1, arity + 1), size):
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


def enumerate_planar_trees_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[tuple]:
    """Enumerate planar-decorated trees in ``B(P)(arity)`` of bar degree ``target_degree``.

    A tree is *planar* (in the quasi-planar cooperad sense) when:

    1. All vertex decorations are planar elements of the base operad.
    2. Children at each vertex have consecutive leaf-label ranges, so that
       ``planarize(T) = T ⊗ id`` (the global permutation is the identity).

    Condition 2 means that the tree is a *standard planar tree*: leaf labels
    run ``1, …, n`` strictly left-to-right, and each child subtree occupies a
    contiguous block.  All such trees are automatically in shuffle form.

    Uses the operad's ``planar_basis_it`` at the exact required degree for
    each vertex, so this is efficient even for operads with many basis elements
    per degree (e.g. Barratt–Eccles).

    Requires the operad to implement ``planar_basis_it``.

    The enumeration mirrors the binary-root topologies produced by
    ``enumerate_trees_by_weight``, extended to planar decorations and
    restricted to consecutive child-leaf ranges.

    Args:
        arity: Number of leaves (arity of the bar-construction component).
        weight_bound: Maximum number of internal vertices.  For connected
            operads, callers should pass ``arity - 1`` (the hard upper bound
            from the branching constraint ``sum(m_v - 1) = n - 1``).
        operad_cls: Operad factory; must supply ``planar_basis_it``.
        base_ring: Coefficient ring.
        target_degree: Exact bar degree to enumerate.  May be any integer,
            including zero or negative, when the base operad has elements
            of negative degree (e.g. a shifted operad).
    """
    if arity < 2:
        return

    connectivity = getattr(operad_cls, "connectivity", 0)

    # ------------------------------------------------------------------
    # Weight 1: single internal vertex with arity leaves.
    # Decoration degree = target_degree - 1.
    # ------------------------------------------------------------------
    if weight_bound >= 1:
        dec_degree = target_degree - 1
        min_dec_degree = connectivity * (arity - 1)
        if dec_degree >= min_dec_degree:
            parent = operad_cls(arity, base_ring)
            if hasattr(parent, "planar_basis_it"):
                for elem in parent.planar_basis_it(dec_degree):
                    for dec in elem.support():
                        yield (dec,) + tuple(range(1, arity + 1))

    # ------------------------------------------------------------------
    # Weight >= 2: binary root, one of whose children is an internal
    # subtree.  We restrict to *consecutive* child-leaf ranges so that
    # both children's subtrees remain planar.  The two ranges are
    # {1,...,a} and {a+1,...,n} for a in 1..n-1.
    # ------------------------------------------------------------------
    if weight_bound >= 2 and arity >= 3:
        root_parent = operad_cls(2, base_ring)
        if not hasattr(root_parent, "planar_basis_it"):
            return

        min_root_dec_degree = connectivity  # v_arity = 2

        for a in range(1, arity):
            part1 = tuple(range(1, a + 1))
            part2 = tuple(range(a + 1, arity + 1))
            min_deg1 = _min_subtree_bar_degree(len(part1), connectivity)
            min_deg2 = _min_subtree_bar_degree(len(part2), connectivity)
            min_child_total = min_deg1 + min_deg2
            max_root_dec_degree = target_degree - 1 - min_child_total
            for root_dec_degree in range(min_root_dec_degree, max_root_dec_degree + 1):
                remaining = target_degree - root_dec_degree - 1
                for root_dec_elem in root_parent.planar_basis_it(root_dec_degree):
                    for root_dec in root_dec_elem.support():
                        max_deg1 = remaining - min_deg2
                        for deg1 in range(min_deg1, max_deg1 + 1):
                            deg2 = remaining - deg1
                            if deg2 < min_deg2:
                                continue
                            for c1 in _planar_subtrees_for_leaves(
                                part1,
                                weight_bound - 1,
                                operad_cls,
                                base_ring,
                                deg1,
                            ):
                                for c2 in _planar_subtrees_for_leaves(
                                    part2,
                                    weight_bound - 1 - weight(c1),
                                    operad_cls,
                                    base_ring,
                                    deg2,
                                ):
                                    yield (root_dec, c1, c2)


def _planar_subtrees_for_leaves(
    leaf_set: tuple,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[tuple | int]:
    """Enumerate planar subtrees for a given leaf set and exact bar degree.

    A single leaf always has bar degree 0.
    """
    n = len(leaf_set)
    if n == 0:
        return
    if n == 1:
        if target_degree == 0:
            yield leaf_set[0]
        return

    if max_weight < 1:
        return

    mapping = {i + 1: leaf_set[i] for i in range(n)}
    for tree in enumerate_planar_trees_in_degree(
        n, max_weight, operad_cls, base_ring, target_degree
    ):
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

    Used in the cobar differential ``d_2``. The infinitesimal cocomposition
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


# =============================================================================
# Shuffle tree normalization for symmetric operads/cooperads
# =============================================================================


def min_leaf(tree) -> int:
    """Return the minimum leaf label in a tree or subtree."""
    if is_leaf(tree):
        return tree
    return min(min_leaf(c) for c in children(tree))


def is_shuffle_tree(tree) -> bool:
    """Check if a tree is in shuffle form.

    A shuffle tree has, at each internal vertex, children ordered so that
    the minimum leaf label in each child subtree increases from left to right.

    Example:
        ((), 1, 2) is shuffle (min({1})=1 < min({2})=2)
        ((), 2, 1) is NOT shuffle (min({2})=2 > min({1})=1)
    """
    if is_leaf(tree):
        return True

    kids = children(tree)
    min_leaves = [min_leaf(c) for c in kids]

    # Children must be sorted by min leaf
    for i in range(len(min_leaves) - 1):
        if min_leaves[i] >= min_leaves[i + 1]:
            return False

    # Recursively check all subtrees
    return all(is_shuffle_tree(c) for c in kids if is_internal(c))


def _koszul_sign_of_permutation(perm: list[int], degrees: list[int]) -> int:
    """Compute the Koszul sign of a permutation acting on graded elements.

    Given a permutation perm and degrees [d_1, ..., d_k], compute the sign
    incurred by permuting elements of degrees d_{perm[0]}, d_{perm[1]}, ...
    back to their original order.

    The sign is (-1)^{sum of d_i * d_j for all inversions (i,j) in perm}.
    """
    n = len(perm)
    exponent = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Inversion: perm[i] > perm[j]
            if perm[i] > perm[j]:
                exponent += degrees[perm[i]] * degrees[perm[j]]
    return 1 if exponent % 2 == 0 else -1


def to_shuffle_tree_bar(tree, operad_cls, base_ring):
    """Normalize a tree to shuffle form for the bar construction B(P).

    Returns ``(shuffle_tree, sign)`` where:
    - ``shuffle_tree`` is the tree with children reordered at each vertex
    - ``sign`` is the accumulated Koszul sign and operad-action coefficient

    At each vertex, children are sorted by min leaf. The decoration is
    acted on by the sorting permutation (using the operad's permute method),
    and Koszul signs are computed based on bar-degrees of subtrees.

    For the implemented bar grading, subtree degrees are computed by
    ``subtree_degree`` (sum of ``deg_P(v) + 1`` over internal vertices).
    """
    if is_leaf(tree):
        return tree, 1

    kids = children(tree)
    dec = decoration(tree)
    k = len(kids)

    # Recursively normalize children first
    normalized_kids = []
    child_sign = 1
    for c in kids:
        if is_internal(c):
            norm_c, s = to_shuffle_tree_bar(c, operad_cls, base_ring)
            normalized_kids.append(norm_c)
            child_sign *= s
        else:
            normalized_kids.append(c)

    # Compute min leaf and bar-degree for each normalized child
    min_leaves = [min_leaf(c) for c in normalized_kids]
    bar_degrees = [subtree_degree(c, operad_cls, base_ring) for c in normalized_kids]

    # Sort children by min leaf
    # indexed_children = [(min_leaf, original_index, child, bar_degree)]
    indexed = list(zip(min_leaves, range(k), normalized_kids, bar_degrees))
    indexed.sort(key=lambda x: x[0])

    # Extract the permutation: perm[new_pos] = old_pos
    # We need the inverse: old_to_new[old_pos] = new_pos
    perm = [item[1] for item in indexed]  # old positions in new order

    # Check if already sorted
    if perm == list(range(k)):
        # Already in shuffle order
        new_tree = (dec,) + tuple(normalized_kids)
        return new_tree, child_sign

    # Compute Koszul sign for reordering
    # The bar-degrees are associated with the OLD positions
    koszul_sign = _koszul_sign_of_permutation(perm, bar_degrees)

    # Compute operad action: the sorting permutation σ acts on decoration
    # σ is the permutation that sends position i to perm[i] (one-indexed)
    # In Sage permutation notation: σ = [perm[0]+1, perm[1]+1, ..., perm[k-1]+1]
    sigma_list = [p + 1 for p in perm]

    # Apply operad action to decoration
    operad_parent = operad_cls(k, base_ring)
    dec_elem = operad_parent.term(dec)

    # The permute method applies σ to the element
    # But we need σ^{-1} because we're reordering children to shuffle form
    # When we sort children, we're effectively applying σ^{-1} to leaf labels
    # So decoration should be acted on by σ^{-1}

    # Compute σ^{-1}: if σ(i) = sigma_list[i-1], then σ^{-1}(j) = position where j appears
    sigma_inv = [0] * k
    for new_pos, old_pos_plus_one in enumerate(sigma_list):
        sigma_inv[old_pos_plus_one - 1] = new_pos + 1
    # Actually, let me reconsider...
    # perm[new_pos] = old_pos means: child at new position came from old position
    # So σ takes new_pos -> old_pos, meaning σ^{-1} takes old_pos -> new_pos
    # The permutation acting on decoration should be the same as acting on children
    # If we reorder children by σ (sorting), decoration gets acted on by σ

    # Let's use the inverse: σ^{-1}[old_pos] = new_pos
    # sigma_inv already computed above

    permuted_dec_elem = dec_elem.permute(sigma_inv)

    # Extract the new decoration (should be a single term for most operads)
    operad_sign = 1
    new_dec = dec
    for new_dec_key, coeff in permuted_dec_elem:
        new_dec = new_dec_key
        operad_sign = coeff
        break  # Assume single term result

    # Build sorted tree
    sorted_kids = tuple(item[2] for item in indexed)
    new_tree = (new_dec,) + sorted_kids

    total_sign = child_sign * koszul_sign * operad_sign
    return new_tree, total_sign


def _operad_basis_keys_in_degree(operad_parent, degree: int) -> Iterator:
    """Yield all basis keys of *operad_parent* in the given degree.

    Handles three calling conventions:

    - ``basis_it(degree)`` — degree-aware (e.g. ``Surjection``, ``BarrattEccles``).
    - Fallback via Sage's ``basis()`` family, filtered by ``degree_on_basis``.
    """
    basis_it = getattr(operad_parent, "basis_it", None)
    if basis_it is not None:
        for elem in basis_it(degree):
            yield from elem.support()
        return
    for key in operad_parent.basis():
        if operad_parent.degree_on_basis(key) == degree:
            yield key


def _min_subtree_bar_degree(part_size: int, connectivity: int) -> int:
    """Minimum bar degree of an internal subtree with ``part_size`` leaves.

    For a tree with weight ≥ 1 and ``part_size`` leaves decorated by an operad
    with given ``connectivity``, the minimum bar degree is
    ``1 + connectivity * (part_size - 1)`` (achieved by a single-vertex corolla).
    Returns 0 for leaves (``part_size == 1``).
    """
    if part_size == 1:
        return 0
    return 1 + connectivity * (part_size - 1)


def _shuffle_partitions(sorted_leaves: tuple, k: int) -> Iterator[list]:
    """Yield all partitions of *sorted_leaves* into *k* non-empty parts sorted by min.

    Elements are processed in ascending order; each new element either joins an
    already-opened part or opens the next new part.  Because elements arrive
    sorted and parts are opened in sequence, ``min(parts[i]) < min(parts[i+1])``
    for all *i*.

    Each yielded value is a list of *k* sorted tuples of leaf labels.
    """
    n = len(sorted_leaves)
    if k <= 0 or k > n:
        return

    parts: list[list[int]] = [[] for _ in range(k)]

    def _backtrack(idx: int, num_open: int) -> Iterator[list]:
        if idx == n:
            if num_open == k:
                yield [tuple(p) for p in parts]
            return
        elem = sorted_leaves[idx]
        # Place elem into an existing open part.
        for i in range(num_open):
            parts[i].append(elem)
            yield from _backtrack(idx + 1, num_open)
            parts[i].pop()
        # Open a new part with this element as its minimum.
        if num_open < k:
            parts[num_open].append(elem)
            yield from _backtrack(idx + 1, num_open + 1)
            parts[num_open].pop()

    parts[0].append(sorted_leaves[0])
    yield from _backtrack(1, 1)


def _shuffle_subtrees_iter(
    leaf_set: tuple,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator:
    """Enumerate all shuffle trees/leaves for a given leaf set and bar degree.

    A *shuffle tree* has children at every internal vertex sorted by their
    minimum leaf label.  Supports any leaf set (not just ``{1, ..., n}``).
    Handles operads with negative-degree elements (e.g. shifted operads).
    """
    n = len(leaf_set)
    if n == 0:
        return
    if n == 1:
        if target_degree == 0:
            yield leaf_set[0]
        return
    if max_weight < 1:
        return

    connectivity = getattr(operad_cls, "connectivity", 0)
    sorted_ls = tuple(sorted(leaf_set))

    # Root has v_arity children (2 ≤ v_arity ≤ n).
    for v_arity in range(2, n + 1):
        root_parent = operad_cls(v_arity, base_ring)
        # Minimum decoration degree for a vertex of this arity.
        min_root_dec_deg = connectivity * (v_arity - 1)
        # The root vertex contributes (root_dec_deg + 1) to the bar degree.
        if v_arity == n:
            # Corolla: all children are individual leaves so child_total = 0.
            root_dec_deg = target_degree - 1
            if root_dec_deg >= min_root_dec_deg:
                for root_dec in _operad_basis_keys_in_degree(root_parent, root_dec_deg):
                    yield (root_dec,) + sorted_ls
        else:
            # Partition sorted_ls into v_arity non-empty shuffle parts.
            for parts in _shuffle_partitions(sorted_ls, v_arity):
                # Minimum child total: sum of minimum bar degrees of internal parts.
                min_child_total = sum(
                    _min_subtree_bar_degree(len(p), connectivity)
                    for p in parts
                    if len(p) >= 2
                )
                # root_dec_deg ranges from min_root_dec_deg up to the value where
                # child_total = target_degree - root_dec_deg - 1 is still achievable.
                max_root_dec_deg = target_degree - 1 - min_child_total
                for root_dec_deg in range(min_root_dec_deg, max_root_dec_deg + 1):
                    child_total = target_degree - root_dec_deg - 1
                    for root_dec in _operad_basis_keys_in_degree(root_parent, root_dec_deg):
                        yield from _shuffle_children_iter(
                            parts,
                            max_weight - 1,
                            operad_cls,
                            base_ring,
                            child_total,
                            root_dec,
                        )


def _shuffle_children_iter(
    parts: list,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    total_deg: int,
    root_dec: tuple,
) -> Iterator[tuple]:
    """Yield complete decorated trees ``(root_dec, t_1, ..., t_k)`` where each
    *t_i* covers *parts[i]* and the bar-degrees of *t_1, ..., t_k* sum to *total_deg*.
    """

    connectivity = getattr(operad_cls, "connectivity", 0)

    # Precompute suffix minimum degrees: min_from[idx] is the minimum total
    # bar-degree contribution from parts[idx:].
    min_from = [0] * (len(parts) + 1)
    for i in range(len(parts) - 1, -1, -1):
        min_from[i] = _min_subtree_bar_degree(len(parts[i]), connectivity) + min_from[i + 1]

    # Build sub-trees for all children incrementally.
    def _children_combinations(idx: int, remaining: int) -> Iterator[list]:
        """Yield lists of sub-trees for parts[idx:] with degrees summing to *remaining*."""
        if idx == len(parts):
            if remaining == 0:
                yield []
            return
        # Early termination: check if remaining can be achieved by parts[idx:].
        if remaining < min_from[idx]:
            return
        part = parts[idx]
        n_part = len(part)
        if n_part == 1:
            # Leaf — contributes 0 to bar degree.
            first = part[0]
            for rest in _children_combinations(idx + 1, remaining):
                yield [first] + rest
        else:
            # Internal subtree — bar degree ≥ min_deg_this.
            min_deg_this = _min_subtree_bar_degree(n_part, connectivity)
            max_d = remaining - min_from[idx + 1]
            if max_d < min_deg_this:
                return
            for d_first in range(min_deg_this, max_d + 1):
                first_trees = list(
                    _shuffle_subtrees_iter(
                        part, max_weight, operad_cls, base_ring, d_first
                    )
                )
                if not first_trees:
                    continue
                for rest in _children_combinations(idx + 1, remaining - d_first):
                    for ft in first_trees:
                        yield [ft] + rest

    for children in _children_combinations(0, total_deg):
        yield (root_dec,) + tuple(children)


def enumerate_shuffle_trees_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[tuple]:
    """Enumerate all shuffle-tree basis elements of ``B(P)(arity)`` in bar degree *target_degree*.

    A *shuffle tree* is a rooted tree with leaves ``{1, ..., arity}`` in which
    children at every vertex are sorted by their minimum leaf label.  Together
    with one decoration per vertex drawn from any basis of ``P(k)``, shuffle
    trees form a **vector-space** basis for the bar construction of any connected
    symmetric operad — not just quasi-planar ones.

    Unlike :func:`enumerate_planar_trees_in_degree`, this function does **not**
    require the base operad to implement ``planarize`` or ``planar_basis_it``.
    It relies only on ``operad_cls(k, base_ring)`` having a ``basis_it``
    method (degree-aware or not) or Sage's ``basis()`` family.

    Args:
        arity: Number of leaves (arity of the bar-construction component).
        weight_bound: Maximum number of internal vertices.  Pass ``arity - 1``
            for connected operads (the hard upper bound from
            ``sum(m_v - 1) = n - 1``).
        operad_cls: Operad factory.
        base_ring: Coefficient ring.
        target_degree: Exact bar degree to enumerate.  May be any integer,
            including zero or negative, when the base operad has elements
            of negative degree (e.g. a shifted operad).

    Yields:
        Decorated shuffle trees as nested tuples valid as basis keys of
        ``BarConstruction(operad_cls)(arity)``.
    """
    if arity < 2 or weight_bound < 1:
        return
    yield from _shuffle_subtrees_iter(
        tuple(range(1, arity + 1)), weight_bound, operad_cls, base_ring, target_degree
    )


def to_shuffle_tree_cobar(tree, cooperad_cls, base_ring):
    """Normalize a tree to shuffle form for the cobar construction Ω(C).

    Returns ``(shuffle_tree, sign)`` where:
    - ``shuffle_tree`` is the tree with children reordered at each vertex
    - ``sign`` is the accumulated Koszul sign and cooperad-action coefficient

    For the implemented cobar grading, subtree degrees are computed by
    ``subtree_degree_cobar`` (sum of ``deg_C(v) - 1`` over internal vertices).
    """
    if is_leaf(tree):
        return tree, 1

    kids = children(tree)
    dec = decoration(tree)
    k = len(kids)

    # Recursively normalize children first
    normalized_kids = []
    child_sign = 1
    for c in kids:
        if is_internal(c):
            norm_c, s = to_shuffle_tree_cobar(c, cooperad_cls, base_ring)
            normalized_kids.append(norm_c)
            child_sign *= s
        else:
            normalized_kids.append(c)

    # Compute min leaf and cobar-degree for each normalized child
    min_leaves = [min_leaf(c) for c in normalized_kids]
    cobar_degrees = [
        subtree_degree_cobar(c, cooperad_cls, base_ring) for c in normalized_kids
    ]

    # Sort children by min leaf
    indexed = list(zip(min_leaves, range(k), normalized_kids, cobar_degrees))
    indexed.sort(key=lambda x: x[0])

    perm = [item[1] for item in indexed]

    if perm == list(range(k)):
        new_tree = (dec,) + tuple(normalized_kids)
        return new_tree, child_sign

    koszul_sign = _koszul_sign_of_permutation(perm, cobar_degrees)

    sigma_list = [p + 1 for p in perm]
    sigma_inv = [0] * k
    for new_pos, old_pos_plus_one in enumerate(sigma_list):
        sigma_inv[old_pos_plus_one - 1] = new_pos + 1

    cooperad_parent = cooperad_cls(k, base_ring)
    dec_elem = cooperad_parent.term(dec)
    permuted_dec_elem = dec_elem.permute(sigma_inv)

    operad_sign = 1
    new_dec = dec
    for new_dec_key, coeff in permuted_dec_elem:
        new_dec = new_dec_key
        operad_sign = coeff
        break

    sorted_kids = tuple(item[2] for item in indexed)
    new_tree = (new_dec,) + sorted_kids

    total_sign = child_sign * koszul_sign * operad_sign
    return new_tree, total_sign
