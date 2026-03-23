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

from html import escape
from typing import Any, Callable, Iterator, Literal


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
    child_deg = sum(subtree_degree_cobar(c, cooperad_cls, base_ring) for c in children(tree))
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
        new_children.append(contract_edge(child, parent_vertex, child_pos, new_decoration))
    return (decoration(tree),) + tuple(new_children)


def graft(tree_top, i: int, tree_bot, relabel_bot: dict | None = None) -> tuple | int:
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


def split_at_vertex(tree: tuple, target_vertex: tuple) -> tuple[tuple | int, int, tuple] | None:
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
        new_children = tuple(find_and_replace(c, replacement_leaf) for c in children(node))
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


def tree_to_string(
    tree,
    operad_name: str = "P",
    decoration_formatter: Callable[[tuple, int], str] | None = None,
) -> str:
    """Return a human-readable string representation of a decorated tree.

    Args:
        tree: A rooted tree encoded by nested tuples.
        operad_name: Prefix used by the default decoration formatter.
        decoration_formatter: Optional callback ``(decoration, arity) -> str``.
            When provided, it is used to render each vertex decoration.
    """
    if is_leaf(tree):
        return str(tree)

    arity = vertex_arity(tree)
    dec = decoration(tree)
    if decoration_formatter is None:
        dec_str = f"{operad_name}{dec}"
    else:
        dec_str = decoration_formatter(dec, arity)

    children_str = ", ".join(
        tree_to_string(c, operad_name, decoration_formatter) for c in children(tree)
    )
    return f"({dec_str}; {children_str})"


def tree_to_latex(
    tree,
    operad_name: str = "P",
    decoration_formatter: Callable[[tuple, int], str] | None = None,
) -> str:
    """Return a LaTeX representation of a decorated rooted tree.

    Args:
        tree: A rooted tree encoded by nested tuples.
        operad_name: Prefix used by the default decoration formatter.
        decoration_formatter: Optional callback ``(decoration, arity) -> str``.
            When provided, it is used to render each vertex decoration.
    """
    if is_leaf(tree):
        return str(tree)

    arity = vertex_arity(tree)
    dec = decoration(tree)
    if decoration_formatter is None:
        dec_indices = ",".join(str(i) for i in dec)
        dec_str = f"\\operatorname{{{operad_name}}}_{{({dec_indices})}}"
    else:
        dec_str = decoration_formatter(dec, arity)

    children_str = ", ".join(
        tree_to_latex(c, operad_name, decoration_formatter) for c in children(tree)
    )
    return f"\\left({dec_str}; {children_str}\\right)"


def tree_to_svg(
    tree,
    operad_name: str = "P",
    decoration_formatter: Callable[[tuple, int], str] | None = None,
    leaf_formatter: Callable[[int], str] | None = None,
    *,
    leaf_dx: int = 70,
    level_dy: int = 88,
    margin: int = 22,
) -> str:
    """Render a decorated rooted tree to standalone SVG markup.

    The layout is deterministic and compact: leaves are equally spaced on the
    bottom row; each internal vertex is centered above its descendant leaves.

    Args:
        tree: A rooted tree encoded by nested tuples.
        operad_name: Prefix used by the default decoration formatter.
        decoration_formatter: Optional callback ``(decoration, arity) -> str``.
        leaf_formatter: Optional callback ``leaf_label -> str``.
        leaf_dx: Horizontal spacing between consecutive leaves.
        level_dy: Vertical spacing between consecutive levels.
        margin: Outer SVG margin.
    """

    def _dec_label(node) -> str:
        dec = decoration(node)
        ar = vertex_arity(node)
        if decoration_formatter is not None:
            return str(decoration_formatter(dec, ar))
        return f"{operad_name}{dec}"

    def _leaf_label(leaf: int) -> str:
        if leaf_formatter is not None:
            return str(leaf_formatter(leaf))
        return str(leaf)

    leaves_sorted = sorted(leaves(tree))
    n_leaves = max(1, len(leaves_sorted))
    leaf_index = {leaf: idx for idx, leaf in enumerate(leaves_sorted)}

    def _depth(node) -> int:
        if is_leaf(node):
            return 0
        if not children(node):
            return 1
        return 1 + max(_depth(c) for c in children(node))

    max_depth = _depth(tree)
    height = 2 * margin + max(1, max_depth) * level_dy
    width = 2 * margin + max(1, n_leaves - 1) * leaf_dx

    coords: dict[int, tuple[float, float]] = {}
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    node_labels: list[tuple[tuple[float, float], str]] = []
    leaf_labels: list[tuple[tuple[float, float], str]] = []

    def _layout(node) -> tuple[float, float]:
        node_id = id(node)
        if node_id in coords:
            return coords[node_id]

        if is_leaf(node):
            x = margin + leaf_index[node] * leaf_dx
            y = margin + max_depth * level_dy
            coords[node_id] = (x, y)
            leaf_labels.append(((x, y + 22), _leaf_label(node)))
            return (x, y)

        child_coords = [_layout(c) for c in children(node)]
        x = sum(cx for cx, _ in child_coords) / len(child_coords)
        y = margin + (max_depth - _depth(node)) * level_dy
        coords[node_id] = (x, y)

        for cx, cy in child_coords:
            edges.append(((x, y), (cx, cy)))
        node_labels.append(((x, y - 10), _dec_label(node)))
        return (x, y)

    _layout(tree)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" ',
        'viewBox="0 0 {w} {h}" role="img" aria-label="decorated rooted tree">'.format(
            w=int(width), h=int(height)
        ),
        "<style>",
        ".edge{stroke:#4b5563;stroke-width:1.6;fill:none;}",
        ".node{fill:#111827;}",
        '.vlabel{font: 13px "STIX Two Text", "Times New Roman", serif; fill:#0f172a; text-anchor:middle;}',
        '.llabel{font: 12px "STIX Two Text", "Times New Roman", serif; fill:#334155; text-anchor:middle;}',
        "</style>",
    ]

    for (x1, y1), (x2, y2) in edges:
        svg_lines.append(
            f'<line class="edge" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" />'
        )

    for (x, y), label in node_labels:
        svg_lines.append(f'<circle class="node" cx="{x:.2f}" cy="{y:.2f}" r="2.6" />')
        svg_lines.append(
            f'<text class="vlabel" x="{x:.2f}" y="{y - 14:.2f}">{escape(label)}</text>'
        )

    for (x, y), label in leaf_labels:
        svg_lines.append(f'<text class="llabel" x="{x:.2f}" y="{y:.2f}">{escape(label)}</text>')

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


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


def replace_vertex_decoration(tree: tuple, target: tuple, new_decoration: tuple) -> tuple:
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

    - ``basis_iter(degree)`` — degree-aware (e.g. ``Surjection``, ``BarrattEccles``).
    - Fallback via Sage's ``basis()`` family, filtered by ``degree_on_basis``.
    """
    basis_iter = getattr(operad_parent, "basis_iter", None)
    if basis_iter is not None:
        for elem in basis_iter(degree):
            yield from elem.support()
        return
    for key in operad_parent.basis():
        if operad_parent.degree_on_basis(key) == degree:
            yield key


def _planar_operad_basis_keys_in_degree(operad_parent, degree: int) -> Iterator:
    """Yield planar basis keys of *operad_parent* in the given degree.

    Uses ``planar_basis_it`` when available (quasi-planar operads such as
    ``Associative``, ``Surjection``, ``BarrattEccles``), otherwise falls back
    to :func:`_operad_basis_keys_in_degree`.

    This is used by the free/cofree tree-module enumeration to avoid
    over-counting: for a quasi-planar operad ``P(n) ≅ P_pl(n) ⊗ k[S_n]``,
    the isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}`` shows
    that only the planar representative from each S_n-orbit is needed.

    When ``planar_basis_it`` is present and returns an empty iterator for a
    given degree, no keys are yielded (correctly indicating no planar basis
    elements exist in that degree).  The full-basis fallback is only used when
    the operad does not expose ``planar_basis_it`` at all.
    """
    planar_basis_it = getattr(operad_parent, "planar_basis_it", None)
    if planar_basis_it is not None:
        for elem in planar_basis_it(degree):
            yield from elem.support()
        return
    yield from _operad_basis_keys_in_degree(operad_parent, degree)


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
                    _min_subtree_bar_degree(len(p), connectivity) for p in parts if len(p) >= 2
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
                    _shuffle_subtrees_iter(part, max_weight, operad_cls, base_ring, d_first)
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
    It relies only on ``operad_cls(k, base_ring)`` having a ``basis_iter``
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


def _min_subtree_degree_generic(part_size: int, connectivity: int, vertex_offset: int) -> int:
    """Minimum tree degree for a subtree with ``part_size`` leaves and given per-vertex offset.

    The three canonical choices are:

    - ``vertex_offset = +1``: bar degree, matches :func:`_min_subtree_bar_degree`.
    - ``vertex_offset = 0``: free degree ``Σ deg_P(dec(v))``.
    - ``vertex_offset = -1``: cobar degree ``Σ (deg_C(dec(v)) - 1)``.

    Returns 0 for leaves (``part_size == 1``).
    """
    if part_size == 1:
        return 0
    return vertex_offset + connectivity * (part_size - 1)


def _shuffle_children_iter_generic(
    parts: list,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    total_deg: int,
    root_dec: tuple,
    vertex_offset: int,
    use_planar_decs: bool = False,
) -> Iterator[tuple]:
    """Yield complete decorated trees ``(root_dec, t_1, ..., t_k)`` with generic per-vertex offset."""
    connectivity = getattr(operad_cls, "connectivity", 0)

    min_from = [0] * (len(parts) + 1)
    for i in range(len(parts) - 1, -1, -1):
        min_from[i] = (
            _min_subtree_degree_generic(len(parts[i]), connectivity, vertex_offset)
            + min_from[i + 1]
        )

    def _children_combinations(idx: int, remaining: int) -> Iterator[list]:
        if idx == len(parts):
            if remaining == 0:
                yield []
            return
        if remaining < min_from[idx]:
            return
        part = parts[idx]
        n_part = len(part)
        if n_part == 1:
            first = part[0]
            for rest in _children_combinations(idx + 1, remaining):
                yield [first] + rest
        else:
            min_deg_this = _min_subtree_degree_generic(n_part, connectivity, vertex_offset)
            max_d = remaining - min_from[idx + 1]
            if max_d < min_deg_this:
                return
            for d_first in range(min_deg_this, max_d + 1):
                first_trees = list(
                    _shuffle_subtrees_iter_generic(
                        part,
                        max_weight,
                        operad_cls,
                        base_ring,
                        d_first,
                        vertex_offset,
                        use_planar_decs,
                    )
                )
                if not first_trees:
                    continue
                for rest in _children_combinations(idx + 1, remaining - d_first):
                    for ft in first_trees:
                        yield [ft] + rest

    for ch in _children_combinations(0, total_deg):
        yield (root_dec,) + tuple(ch)


def _shuffle_subtrees_iter_generic(
    leaf_set: tuple,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
    vertex_offset: int,
    use_planar_decs: bool = False,
) -> Iterator:
    """Generic shuffle-tree enumerator with configurable per-vertex degree offset.

    The three canonical choices for ``vertex_offset`` are:

    - ``+1``: bar degree ``Σ (deg_P(v) + 1)``.
    - ``0``: free degree ``Σ deg_P(v)`` (no suspension).
    - ``-1``: cobar degree ``Σ (deg_C(v) - 1)``.

    When ``use_planar_decs=True``, only the planar basis of each vertex
    decoration is used (via ``planar_basis_it`` when available).  This is
    the correct choice for the composite product ``P ∘ M``, where only one
    representative per ``S_n``-orbit is needed.
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
    _dec_iter = (
        _planar_operad_basis_keys_in_degree if use_planar_decs else _operad_basis_keys_in_degree
    )

    for v_arity in range(2, n + 1):
        root_parent = operad_cls(v_arity, base_ring)
        min_root_dec_deg = connectivity * (v_arity - 1)
        if v_arity == n:
            root_dec_deg = target_degree - vertex_offset
            if root_dec_deg >= min_root_dec_deg:
                for root_dec in _dec_iter(root_parent, root_dec_deg):
                    yield (root_dec,) + sorted_ls
        else:
            for parts in _shuffle_partitions(sorted_ls, v_arity):
                min_child_total = sum(
                    _min_subtree_degree_generic(len(p), connectivity, vertex_offset)
                    for p in parts
                    if len(p) >= 2
                )
                max_root_dec_deg = target_degree - vertex_offset - min_child_total
                for root_dec_deg in range(min_root_dec_deg, max_root_dec_deg + 1):
                    child_total = target_degree - root_dec_deg - vertex_offset
                    for root_dec in _dec_iter(root_parent, root_dec_deg):
                        yield from _shuffle_children_iter_generic(
                            parts,
                            max_weight - 1,
                            operad_cls,
                            base_ring,
                            child_total,
                            root_dec,
                            vertex_offset,
                            use_planar_decs,
                        )


def enumerate_shuffle_trees_free_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[tuple]:
    """Enumerate shuffle trees of ``arity`` leaves with free degree *target_degree*.

    The *free degree* of a decorated tree is ``Σ_v deg_P(dec(v))`` — the sum of
    operad/cooperad element degrees over all internal vertices, **without** the
    per-vertex ``+1`` offset used by the bar construction.  This is the natural
    degree for the free P-algebra composite product ``P ∘ M`` and the cofree
    conilpotent C-coalgebra ``T^c_C(M)``.

    Args:
        arity: Number of leaves.
        weight_bound: Maximum number of internal vertices (pass ``arity - 1``
            for connected operads).
        operad_cls: Operad or cooperad factory used for vertex decorations.
        base_ring: Coefficient ring.
        target_degree: Exact free degree to enumerate.

    Yields:
        Decorated shuffle trees (nested tuples) as valid tree basis keys.
    """
    if arity < 2 or weight_bound < 1:
        return
    yield from _shuffle_subtrees_iter_generic(
        tuple(range(1, arity + 1)), weight_bound, operad_cls, base_ring, target_degree, 0
    )


def enumerate_shuffle_trees_cobar_in_degree(
    arity: int,
    weight_bound: int,
    cooperad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[tuple]:
    """Enumerate shuffle trees of ``arity`` leaves with cobar degree *target_degree*.

    The *cobar degree* of a decorated tree is ``Σ_v (deg_C(dec(v)) - 1)`` — the
    sum of desuspended cooperad element degrees over all internal vertices.  This
    is the natural degree for the cobar construction ``Ω(C)`` and the cobar
    complex ``Ω_C(V)``.

    Args:
        arity: Number of leaves.
        weight_bound: Maximum number of internal vertices (pass ``arity - 1``
            for connected cooperads).
        cooperad_cls: Cooperad factory used for vertex decorations.
        base_ring: Coefficient ring.
        target_degree: Exact cobar degree to enumerate.  May be negative when
            the cooperad has elements of low degree (e.g. ``CoAssociative`` in
            degree 0 contributes ``-1`` per vertex).

    Yields:
        Decorated shuffle trees (nested tuples) as valid tree basis keys.
    """
    if arity < 2 or weight_bound < 1:
        return
    yield from _shuffle_subtrees_iter_generic(
        tuple(range(1, arity + 1)), weight_bound, cooperad_cls, base_ring, target_degree, -1
    )


def enumerate_shuffle_trees_generic_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
    vertex_offset: int,
    use_planar_decs: bool = False,
) -> Iterator[tuple]:
    """Enumerate shuffle trees with an arbitrary per-vertex degree offset.

    The three canonical choices for ``vertex_offset`` are:

    - ``+1``: bar degree ``Σ (deg_P(v) + 1)``.
    - ``0``: free degree ``Σ deg_P(v)`` (no suspension).
    - ``-1``: cobar degree ``Σ (deg_C(v) - 1)``.

    Args:
        arity: Number of leaves.
        weight_bound: Maximum number of internal vertices.
        operad_cls: Operad or cooperad factory for vertex decorations.
        base_ring: Coefficient ring.
        target_degree: Exact tree degree to enumerate.
        vertex_offset: Per-vertex degree offset.
        use_planar_decs: When ``True``, restrict vertex decorations to the
            planar basis (via ``planar_basis_it``) instead of the full basis.
            Use this for composite-product ``P ∘ M`` enumeration to obtain
            one representative per ``S_n``-orbit.

    Yields:
        Decorated shuffle trees (nested tuples) as valid tree basis keys.
    """
    if arity < 2 or weight_bound < 1:
        return
    yield from _shuffle_subtrees_iter_generic(
        tuple(range(1, arity + 1)),
        weight_bound,
        operad_cls,
        base_ring,
        target_degree,
        vertex_offset,
        use_planar_decs,
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
    cobar_degrees = [subtree_degree_cobar(c, cooperad_cls, base_ring) for c in normalized_kids]

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
