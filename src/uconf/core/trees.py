"""Rooted tree utilities for bar/cobar constructions.

Trees are represented using the :class:`RootedTree` class:

- A **leaf** is a plain ``int`` in ``{1, …, n}`` (the arity).
- An **internal vertex** is a :class:`RootedTree` instance holding a
  ``decoration`` (an operad/cooperad basis key, always a tuple) and an
  ordered sequence of ``children`` (each a leaf or another ``RootedTree``).
  For connected operads, every vertex has at least 2 children.

``RootedTree`` is **immutable** and **hashable**, making it suitable as a
basis key in SageMath's ``CombinatorialFreeModule`` and as an argument to
``@cached_method``.

Structural properties that the old tuple encoding recomputed from scratch
on every call — ``weight``, ``leaves``, ``min_leaf``, ``tree_arity`` — are
now **eagerly cached** at construction time and available in O(1).

Example (arity 3, weight 2)::

    RootedTree((1, 2), RootedTree((1,), 1, 2), 3)

represents a tree with root decorated by the Lie basis key ``(1, 2)``
(a binary bracket), whose first child is another internal vertex
decorated by ``(1,)`` with leaves 1 and 2, and whose second child is leaf 3.
"""

from __future__ import annotations

from functools import total_ordering
from typing import Any, Callable, Iterator, Literal

from uconf.core.signs import koszul_sign_of_permutation


# =====================================================================
# RootedTree class
# =====================================================================


@total_ordering
class RootedTree:
    """Immutable rooted tree with eagerly cached structural properties.

    Parameters
    ----------
    decoration : tuple
        Operad/cooperad basis key decorating this vertex.
    *children : int | RootedTree
        Ordered children.  Each child is either a leaf (``int``) or
        another ``RootedTree``.

    Cached attributes (O(1) access after construction):

    - ``_decoration``: the vertex decoration.
    - ``_children``: tuple of children.
    - ``_arity``: number of children (vertex arity).
    - ``_weight``: total number of internal vertices in this subtree.
    - ``_leaves``: ``frozenset`` of all leaf labels.
    - ``_min_leaf``: minimum leaf label.
    - ``_tree_arity``: total number of leaves.

    ``RootedTree`` is immutable: all ``__slots__`` are set once in
    ``__init__`` and ``__setattr__`` is overridden to prevent mutation.
    """

    __slots__ = (
        "_decoration",
        "_children",
        "_arity",
        "_weight",
        "_leaves",
        "_min_leaf",
        "_tree_arity",
        "_hash",
    )

    def __init__(self, decoration: tuple, *children: int | RootedTree):
        object.__setattr__(self, "_decoration", decoration)
        object.__setattr__(self, "_children", children)
        object.__setattr__(self, "_arity", len(children))

        w = 1
        lv: set[int] = set()
        ml = float("inf")
        for c in children:
            if isinstance(c, RootedTree):
                w += c._weight
                lv.update(c._leaves)
                cml = c._min_leaf
            else:
                lv.add(c)
                cml = c
            if cml < ml:
                ml = cml

        object.__setattr__(self, "_weight", w)
        object.__setattr__(self, "_leaves", frozenset(lv))
        object.__setattr__(self, "_min_leaf", ml)
        object.__setattr__(self, "_tree_arity", len(lv))
        object.__setattr__(self, "_hash", hash((decoration, children)))

    # -- immutability ---------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RootedTree instances are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RootedTree instances are immutable")

    # -- hashing and equality -------------------------------------------

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, RootedTree):
            return NotImplemented
        if self._hash != other._hash:
            return False
        return self._decoration == other._decoration and self._children == other._children

    def __lt__(self, other: object) -> bool:
        if isinstance(other, int):
            return False  # internal vertices sort after leaves
        if not isinstance(other, RootedTree):
            return NotImplemented
        # Lexicographic on (decoration, children), matching old tuple order.
        if self._decoration != other._decoration:
            return self._decoration < other._decoration
        return self._cmp_children() < other._cmp_children()

    def _cmp_children(self) -> tuple:
        """Return a tuple suitable for lexicographic child comparison.

        Leaves (int) sort before ``RootedTree`` instances.  Within each
        type the natural ordering is used.  We wrap each child in a
        ``(type_tag, value)`` pair so that ``int < RootedTree`` is
        well-defined even though Python 3 forbids direct ``int < object``.
        """
        return tuple(
            (0, c) if isinstance(c, int) else (1, c._decoration, c._cmp_children())
            for c in self._children
        )

    # -- representation -------------------------------------------------

    def __repr__(self) -> str:
        kids = ", ".join(repr(c) for c in self._children)
        return f"T({self._decoration!r}; {kids})"

    # -- conversion -----------------------------------------------------

    def to_tuple(self) -> tuple:
        """Convert back to the legacy nested-tuple representation."""
        kids = tuple(c.to_tuple() if isinstance(c, RootedTree) else c for c in self._children)
        return (self._decoration,) + kids

    @classmethod
    def from_tuple(cls, t):
        """Create a ``RootedTree`` from a nested-tuple representation.

        Leaves (``int``) are returned as-is.
        """
        if isinstance(t, int):
            return t
        if isinstance(t, cls):
            return t
        dec = t[0]
        kids = tuple(cls.from_tuple(c) for c in t[1:])
        return cls(dec, *kids)

    # -- pickling (needed by SageMath serialisation) --------------------

    def __reduce__(self):
        return (RootedTree, (self._decoration,) + self._children)


# =====================================================================
# Predicates and accessors  (thin wrappers for uniform API)
# =====================================================================


def is_leaf(node) -> bool:
    """Return True if ``node`` is a leaf (an integer)."""
    return isinstance(node, int)


def is_internal(node) -> bool:
    """Return True if ``node`` is an internal vertex."""
    return isinstance(node, RootedTree)


def decoration(vertex: RootedTree) -> tuple:
    """Return the decoration (operad basis key) of an internal vertex."""
    return vertex._decoration


def children(vertex: RootedTree) -> tuple:
    """Return the children of an internal vertex as a tuple."""
    return vertex._children


def vertex_arity(vertex: RootedTree) -> int:
    """Return the arity (number of children) of an internal vertex."""
    return vertex._arity


def leaves(tree) -> frozenset[int]:
    """Return the (frozen)set of all leaf labels in the tree.

    O(1) for ``RootedTree`` nodes (cached); O(n) for bare ``int`` leaves.
    """
    if isinstance(tree, int):
        return frozenset({tree})
    return tree._leaves


def weight(tree) -> int:
    """Return the weight (number of internal vertices) of the tree.

    O(1) for ``RootedTree`` nodes (cached).
    """
    if isinstance(tree, int):
        return 0
    return tree._weight


def tree_arity(tree) -> int:
    """Return the arity of the tree (number of leaves).

    O(1) for ``RootedTree`` nodes (cached).
    """
    if isinstance(tree, int):
        return 1
    return tree._tree_arity


def vertices_dfs(tree) -> list[RootedTree]:
    """Return all internal vertices in depth-first (pre-order) traversal.

    Each entry is the ``RootedTree`` object itself (identity-safe).
    """
    if isinstance(tree, int):
        return []
    result: list[RootedTree] = [tree]
    for child in tree._children:
        if isinstance(child, RootedTree):
            result.extend(vertices_dfs(child))
    return result


def subtree_degree(tree, operad_cls, base_ring) -> int:
    """Compute the total shifted bar degree of a subtree.

    In the conventions used by ``BarConstruction``, each internal vertex
    contributes ``deg_P(decoration) + 1``.
    """
    if isinstance(tree, int):
        return 0
    parent = operad_cls(tree._arity, base_ring)
    dec = tree._decoration
    vertex_deg = parent.degree_on_basis(dec) + 1
    child_deg = sum(subtree_degree(c, operad_cls, base_ring) for c in tree._children)
    return vertex_deg + child_deg


def subtree_degree_cobar(tree, cooperad_cls, base_ring) -> int:
    """Compute the total shifted cobar degree of a subtree.

    In the conventions used by ``CobarConstruction``, each internal vertex
    contributes ``deg_C(decoration) - 1``.
    """
    if isinstance(tree, int):
        return 0
    parent = cooperad_cls(tree._arity, base_ring)
    dec = tree._decoration
    vertex_deg = parent.degree_on_basis(dec) - 1
    child_deg = sum(subtree_degree_cobar(c, cooperad_cls, base_ring) for c in tree._children)
    return vertex_deg + child_deg


def after_cobar_deg(tree, leaf_i: int, cooperad_cls, base_ring) -> int:
    """Total cobar degree of internal vertices after *leaf_i* in DFS order.

    In the DFS of a shuffle tree, the root vertex is visited first, then
    children are recursed left-to-right.  A leaf is visited in place.
    ``after_cobar_deg`` returns the sum of ``(deg_C(v) - 1)`` for every
    internal vertex ``v`` whose DFS visit occurs **after** the visit of
    *leaf_i*.

    For a corolla (single internal vertex) this is always 0, since the
    root is visited before any leaf.
    """
    if isinstance(tree, int):
        return 0
    kids = tree._children
    # Find which child subtree contains leaf_i (O(1) via cached leaves)
    for p, child in enumerate(kids):
        if isinstance(child, int):
            child_leaves = frozenset({child})
        else:
            child_leaves = child._leaves
        if leaf_i in child_leaves:
            # Vertices after leaf_i:
            # 1. vertices after leaf_i within child p (recursion)
            # 2. all internal vertices in children p+1, ..., k-1
            result = after_cobar_deg(child, leaf_i, cooperad_cls, base_ring)
            for later_child in kids[p + 1 :]:
                result += subtree_degree_cobar(later_child, cooperad_cls, base_ring)
            return result
    raise ValueError(f"Leaf {leaf_i} not found in tree")


def internal_edges_dfs(tree) -> list[tuple[RootedTree, int, RootedTree]]:
    """Enumerate all internal edges (parent-child pairs between internal vertices).

    Returns a list of ``(parent_vertex, child_position, child_vertex)`` tuples
    where ``child_position`` is 1-indexed (the slot in the operad composition).
    """
    if isinstance(tree, int):
        return []
    result: list[tuple[RootedTree, int, RootedTree]] = []
    for i, child in enumerate(tree._children, start=1):
        if isinstance(child, RootedTree):
            result.append((tree, i, child))
            result.extend(internal_edges_dfs(child))
    return result


def contract_edge(
    tree: RootedTree, parent_vertex: RootedTree, child_pos: int, new_decoration: tuple
) -> RootedTree:
    """Contract one internal edge, merging parent and child vertices.

    The ``child_pos``-th child of ``parent_vertex`` is an internal vertex.
    Replace both with a single vertex having ``new_decoration`` and the combined
    children (children of parent before child_pos, children of child, children
    of parent after child_pos).

    This function recursively finds and contracts the specified edge in the tree.
    """
    if isinstance(tree, int):
        return tree

    if tree is parent_vertex:
        # This is the parent vertex to contract
        child_vertex = tree._children[child_pos - 1]
        assert isinstance(child_vertex, RootedTree), "Child must be internal for contraction"

        # Build new children list
        new_children = []
        for i, c in enumerate(tree._children, start=1):
            if i < child_pos:
                new_children.append(c)
            elif i == child_pos:
                # Insert children of the child vertex here
                new_children.extend(child_vertex._children)
            else:
                new_children.append(c)

        return RootedTree(new_decoration, *new_children)

    # Recurse into children
    new_children = tuple(
        contract_edge(c, parent_vertex, child_pos, new_decoration)
        if isinstance(c, RootedTree)
        else c
        for c in tree._children
    )
    return RootedTree(tree._decoration, *new_children)


def graft(tree_top, i: int, tree_bot, relabel_bot: dict | None = None):
    """Graft ``tree_bot`` onto leaf ``i`` of ``tree_top``.

    If ``relabel_bot`` is provided, it maps each leaf of ``tree_bot`` to its
    new label in the grafted tree.  If not provided, leaves of ``tree_bot``
    are relabeled to ``{i, i+1, ..., i+n_bot-1}`` and leaves > i in ``tree_top``
    are shifted by ``n_bot - 1``.

    Returns the grafted tree.
    """
    if isinstance(tree_bot, int):
        n_bot = 1
    else:
        n_bot = tree_bot._tree_arity

    if relabel_bot is None:
        # Default relabeling: tree_bot leaves become i, i+1, ..., i+n_bot-1
        if isinstance(tree_bot, int):
            bot_leaves_sorted = [tree_bot]
        else:
            bot_leaves_sorted = sorted(tree_bot._leaves)
        relabel_bot = {old: i + idx for idx, old in enumerate(bot_leaves_sorted)}

    def relabel_top(leaf: int) -> int:
        if leaf < i:
            return leaf
        elif leaf > i:
            return leaf + n_bot - 1
        else:
            raise ValueError("Leaf i should be replaced, not relabeled")

    def graft_rec(node):
        if isinstance(node, int):
            if node == i:
                # Replace with tree_bot (relabeled)
                return relabel_leaves(tree_bot, relabel_bot)
            else:
                return relabel_top(node)
        # Internal vertex: recurse
        new_children = tuple(graft_rec(c) for c in node._children)
        return RootedTree(node._decoration, *new_children)

    return graft_rec(tree_top)


def relabel_leaves(tree, mapping: dict):
    """Apply a leaf relabeling according to ``mapping``."""
    if isinstance(tree, int):
        return mapping.get(tree, tree)
    new_children = tuple(relabel_leaves(c, mapping) for c in tree._children)
    return RootedTree(tree._decoration, *new_children)


def split_at_vertex(
    tree: RootedTree, target_vertex: RootedTree
) -> tuple[RootedTree | int, int, RootedTree] | None:
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
        if isinstance(node, int):
            return node
        if node is target_vertex:
            return replacement_leaf
        new_children = tuple(find_and_replace(c, replacement_leaf) for c in node._children)
        return RootedTree(node._decoration, *new_children)

    # Determine what leaf label to use as placeholder
    # Use min leaf of target_vertex's subtree
    placeholder = target_vertex._min_leaf

    tree_top = find_and_replace(tree, placeholder)
    tree_bot = target_vertex

    return (tree_top, placeholder, tree_bot)


def validate_tree(tree, arity: int, operad_cls, base_ring) -> RootedTree | Literal[1] | None:
    """Validate a tree for use in bar/cobar constructions.

    Checks:
    - Leaves are exactly {1, ..., arity}
    - All internal vertices have arity >= 2 (connected assumption)
    - All decorations are valid for the given operad/cooperad

    Returns a validated tree with cleaned decorations, raises if invalid, or None if decorations are invalid but the tree structure is fine.
    """
    # Check that tree_arity matches
    if isinstance(tree, int):
        # A single leaf is only valid for arity 1
        if arity == 1 and tree == 1:
            return tree
        raise ValueError(f"Invalid leaf: {tree}, expected 1 for arity 1")

    # Check leaves
    tree_leaves = tree._leaves
    if tree_leaves != frozenset(range(1, arity + 1)):
        raise ValueError(f"Invalid leaves: {tree_leaves}, should be {set(range(1, arity + 1))}")

    # Validate all internal vertices
    def validate_vertex(node):
        if isinstance(node, int):
            return node

        v_arity = node._arity
        if v_arity < 2:
            raise ValueError(f"Invalid vertex with arity {v_arity} < 2: {node}")

        dec = node._decoration
        parent = operad_cls(v_arity, base_ring)

        validate_fn = getattr(parent, "_validate_basis_key", None)
        if validate_fn is not None:
            clean_dec = validate_fn(dec)
            if clean_dec is None:
                return None
        else:
            clean_dec = dec

        # Validate children, tracking whether anything changed
        changed = clean_dec is not dec
        new_children = []
        for child in node._children:
            validated = validate_vertex(child)
            if validated is None:
                return None
            if validated is not child:
                changed = True
            new_children.append(validated)

        # Avoid reconstructing the tree if nothing changed
        if not changed:
            return node
        return RootedTree(clean_dec, *new_children)

    return validate_vertex(tree)


def enumerate_planar_trees_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[RootedTree]:
    """Enumerate planar-decorated trees in ``B(P)(arity)`` of bar degree ``target_degree``.

    A tree is *planar* (in the quasi-planar cooperad sense) when:

    1. All vertex decorations are planar elements of the base operad.
    2. Children at each vertex have consecutive leaf-label ranges, so that
       ``planarize(T) = T ⊗ id`` (the global permutation is the identity).

    Condition 2 means that the tree is a *standard planar tree*: leaf labels
    run ``1, …, n`` strictly left-to-right, and each child subtree occupies a
    contiguous block.  All such trees are automatically in shuffle form.

    Uses the operad's ``planar_basis_iter`` at the exact required degree for
    each vertex, so this is efficient even for operads with many basis elements
    per degree (e.g. Barratt–Eccles).

    Requires the operad to implement ``planar_basis_iter``.

    The enumeration mirrors the binary-root topologies produced by
    ``enumerate_trees_by_weight``, extended to planar decorations and
    restricted to consecutive child-leaf ranges.

    Args:
        arity: Number of leaves (arity of the bar-construction component).
        weight_bound: Maximum number of internal vertices.  For connected
            operads, callers should pass ``arity - 1`` (the hard upper bound
            from the branching constraint ``sum(m_v - 1) = n - 1``).
        operad_cls: Operad factory; must supply ``planar_basis_iter``.
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
            if hasattr(parent, "planar_basis_iter"):
                for elem in parent.planar_basis_iter(dec_degree):
                    for dec in elem.support():
                        yield RootedTree(dec, *range(1, arity + 1))

    # ------------------------------------------------------------------
    # Weight >= 2: binary root, one of whose children is an internal
    # subtree.  We restrict to *consecutive* child-leaf ranges so that
    # both children's subtrees remain planar.  The two ranges are
    # {1,...,a} and {a+1,...,n} for a in 1..n-1.
    # ------------------------------------------------------------------
    if weight_bound >= 2 and arity >= 3:
        root_parent = operad_cls(2, base_ring)
        if not hasattr(root_parent, "planar_basis_iter"):
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
                for root_dec_elem in root_parent.planar_basis_iter(root_dec_degree):
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
                                    yield RootedTree(root_dec, c1, c2)


def _planar_subtrees_for_leaves(
    leaf_set: tuple,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
) -> Iterator[RootedTree | int]:
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
    decoration_formatter: Callable[[tuple, int], str],
) -> str:
    """Return a human-readable string representation of a decorated tree.

    Args:
        tree: A rooted tree (``RootedTree`` or leaf ``int``).
        decoration_formatter: Optional callback ``(decoration, arity) -> str``.
            When provided, it is used to render each vertex decoration.
    """
    if isinstance(tree, int):
        return str(tree)

    arity = tree._arity
    dec = tree._decoration
    dec_str = decoration_formatter(dec, arity)

    children_str = ", ".join(tree_to_string(c, decoration_formatter) for c in tree._children)
    return f"({dec_str}; {children_str})"


def tree_to_latex(
    tree,
    decoration_formatter: Callable[[tuple, int], str],
) -> str:
    """Return a LaTeX representation of a decorated rooted tree.

    Args:
        tree: A rooted tree (``RootedTree`` or leaf ``int``).
        decoration_formatter: Optional callback ``(decoration, arity) -> str``.
            When provided, it is used to render each vertex decoration.
    """
    if isinstance(tree, int):
        return str(tree)

    arity = tree._arity
    dec = tree._decoration
    dec_str = decoration_formatter(dec, arity)

    children_str = ", ".join(tree_to_latex(c, decoration_formatter) for c in tree._children)
    return f"\\left({dec_str}; {children_str}\\right)"


def copy_tree_structure(old_tree, new_decorations: list[tuple]) -> RootedTree:
    """Create a tree with the same structure but new decorations.

    ``new_decorations`` should be in DFS order matching ``vertices_dfs(old_tree)``.
    """
    if isinstance(old_tree, int):
        return old_tree

    dec_iter = iter(new_decorations)

    def rebuild(node):
        if isinstance(node, int):
            return node
        new_dec = next(dec_iter)
        new_children = tuple(rebuild(c) for c in node._children)
        return RootedTree(new_dec, *new_children)

    return rebuild(old_tree)


def replace_vertex_decoration(tree: RootedTree, target: RootedTree, new_decoration: tuple):
    """Replace the decoration of a specific vertex in the tree."""
    if isinstance(tree, int):
        return tree

    if tree is target:
        return RootedTree(new_decoration, *tree._children)

    new_children = tuple(
        replace_vertex_decoration(c, target, new_decoration) if isinstance(c, RootedTree) else c
        for c in tree._children
    )
    return RootedTree(tree._decoration, *new_children)


def expand_vertex(
    tree: RootedTree,
    target_vertex: RootedTree,
    child_pos: int,
    left_decoration: tuple,
    right_decoration: tuple,
    left_arity: int,
    right_arity: int,
) -> RootedTree:
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
    if isinstance(tree, int):
        return tree

    if tree is target_vertex:
        orig_children = tree._children
        k = len(orig_children)
        a, b = left_arity, right_arity
        l = child_pos
        assert k == a + b - 1, f"Arity mismatch: {k} != {a} + {b} - 1"

        # Build bottom vertex children: original children l, l+1, ..., l+b-1
        bottom_children = orig_children[l - 1 : l - 1 + b]
        bottom_vertex = RootedTree(right_decoration, *bottom_children)

        # Build top vertex children
        top_children = (
            orig_children[: l - 1]  # positions 1..l-1
            + (bottom_vertex,)  # position l
            + orig_children[l - 1 + b :]  # remaining
        )
        return RootedTree(left_decoration, *top_children)

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
            if isinstance(c, RootedTree)
            else c
        )
        for c in tree._children
    )
    return RootedTree(tree._decoration, *new_children)


# =============================================================================
# Shuffle tree normalization for symmetric operads/cooperads
# =============================================================================


def min_leaf(tree) -> int:
    """Return the minimum leaf label in a tree or subtree.

    O(1) for ``RootedTree`` nodes (cached).
    """
    if isinstance(tree, int):
        return tree
    return tree._min_leaf


def is_shuffle_tree(tree) -> bool:
    """Check if a tree is in shuffle form.

    A shuffle tree has, at each internal vertex, children ordered so that
    the minimum leaf label in each child subtree increases from left to right.

    Example:
        ``RootedTree((), 1, 2)`` is shuffle (min({1})=1 < min({2})=2)
        ``RootedTree((), 2, 1)`` is NOT shuffle (min({2})=2 > min({1})=1)

    """
    if isinstance(tree, int):
        return True

    kids = tree._children
    min_leaves = [min_leaf(c) for c in kids]

    # Children must be sorted by min leaf
    for i in range(len(min_leaves) - 1):
        if min_leaves[i] >= min_leaves[i + 1]:
            return False

    # Recursively check all subtrees
    return all(is_shuffle_tree(c) for c in kids if isinstance(c, RootedTree))


def to_shuffle_tree_bar(tree, operad_cls, base_ring):
    """Normalize a tree to shuffle form for the bar construction B(P).

    Returns a list of ``(shuffle_tree, coeff)`` pairs representing a
    (possibly multi-term) linear combination.

    At each vertex, children are sorted by min leaf. The decoration is
    acted on by the sorting permutation (using the operad's permute method),
    and Koszul signs are computed based on bar-degrees of subtrees.

    When the operad's ``permute`` produces multiple basis terms (e.g. for
    Lie-based operads), each term gives a separate output tree.
    """
    if isinstance(tree, int):
        return [(tree, 1)]

    kids = tree._children
    dec = tree._decoration
    k = len(kids)

    # Recursively normalize children first.
    # Each child may expand into multiple terms.
    from itertools import product as iter_product

    child_term_lists: list[list[tuple]] = []
    for c in kids:
        if isinstance(c, RootedTree):
            child_term_lists.append(to_shuffle_tree_bar(c, operad_cls, base_ring))
        else:
            child_term_lists.append([(c, 1)])

    results = []
    for combo in iter_product(*child_term_lists):
        # combo is a tuple of (normalized_child, child_coeff)
        normalized_kids = [item[0] for item in combo]
        child_coeff = 1
        for item in combo:
            child_coeff *= item[1]

        # Compute min leaf and bar-degree for each normalized child (O(1) each)
        min_leaves = [min_leaf(c) for c in normalized_kids]
        bar_degrees = [subtree_degree(c, operad_cls, base_ring) for c in normalized_kids]

        # Sort children by min leaf
        indexed = list(zip(min_leaves, range(k), normalized_kids, bar_degrees))
        indexed.sort(key=lambda x: x[0])

        perm = [item[1] for item in indexed]

        if perm == list(range(k)):
            new_tree = RootedTree(dec, *normalized_kids)
            results.append((new_tree, child_coeff))
            continue

        koszul_sign = koszul_sign_of_permutation(perm, bar_degrees)

        # Compute σ^{-1} for the sorting permutation
        sigma_list = [p + 1 for p in perm]
        sigma_inv = [0] * k
        for new_pos, old_pos_plus_one in enumerate(sigma_list):
            sigma_inv[old_pos_plus_one - 1] = new_pos + 1

        operad_parent = operad_cls(k, base_ring)
        dec_elem = operad_parent(dec)
        permuted_dec_elem = dec_elem.permute(sigma_inv)

        sorted_kids = tuple(item[2] for item in indexed)
        for new_dec_key, dec_coeff in permuted_dec_elem:
            new_tree = RootedTree(new_dec_key, *sorted_kids)
            total_coeff = child_coeff * koszul_sign * dec_coeff
            results.append((new_tree, total_coeff))

    return results


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

    Uses ``planar_basis_iter`` when available (quasi-planar operads such as
    ``Associative``, ``Surjection``, ``BarrattEccles``), otherwise falls back
    to :func:`_operad_basis_keys_in_degree`.

    This is used by the free/cofree tree-module enumeration to avoid
    over-counting: for a quasi-planar operad ``P(n) ≅ P_pl(n) ⊗ k[S_n]``,
    the isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}`` shows
    that only the planar representative from each S_n-orbit is needed.

    When ``planar_basis_iter`` is present and returns an empty iterator for a
    given degree, no keys are yielded (correctly indicating no planar basis
    elements exist in that degree).  The full-basis fallback is only used when
    the operad does not expose ``planar_basis_iter`` at all.
    """
    planar_basis_iter = getattr(operad_parent, "planar_basis_iter", None)
    if planar_basis_iter is not None:
        for elem in planar_basis_iter(degree):
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
                    yield RootedTree(root_dec, *sorted_ls)
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
) -> Iterator[RootedTree]:
    """Yield complete decorated trees ``RootedTree(root_dec, t_1, ..., t_k)`` where each
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

    for ch_list in _children_combinations(0, total_deg):
        yield RootedTree(root_dec, *ch_list)


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
    require the base operad to implement ``planarize`` or ``planar_basis_iter``.
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
) -> Iterator[RootedTree]:
    """Yield complete decorated trees ``RootedTree(root_dec, t_1, ..., t_k)`` with generic per-vertex offset."""
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
        yield RootedTree(root_dec, *ch)


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
    decoration is used (via ``planar_basis_iter`` when available).  This is
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
                    yield RootedTree(root_dec, *sorted_ls)
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


def _consecutive_parts_iter(leaf_range: tuple, k: int) -> Iterator[list[tuple]]:
    """Yield all partitions of a consecutive leaf range into *k* consecutive sub-ranges.

    Unlike :func:`_shuffle_partitions`, which partitions into parts sorted only
    by their minimum element, this function insists that each part is a
    *consecutive* block ``(leaf_range[a], ..., leaf_range[b])`` for some
    ``a ≤ b``.  This is the correct partition type for *planar* (DFS-canonical)
    trees, where each subtree occupies a contiguous interval of leaf labels.

    Args:
        leaf_range: A tuple of consecutive leaf labels, e.g. ``(1, 2, 3, 4)``.
        k: Number of parts.

    Yields:
        Lists of *k* consecutive sub-tuples that together cover *leaf_range*.
    """
    from itertools import combinations

    n = len(leaf_range)
    if k <= 0 or k > n:
        return
    if k == 1:
        yield [leaf_range]
        return
    # Choose k-1 split points from positions 1..n-1 (exclusive start/end)
    for split_positions in combinations(range(1, n), k - 1):
        parts = []
        prev = 0
        for s in split_positions:
            parts.append(leaf_range[prev:s])
            prev = s
        parts.append(leaf_range[prev:])
        yield parts


def _planar_children_iter_generic(
    parts: list,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    total_deg: int,
    root_dec: tuple,
    vertex_offset: int,
    use_planar_decs: bool = True,
) -> Iterator[RootedTree]:
    """Yield complete planar-decorated trees ``RootedTree(root_dec, t_1, …, t_k)`` for consecutive parts."""
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
                    _planar_subtrees_iter_generic(
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
        yield RootedTree(root_dec, *ch)


def _planar_subtrees_iter_generic(
    leaf_range: tuple,
    max_weight: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
    vertex_offset: int,
    use_planar_decs: bool = True,
) -> Iterator:
    """Enumerate planar trees over *leaf_range* (a consecutive block) with generic offset.

    A *planar tree* (in the DFS-canonical sense) has children at every vertex
    occupying *consecutive* sub-ranges of its leaf range.  This guarantees that
    the DFS leaf order equals ``1, 2, …, n`` and that
    ``planarize(T) = T ⊗ id`` (sigma_global = identity) when all vertex
    decorations are already in planar form.

    Compared to :func:`_shuffle_subtrees_iter_generic`, this function replaces
    :func:`_shuffle_partitions` with :func:`_consecutive_parts_iter`, excluding
    shuffle-tree configurations where subtrees carry non-consecutive leaf sets.

    Args:
        leaf_range: Tuple of consecutive leaf labels (e.g. ``(1, 2, 3)``).
        max_weight: Maximum number of internal vertices.
        operad_cls: Operad/cooperad factory for vertex decorations.
        base_ring: Coefficient ring.
        target_degree: Exact total degree to enumerate.
        vertex_offset: Per-vertex degree contribution (+1 bar, 0 free, −1 cobar).
        use_planar_decs: When True, restrict each vertex decoration to planar
            basis elements via ``planar_basis_iter``.
    """
    n = len(leaf_range)
    if n == 0:
        return
    if n == 1:
        if target_degree == 0:
            yield leaf_range[0]
        return
    if max_weight < 1:
        return

    connectivity = getattr(operad_cls, "connectivity", 0)
    _dec_iter = (
        _planar_operad_basis_keys_in_degree if use_planar_decs else _operad_basis_keys_in_degree
    )

    for v_arity in range(2, n + 1):
        root_parent = operad_cls(v_arity, base_ring)
        min_root_dec_deg = connectivity * (v_arity - 1)
        if v_arity == n:
            # Corolla: all leaves are direct children
            root_dec_deg = target_degree - vertex_offset
            if root_dec_deg >= min_root_dec_deg:
                for root_dec in _dec_iter(root_parent, root_dec_deg):
                    yield RootedTree(root_dec, *leaf_range)
        else:
            # Split leaf_range into v_arity consecutive sub-ranges
            for parts in _consecutive_parts_iter(leaf_range, v_arity):
                min_child_total = sum(
                    _min_subtree_degree_generic(len(p), connectivity, vertex_offset)
                    for p in parts
                    if len(p) >= 2
                )
                max_root_dec_deg = target_degree - vertex_offset - min_child_total
                for root_dec_deg in range(min_root_dec_deg, max_root_dec_deg + 1):
                    child_total = target_degree - root_dec_deg - vertex_offset
                    for root_dec in _dec_iter(root_parent, root_dec_deg):
                        yield from _planar_children_iter_generic(
                            parts,
                            max_weight - 1,
                            operad_cls,
                            base_ring,
                            child_total,
                            root_dec,
                            vertex_offset,
                            use_planar_decs,
                        )


def enumerate_planar_trees_generic_in_degree(
    arity: int,
    weight_bound: int,
    operad_cls: Any,
    base_ring: Any,
    target_degree: int,
    vertex_offset: int,
    use_planar_decs: bool = True,
) -> Iterator[tuple]:
    """Enumerate DFS-canonical planar trees with an arbitrary per-vertex degree offset.

    A *planar tree* in the DFS-canonical sense has leaves ``1, …, n`` in strict
    left-to-right order and every child subtree occupies a contiguous block of
    leaf labels.  Together with planar vertex decorations (one per ``S_k``-orbit),
    these trees form a basis for the *planar part* ``P_pl(n)`` of a quasi-planar
    symmetric sequence ``P``.

    Unlike :func:`enumerate_shuffle_trees_generic_in_degree`, which generates
    *all* shuffle trees (and thus over-counts orbits when the canonical form of
    ``P(n)`` under ``planarize`` is DFS-canonical rather than shuffle), this
    function yields exactly one representative per ``S_n``-orbit, matching the
    output of ``P.Component.planarize(T)`` when ``sigma_global == id``.

    The three canonical choices for ``vertex_offset`` are:

    - ``+1``: bar degree ``Σ (deg_P(v) + 1)``.
    - ``0``:  free degree ``Σ deg_P(v)`` (no suspension).
    - ``-1``: cobar degree ``Σ (deg_C(v) − 1)``.

    Args:
        arity: Number of leaves.
        weight_bound: Maximum number of internal vertices.
        operad_cls: Operad or cooperad factory for vertex decorations.
        base_ring: Coefficient ring.
        target_degree: Exact total degree to enumerate.
        vertex_offset: Per-vertex degree contribution.
        use_planar_decs: When ``True`` (default), restrict vertex decorations
            to the planar basis via ``planar_basis_iter``.

    Yields:
        Decorated planar trees (nested tuples) as valid tree basis keys.
    """
    if arity < 2 or weight_bound < 1:
        return
    yield from _planar_subtrees_iter_generic(
        tuple(range(1, arity + 1)),
        weight_bound,
        operad_cls,
        base_ring,
        target_degree,
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
            planar basis (via ``planar_basis_iter``) instead of the full basis.
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

    Returns a list of ``(shuffle_tree, coeff)`` pairs representing a
    (possibly multi-term) linear combination.

    For the implemented cobar grading, subtree degrees are computed by
    ``subtree_degree_cobar`` (sum of ``deg_C(v) - 1`` over internal vertices).
    """
    if isinstance(tree, int):
        return [(tree, 1)]

    kids = tree._children
    dec = tree._decoration
    k = len(kids)

    # Recursively normalize children first.
    from itertools import product as iter_product

    child_term_lists: list[list[tuple]] = []
    for c in kids:
        if isinstance(c, RootedTree):
            child_term_lists.append(to_shuffle_tree_cobar(c, cooperad_cls, base_ring))
        else:
            child_term_lists.append([(c, 1)])

    results = []
    for combo in iter_product(*child_term_lists):
        normalized_kids = [item[0] for item in combo]
        child_coeff = 1
        for item in combo:
            child_coeff *= item[1]

        min_leaves = [min_leaf(c) for c in normalized_kids]
        cobar_degrees = [subtree_degree_cobar(c, cooperad_cls, base_ring) for c in normalized_kids]

        indexed = list(zip(min_leaves, range(k), normalized_kids, cobar_degrees))
        indexed.sort(key=lambda x: x[0])

        perm = [item[1] for item in indexed]

        if perm == list(range(k)):
            new_tree = RootedTree(dec, *normalized_kids)
            results.append((new_tree, child_coeff))
            continue

        koszul_sign = koszul_sign_of_permutation(perm, cobar_degrees)

        sigma_list = [p + 1 for p in perm]
        sigma_inv = [0] * k
        for new_pos, old_pos_plus_one in enumerate(sigma_list):
            sigma_inv[old_pos_plus_one - 1] = new_pos + 1

        cooperad_parent = cooperad_cls(k, base_ring)
        dec_elem = cooperad_parent(dec)
        permuted_dec_elem = dec_elem.permute(sigma_inv)

        sorted_kids = tuple(item[2] for item in indexed)
        for new_dec_key, dec_coeff in permuted_dec_elem:
            new_tree = RootedTree(new_dec_key, *sorted_kids)
            total_coeff = child_coeff * koszul_sign * dec_coeff
            results.append((new_tree, total_coeff))

    return results
