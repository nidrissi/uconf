"""Cofree conilpotent C-coalgebra on a dg-module M.

The cofree conilpotent C-coalgebra on a dg-module M is

    T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(tree, m_tuple) = Σ_{v internal} deg_C(dec(v)) + Σ_i deg_M(m_i)
  (no desuspension; compare with the cobar construction which uses deg_C - 1).
- Differential d = d_C + d_M from the Koszul sign rule on the interleaved
  DFS pre-order of vertices and leaves.
- C-coalgebra costructure: the infinitesimal cocomposition Δ^{i;m,n} splits
  a tree at the unique internal vertex whose subtree leaves are {i,...,i+n-1},
  mirroring the BarConstruction infinitesimal cocomposition but also splitting
  the M-label tuple accordingly.

The basis keys are pairs ``(tree, m_tuple)`` where:

- ``tree`` is an integer leaf (= 1 in arity 1) or a tuple representing a
  decorated rooted tree with leaves labeled ``1, ..., n``.
- ``m_tuple`` is a tuple of ``n`` basis keys of the inner module M.

The coprojection π: T^c_C(M) → M kills all elements of weight ≥ 2 and sends
(1, (m,)) ↦ m.

Reference: Loday-Vallette "Algebraic Operads", Section 5.8.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, tensor

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import (
    _dfs_all_iter,
    _module_basis_keys_in_degree,
    _tuples_in_degree,
)
from uconf.core.cooperad import CooperadLike
from uconf.core.signs import sign_from_exponent
from uconf.core.vertex_decorated import VertexDecoratedLike
from uconf.core.trees import (
    children,
    decoration,
    enumerate_shuffle_trees_generic_in_degree,
    is_leaf,
    leaves,
    relabel_leaves,
    split_at_vertex,
    tree_arity,
    vertex_arity,
    vertices_dfs,
)


class CofreeCoalgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the cofree conilpotent C-coalgebra ``T^c_C(M)``.

    Basis keys are ``(tree, m_tuple)`` pairs.  The differential is
    ``d = d_C + d_M`` using the DFS-interleaved Koszul sign rule.

    This class is normally not instantiated directly; use
    :class:`CofreeConilpotentCoalgebra` instead.
    """

    name: ClassVar[str] = "T^c"

    def __init__(
        self,
        cooperad_cls: VertexDecoratedLike,
        inner_module,
        base_ring,
        *,
        vertex_degree_shift: int = 0,
        name: str | None = None,
    ):
        """Initialize the cofree conilpotent C-coalgebra module ``T^c_C(M)``.


        Args:
            cooperad_cls: Arity-indexed vertex-decoration provider used on
                internal vertices (typically a cooperad, but may be any object
                matching the shared structural protocol).
            inner_module: Cogenerating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring.
            vertex_degree_shift: Per-vertex degree offset (0 = standard cofree,
                +1 = bar/suspension convention, -1 = cobar/desuspension).
            name: Display name override.  Defaults to ``T^c_C(M)``.

        """
        self._cooperad_cls = cooperad_cls
        self._inner_module = inner_module
        self._vertex_degree_shift = vertex_degree_shift

        if name is None:
            name = f"T^c_{cooperad_cls.name}({inner_module})"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)

        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Validation and element construction
    # -----------------------------------------------------------------------

    def _validate_basis_key(self, key):
        """Validate a ``(tree, m_tuple)`` basis key.

        Returns the normalized key as a ``(tree, tuple(m_tuple))`` pair, or
        ``None`` if the key is structurally invalid or contains invalid
        inner-module keys.
        """
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        tree, m_tuple = key[0], key[1]
        if not isinstance(m_tuple, (tuple, list)):
            return None

        if is_leaf(tree):
            if tree != 1 or len(m_tuple) != 1:
                return None
            m_key = self._validate_m_key(m_tuple[0])
            if m_key is None:
                return None
            return (1, (m_key,))

        n = tree_arity(tree)
        if len(m_tuple) != n:
            return None
        new_m = []
        for m_key in m_tuple:
            vk = self._validate_m_key(m_key)
            if vk is None:
                return None
            new_m.append(vk)
        return (tree, tuple(new_m))

    def _validate_m_key(self, m_key):
        """Validate one inner-module basis key.

        Delegates to ``inner_module._validate_basis_key`` if available;
        otherwise returns ``m_key`` unchanged.
        """
        if hasattr(self._inner_module, "_validate_basis_key"):
            return self._inner_module._validate_basis_key(m_key)
        return m_key

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return self.sum_of_terms(clean.items())

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)

        return super()._element_constructor_(x)

    # -----------------------------------------------------------------------
    # Degree
    # -----------------------------------------------------------------------

    def degree_on_basis(self, key) -> int:
        """Degree = Σ_{v internal} (deg_C(dec(v)) + shift) + Σ_i deg_M(m_i).

        When ``vertex_degree_shift`` is 0 (default), vertices contribute their
        C-degree directly.  With shift +1 this gives the bar convention
        ``Σ (deg_P + 1)``; with shift -1 the cobar convention ``Σ (deg_C - 1)``.
        """
        tree, m_tuple = key
        shift = self._vertex_degree_shift
        v_deg = (
            0
            if is_leaf(tree)
            else sum(
                self._cooperad_cls(vertex_arity(v), self.base_ring()).degree_on_basis(decoration(v))
                + shift
                for v in vertices_dfs(tree)
            )
        )
        m_deg = sum(self._inner_module.degree_on_basis(m) for m in m_tuple)
        return v_deg + m_deg

    # -----------------------------------------------------------------------
    # Basis iteration
    # -----------------------------------------------------------------------

    def basis_it(self, d: int) -> Iterator["CofreeCoalgebraModule.Element"]:
        """Iterate over basis elements of degree *d*.

        Yields all ``(tree, m_tuple)`` pairs with total degree ``d``, where
        ``tree`` is a shuffle tree decorated by the cooperad *C* and
        ``m_tuple`` is a tuple of basis keys of the inner module *M*.

        The same arity-bounding logic as :meth:`FreeAlgebraModule.basis_it`
        applies (see that method for details).  In particular, when both the
        inner module and cooperad admit degree-0 generators, exhaustive
        fixed-degree enumeration is not guaranteed and this method raises
        ``ValueError``.

        Args:
            d: Homological degree to enumerate.

        Yields:
            Elements of this module with degree ``d``.
        """
        M = self._inner_module
        C = self._cooperad_cls
        R = self.base_ring()

        # Pre-collect M-keys by degree from 0 to d.
        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(d + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        # Arity 1: single leaf
        for m_key in m_keys_by_deg.get(d, []):
            yield self.term((1, (m_key,)))

        if not m_keys_by_deg:
            return

        # Arity n ≥ 2: determine upper arity bound.
        # Any n≥2 tree has tree degree ≥ vertex_degree_shift + connectivity
        # (a single corolla of arity 2 with minimum decoration degree).
        min_m_deg = min(m_keys_by_deg.keys())
        connectivity = getattr(C, "connectivity", 0)
        min_tree_deg_n2 = self._vertex_degree_shift + connectivity

        if d < min_tree_deg_n2:
            # Total degree can't accommodate even a single vertex → only
            # single-leaf elements exist (already yielded above).
            return

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = (d - self._vertex_degree_shift) // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_it(d): both the inner module "
                "and cooperad admit degree-0 generators (min_deg=0, connectivity=0), "
                "so arity is unbounded in fixed degree."
            )

        for n in range(2, max_n + 1):
            max_weight = n - 1
            for d_M in range(d + 1):
                d_tree = d - d_M
                if d_tree < 0:
                    continue
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_M))
                if not m_tuples:
                    continue
                for tree in enumerate_shuffle_trees_generic_in_degree(
                    n, max_weight, C, R, d_tree, self._vertex_degree_shift
                ):
                    for m_tuple in m_tuples:
                        yield self.term((tree, m_tuple))

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "CofreeCoalgebraModule.Element":
        """Differential d = d_C + d_M using the interleaved DFS sign rule.

        For each node in DFS pre-order (vertices and leaves interleaved),
        the sign when applying ∂ at that node is

            ``(-1)^{Σ_{l before this node in DFS all order} deg(x_l)}``

        where ``deg(x_l) = deg_C(dec(v)) + shift`` for a vertex and
        ``deg_M(m)`` for a leaf.
        """
        tree, m_tuple = key
        result = self.zero()
        base_ring = self.base_ring()
        shift = self._vertex_degree_shift
        cumulative = 0

        for node, leaf_0idx in _dfs_all_iter(tree):
            sign = sign_from_exponent(cumulative)

            if leaf_0idx is not None:
                # d_M: apply ∂_M to this leaf
                m_key = m_tuple[leaf_0idx]
                m_elem = self._inner_module.term(m_key)
                bdry = self._inner_module.boundary(m_elem)
                for new_m_key, coeff in bdry:
                    new_m = m_tuple[:leaf_0idx] + (new_m_key,) + m_tuple[leaf_0idx + 1 :]
                    result += sign * coeff * self.term((tree, new_m))
                cumulative += self._inner_module.degree_on_basis(m_key)
            else:
                # d_C: apply ∂_C to this vertex
                v_arity = vertex_arity(node)
                dec = decoration(node)
                coop_parent = self._cooperad_cls(v_arity, base_ring)
                coop_elem: Any = coop_parent.term(dec)
                bdry = coop_parent.boundary(coop_elem)
                for new_dec, coeff in bdry:
                    new_tree = self._replace_dec(tree, node, new_dec)
                    result += sign * coeff * self.term((new_tree, m_tuple))
                cumulative += coop_parent.degree_on_basis(dec) + shift

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _replace_dec(self, tree, target_vertex, new_dec):
        """Replace the decoration of ``target_vertex`` in ``tree``.

        Returns a new tree identical to ``tree`` except that the decoration
        tuple of ``target_vertex`` is replaced by ``new_dec``.
        """
        if is_leaf(tree):
            return tree
        if tree is target_vertex:
            return (new_dec,) + children(tree)
        new_children = tuple(self._replace_dec(c, target_vertex, new_dec) for c in children(tree))
        return (decoration(tree),) + new_children

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the cofree conilpotent C-coalgebra module."""

        def boundary(self) -> "CofreeCoalgebraModule.Element":
            """Apply the differential d = d_C + d_M."""
            return self.parent().boundary(self)


class CofreeConilpotentCoalgebra(CooperadCoalgebra):
    """Cofree conilpotent C-coalgebra on a dg-module M.

    Constructs ``T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}`` as a
    :class:`CofreeCoalgebraModule` and equips it with the canonical
    C-coalgebra infinitesimal cocomposition given by tree-splitting.

    Args:
        cooperad_cls: Cooperad provider C (class or wrapper instance).
        inner_module: The cogenerating dg-module M.
        base_ring: Coefficient ring

    The coprojection ``π: T^c_C(M) → M`` is given by ``project()``.

    The coaction ``δ_k`` is implemented via ``coact(v_elem, k)`` which returns
    an element of ``C(k) ⊗ T^c_C(M)^{⊗k}``.  The infinitesimal cocomposition
    ``Δ^{i;m,n}`` is accessible via ``infinitesimal_cocompose(x, i, m, n)``.

    Examples::

        cofree_coass = CofreeConilpotentCoalgebra(CoAssociative, module_M)
        # Coaction on a binary-tree element:
        elem = cofree_coass.module.term(((1,2), 1, 2), (m1, m2))
        cofree_coass.coact(elem, 2)   # splits at root

    """

    def __init__(self, cooperad_cls: CooperadLike, inner_module, base_ring):
        cofree_module = CofreeCoalgebraModule(cooperad_cls, inner_module, base_ring)
        super().__init__(cofree_module, cooperad_cls, self._coact_impl)
        self._inner_module = inner_module

    def _coact_impl(self, v_element, n: int):
        """C-coalgebra coaction δ_n: split each tree at its root when root arity = n.

        For each basis element ``(tree, m_tuple)`` with ``vertex_arity(root) == n``:

            δ_n((tree, m_tuple)) = root_dec ⊗ (subtree_1, m_1) ⊗ … ⊗ (subtree_n, m_n)

        where ``root_dec ∈ C(n)`` is the root decoration, ``subtree_j`` are the n
        root children (relabeled to leaves ``{1, …, n_j}``), and ``m_j`` is the
        sub-tuple of M-labels for the leaves of ``subtree_j``.

        For a single leaf ``(1, (m,))``: δ_n = 0 for all n (no root arity to split).

        Returns an element of ``C(n) ⊗ T^c_C(M)^{⊗n}`` (a Sage tensor module element).
        The basis keys are ``(c_key, cofree_key_1, ..., cofree_key_n)`` as required by
        :meth:`uconf.coalgebra_cobar.CobarComplexCoalgebra._dcoact_on_basis`.
        """
        base_ring = self.module.base_ring()
        coop_parent = self.cooperad_cls(n, base_ring)
        cofree_mod = self.module  # CofreeCoalgebraModule

        right_factors = [cofree_mod] * n
        target = tensor([coop_parent] + right_factors)
        result = target.zero()

        for (tree, m_tuple), v_coeff in v_element:
            if is_leaf(tree):
                continue  # no root vertex, δ_n = 0
            if vertex_arity(tree) != n:
                continue  # root arity doesn't match

            root_dec = decoration(tree)
            root_children = children(tree)

            # Split m_tuple among the n children according to leaf membership
            child_leaf_lists = []
            for child in root_children:
                child_leaves = sorted(leaves(child) if not is_leaf(child) else {child})
                child_leaf_lists.append(child_leaves)

            # Relabel each child to have leaves {1, ..., n_j}
            relabeled_children = []
            child_m_tuples = []
            for j, (child, child_leaves) in enumerate(zip(root_children, child_leaf_lists)):
                relabel = {old: new for new, old in enumerate(child_leaves, start=1)}
                if is_leaf(child):
                    relabeled_children.append(1)
                else:
                    relabeled_children.append(relabel_leaves(child, relabel))
                # m-labels for this child (by original leaf order)
                child_m_tuples.append(tuple(m_tuple[l - 1] for l in child_leaves))

            # Build tensor product term: c_key ⊗ (sub_1, m_1) ⊗ ... ⊗ (sub_n, m_n)
            coop_elem = coop_parent.term(root_dec)
            term = coop_elem
            for j in range(n):
                sub_key = (relabeled_children[j], child_m_tuples[j])
                term = tensor([term, cofree_mod.term(sub_key)])

            result += v_coeff * term

        return result

    def infinitesimal_cocompose(self, x, i: int, m: int, n: int):
        """Δ^{i;m,n}: T^c_C(M)(m+n-1) → T^c_C(M)(m) ⊗ T^c_C(M)(n).

        Splits each tree at every internal vertex whose subtree leaves are
        exactly ``{i, i+1, ..., i+n-1}``, then relabels the two resulting
        subtrees to have leaves ``{1,...,m}`` and ``{1,...,n}`` respectively.
        The M-label tuple is split accordingly.

        This mirrors the cocomposition in
        :class:`uconf.bar_construction.BarConstruction.Component`, extended to
        carry the M-label tuple.

        Args:
            x: An element of the cofree module.
            i: Starting position of the right factor's leaves (1-indexed).
            m: Arity of the left factor.
            n: Arity of the right factor.

        Returns:
            An element of ``T^c_C(M)(m) ⊗ T^c_C(M)(n)``.

        """
        if m <= 0 or n <= 0:
            raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")

        cofree_mod = self.module
        left_parent = cofree_mod
        right_parent = cofree_mod
        target = tensor([left_parent, right_parent])
        result = target.zero()

        target_leaves = set(range(i, i + n))

        for (tree, m_tuple), coeff in x:
            for vertex in vertices_dfs(tree):
                if leaves(vertex) != target_leaves:
                    continue
                split = split_at_vertex(tree, vertex)
                if split is None:
                    continue

                tree_top, placeholder, tree_bot = split

                # Build top relabeling: {original leaves} → {1,...,m}
                top_relabel: dict[int, int] = {}
                for leaf in leaves(tree_top):
                    if leaf < i:
                        top_relabel[leaf] = leaf
                    elif leaf == placeholder:
                        top_relabel[leaf] = i
                    else:
                        top_relabel[leaf] = leaf - n + 1

                # Build bot relabeling: {i,...,i+n-1} → {1,...,n}
                bot_relabel = {leaf: leaf - i + 1 for leaf in target_leaves}

                relabeled_top = relabel_leaves(tree_top, top_relabel)
                relabeled_bot = relabel_leaves(tree_bot, bot_relabel)

                # Split m_tuple: top gets all leaves except target, bot gets target
                top_relabel_inv = {new: old for old, new in top_relabel.items()}
                bot_relabel_inv = {new: old for old, new in bot_relabel.items()}
                top_m = tuple(m_tuple[top_relabel_inv[j] - 1] for j in range(1, m + 1))
                bot_m = tuple(m_tuple[bot_relabel_inv[j] - 1] for j in range(1, n + 1))

                # Validate both parts
                top_key = (relabeled_top, top_m)
                bot_key = (relabeled_bot, bot_m)
                if cofree_mod._validate_basis_key(top_key) is None:
                    continue
                if cofree_mod._validate_basis_key(bot_key) is None:
                    continue

                result += coeff * cofree_mod.term(top_key).tensor(cofree_mod.term(bot_key))

        return result

    def project(self, x):
        """Coprojection π: T^c_C(M) → M, projecting onto weight-1 generators.

        Returns the image in the inner module M.  Non-zero only for elements
        of the form ``(1, (m_key,))`` (single leaf, weight 0).

        Args:
            x: An element of the cofree module.

        Returns:
            An element of the inner module M.

        """
        inner = self._inner_module
        result = inner.zero()
        for (tree, m_tuple), coeff in x:
            if is_leaf(tree) and len(m_tuple) == 1:
                result += coeff * inner.term(m_tuple[0])
        return result
