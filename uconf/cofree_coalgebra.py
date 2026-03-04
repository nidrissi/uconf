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

from typing import ClassVar

from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis, tensor

from .coalgebra import CooperadCoalgebra
from .free_algebra import _dfs_all_iter
from .signs import sign_from_exponent
from .trees import (
    children,
    decoration,
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

    def __init__(self, cooperad_cls, inner_module, base_ring=QQ):
        """Initialize the cofree conilpotent C-coalgebra module ``T^c_C(M)``.

        Args:
            cooperad_cls: Cooperad class C (e.g. ``CoAssociative``, ``CoCommutative``).
            inner_module: Cogenerating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring (default ``QQ``).
        """
        self._cooperad_cls = cooperad_cls
        self._inner_module = inner_module

        name = f"T^c_{cooperad_cls.name}({inner_module})"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)

        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

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
        """Degree = Σ_{v internal} deg_C(dec(v)) + Σ_i deg_M(m_i).

        No desuspension: vertices contribute their C-degree directly, not
        deg_C - 1 as in the cobar construction.
        """
        tree, m_tuple = key
        v_deg = (
            0
            if is_leaf(tree)
            else sum(
                self._cooperad_cls(
                    vertex_arity(v), self.base_ring()
                ).degree_on_basis(decoration(v))
                for v in vertices_dfs(tree)
            )
        )
        m_deg = sum(self._inner_module.degree_on_basis(m) for m in m_tuple)
        return v_deg + m_deg

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "CofreeCoalgebraModule.Element":
        """Differential d = d_C + d_M using the interleaved DFS sign rule.

        For each node in DFS pre-order (vertices and leaves interleaved),
        the sign when applying ∂ at that node is

            ``(-1)^{Σ_{l before this node in DFS all order} deg(x_l)}``

        where ``deg(x_l) = deg_C(dec(v))`` for a vertex and ``deg_M(m)`` for a leaf.
        """
        tree, m_tuple = key
        result = self.zero()
        base_ring = self.base_ring()
        cumulative = 0

        for node, is_leaf_flag, leaf_0idx in _dfs_all_iter(tree):
            sign = sign_from_exponent(cumulative)

            if is_leaf_flag:
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
                bdry = coop_parent.boundary(coop_parent.term(dec))
                for new_dec, coeff in bdry:
                    new_tree = self._replace_dec(tree, node, new_dec)
                    result += sign * coeff * self.term((new_tree, m_tuple))
                cumulative += coop_parent.degree_on_basis(dec)

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
        new_children = tuple(
            self._replace_dec(c, target_vertex, new_dec) for c in children(tree)
        )
        return (decoration(tree),) + new_children

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the cofree conilpotent C-coalgebra module."""

        def boundary(self) -> "CofreeCoalgebraModule.Element":
            """Apply the differential d = d_C + d_M."""
            return self.parent().boundary(self)


CofreeCoalgebraModule.Element = CofreeCoalgebraModule.Element


class CofreeConilpotentCoalgebra(CooperadCoalgebra):
    """Cofree conilpotent C-coalgebra on a dg-module M.

    Constructs ``T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}`` as a
    :class:`CofreeCoalgebraModule` and equips it with the canonical
    C-coalgebra infinitesimal cocomposition given by tree-splitting.

    Args:
        cooperad_cls: Cooperad class C (e.g. ``CoAssociative``, ``CoCommutative``).
        inner_module: The cogenerating dg-module M.
        base_ring: Coefficient ring (default ``QQ``).

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

    def __init__(self, cooperad_cls, inner_module, base_ring=QQ):
        cofree_module = CofreeCoalgebraModule(cooperad_cls, inner_module, base_ring)
        # Pass a placeholder costructure_map; we override coact() directly.
        super().__init__(cofree_module, cooperad_cls, costructure_map=None)
        self._inner_module = inner_module

    def coact(self, v_element, k: int):
        """C-coalgebra coaction δ_k: split each tree at its root when root arity = k.

        For each basis element ``(tree, m_tuple)`` with ``vertex_arity(root) == k``:

            δ_k((tree, m_tuple)) = root_dec ⊗ (subtree_1, m_1) ⊗ … ⊗ (subtree_k, m_k)

        where ``root_dec ∈ C(k)`` is the root decoration, ``subtree_j`` are the k
        root children (relabeled to leaves ``{1, …, n_j}``), and ``m_j`` is the
        sub-tuple of M-labels for the leaves of ``subtree_j``.

        For a single leaf ``(1, (m,))``: δ_k = 0 for all k (no root arity to split).

        Returns an element of ``C(k) ⊗ T^c_C(M)^{⊗k}`` (a Sage tensor module element).
        The basis keys are ``(c_key, cofree_key_1, ..., cofree_key_k)`` as required by
        :meth:`uconf.coalgebra_cobar.CobarComplexCoalgebra._dcoact_on_basis`.
        """
        base_ring = self.module.base_ring()
        coop_parent = self.cooperad_cls(k, base_ring)
        cofree_mod = self.module  # CofreeCoalgebraModule

        right_factors = [cofree_mod] * k
        target = tensor([coop_parent] + right_factors)
        result = target.zero()

        for (tree, m_tuple), v_coeff in v_element:
            if is_leaf(tree):
                continue  # no root vertex, δ_k = 0
            if vertex_arity(tree) != k:
                continue  # root arity doesn't match

            root_dec = decoration(tree)
            root_children = children(tree)

            # Split m_tuple among the k children according to leaf membership
            child_leaf_lists = []
            for child in root_children:
                child_leaves = sorted(leaves(child) if not is_leaf(child) else {child})
                child_leaf_lists.append(child_leaves)

            # Relabel each child to have leaves {1, ..., n_j}
            relabeled_children = []
            child_m_tuples = []
            for j, (child, child_leaves) in enumerate(
                zip(root_children, child_leaf_lists)
            ):
                n_j = len(child_leaves)
                relabel = {old: new for new, old in enumerate(child_leaves, start=1)}
                if is_leaf(child):
                    relabeled_children.append(1)
                else:
                    relabeled_children.append(relabel_leaves(child, relabel))
                # m-labels for this child (by original leaf order)
                child_m_tuples.append(tuple(m_tuple[l - 1] for l in child_leaves))

            # Build tensor product term: c_key ⊗ (sub_1, m_1) ⊗ ... ⊗ (sub_k, m_k)
            coop_elem = coop_parent.term(root_dec)
            term = coop_elem
            for j in range(k):
                sub_key = (relabeled_children[j], child_m_tuples[j])
                term = term.tensor(cofree_mod.term(sub_key))

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

                result += coeff * cofree_mod.term(top_key).tensor(
                    cofree_mod.term(bot_key)
                )

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
