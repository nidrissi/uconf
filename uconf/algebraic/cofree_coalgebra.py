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

from sage.all import CombinatorialFreeModule, tensor

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.tree_module import TreeModule
from uconf.core.cooperad import CooperadLike
from uconf.core.vertex_decorated import VertexDecoratedLike
from uconf.core.trees import (
    children,
    decoration,
    is_leaf,
    leaves,
    relabel_leaves,
    split_at_vertex,
    vertex_arity,
    vertices_dfs,
)


class CofreeCoalgebraModule(TreeModule):
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
        inner_module: CombinatorialFreeModule,
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
        if name is None:
            name = f"T^c_{cooperad_cls.name}({inner_module})"
        super().__init__(
            symmetric_sequence_cls=cooperad_cls,
            inner_module=inner_module,
            base_ring=base_ring,
            vertex_degree_shift=vertex_degree_shift,
            name=name,
        )
        # Backward-compatible alias expected by subclasses and callers.
        self._cooperad_cls = cooperad_cls


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
