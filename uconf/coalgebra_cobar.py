"""Cobar complex Ω_C(V) of a C-coalgebra V.

For a connected coaugmented dg-cooperad C and a C-coalgebra (V, δ), the cobar
complex is the dg-module

    Ω_C(V) = (Ω(C) ∘' V, d)

where Ω(C) ∘' V is the left composite product and d has four components:

- d_1  : apply ∂_C to each internal vertex decoration.
- d_2  : structural cobar differential (expand internal vertices via cooperad
         cocomposition).
- d_V  : apply ∂_V to each leaf decoration (Koszul sign from tensor product rule).
- d_coact: apply the C-coalgebra coaction δ at each leaf (sign from vertex-degree
           accumulation in DFS order).

The basis elements are pairs ``(tree, v_tuple)`` where:

- ``tree`` is a cobar-construction tree in ``Ω(C)(n)`` (integer leaf or decorated
  tuple as in :class:`uconf.cobar_construction.CobarConstruction`).
- ``v_tuple`` is a tuple of ``n`` basis keys of the module V, one per leaf.

The homological degree is

    deg(tree, v_tuple) = deg_cobar(tree) + Σ_i deg_V(v_tuple[i])

where ``deg_cobar(tree) = Σ_{v internal} (deg_C(dec(v)) - 1)``.

Reference: Loday-Vallette "Algebraic Operads", Section 11.2.
"""

from __future__ import annotations

from typing import ClassVar

from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis

from .coalgebra import CooperadCoalgebra
from .signs import sign_from_exponent
from .trees import (
    children,
    decoration,
    expand_vertex,
    is_leaf,
    subtree_degree_cobar,
    tree_arity,
    vertex_arity,
    vertices_dfs,
)


class CobarComplexCoalgebra(CombinatorialFreeModule):
    """Cobar complex Ω_C(V) of a C-coalgebra V.

    Args:
        coalgebra: A :class:`uconf.coalgebra.CooperadCoalgebra` instance.
        base_ring: Coefficient ring (default ``QQ``).

    Basis keys are pairs ``(tree, v_tuple)`` where:

    - ``tree`` is an integer (single leaf, arity 1) or a tuple representing a
      decorated rooted tree with leaves labeled ``1, ..., n``.
    - ``v_tuple`` is a tuple of ``n`` basis keys of the underlying module V.

    The degree is ``deg_cobar(tree) + Σ_i deg_V(v_tuple[i])``.
    """

    name: ClassVar[str] = "Ω"

    def __init__(self, coalgebra: CooperadCoalgebra, base_ring=QQ):
        self._coalgebra = coalgebra
        self._cooperad_cls = coalgebra.cooperad_cls
        self._module = coalgebra.module

        name = f"Ω({coalgebra.cooperad_cls.name}; {coalgebra.module})"
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
        self._d1 = self.module_morphism(on_basis=self._d1_on_basis, codomain=self)
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dV = self.module_morphism(on_basis=self._dV_on_basis, codomain=self)
        self._dcoact = self.module_morphism(
            on_basis=self._dcoact_on_basis, codomain=self
        )

    # -----------------------------------------------------------------------
    # Basis key validation and element construction
    # -----------------------------------------------------------------------

    def _validate_basis_key(self, key):
        """Validate a basis key ``(tree, v_tuple)``."""
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        tree, v_tuple = key[0], key[1]
        if not isinstance(v_tuple, (tuple, list)):
            return None

        if is_leaf(tree):
            if tree != 1 or len(v_tuple) != 1:
                return None
            v_key = self._validate_v_key(v_tuple[0])
            if v_key is None:
                return None
            return (1, (v_key,))

        n = tree_arity(tree)
        if len(v_tuple) != n:
            return None
        new_v = []
        for v_key in v_tuple:
            validated = self._validate_v_key(v_key)
            if validated is None:
                return None
            new_v.append(validated)
        return (tree, tuple(new_v))

    def _validate_v_key(self, v_key):
        """Validate one V-module basis key."""
        if hasattr(self._module, "_validate_basis_key"):
            return self._module._validate_basis_key(v_key)
        return v_key

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return super()._element_constructor_(clean)

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)

        return super()._element_constructor_(x)

    # -----------------------------------------------------------------------
    # Grading
    # -----------------------------------------------------------------------

    def degree_on_basis(self, key) -> int:
        """Degree = deg_cobar(tree) + Σ_i deg_V(v_i)."""
        tree, v_tuple = key
        tree_deg = (
            0
            if is_leaf(tree)
            else subtree_degree_cobar(tree, self._cooperad_cls, self.base_ring())
        )
        v_deg = sum(self._module.degree_on_basis(v) for v in v_tuple)
        return tree_deg + v_deg

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Total differential d = d_1 + d_2 + d_V + d_coact."""
        return (
            self._d1_on_basis(key)
            + self._d2_on_basis(key)
            + self._dV_on_basis(key)
            + self._dcoact_on_basis(key)
        )

    def _d1_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Apply ∂_C to each internal vertex decoration.

        Sign at DFS vertex v_j: ``(-1)^{Σ_{l<j} (deg_C(dec(v_l)) - 1)}``.
        """
        tree, v_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        verts = vertices_dfs(tree)
        base_ring = self.base_ring()
        cumulative = 0

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            coop_parent = self._cooperad_cls(v_arity, base_ring)
            vertex_sinv_deg = coop_parent.degree_on_basis(dec) - 1
            sign = sign_from_exponent(cumulative)

            bdry = coop_parent.boundary(coop_parent.term(dec))
            for new_dec, coeff in bdry:
                new_tree = self._replace_decoration(tree, v, new_dec)
                result += sign * coeff * self.term((new_tree, v_tuple))

            cumulative += vertex_sinv_deg

        return result

    def _d2_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Structural differential: expand each vertex via cooperad cocomposition.

        Sign at vertex v with split ``Δ^{i;m,n}(dec(v)) = Σ c_L ⊗ c_R``:

            ``(-1)^{global_accum + deg_C(c_L) + (deg_C(c_R) - 1) * before_deg}``

        where ``global_accum = Σ_{v' before v in DFS} (deg_C(v') - 1)`` and
        ``before_deg = Σ_{j<i} deg_cobar(subtree_j of v)``.

        This is the same sign convention as
        :meth:`uconf.cobar_construction.CobarConstruction.Component._d2_on_basis`.
        The V-leaf decorations are unaffected (leaf labels shift but order is
        preserved).
        """
        tree, v_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)

        for curr_vertex in verts:
            curr_arity = vertex_arity(curr_vertex)
            curr_dec = decoration(curr_vertex)
            curr_parent = self._cooperad_cls(curr_arity, base_ring)
            curr_elem = curr_parent.term(curr_dec)

            # Global accumulation: deg_C(v) - 1 for DFS vertices before curr
            global_accum = 0
            for v in verts:
                if v is curr_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = self._cooperad_cls(v_arity, base_ring).degree_on_basis(
                    decoration(v)
                )
                global_accum += v_deg - 1

            for m in range(2, curr_arity):
                n_right = curr_arity - m + 1
                for i in range(1, m + 1):
                    cocomp = curr_parent.infinitesimal_cocompose(
                        curr_elem, i, m, n_right
                    )
                    for (dec_left, dec_right), coeff in cocomp:
                        right_parent = self._cooperad_cls(n_right, base_ring)
                        left_parent = self._cooperad_cls(m, base_ring)
                        right_sinv_deg = right_parent.degree_on_basis(dec_right) - 1
                        left_degree = left_parent.degree_on_basis(dec_left)

                        before_deg = sum(
                            subtree_degree_cobar(ch, self._cooperad_cls, base_ring)
                            for j, ch in enumerate(children(curr_vertex), start=1)
                            if j < i
                        )
                        koszul_exp = right_sinv_deg * before_deg
                        total_sign = sign_from_exponent(
                            global_accum + left_degree + koszul_exp
                        )

                        new_tree = expand_vertex(
                            tree, curr_vertex, i, dec_left, dec_right, m, n_right
                        )
                        # v_tuple is unchanged (leaf labels shift but order preserved)
                        result += total_sign * coeff * self.term((new_tree, v_tuple))

        return result

    def _dV_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Apply ∂_V to each leaf decoration.

        Sign at leaf with label ``i`` (1-indexed):

            ``(-1)^{deg_cobar(tree) + Σ_{j < i} deg_V(v_j)}``
        """
        tree, v_tuple = key
        deg_cobar = (
            0
            if is_leaf(tree)
            else subtree_degree_cobar(tree, self._cooperad_cls, self.base_ring())
        )

        result = self.zero()
        cumulative_v = 0

        for i in range(1, len(v_tuple) + 1):
            v_key = v_tuple[i - 1]
            sign_exp = deg_cobar + cumulative_v
            sign = sign_from_exponent(sign_exp)

            v_elem = self._module.term(v_key)
            bdry = self._module.boundary(v_elem)

            for new_v_key, coeff in bdry:
                new_v_tuple = v_tuple[: i - 1] + (new_v_key,) + v_tuple[i:]
                result += sign * coeff * self.term((tree, new_v_tuple))

            cumulative_v += self._module.degree_on_basis(v_key)

        return result

    def _dcoact_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Expand each leaf using the C-coalgebra coaction δ.

        For leaf ``l`` decorated by ``v_l``, the coaction
        ``δ_k(v_l) = Σ c_k ⊗ v_1 ⊗ ... ⊗ v_k`` inserts a new internal vertex
        (decorated by ``c_k``) with ``k`` new leaf children at position ``l``.
        Old leaves ``> l`` are shifted by ``k - 1``.

        Sign at leaf ``l``:

            ``(-1)^{deg_cobar(tree) + Σ_{j < l} deg_V(v_j)}``
        """
        tree, v_tuple = key
        n = len(v_tuple)

        deg_cobar = (
            0
            if is_leaf(tree)
            else subtree_degree_cobar(tree, self._cooperad_cls, self.base_ring())
        )

        result = self.zero()
        base_ring = self.base_ring()
        cumulative_v = 0

        for leaf_l in range(1, n + 1):
            v_key = v_tuple[leaf_l - 1]
            sign_exp = deg_cobar + cumulative_v
            sign = sign_from_exponent(sign_exp)

            v_elem = self._module.term(v_key)

            # Try all coaction arities k >= 2
            for k in range(2, n + 2):
                coaction = self._coalgebra.coact(v_elem, k)
                for (c_key, *new_v_keys_raw), coeff in coaction:
                    # coaction returns terms in C(k) ⊗ V^⊗k
                    # c_key is the C(k) basis key, new_v_keys is the V^⊗k part
                    new_v_keys = tuple(new_v_keys_raw) if new_v_keys_raw else ()
                    if len(new_v_keys) != k:
                        continue

                    # Build the new tree by grafting a new internal vertex at leaf_l
                    # The new vertex is decorated by c_key with k leaf children
                    new_tree = self._expand_leaf(tree, leaf_l, c_key, k)

                    # Build new v_tuple: insert new_v_keys at position leaf_l
                    new_v_tuple = v_tuple[: leaf_l - 1] + new_v_keys + v_tuple[leaf_l:]

                    result += sign * coeff * self.term((new_tree, new_v_tuple))

            cumulative_v += self._module.degree_on_basis(v_key)

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _replace_decoration(self, tree, target_vertex, new_dec):
        """Replace the decoration of ``target_vertex`` in ``tree``."""
        if is_leaf(tree):
            return tree
        if tree is target_vertex:
            return (new_dec,) + children(tree)
        new_children = tuple(
            self._replace_decoration(c, target_vertex, new_dec) for c in children(tree)
        )
        return (decoration(tree),) + new_children

    def _expand_leaf(self, tree, leaf_l: int, new_dec, k: int):
        """Replace leaf ``leaf_l`` with a new internal vertex having ``k`` children.

        The new vertex is decorated by ``new_dec``.  Old leaves ``> leaf_l``
        are shifted by ``k - 1``.
        """
        # Build a small subtree: (new_dec, 1, 2, ..., k) with leaf labels 1..k
        new_subtree = (new_dec,) + tuple(range(1, k + 1))

        def _expand_rec(node):
            if is_leaf(node):
                if node == leaf_l:
                    # Replace with new subtree, relabeled to leaf_l, leaf_l+1, ..., leaf_l+k-1
                    relabel = {j: leaf_l + j - 1 for j in range(1, k + 1)}
                    relabeled_sub = (new_dec,) + tuple(
                        relabel[c] for c in range(1, k + 1)
                    )
                    return relabeled_sub
                elif node > leaf_l:
                    return node + k - 1
                else:
                    return node
            new_children = tuple(_expand_rec(c) for c in children(node))
            return (decoration(node),) + new_children

        if is_leaf(tree):
            # The whole tree is just leaf_l
            relabeled_sub = (new_dec,) + tuple(range(leaf_l, leaf_l + k))
            return relabeled_sub
        return _expand_rec(tree)

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the cobar complex Ω_C(V)."""

        def boundary(self) -> "CobarComplexCoalgebra.Element":
            """Apply the full cobar differential d = d_1 + d_2 + d_V + d_coact."""
            return self.parent().boundary(self)

        def d1(self) -> "CobarComplexCoalgebra.Element":
            """Apply ∂_C to each vertex decoration."""
            return self.parent()._d1(self)

        def d2(self) -> "CobarComplexCoalgebra.Element":
            """Apply the structural cobar differential (expand internal vertices)."""
            return self.parent()._d2(self)

        def dV(self) -> "CobarComplexCoalgebra.Element":
            """Apply ∂_V to each leaf decoration."""
            return self.parent()._dV(self)

        def dcoact(self) -> "CobarComplexCoalgebra.Element":
            """Apply the C-coalgebra coaction at each leaf."""
            return self.parent()._dcoact(self)
