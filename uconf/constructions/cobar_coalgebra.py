"""Cobar complex Ω_C(V) of a C-coalgebra V.

For a connected coaugmented dg-cooperad C and a C-coalgebra (V, δ), the cobar
complex is the dg-module

    Ω_C(V) = (T_{s⁻¹C̄}(V), d_internal + d_2 + d_coact)

where ``T_{s⁻¹C̄}(V)`` is the free algebra on V under the desuspended
coaugmentation coideal ``s⁻¹C̄``, and:

- d_internal : the interleaved DFS differential inherited from
  :class:`~uconf.algebraic.free_algebra.FreeAlgebraModule`, applying ``∂_C``
  to vertex decorations and ``∂_V`` to leaf decorations with the Koszul sign
  rule.
- d_2  : structural cobar differential (expand internal vertices via cooperad
  cocomposition).
- d_coact: apply the C-coalgebra coaction δ at each leaf.

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

from typing import ClassVar, cast

from sage.all import CombinatorialFreeModule

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import FreeAlgebraModule
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.cooperad import CooperadLike
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    decoration,
    expand_vertex,
    is_leaf,
    subtree_degree_cobar,
    vertex_arity,
    vertices_dfs,
)


class CobarComplexCoalgebra(FreeAlgebraModule):
    """Cobar complex Ω_C(V) of a C-coalgebra V.

    Subclasses :class:`~uconf.algebraic.free_algebra.FreeAlgebraModule`
    with ``vertex_degree_shift = -1`` (desuspension convention).  Inherits basis
    key validation, degree computation, basis iteration, and the internal
    differential (d_1 + d_V) from the free module.  Only the structural
    differential ``d_2`` and coalgebra coaction ``d_coact`` are implemented here.

    Args:
        coalgebra: A :class:`uconf.coalgebra.CooperadCoalgebra` instance.
        base_ring: Coefficient ring.

    Basis keys are pairs ``(tree, v_tuple)`` where:

    - ``tree`` is an integer (single leaf, arity 1) or a tuple representing a
      decorated rooted tree with leaves labeled ``1, ..., n``.
    - ``v_tuple`` is a tuple of ``n`` basis keys of the underlying module V.

    The degree is ``deg_cobar(tree) + Σ_i deg_V(v_tuple[i])``.
    """

    name: ClassVar[str] = "Ω"

    def __init__(self, coalgebra: CooperadCoalgebra, base_ring):
        self._coalgebra = coalgebra
        self._cooperad_cls = CobarConstruction(coalgebra.cooperad_cls)
        self._module = coalgebra.module

        super().__init__(
            operad_cls=coalgebra.cooperad_cls,
            inner_module=coalgebra.module,
            base_ring=base_ring,
            vertex_degree_shift=-1,
            name=f"Ω({coalgebra.cooperad_cls.name}; {coalgebra.module})",
        )

        # Override the inherited boundary with the cobar-specific differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: FreeAlgebraModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dcoact = self.module_morphism(on_basis=self._dcoact_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "CobarComplexCoalgebra.Element":
        """Total differential d = d_internal + d_2 + d_coact.

        ``d_internal`` is the interleaved DFS differential inherited from
        :class:`~uconf.algebraic.free_algebra.FreeAlgebraModule`, applying
        ``∂_C`` to vertex decorations and ``∂_V`` to leaf decorations with
        the Koszul sign rule.
        """
        return (
            FreeAlgebraModule._boundary_on_basis(self, key)
            + self._d2_on_basis(key)
            + self._dcoact_on_basis(key)
        )

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
        # Raw cooperad C is stored as operad_cls by the base class
        raw_C = cast(CooperadLike, self._operad_cls)

        for curr_vertex in verts:
            curr_arity = vertex_arity(curr_vertex)
            curr_dec = decoration(curr_vertex)
            curr_parent = raw_C(curr_arity, base_ring)
            curr_elem = curr_parent.term(curr_dec)

            # Global accumulation: deg_C(v) - 1 for DFS vertices before curr
            global_accum = 0
            for v in verts:
                if v is curr_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = raw_C(v_arity, base_ring).degree_on_basis(decoration(v))
                global_accum += v_deg - 1

            for m in range(2, curr_arity):
                n_right = curr_arity - m + 1
                for i in range(1, m + 1):
                    cocomp = curr_parent.infinitesimal_cocompose(curr_elem, i, m, n_right)
                    for (dec_left, dec_right), coeff in cocomp:
                        right_parent = raw_C(n_right, base_ring)
                        left_parent = raw_C(m, base_ring)
                        right_sinv_deg = right_parent.degree_on_basis(dec_right) - 1
                        left_degree = left_parent.degree_on_basis(dec_left)

                        before_deg = sum(
                            subtree_degree_cobar(ch, raw_C, base_ring)
                            for j, ch in enumerate(children(curr_vertex), start=1)
                            if j < i
                        )
                        koszul_exp = right_sinv_deg * before_deg
                        total_sign = sign_from_exponent(global_accum + left_degree + koszul_exp)

                        new_tree = expand_vertex(
                            tree, curr_vertex, i, dec_left, dec_right, m, n_right
                        )
                        # v_tuple is unchanged (leaf labels shift but order preserved)
                        result += total_sign * coeff * self.term((new_tree, v_tuple))

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
        # Raw cooperad C is stored as operad_cls by the base class
        raw_C = cast(CooperadLike, self._operad_cls)

        deg_cobar = 0 if is_leaf(tree) else subtree_degree_cobar(tree, raw_C, self.base_ring())

        result = self.zero()
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
                    new_v_keys = tuple(new_v_keys_raw) if new_v_keys_raw else ()
                    if len(new_v_keys) != k:
                        continue

                    new_tree = self._expand_leaf(tree, leaf_l, c_key, k)
                    new_v_tuple = v_tuple[: leaf_l - 1] + new_v_keys + v_tuple[leaf_l:]

                    result += sign * coeff * self.term((new_tree, new_v_tuple))

            cumulative_v += self._module.degree_on_basis(v_key)

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _expand_leaf(self, tree, leaf_l: int, new_dec, k: int):
        """Replace leaf ``leaf_l`` with a new internal vertex having ``k`` children.

        The new vertex is decorated by ``new_dec``.  Old leaves ``> leaf_l``
        are shifted by ``k - 1``.
        """

        def _expand_rec(node):
            if is_leaf(node):
                if node == leaf_l:
                    return (new_dec,) + tuple(range(leaf_l, leaf_l + k))
                elif node > leaf_l:
                    return node + k - 1
                else:
                    return node
            new_children = tuple(_expand_rec(c) for c in children(node))
            return (decoration(node),) + new_children

        if is_leaf(tree):
            return (new_dec,) + tuple(range(leaf_l, leaf_l + k))
        return _expand_rec(tree)

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(ParentedElementMixin["CobarComplexCoalgebra"], CombinatorialFreeModule.Element):
        """An element of the cobar complex Ω_C(V)."""

        def boundary(self) -> "CobarComplexCoalgebra.Element":
            """Apply the full cobar differential d = d_internal + d_2 + d_coact."""
            parent = self.parent()
            return parent.boundary(self)

        def d_internal(self) -> "CobarComplexCoalgebra.Element":
            """Apply the internal differential (∂_C on vertices + ∂_V on leaves)."""
            parent = self.parent()
            return parent._d_internal(self)

        def d2(self) -> "CobarComplexCoalgebra.Element":
            """Apply the structural cobar differential (expand internal vertices)."""
            parent = self.parent()
            return parent._d2(self)

        def dcoact(self) -> "CobarComplexCoalgebra.Element":
            """Apply the C-coalgebra coaction at each leaf."""
            parent = self.parent()
            return parent._dcoact(self)
