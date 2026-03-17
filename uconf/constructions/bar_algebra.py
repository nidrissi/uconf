"""Bar complex B_P(A) of a P-algebra A.

For a connected augmented dg-operad P and a P-algebra (A, γ), the bar complex
is the dg-module

    B_P(A) = (T^c_{sP̄}(A), d_internal + d_2 + d_act)

where ``T^c_{sP̄}(A)`` is the cofree conilpotent coalgebra on A cogenerating
under the suspended augmentation ideal ``sP̄``, and:

- d_internal : the interleaved DFS differential inherited from
  :class:`~uconf.algebraic.cofree_coalgebra.CofreeCoalgebraModule`, applying
  ``∂_P`` to vertex decorations and ``∂_A`` to leaf decorations with the
  Koszul sign rule.
- d_2  : structural bar differential (contract internal edges via operad
  composition).
- d_act: apply the P-algebra action γ at each internal vertex all of whose
  children are leaves.

The basis elements are pairs ``(tree, a_tuple)`` where:

- ``tree`` is a bar-construction tree in ``B(P)(n)`` (integer leaf or decorated
  tuple as in :class:`uconf.bar_construction.BarConstruction`).
- ``a_tuple`` is a tuple of ``n`` basis keys of the module A, one per leaf.

The homological degree is

    deg(tree, a_tuple) = deg_bar(tree) + Σ_i deg_A(a_tuple[i])

where ``deg_bar(tree) = Σ_{v internal} (deg_P(dec(v)) + 1)``.

Reference: Loday-Vallette "Algebraic Operads", Section 11.2.
"""

from __future__ import annotations

from typing import ClassVar, cast

from sage.all import CombinatorialFreeModule

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.constructions.bar_construction import BarConstruction
from uconf.core.operad import OperadLike
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    contract_edge,
    decoration,
    internal_edges_dfs,
    is_leaf,
    subtree_degree,
    vertices_dfs,
    vertex_arity,
)


class BarComplexAlgebra(CofreeCoalgebraModule):
    """Bar complex B_P(A) of a P-algebra A.

    Subclasses :class:`~uconf.algebraic.cofree_coalgebra.CofreeCoalgebraModule`
    with ``vertex_degree_shift = +1`` (suspension convention).  Inherits basis
    key validation, degree computation, basis iteration, and the internal
    differential (d_1 + d_A) from the cofree module.  Only the structural
    differential ``d_2`` and algebra action ``d_act`` are implemented here.

    Args:
        algebra: An :class:`uconf.algebra.OperadAlgebra` instance (the P-algebra A).
        base_ring: Coefficient ring.

    Basis keys are pairs ``(tree, a_tuple)`` where:

    - ``tree`` is an integer (single leaf, arity 1) or a tuple representing a
      decorated rooted tree with leaves labeled ``1, ..., n``.
    - ``a_tuple`` is a tuple of ``n`` basis keys of the underlying module A.

    The degree is ``deg_bar(tree) + Σ_i deg_A(a_tuple[i])``.
    """

    name: ClassVar[str] = "B"

    def __init__(self, algebra: OperadAlgebra, base_ring):
        self._algebra = algebra
        self._operad_cls = BarConstruction(algebra.operad_cls)
        self._module = algebra.module

        super().__init__(
            cooperad_cls=algebra.operad_cls,
            inner_module=algebra.module,
            base_ring=base_ring,
            vertex_degree_shift=1,
            name=f"B({algebra.operad_cls.name}; {algebra.module})",
        )

        # Override the inherited boundary with the bar-specific differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: CofreeCoalgebraModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dact = self.module_morphism(on_basis=self._dact_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Total differential d = d_internal + d_2 + d_act.

        ``d_internal`` is the interleaved DFS differential inherited from
        :class:`~uconf.algebraic.cofree_coalgebra.CofreeCoalgebraModule`,
        applying ``∂_P`` to vertex decorations and ``∂_A`` to leaf
        decorations with the Koszul sign rule.
        """
        return super()._boundary_on_basis(key) + self._d2_on_basis(key) + self._dact_on_basis(key)

    def _d2_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Structural differential: contract each internal edge via operad composition.

        Sign at edge (parent p, slot l, child c):

            ``(-1)^{global_accum + deg_P(p) + (deg_P(c) - 1) * before_deg}``

        where ``global_accum = Σ_{v before p in DFS} (deg_P(v) - 1)`` and
        ``before_deg = Σ_{j<l} deg_bar(subtree_j of p)``.

        This is the same sign convention as
        :meth:`uconf.bar_construction.BarConstruction.Component._d2_on_basis`.
        The A-leaf decorations are unaffected (leaf labels do not change when
        an internal edge is contracted).
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        edges = internal_edges_dfs(tree)
        if not edges:
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        raw_P = cast(OperadLike, self._symmetric_sequence_cls)

        for parent_vertex, child_pos, child_vertex in edges:
            p_arity = vertex_arity(parent_vertex)
            c_arity = vertex_arity(child_vertex)
            p_dec = decoration(parent_vertex)
            c_dec = decoration(child_vertex)

            p_parent = raw_P(p_arity, base_ring)
            c_parent = raw_P(c_arity, base_ring)

            p_deg_P = p_parent.degree_on_basis(p_dec)
            c_deg_P = c_parent.degree_on_basis(c_dec)

            # Global accumulation: deg_P(v) - 1 for each DFS vertex before parent
            global_accum = 0
            for v in verts:
                if v is parent_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = raw_P(v_arity, base_ring).degree_on_basis(decoration(v))
                global_accum += v_deg - 1

            # Koszul sign from the child's position among parent's children
            c_sp_deg = c_deg_P - 1
            before_deg = sum(
                subtree_degree(ch, raw_P, base_ring)
                for i, ch in enumerate(children(parent_vertex), start=1)
                if i < child_pos
            )
            koszul_exp = c_sp_deg * before_deg

            total_sign = sign_from_exponent(global_accum + p_deg_P + koszul_exp)

            # Compose and build contracted tree (a_tuple unchanged)
            p_elem = p_parent.term(p_dec)
            c_elem = c_parent.term(c_dec)
            composed = raw_P.compose(p_elem, child_pos, c_elem)

            for new_dec, coeff in composed:
                new_tree = contract_edge(tree, parent_vertex, child_pos, new_dec)
                result += total_sign * coeff * self.term((new_tree, a_tuple))

        return result

    def _dact_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Apply the P-algebra action γ at each vertex all of whose children are leaves.

        Sign at DFS vertex v_j (all-leaf-children):

            ``(-1)^{Σ_{l<j} (deg_P(dec(v_l)) + 1)}``

        The vertex v and its k leaf children are replaced by a single new leaf
        decorated by γ(dec(v); a_{c_1}, ..., a_{c_k}).
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        cumulative = 0
        raw_P = cast(OperadLike, self._symmetric_sequence_cls)

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            operad_parent = raw_P(v_arity, base_ring)
            vertex_sp_deg = operad_parent.degree_on_basis(dec) + 1

            # Check if all children of v are leaves
            v_children = children(v)
            if all(is_leaf(c) for c in v_children):
                sign = sign_from_exponent(cumulative)
                leaf_labels = list(v_children)  # integer leaf labels
                # Algebra elements at each leaf child
                a_elems = [self._module.term(a_tuple[l - 1]) for l in leaf_labels]
                operad_elem = operad_parent.term(dec)
                action_result = self._algebra.act(operad_elem, a_elems)

                for new_a_key, coeff in action_result:
                    new_tree, new_a_tuple = self._contract_leaf_vertex(tree, v, a_tuple, new_a_key)
                    result += sign * coeff * self.term((new_tree, new_a_tuple))

            cumulative += vertex_sp_deg

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _contract_leaf_vertex(self, tree, target_vertex, a_tuple, new_a_key):
        """Contract ``target_vertex`` (whose children are all leaves) to a single leaf.

        The k leaf children ``l_1 < l_2 < ... < l_k`` are removed; the vertex
        is replaced by a single new leaf with the minimum label.  All remaining
        leaf labels are compacted to ``1, ..., n - k + 1``.

        Returns ``(new_tree, new_a_tuple)``.
        """
        leaf_children = sorted(children(target_vertex))  # all are leaf integers
        n = len(a_tuple)

        new_leaf_label = leaf_children[0]  # minimum child label
        removed_leaves = set(leaf_children[1:])  # all except the minimum

        # Build relabeling map: compact {1,...,n} \ removed_leaves → {1,...,n-k+1}
        relabel: dict[int, int] = {}
        counter = 1
        for leaf in range(1, n + 1):
            if leaf not in removed_leaves:
                relabel[leaf] = counter
                counter += 1

        # Build new a_tuple
        new_a: dict[int, object] = {}
        for leaf in range(1, n + 1):
            if leaf in removed_leaves:
                continue
            new_l = relabel[leaf]
            if leaf == new_leaf_label:
                new_a[new_l] = new_a_key
            else:
                new_a[new_l] = a_tuple[leaf - 1]

        max_new = max(relabel.values()) if relabel else 0
        new_a_tuple = tuple(new_a[i] for i in range(1, max_new + 1))

        # Build new tree: replace target_vertex with its minimum leaf label,
        # then apply the relabeling to all leaves.
        def _replace_vertex_with_leaf(node):
            if is_leaf(node):
                return relabel.get(node, node)
            if node is target_vertex:
                return relabel[new_leaf_label]
            new_children = tuple(_replace_vertex_with_leaf(c) for c in children(node))
            return (decoration(node),) + new_children

        new_tree = _replace_vertex_with_leaf(tree)
        return new_tree, new_a_tuple

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(ParentedElementMixin["BarComplexAlgebra"], CombinatorialFreeModule.Element):
        """An element of the bar complex B_P(A)."""

        def boundary(self) -> "BarComplexAlgebra.Element":
            """Apply the full bar differential d = d_internal + d_2 + d_act."""
            parent = self.parent()
            return parent.boundary(self)

        def d_internal(self) -> "BarComplexAlgebra.Element":
            """Apply the internal differential (∂_P on vertices + ∂_A on leaves)."""
            parent = self.parent()
            return parent._d_internal(self)

        def d2(self) -> "BarComplexAlgebra.Element":
            """Apply the structural bar differential (contract internal edges)."""
            parent = self.parent()
            return parent._d2(self)

        def dact(self) -> "BarComplexAlgebra.Element":
            """Apply the P-algebra action at all-leaf-children vertices."""
            parent = self.parent()
            return parent._dact(self)
