"""Twisted bar complex B_ι(A) of an ΩB(P)-algebra A.

For a connected augmented dg-operad P and an ΩB(P)-algebra (A, γ), the twisted
bar complex is the dg-module

    B_ι(A) = (T^c_{sP̄}(A), d_internal + d_2 + d_twist)

where ``T^c_{sP̄}(A)`` is the cofree conilpotent B(P)-coalgebra on A (the same
underlying module as the standard bar complex of a P-algebra), and:

- d_internal : the interleaved DFS differential inherited from
  :class:`~uconf.algebraic.tree_module.TreeModule`, applying ``∂_P`` to vertex
  decorations and ``∂_A`` to leaf decorations with the Koszul sign rule.
- d_2 : structural bar differential (contract internal edges via operad
  composition), identical to that of
  :class:`~uconf.constructions.bar_algebra.BarComplexAlgebra`.
- d_twist: apply the ΩB(P)-algebra action via the universal twisting morphism
  ι: B(P) → ΩB(P) at each internal vertex all of whose children are leaves.

The universal twisting morphism ι: B(P) → ΩB(P) sends a P(n)-element p (viewed
as a single-vertex bar tree [p] ∈ B(P)(n)) to the single-vertex cobar tree
ι([p]) ∈ ΩB(P)(n) decorated by [p] itself.  Concretely, in basis-key notation:

    ι(dec) = ((dec, 1, …, n), 1, …, n)   for dec ∈ P(n)

The result B_ι(A) is a dg-B(P)-coalgebra.  This construction implements the
left adjoint of the bar-cobar adjunction

    B_ι : dg-ΩB(P)-alg ⇌ dg-B(P)-coalg : Ω_ι

induced by ι (see article, Section 3.4).

Reference: Loday-Vallette "Algebraic Operads", Section 11.4.
"""

from __future__ import annotations

from typing import ClassVar, cast

from sage.all import CombinatorialFreeModule

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import TreeModule
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
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


class TwistedBarComplexAlgebra(TreeModule):
    """Twisted bar complex B_ι(A) of an ΩB(P)-algebra A.

    Given a connected augmented dg-operad P and a
    ``CobarConstruction(BarConstruction(P))``-algebra (A, γ), this class
    produces a B(P)-coalgebra via the universal twisting morphism
    ι: B(P) → ΩB(P).

    The underlying module is T^c_{B(P)}(A), the cofree conilpotent B(P)-
    coalgebra on A, which coincides with the underlying module of the standard
    bar complex of a P-algebra.  The differential d = d_internal + d_2 +
    d_twist replaces the algebra-action term d_act of
    :class:`~uconf.constructions.bar_algebra.BarComplexAlgebra`:

    - d_internal: applies ∂_P to vertex decorations and ∂_A to leaf values.
    - d_2: contracts internal edges via P-composition (same as BarComplexAlgebra).
    - d_twist: at each corolla vertex (all-leaf children) with P(n)-decoration
      dec, builds the cobar element ι([dec]) ∈ ΩB(P)(n) and applies the
      ΩB(P)-algebra action γ(ι([dec]); a_1,…,a_n).

    Args:
        algebra: An :class:`~uconf.algebraic.algebra.OperadAlgebra` instance
            whose ``operad_cls`` is ``CobarConstruction(BarConstruction(P))``
            for some connected augmented dg-operad P.
        base_ring: Coefficient ring.

    Raises:
        TypeError: If ``algebra.operad_cls`` is not a
            ``CobarConstruction(BarConstruction(P))`` instance.

    Basis keys are pairs ``(tree, a_tuple)`` where:

    - ``tree`` is an integer (single leaf) or a P-decorated rooted tree with
      leaves labeled ``1, …, n``, exactly as in
      :class:`~uconf.constructions.bar_algebra.BarComplexAlgebra`.
    - ``a_tuple`` is a tuple of ``n`` basis keys of the module A.

    The degree is ``deg_bar(tree) + Σ_i deg_A(a_tuple[i])``.
    """

    name: ClassVar[str] = "B_ι"

    def __init__(self, algebra: OperadAlgebra, base_ring):
        self._algebra = algebra

        # Validate and extract P from ΩB(P) = CobarConstruction(BarConstruction(P))
        cobar_bar = algebra.operad_cls
        if not isinstance(cobar_bar, CobarConstruction):
            raise TypeError(
                f"Expected algebra.operad_cls to be a CobarConstruction instance, "
                f"got {type(cobar_bar).__name__}. "
                f"The algebra must be an ΩB(P)-algebra, i.e. its operad_cls must be "
                f"CobarConstruction(BarConstruction(P)) for some operad P."
            )
        bar_cls = cobar_bar.cooperad_cls
        if not isinstance(bar_cls, BarConstruction):
            raise TypeError(
                f"Expected algebra.operad_cls.cooperad_cls to be a BarConstruction "
                f"instance, got {type(bar_cls).__name__}. "
                f"The algebra must be an ΩB(P)-algebra, i.e. its operad_cls must be "
                f"CobarConstruction(BarConstruction(P)) for some operad P."
            )

        self._cobar_bar_cls = cobar_bar  # Ω(B(P))
        self._bar_cls = bar_cls           # B(P)
        self._base_operad_cls = bar_cls.operad_cls  # P
        self._module = algebra.module

        super().__init__(
            symmetric_sequence_cls=self._base_operad_cls,  # P for vertex decorations
            inner_module=algebra.module,
            base_ring=base_ring,
            vertex_degree_shift=1,  # bar suspension convention
            name=f"B_ι({cobar_bar.name}; {algebra.module})",
        )

        # Override the inherited boundary with the twisted bar differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: TreeModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dtwist = self.module_morphism(on_basis=self._dtwist_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "TwistedBarComplexAlgebra.Element":
        """Total differential d = d_internal + d_2 + d_twist.

        ``d_internal`` is the interleaved DFS differential inherited from
        :class:`~uconf.algebraic.tree_module.TreeModule`, applying ``∂_P``
        to vertex decorations and ``∂_A`` to leaf decorations with the
        Koszul sign rule.
        """
        return (
            TreeModule._boundary_on_basis(self, key)
            + self._d2_on_basis(key)
            + self._dtwist_on_basis(key)
        )

    def _d2_on_basis(self, key) -> "TwistedBarComplexAlgebra.Element":
        """Structural differential: contract each internal edge via P-composition.

        Identical to
        :meth:`~uconf.constructions.bar_algebra.BarComplexAlgebra._d2_on_basis`.

        Sign at edge (parent p, slot l, child c):

            ``(-1)^{global_accum + deg_P(p) + (deg_P(c) - 1) * before_deg}``

        where ``global_accum = Σ_{v before p in DFS} (deg_P(v) - 1)`` and
        ``before_deg = Σ_{j<l} deg_bar(subtree_j of p)``.
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

    def _dtwist_on_basis(self, key) -> "TwistedBarComplexAlgebra.Element":
        """Twisted differential via the universal twisting morphism ι: B(P) → ΩB(P).

        At each vertex v with all-leaf children and P(n)-decoration ``dec``:

        1. Build the bar corolla ``[dec] ∈ B(P)(n)`` with key
           ``(dec, 1, …, n)``.
        2. Build the cobar single-vertex tree
           ``ι([dec]) ∈ ΩB(P)(n)`` with key ``((dec, 1, …, n), 1, …, n)``.
        3. Apply the ΩB(P)-algebra action
           ``γ(ι([dec]); a_{c_1}, …, a_{c_n}) ∈ A``.

        Sign at DFS vertex v_j (all-leaf-children):

            ``(-1)^{Σ_{l<j} (deg_P(dec(v_l)) + 1)}``
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        raw_P = cast(OperadLike, self._symmetric_sequence_cls)
        cumulative = 0

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            operad_parent = raw_P(v_arity, base_ring)
            vertex_sp_deg = operad_parent.degree_on_basis(dec) + 1  # bar suspension

            # Apply at corolla vertices (all children are leaves)
            v_children = children(v)
            if all(is_leaf(c) for c in v_children):
                sign = sign_from_exponent(cumulative)
                leaf_labels = list(v_children)  # integer leaf labels
                a_elems = [self._module.term(a_tuple[l - 1]) for l in leaf_labels]

                # Build ι([dec]) ∈ ΩB(P)(v_arity):
                # Step 1: bar corolla key = (dec, 1, …, v_arity) in B(P)(v_arity)
                bar_corolla_key = (dec,) + tuple(range(1, v_arity + 1))
                # Step 2: cobar single-vertex key = (bar_corolla_key, 1, …, v_arity)
                cobar_vertex_key = (bar_corolla_key,) + tuple(range(1, v_arity + 1))
                cobar_parent = self._cobar_bar_cls(v_arity, base_ring)
                iota_elem = cobar_parent.term(cobar_vertex_key)

                # Step 3: apply the ΩB(P)-algebra action
                action_result = self._algebra.act(iota_elem, a_elems)

                for new_a_key, coeff in action_result:
                    new_tree, new_a_tuple = self._contract_leaf_vertex(
                        tree, v, a_tuple, new_a_key
                    )
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

    class Element(ParentedElementMixin["TwistedBarComplexAlgebra"], CombinatorialFreeModule.Element):
        """An element of the twisted bar complex B_ι(A)."""

        def boundary(self) -> "TwistedBarComplexAlgebra.Element":
            """Apply the full twisted bar differential d = d_internal + d_2 + d_twist."""
            parent = self.parent()
            return parent.boundary(self)

        def d_internal(self) -> "TwistedBarComplexAlgebra.Element":
            """Apply the internal differential (∂_P on vertices + ∂_A on leaves)."""
            parent = self.parent()
            return parent._d_internal(self)

        def d2(self) -> "TwistedBarComplexAlgebra.Element":
            """Apply the structural bar differential (contract internal edges)."""
            parent = self.parent()
            return parent._d2(self)

        def dtwist(self) -> "TwistedBarComplexAlgebra.Element":
            """Apply the twisted differential (ΩB(P)-algebra action via ι)."""
            parent = self.parent()
            return parent._dtwist(self)
