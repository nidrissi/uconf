"""Twisted bar and cobar complexes for a general twisting morphism α: C → P.

Given a twisting morphism α: C → P satisfying the Maurer-Cartan equation
∂α + α ⋆ α = 0, the twisted bar complex B_α(A) and twisted cobar complex
Ω_α(V) provide an adjunction:

    B_α : P-alg ⇌ C-coalg : Ω_α

**Twisted bar complex** B_α(A) for a P-algebra (A, γ):
    B_α(A) = (T^c_{C}(A), d_internal + d_2 + d_α)

where T^c_{C}(A) is the cofree conilpotent C-coalgebra on A, and:
- d_internal: interleaved DFS differential (∂_C on vertex decorations + ∂_A on
  leaves)
- d_2: structural cobar differential (expand vertices via C-cocomposition)
  [Note: this is the cooperad structural differential on the C-decorated trees]
- d_α: at each corolla vertex with C(n)-decoration c, apply α(c) ∈ P(n) via
  the P-algebra action γ(α(c); a_1,...,a_n)

The vertex degree shift is determined by the cooperad C (which controls the
degree convention for basis elements).

**Twisted cobar complex** Ω_α(V) for a C-coalgebra (V, δ):
    Ω_α(V) = (T_{P}(V), d_internal + d_2 + d_α)

where T_{P}(V) is the free P-algebra on V, and:
- d_internal: interleaved DFS differential (∂_P on vertex decorations + ∂_V on
  leaves)
- d_2: structural bar differential (contract internal edges via P-composition)
- d_α: at each leaf l with value v_l, apply the C-coalgebra coaction
  δ_k(v_l) = Σ c_k ⊗ v'_1 ⊗ ... ⊗ v'_k, then convert via α:
  insert new vertex decorated by α(c_k) ∈ P(k).

**Canonical specializations**:

- ``BarComplexAlgebra(alg, base_ring)`` ≅ ``TwistedBarComplex(canonical_projection(P), alg, base_ring)``
- ``CobarComplexCoalgebra(coalg, base_ring)`` ≅ ``TwistedCobarComplex(canonical_inclusion(C), coalg, base_ring)``
- ``TwistedBarComplexAlgebra(alg, base_ring)`` ≅ ``TwistedBarComplex(canonical_inclusion(B(P)), alg, base_ring)``

Reference: Loday-Vallette "Algebraic Operads", Section 11.2 and 11.4.
"""

from __future__ import annotations

from typing import ClassVar, cast

from sage.all import CombinatorialFreeModule

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.tree_module import TreeModule
from uconf.core.cooperad import CooperadLike
from uconf.core.operad import OperadLike
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    contract_edge,
    decoration,
    expand_vertex,
    internal_edges_dfs,
    is_leaf,
    subtree_degree,
    subtree_degree_cobar,
    vertices_dfs,
    vertex_arity,
)
from uconf.core.twisting import TwistingMorphism


# ===========================================================================
# TwistedBarComplex: B_α(A)
# ===========================================================================


class TwistedBarComplex(TreeModule):
    """Twisted bar complex B_α(A) for a twisting morphism α: C → P and P-algebra A.

    The underlying module is T^c_C(A), the cofree conilpotent C-coalgebra on A.
    Basis keys are pairs ``(tree, a_tuple)`` where tree has C-decorated vertices
    and a_tuple carries A-module values at the leaves.

    The differential is d = d_internal + d_2 + d_α where:

    - d_internal: applies ∂_C to vertex decorations and ∂_A to leaf decorations
      with the Koszul sign rule (inherited from TreeModule).
    - d_2: expands internal vertices via C-cocomposition (cobar structural
      differential on the cooperad side).
    - d_α: at each corolla vertex (all-leaf children) with C(n)-decoration c,
      applies the P-algebra action γ(α(c); a_1,...,a_n).

    Args:
        alpha: A :class:`~uconf.core.twisting.TwistingMorphism` α: C → P.
        algebra: An :class:`~uconf.algebraic.algebra.OperadAlgebra` (P-algebra).
        base_ring: Coefficient ring.

    Note:
        The ``vertex_degree_shift`` is -1 (cobar/desuspension convention) since
        the trees are decorated by the cooperad C.  The cooperad C carries the
        s⁻¹ C̄ decoration, and the bar complex B_α(A) lives on the "cofree
        C-coalgebra" side of the adjunction.
    """

    name: ClassVar[str] = "B_α"

    def __init__(self, alpha: TwistingMorphism, algebra: OperadAlgebra, base_ring):
        self._alpha = alpha
        self._algebra = algebra
        self._cooperad_cls = alpha.cooperad
        self._operad_cls = alpha.operad
        self._module = algebra.module

        # Determine vertex degree shift from the cooperad
        # For B(P) cooperad: trees use bar suspension (+1)
        # For a "raw" cooperad C: trees use cobar desuspension (-1)
        from uconf.constructions.bar_construction import BarConstruction
        if isinstance(self._cooperad_cls, BarConstruction):
            vertex_shift = 1  # bar trees: degree = Σ (deg_P(v) + 1)
            sym_seq = self._cooperad_cls.operad_cls  # P for decorations
        else:
            vertex_shift = -1  # cobar trees: degree = Σ (deg_C(v) - 1)
            sym_seq = self._cooperad_cls  # C for decorations

        self._vertex_shift = vertex_shift
        self._raw_sym_seq = sym_seq

        super().__init__(
            symmetric_sequence_cls=sym_seq,
            inner_module=algebra.module,
            base_ring=base_ring,
            vertex_degree_shift=vertex_shift,
            name=f"B_{{{alpha.name}}}({algebra.module})",
        )

        # Override the inherited boundary with the twisted differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: TreeModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dalpha = self.module_morphism(on_basis=self._dalpha_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "TwistedBarComplex.Element":
        """Total differential d = d_internal + d_2 + d_α."""
        return (
            TreeModule._boundary_on_basis(self, key)
            + self._d2_on_basis(key)
            + self._dalpha_on_basis(key)
        )

    def _d2_on_basis(self, key) -> "TwistedBarComplex.Element":
        """Structural differential on cooperad-decorated trees.

        When cooperad = B(P), contracts internal edges via P-composition (bar d_2).
        When cooperad = C (raw), expands vertices via C-cocomposition (cobar d_2).
        """
        from uconf.constructions.bar_construction import BarConstruction

        if isinstance(self._cooperad_cls, BarConstruction):
            return self._d2_bar(key)
        else:
            return self._d2_cobar(key)

    def _d2_bar(self, key) -> "TwistedBarComplex.Element":
        """Structural bar differential: contract internal edges via operad composition.

        Sign at edge (parent p, slot l, child c):
            ``(-1)^{global_accum + deg_P(p) + (deg_P(c) - 1) * before_deg}``
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
        raw_P = cast(OperadLike, self._raw_sym_seq)

        for parent_vertex, child_pos, child_vertex in edges:
            p_arity = vertex_arity(parent_vertex)
            c_arity = vertex_arity(child_vertex)
            p_dec = decoration(parent_vertex)
            c_dec = decoration(child_vertex)

            p_parent = raw_P(p_arity, base_ring)
            c_parent = raw_P(c_arity, base_ring)

            p_deg_P = p_parent.degree_on_basis(p_dec)
            c_deg_P = c_parent.degree_on_basis(c_dec)

            global_accum = 0
            for v in verts:
                if v is parent_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = raw_P(v_arity, base_ring).degree_on_basis(decoration(v))
                global_accum += v_deg - 1

            c_sp_deg = c_deg_P - 1
            before_deg = sum(
                subtree_degree(ch, raw_P, base_ring)
                for i, ch in enumerate(children(parent_vertex), start=1)
                if i < child_pos
            )
            koszul_exp = c_sp_deg * before_deg
            total_sign = sign_from_exponent(global_accum + p_deg_P + koszul_exp)

            p_elem = p_parent.term(p_dec)
            c_elem = c_parent.term(c_dec)
            composed = raw_P.compose(p_elem, child_pos, c_elem)

            for new_dec, coeff in composed:
                new_tree = contract_edge(tree, parent_vertex, child_pos, new_dec)
                result += total_sign * coeff * self.term((new_tree, a_tuple))

        return result

    def _d2_cobar(self, key) -> "TwistedBarComplex.Element":
        """Structural cobar differential: expand vertices via cooperad cocomposition.

        Sign at vertex v with split ``Δ^{i;m,n}(dec(v)) = Σ c_L ⊗ c_R``:
            ``(-1)^{global_accum + deg_C(c_L) + (deg_C(c_R) - 1) * before_deg}``
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        raw_C = cast(CooperadLike, self._raw_sym_seq)

        for curr_vertex in verts:
            curr_arity = vertex_arity(curr_vertex)
            curr_dec = decoration(curr_vertex)
            curr_parent = raw_C(curr_arity, base_ring)
            curr_elem = curr_parent.term(curr_dec)

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
                        result += total_sign * coeff * self.term((new_tree, a_tuple))

        return result

    def _dalpha_on_basis(self, key) -> "TwistedBarComplex.Element":
        """Twisted action differential via α: C → P.

        At each corolla vertex v (all children are leaves) with decoration c:
        1. Apply α(c) ∈ P(n)
        2. Apply the P-algebra action γ(α(c); a_1,...,a_n)
        3. Replace v and its k leaf children with a single leaf.

        Sign at DFS vertex v_j:
            ``(-1)^{Σ_{l<j} (deg(dec(v_l)) + vertex_shift)}``
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        cumulative = 0
        raw_S = cast(type, self._symmetric_sequence_cls)
        cooperad = self._cooperad_cls

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            s_parent = raw_S(v_arity, base_ring)
            vertex_sp_deg = s_parent.degree_on_basis(dec) + self._vertex_shift

            v_children = children(v)
            if all(is_leaf(c) for c in v_children):
                sign = sign_from_exponent(cumulative)
                leaf_labels = list(v_children)
                a_elems = [self._module.term(a_tuple[l - 1]) for l in leaf_labels]

                # Build the cooperad element for this vertex
                from uconf.constructions.bar_construction import BarConstruction
                if isinstance(cooperad, BarConstruction):
                    # Vertex decoration is a P-basis key; build bar corolla
                    bar_corolla_key = (dec,) + tuple(range(1, v_arity + 1))
                    c_parent = cooperad(v_arity, base_ring)
                    c_elem = c_parent.term(bar_corolla_key)
                else:
                    # Vertex decoration is a C-basis key directly
                    c_parent = cooperad(v_arity, base_ring)
                    c_elem = c_parent.term(dec)

                # Apply α to get P-element
                alpha_c = self._alpha(c_elem)

                # Apply P-algebra action
                action_result = self._algebra.act(alpha_c, a_elems)

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
        """Contract target_vertex (all-leaf children) to a single leaf.

        Returns (new_tree, new_a_tuple).
        """
        leaf_children = sorted(children(target_vertex))
        n = len(a_tuple)

        new_leaf_label = leaf_children[0]
        removed_leaves = set(leaf_children[1:])

        relabel: dict[int, int] = {}
        counter = 1
        for leaf in range(1, n + 1):
            if leaf not in removed_leaves:
                relabel[leaf] = counter
                counter += 1

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

    class Element(ParentedElementMixin["TwistedBarComplex"], CombinatorialFreeModule.Element):
        """An element of the twisted bar complex B_α(A)."""

        def boundary(self) -> "TwistedBarComplex.Element":
            """Apply the full twisted bar differential d = d_internal + d_2 + d_α."""
            parent = self.parent()
            return parent.boundary(self)

        def d_internal(self) -> "TwistedBarComplex.Element":
            """Apply the internal differential."""
            parent = self.parent()
            return parent._d_internal(self)

        def d2(self) -> "TwistedBarComplex.Element":
            """Apply the structural differential."""
            parent = self.parent()
            return parent._d2(self)

        def dalpha(self) -> "TwistedBarComplex.Element":
            """Apply the twisted action differential via α."""
            parent = self.parent()
            return parent._dalpha(self)


# ===========================================================================
# TwistedCobarComplex: Ω_α(V)
# ===========================================================================


class TwistedCobarComplex(TreeModule):
    """Twisted cobar complex Ω_α(V) for a twisting morphism α: C → P and C-coalgebra V.

    The underlying module is T_P(V), the free P-algebra on V.
    Basis keys are pairs ``(tree, v_tuple)`` where tree has P-decorated vertices
    and v_tuple carries V-module values at the leaves.

    The differential is d = d_internal + d_2 + d_α where:

    - d_internal: applies ∂_P to vertex decorations and ∂_V to leaf decorations
      with the Koszul sign rule (inherited from TreeModule).
    - d_2: contracts internal edges via P-composition (bar structural
      differential on the operad side).
    - d_α: at each leaf l with value v_l, applies the C-coalgebra coaction
      δ_k(v_l) = Σ c_k ⊗ v'_1 ⊗ ... ⊗ v'_k, then inserts a new vertex
      decorated by α(c_k) ∈ P(k).

    Args:
        alpha: A :class:`~uconf.core.twisting.TwistingMorphism` α: C → P.
        coalgebra: A :class:`~uconf.algebraic.coalgebra.CooperadCoalgebra` (C-coalgebra).
        base_ring: Coefficient ring.

    Note:
        The ``vertex_degree_shift`` is +1 (bar/suspension convention) when the
        operad is a BarConstruction, and -1 (cobar convention) for raw operads.
    """

    name: ClassVar[str] = "Ω_α"

    def __init__(self, alpha: TwistingMorphism, coalgebra: CooperadCoalgebra, base_ring):
        self._alpha = alpha
        self._coalgebra = coalgebra
        self._cooperad_cls = alpha.cooperad
        self._operad_cls = alpha.operad
        self._module = coalgebra.module

        # Determine vertex degree shift from the operad
        from uconf.constructions.cobar_construction import CobarConstruction
        if isinstance(self._operad_cls, CobarConstruction):
            vertex_shift = -1  # cobar trees: degree = Σ (deg_C(v) - 1)
            sym_seq = self._operad_cls.cooperad_cls  # C for decorations
        else:
            vertex_shift = 1  # bar trees: degree = Σ (deg_P(v) + 1)
            sym_seq = self._operad_cls  # P for decorations

        self._vertex_shift = vertex_shift
        self._raw_sym_seq = sym_seq

        super().__init__(
            symmetric_sequence_cls=sym_seq,
            inner_module=coalgebra.module,
            base_ring=base_ring,
            vertex_degree_shift=vertex_shift,
            name=f"Ω_{{{alpha.name}}}({coalgebra.module})",
        )

        # Override the inherited boundary with the twisted differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: TreeModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dalpha = self.module_morphism(on_basis=self._dalpha_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "TwistedCobarComplex.Element":
        """Total differential d = d_internal + d_2 + d_α."""
        return (
            TreeModule._boundary_on_basis(self, key)
            + self._d2_on_basis(key)
            + self._dalpha_on_basis(key)
        )

    def _d2_on_basis(self, key) -> "TwistedCobarComplex.Element":
        """Structural differential on operad-decorated trees.

        When operad = Ω(C), expands vertices via C-cocomposition (cobar d_2).
        When operad = P (raw), contracts internal edges via P-composition (bar d_2).
        """
        from uconf.constructions.cobar_construction import CobarConstruction

        if isinstance(self._operad_cls, CobarConstruction):
            return self._d2_cobar(key)
        else:
            return self._d2_bar(key)

    def _d2_bar(self, key) -> "TwistedCobarComplex.Element":
        """Structural bar differential: contract internal edges via P-composition."""
        tree, v_tuple = key
        if is_leaf(tree):
            return self.zero()

        edges = internal_edges_dfs(tree)
        if not edges:
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        raw_P = cast(OperadLike, self._raw_sym_seq)

        for parent_vertex, child_pos, child_vertex in edges:
            p_arity = vertex_arity(parent_vertex)
            c_arity = vertex_arity(child_vertex)
            p_dec = decoration(parent_vertex)
            c_dec = decoration(child_vertex)

            p_parent = raw_P(p_arity, base_ring)
            c_parent = raw_P(c_arity, base_ring)

            p_deg_P = p_parent.degree_on_basis(p_dec)
            c_deg_P = c_parent.degree_on_basis(c_dec)

            global_accum = 0
            for v in verts:
                if v is parent_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = raw_P(v_arity, base_ring).degree_on_basis(decoration(v))
                global_accum += v_deg - 1

            c_sp_deg = c_deg_P - 1
            before_deg = sum(
                subtree_degree(ch, raw_P, base_ring)
                for i, ch in enumerate(children(parent_vertex), start=1)
                if i < child_pos
            )
            koszul_exp = c_sp_deg * before_deg
            total_sign = sign_from_exponent(global_accum + p_deg_P + koszul_exp)

            p_elem = p_parent.term(p_dec)
            c_elem = c_parent.term(c_dec)
            composed = raw_P.compose(p_elem, child_pos, c_elem)

            for new_dec, coeff in composed:
                new_tree = contract_edge(tree, parent_vertex, child_pos, new_dec)
                result += total_sign * coeff * self.term((new_tree, v_tuple))

        return result

    def _d2_cobar(self, key) -> "TwistedCobarComplex.Element":
        """Structural cobar differential: expand vertices via cooperad cocomposition."""
        tree, v_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        base_ring = self.base_ring()
        verts = vertices_dfs(tree)
        raw_C = cast(CooperadLike, self._raw_sym_seq)

        for curr_vertex in verts:
            curr_arity = vertex_arity(curr_vertex)
            curr_dec = decoration(curr_vertex)
            curr_parent = raw_C(curr_arity, base_ring)
            curr_elem = curr_parent.term(curr_dec)

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
                        result += total_sign * coeff * self.term((new_tree, v_tuple))

        return result

    def _dalpha_on_basis(self, key) -> "TwistedCobarComplex.Element":
        """Twisted coaction differential via α: C → P.

        At each leaf l with value v_l:
        1. Apply C-coalgebra coaction δ_k(v_l) = Σ c_k ⊗ v'_1 ⊗ ... ⊗ v'_k
        2. Apply α(c_k) ∈ P(k) to get the decoration for the new vertex
        3. Insert new internal vertex at leaf l position

        Sign at leaf l:
            ``(-1)^{deg_tree(tree) + Σ_{j < l} deg_V(v_j)}``
        """
        tree, v_tuple = key
        n = len(v_tuple)
        raw_S = cast(type, self._symmetric_sequence_cls)

        # Compute tree degree for sign
        if is_leaf(tree):
            deg_tree = 0
        else:
            if self._vertex_shift == -1:
                deg_tree = subtree_degree_cobar(tree, raw_S, self.base_ring())
            else:
                deg_tree = subtree_degree(tree, raw_S, self.base_ring())

        result = self.zero()
        base_ring = self.base_ring()
        cumulative_v = 0

        for leaf_l in range(1, n + 1):
            v_key = v_tuple[leaf_l - 1]
            sign_exp = deg_tree + cumulative_v
            sign = sign_from_exponent(sign_exp)

            v_elem = self._module.term(v_key)

            # Try all coaction arities k >= 2
            for k in range(2, n + 2):
                coaction = self._coalgebra.coact(v_elem, k)
                for (c_key, *new_v_keys_raw), coeff in coaction:
                    new_v_keys = tuple(new_v_keys_raw) if new_v_keys_raw else ()
                    if len(new_v_keys) != k:
                        continue

                    # Apply α to get P-decoration for the new vertex
                    c_parent = self._cooperad_cls(k, base_ring)
                    c_elem = c_parent.term(c_key)
                    alpha_c = self._alpha(c_elem)

                    # Extract the P-key from α(c)
                    for p_key, alpha_coeff in alpha_c:
                        new_tree = self._expand_leaf(tree, leaf_l, p_key, k)
                        new_v_tuple = v_tuple[:leaf_l - 1] + new_v_keys + v_tuple[leaf_l:]
                        result += sign * coeff * alpha_coeff * self.term((new_tree, new_v_tuple))

            cumulative_v += self._module.degree_on_basis(v_key)

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _expand_leaf(self, tree, leaf_l: int, new_dec, k: int):
        """Replace leaf leaf_l with a new internal vertex having k children."""

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

    class Element(ParentedElementMixin["TwistedCobarComplex"], CombinatorialFreeModule.Element):
        """An element of the twisted cobar complex Ω_α(V)."""

        def boundary(self) -> "TwistedCobarComplex.Element":
            """Apply the full twisted cobar differential d = d_internal + d_2 + d_α."""
            parent = self.parent()
            return parent.boundary(self)

        def d_internal(self) -> "TwistedCobarComplex.Element":
            """Apply the internal differential."""
            parent = self.parent()
            return parent._d_internal(self)

        def d2(self) -> "TwistedCobarComplex.Element":
            """Apply the structural differential."""
            parent = self.parent()
            return parent._d2(self)

        def dalpha(self) -> "TwistedCobarComplex.Element":
            """Apply the twisted coaction differential via α."""
            parent = self.parent()
            return parent._dalpha(self)
