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

from sage.all import CombinatorialFreeModule, Family, cached_method

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import TreeModule
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


class BarComplexAlgebra(TreeModule):
    """Bar complex B_P(A) of a P-algebra A.

    Subclasses :class:`~uconf.algebraic.tree_module.TreeModule`
    with ``vertex_degree_shift = +1`` (suspension convention).  Inherits basis
    key validation, degree computation, basis iteration, and the internal
    differential (d_1 + d_A) from TreeModule.  Only the structural
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
        self._n_factors: int | None = None

        super().__init__(
            symmetric_sequence_cls=algebra.operad_cls,
            inner_module=algebra.module,
            base_ring=base_ring,
            vertex_degree_shift=1,
            name=f"B({algebra.operad_cls.name}; {algebra.module})",
        )

        # Override the inherited boundary with the bar-specific differential
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self._d_internal = self.module_morphism(
            on_basis=lambda key: TreeModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d2 = self.module_morphism(on_basis=self._d2_on_basis, codomain=self)
        self._dact = self.module_morphism(on_basis=self._dact_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # n_factors filtering
    # -----------------------------------------------------------------------

    def set_n_factors(self, n_factors: int | None) -> None:
        """Restrict basis enumeration to elements with exactly *n_factors*
        occurrences of the coefficient module.

        When the inner module of the underlying algebra is a tensor product
        ``A ⊗ Free_P(M)`` (as produced by :func:`~uconf.algebraic.conf.labelled_configuration_model`),
        each leaf of the bar tree carries a tensor key ``(a_key, (p_key, m_tuple))``
        where ``m_tuple`` is a tuple of coefficient-module basis keys.
        The *total* number of coefficient-module keys across all leaves is
        ``Σ_i len(m_tuple_i)``.  Setting ``n_factors`` restricts the basis
        enumeration to exactly that total.

        This also implies a finite arity bound on the bar tree (at most
        ``n_factors`` leaves) and enables automatic connectivity computation.

        Passing ``None`` removes the restriction.

        Clears cached ``graded_basis`` results.
        """
        self._n_factors = n_factors
        if n_factors is not None:
            self.set_max_arity(n_factors)
        else:
            self.set_max_arity(None)
        self.graded_basis.clear_cache()

    @property
    def connectivity(self) -> int:
        """Minimum degree of any basis element.

        When ``_n_factors`` is set, computes a lower bound on the degree of
        elements with exactly that many coefficient-module keys.  The bound
        accounts for the minimum contributions from:

        - Coefficient factors: ``n_factors * coeff_conn``
        - Manifold/left factors: ``k * left_conn`` (for ``k`` bar-tree leaves)
        - Free-algebra operad factors at various arities
        - Bar-tree vertex decorations with degree shift +1

        The analytical bound is conservative; the true connectivity may be
        higher.

        TODO: The formula below is a rough lower bound.  For specific models
        (e.g. the sphere algebra in dimension N with F coefficient factors),
        a tighter bound may be computable.  The bound assumes that all
        non-coefficient degree contributions can be simultaneously minimised,
        which may not always be achievable.
        """
        if self._n_factors is None:
            return super().connectivity

        F = self._n_factors

        # Try to extract coefficient-module connectivity and left-module
        # connectivity from the tensor product structure.
        inner = self._inner_module
        coeff_conn = 0
        left_conn = 0
        operad_conn = 0

        # The inner module is typically a Sage tensor product A ⊗ B where
        # A = manifold model module, B = free algebra module.
        # HadamardTensorAlgebra sets module = tensor([left_module, right_module])
        if hasattr(inner, '_sets') and len(inner._sets) == 2:
            left_mod, right_mod = inner._sets
            left_conn = int(getattr(left_mod, "connectivity", 0))
            # The right module is a FreeAlgebraModule; its connectivity comes
            # from the coefficient module it wraps.
            if hasattr(right_mod, "_inner_module"):
                coeff_conn = int(getattr(right_mod._inner_module, "connectivity", 0))
            if hasattr(right_mod, "_operad_cls"):
                operad_conn = int(getattr(right_mod._operad_cls, "connectivity", 0))
        else:
            left_conn = int(getattr(inner, "connectivity", 0))

        # Operad connectivity for the bar tree vertices
        bar_operad_conn = int(getattr(self._symmetric_sequence_cls, "connectivity", 0))

        # Lower bound on degree for F coefficient factors distributed across
        # k bar-tree leaves (1 ≤ k ≤ F):
        #
        # degree ≥ bar_vertex_contrib + k * left_conn + free_alg_operad_contrib + F * coeff_conn
        #
        # bar_vertex_contrib ≥ 0 for a single vertex at minimum arity (bar_operad_conn + 1 ≥ 0
        # when bar_operad_conn ≥ -1); for a leaf (k=1) it's 0.
        #
        # free_alg_operad_contrib depends on how F factors are distributed:
        # - If 1 leaf with F factors: operad at arity F, contrib ≥ operad_conn * (F-1)
        # - If F leaves with 1 factor each: operad at arity 1, contrib = 0 for each
        #
        # We take the minimum over all valid k.
        min_deg = None
        for k in range(1, F + 1):
            # Bar tree with k leaves: minimum vertex contribution
            # A tree with k leaves needs (k-1) compositions; at minimum a single
            # arity-k corolla.  Its vertex degree = bar_operad_conn * (min vertex weight)
            # but the shift is +1.  For a corolla: deg = bar_operad_conn + 1
            # For k=1 (leaf only): no vertex, contrib = 0
            if k == 1:
                bar_contrib = 0
            else:
                # Corolla at arity k: deg ≥ bar_operad_conn + 1
                # More complex trees have more vertices, each contributing ≥ bar_operad_conn + 1
                bar_contrib = bar_operad_conn + 1

            # Left (manifold) contribution: k leaves × min_left_deg
            left_contrib = k * left_conn

            # Free-algebra operad contribution: distribute F factors across k leaves.
            # Each leaf i gets f_i factors (f_i ≥ 1, Σf_i = F).
            # The operad at arity f_i contributes ≥ operad_conn * (f_i - 1) if f_i ≥ 2, else 0.
            # Total: operad_conn * Σ(f_i - 1) = operad_conn * (F - k)
            operad_contrib = operad_conn * (F - k)

            # Coefficient contribution: F factors × coeff_conn
            coeff_contrib = F * coeff_conn

            total = bar_contrib + left_contrib + operad_contrib + coeff_contrib
            if min_deg is None or total < min_deg:
                min_deg = total

        return min_deg if min_deg is not None else 0

    def count_factors(self, key) -> int:
        """Count the total number of coefficient-module keys in a basis key.

        For a basis key ``(tree, a_tuple)`` where each ``a_tuple[i]`` is a
        tensor key ``(left_key, right_key)``, and ``right_key = (p_key, m_tuple)``
        is a free-algebra key, returns ``Σ_i len(m_tuple_i)``.

        If the inner module is not a tensor product with a free algebra
        right factor, falls back to counting one factor per leaf.
        """
        _tree, a_tuple = key
        total = 0
        for a_entry in a_tuple:
            if isinstance(a_entry, (tuple, list)) and len(a_entry) == 2:
                _left_key, right_key = a_entry
                if isinstance(right_key, (tuple, list)) and len(right_key) == 2:
                    _p_key, m_tuple = right_key
                    if isinstance(m_tuple, (tuple, list)):
                        total += len(m_tuple)
                        continue
            # Fallback: each leaf counts as 1 factor
            total += 1
        return total

    def basis_it(self, d: int):
        """Iterate over basis elements of total degree ``d``.

        When ``_n_factors`` is set, only yields elements whose total number
        of coefficient-module keys equals ``_n_factors``.  Otherwise delegates
        to the parent :class:`TreeModule` implementation.
        """
        for elem in super().basis_it(d):
            if self._n_factors is not None:
                for key, _coeff in elem:
                    if self.count_factors(key) != self._n_factors:
                        break
                else:
                    yield elem
            else:
                yield elem

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_it(d))

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Total differential d = d_internal + d_2 + d_act.

        ``d_internal`` is the interleaved DFS differential inherited from
        :class:`~uconf.algebraic.tree_module.TreeModule`, applying ``∂_P``
        to vertex decorations and ``∂_A`` to leaf decorations with the
        Koszul sign rule.
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
