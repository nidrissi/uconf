"""Bar complex B_P(A) of a P-algebra A.

For a connected augmented dg-operad P and a P-algebra (A, γ), the bar complex
is the dg-module

    B_P(A) = (B(P) ∘ A, d)

where B(P) ∘ A is the composite product and d has four components:

- d_1  : apply ∂_P to each internal vertex decoration.
- d_2  : structural bar differential (contract internal edges via operad composition).
- d_A  : apply ∂_A to each leaf decoration (Koszul sign from tensor product rule).
- d_act: apply the P-algebra action γ at each internal vertex all of whose
         children are leaves (sign from vertex-degree accumulation in DFS order).

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

from typing import ClassVar, Iterator

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.free_algebra import (
    _module_basis_keys_in_degree,
    _tuples_in_degree_precomputed,
)
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    contract_edge,
    decoration,
    internal_edges_dfs,
    is_leaf,
    subtree_degree,
    tree_arity,
    vertices_dfs,
    vertex_arity,
)


class BarComplexAlgebra(CombinatorialFreeModule):
    """Bar complex B_P(A) of a P-algebra A.

    Args:
        algebra: An :class:`uconf.algebra.OperadAlgebra` instance (the P-algebra A).
        base_ring: Coefficient ring

    Basis keys are pairs ``(tree, a_tuple)`` where:

    - ``tree`` is an integer (single leaf, arity 1) or a tuple representing a
      decorated rooted tree with leaves labeled ``1, ..., n``.
    - ``a_tuple`` is a tuple of ``n`` basis keys of the underlying module A.

    The degree is ``deg_bar(tree) + Σ_i deg_A(a_tuple[i])``.
    """

    name: ClassVar[str] = "B"

    def __init__(self, algebra: OperadAlgebra, base_ring):
        self._algebra = algebra
        self._operad_cls = algebra.operad_cls
        self._module = algebra.module

        name = f"B({algebra.operad_cls.name}; {algebra.module})"
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
        self._dA = self.module_morphism(on_basis=self._dA_on_basis, codomain=self)
        self._dact = self.module_morphism(on_basis=self._dact_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Basis key validation and element construction
    # -----------------------------------------------------------------------

    def _validate_basis_key(self, key):
        """Validate a basis key ``(tree, a_tuple)``."""
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        tree, a_tuple = key[0], key[1]
        if not isinstance(a_tuple, (tuple, list)):
            return None

        if is_leaf(tree):
            if tree != 1 or len(a_tuple) != 1:
                return None
            a_key = self._validate_a_key(a_tuple[0])
            if a_key is None:
                return None
            return (1, (a_key,))

        # Internal vertex tree
        n = tree_arity(tree)
        if len(a_tuple) != n:
            return None
        new_a = []
        for a_key in a_tuple:
            validated = self._validate_a_key(a_key)
            if validated is None:
                return None
            new_a.append(validated)
        return (tree, tuple(new_a))

    def _validate_a_key(self, a_key):
        """Validate one A-module basis key."""
        if hasattr(self._module, "_validate_basis_key"):
            return self._module._validate_basis_key(a_key)
        return a_key

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
        """Degree of ``(tree, a_tuple)`` = deg_bar(tree) + Σ_i deg_A(a_i)."""
        tree, a_tuple = key
        tree_deg = (
            0
            if is_leaf(tree)
            else subtree_degree(tree, self._operad_cls, self.base_ring())
        )
        a_deg = sum(self._module.degree_on_basis(a) for a in a_tuple)
        return tree_deg + a_deg

    # -----------------------------------------------------------------------
    # Basis iteration
    # -----------------------------------------------------------------------

    def basis_it(self, d: int) -> Iterator["BarComplexAlgebra.Element"]:
        """Iterate over basis elements of degree *d*.

        Yields all ``(tree, a_tuple)`` pairs with total degree ``d``, where
        ``tree`` is a bar-construction shuffle tree of bar degree ``d_bar``
        and ``a_tuple`` is a tuple of ``n`` basis keys of the algebra module
        *A* with total A-degree ``d - d_bar``.

        The iteration covers all arities ``n ≥ 1``.  For connected operads,
        ``deg_bar(tree) ≥ weight ≥ 1`` for ``n ≥ 2``, so the arity is bounded
        by ``d + 1`` when *A* is non-negatively graded.

        Args:
            d: Homological degree to enumerate.

        Yields:
            Elements of this module with degree ``d``.
        """
        from uconf.constructions.bar_construction import BarConstruction

        A = self._module
        P = self._operad_cls
        R = self.base_ring()
        bar_cls = BarConstruction(P)

        # Pre-collect A-keys by degree from 0 to d.
        a_keys_by_deg: dict[int, list] = {}
        for d_a in range(d + 1):
            keys = list(_module_basis_keys_in_degree(A, d_a))
            if keys:
                a_keys_by_deg[d_a] = keys

        # Arity 1: single leaf (bar tree = leaf "1", bar degree = 0)
        for a_key in a_keys_by_deg.get(d, []):
            yield self.term((1, (a_key,)))

        # Arity n ≥ 2: tree from BarConstruction, A-tuple from module
        # For connected P, bar degree ≥ weight ≥ 1, so d_bar ≥ 1 and
        # arity n ≤ d_bar + 1 ≤ d + 1 when A is non-negatively graded.
        for n in range(2, d + 2):
            bar_comp = bar_cls(n, R)
            for d_bar in range(1, d + 1):
                d_A = d - d_bar
                if d_A < 0:
                    continue
                a_tuples = list(_tuples_in_degree_precomputed(a_keys_by_deg, n, d_A))
                if not a_tuples:
                    continue
                for bar_elem in bar_comp.basis_it(d_bar):
                    for tree_key in bar_elem.support():
                        for a_tuple in a_tuples:
                            yield self.term((tree_key, a_tuple))

    # -----------------------------------------------------------------------
    # Differential
    # -----------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Total differential d = d_1 + d_2 + d_A + d_act."""
        return (
            self._d1_on_basis(key)
            + self._d2_on_basis(key)
            + self._dA_on_basis(key)
            + self._dact_on_basis(key)
        )

    def _d1_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Apply ∂_P to each internal vertex decoration.

        Sign at DFS vertex v_j: ``(-1)^{Σ_{l<j} (deg_P(dec(v_l)) + 1)}``.
        """
        tree, a_tuple = key
        if is_leaf(tree):
            return self.zero()

        result = self.zero()
        verts = vertices_dfs(tree)
        base_ring = self.base_ring()
        cumulative = 0

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            operad_parent = self._operad_cls(v_arity, base_ring)
            vertex_sp_deg = operad_parent.degree_on_basis(dec) + 1
            sign = sign_from_exponent(cumulative)

            bdry = operad_parent.boundary(operad_parent.term(dec))
            for new_dec, coeff in bdry:
                new_tree = self._replace_decoration(tree, v, new_dec)
                result += sign * coeff * self.term((new_tree, a_tuple))

            cumulative += vertex_sp_deg

        return result

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

        for parent_vertex, child_pos, child_vertex in edges:
            p_arity = vertex_arity(parent_vertex)
            c_arity = vertex_arity(child_vertex)
            p_dec = decoration(parent_vertex)
            c_dec = decoration(child_vertex)

            p_parent = self._operad_cls(p_arity, base_ring)
            c_parent = self._operad_cls(c_arity, base_ring)

            p_deg_P = p_parent.degree_on_basis(p_dec)
            c_deg_P = c_parent.degree_on_basis(c_dec)

            # Global accumulation: deg_P(v) - 1 for each DFS vertex before parent
            global_accum = 0
            for v in verts:
                if v is parent_vertex:
                    break
                v_arity = vertex_arity(v)
                v_deg = self._operad_cls(v_arity, base_ring).degree_on_basis(
                    decoration(v)
                )
                global_accum += v_deg - 1

            # Koszul sign from the child's position among parent's children
            c_sp_deg = c_deg_P - 1
            before_deg = sum(
                subtree_degree(ch, self._operad_cls, base_ring)
                for i, ch in enumerate(children(parent_vertex), start=1)
                if i < child_pos
            )
            koszul_exp = c_sp_deg * before_deg

            total_sign = sign_from_exponent(global_accum + p_deg_P + koszul_exp)

            # Compose and build contracted tree (a_tuple unchanged)
            p_elem = p_parent.term(p_dec)
            c_elem = c_parent.term(c_dec)
            composed = self._operad_cls.compose(p_elem, child_pos, c_elem)

            for new_dec, coeff in composed:
                new_tree = contract_edge(tree, parent_vertex, child_pos, new_dec)
                result += total_sign * coeff * self.term((new_tree, a_tuple))

        return result

    def _dA_on_basis(self, key) -> "BarComplexAlgebra.Element":
        """Apply ∂_A to each leaf decoration.

        Sign at leaf with label ``i`` (1-indexed):

            ``(-1)^{deg_bar(tree) + Σ_{j < i} deg_A(a_j)}``
        """
        tree, a_tuple = key
        if is_leaf(tree):
            # Single leaf: no internal vertices, deg_bar = 0
            deg_bar = 0
        else:
            deg_bar = subtree_degree(tree, self._operad_cls, self.base_ring())

        result = self.zero()
        cumulative_a = 0

        for i in range(1, len(a_tuple) + 1):
            a_key = a_tuple[i - 1]
            sign_exp = deg_bar + cumulative_a
            sign = sign_from_exponent(sign_exp)

            a_elem = self._module.term(a_key)
            bdry = self._module.boundary(a_elem)

            for new_a_key, coeff in bdry:
                new_a_tuple = a_tuple[: i - 1] + (new_a_key,) + a_tuple[i:]
                result += sign * coeff * self.term((tree, new_a_tuple))

            cumulative_a += self._module.degree_on_basis(a_key)

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

        for v in verts:
            v_arity = vertex_arity(v)
            dec = decoration(v)
            operad_parent = self._operad_cls(v_arity, base_ring)
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
                    new_tree, new_a_tuple = self._contract_leaf_vertex(
                        tree, v, a_tuple, new_a_key
                    )
                    result += sign * coeff * self.term((new_tree, new_a_tuple))

            cumulative += vertex_sp_deg

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

    class Element(
        ParentedElementMixin["BarComplexAlgebra"], CombinatorialFreeModule.Element
    ):
        """An element of the bar complex B_P(A)."""

        def boundary(self) -> "BarComplexAlgebra.Element":
            """Apply the full bar differential d = d_1 + d_2 + d_A + d_act."""
            parent = self._parent()
            return parent.boundary(self)

        def d1(self) -> "BarComplexAlgebra.Element":
            """Apply ∂_P to each vertex decoration."""
            parent = self._parent()
            return parent._d1(self)

        def d2(self) -> "BarComplexAlgebra.Element":
            """Apply the structural bar differential (contract internal edges)."""
            parent = self._parent()
            return parent._d2(self)

        def dA(self) -> "BarComplexAlgebra.Element":
            """Apply ∂_A to each leaf decoration."""
            parent = self._parent()
            return parent._dA(self)

        def dact(self) -> "BarComplexAlgebra.Element":
            """Apply the P-algebra action at all-leaf-children vertices."""
            parent = self._parent()
            return parent._dact(self)
