"""Cobar construction for connected dg-cooperads.

The cobar construction Ω(C) of a connected coaugmented dg-cooperad C is the
free operad on the desuspension of the coaugmentation coideal:

    Ω(C) = (T(s⁻¹C̄), d_1 + d_2)

where:
- C̄ is the coaugmentation coideal (C̄(1) = 0 for connected cooperads)
- s⁻¹C̄ denotes the desuspension (degree shift by -1 per arity-1)
- T denotes the free operad (decorated rooted trees)
- d_1 is the internal differential from C
- d_2 is the structural differential from vertex expansions

Reference: Loday-Vallette "Algebraic Operads", Chapter 6.
"""

from __future__ import annotations

from typing import ClassVar, Iterator

from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis, SymmetricGroup

from .signs import shifted_boundary_sign, sign_from_exponent
from .trees import (
    children,
    decoration,
    expand_vertex,
    graft,
    internal_edges_dfs,
    is_internal,
    is_leaf,
    leaves,
    relabel_leaves,
    subtree_degree,
    subtree_degree_cobar,
    tree_arity,
    tree_to_latex,
    tree_to_string,
    validate_tree,
    vertex_arity,
    vertices_dfs,
    weight,
)


class CobarConstruction:
    """Factory for cobar construction components of a connected dg-cooperad.

    Args:
        cooperad_cls: Base cooperad class (e.g., ``SurjectionLinearDual``).
        max_weight: Maximum tree weight for enumeration helpers (default 3).

    The cobar construction Ω(C) is a dg-operad whose arity-n component has
    basis elements given by rooted trees with n leaves, where internal
    vertices are decorated by elements of C̄ (the coaugmentation coideal).

    For connected cooperads, C̄(1) = 0, so all internal vertices have arity >= 2.

    Note:
        Unlike BarConstruction, this does not currently implement automatic
        shuffle tree normalization. The differential sign formulas need to be
        reworked to be compatible with shuffle normalization. Use the functions
        ``to_shuffle_tree_cobar`` and ``is_shuffle_tree`` from ``trees`` module
        for manual normalization if needed.
    """

    def __init__(self, cooperad_cls, max_weight: int = 3):
        self.cooperad_cls = cooperad_cls
        self.max_weight = int(max_weight)
        self.name = f"Ω({cooperad_cls.name})"

    def __call__(self, n: int, base_ring=QQ) -> "CobarConstruction.Component":
        return CobarConstruction.Component(self, n, base_ring)

    def unit(self, base_ring=QQ) -> "CobarConstruction.Element":
        """Return the unit element (identity in arity 1).

        For the free operad, the unit is represented by a single leaf.
        """
        component = self(1, base_ring)
        # Unit is the single leaf "1" (a trivial tree with no internal vertices)
        return component.term(1)

    def compose(
        self, x: "CobarConstruction.Element", i: int, y: "CobarConstruction.Element"
    ) -> "CobarConstruction.Element":
        """Free operad composition: graft y onto leaf i of x.

        In the free operad T(s⁻¹C̄), composition is pure tree grafting
        with no signs (signs are encoded in the s⁻¹C̄ structure).
        """
        x_parent = x.parent()
        y_parent = y.parent()

        if x_parent.factory is not self or y_parent.factory is not self:
            raise TypeError("Both elements must belong to this cobar construction.")
        if x_parent.base_ring() != y_parent.base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x_parent.arity()
        n = y_parent.arity()

        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")

        target = self(m + n - 1, x_parent.base_ring())
        result = target.zero()

        for x_tree, x_coeff in x:
            for y_tree, y_coeff in y:
                # Graft y_tree onto leaf i of x_tree
                grafted = graft(x_tree, i, y_tree)
                result += x_coeff * y_coeff * target.term(grafted)

        return result

    class Component(CombinatorialFreeModule):
        """A fixed-arity component of the cobar construction operad."""

        name: ClassVar[str] = "Ω"

        def __init__(self, factory: "CobarConstruction", n: int, base_ring=QQ):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            self.factory = factory
            self._arity = int(n)
            self._cooperad_cls = factory.cooperad_cls
            self._max_weight = factory.max_weight

            name = f"{factory.name}{n}"
            super().__init__(
                base_ring,
                tuple,
                prefix=name,
                category=GradedModulesWithBasis(base_ring),
            )
            self.rename(name)
            self._symmetric_group = SymmetricGroup(n) if n > 0 else None

            self.boundary = self.module_morphism(
                on_basis=self._boundary_on_basis,
                codomain=self,
            )
            self._d1 = self.module_morphism(
                on_basis=self._d1_on_basis,
                codomain=self,
            )
            self._d2 = self.module_morphism(
                on_basis=self._d2_on_basis,
                codomain=self,
            )

        def _validate_basis_key(self, basis_key):
            """Validate a tree basis key.

            For arity 1, the unit (leaf 1) is valid.
            For arity >= 2, must be a valid decorated tree.
            """
            if self._arity == 1:
                if basis_key == 1:
                    return 1
                # Check if it's a tree with no internal vertices
                if is_leaf(basis_key) and basis_key == 1:
                    return 1
                return None

            return validate_tree(
                basis_key, self._arity, self._cooperad_cls, self.base_ring()
            )

        def _element_constructor_(self, x):
            """Build elements from tree basis keys or sparse dictionaries."""
            if isinstance(x, CobarConstruction.Element):
                if x.parent().factory is self.factory:
                    return self.sum_of_terms((basis, coeff) for basis, coeff in x)
                raise TypeError("Element from different cobar construction")

            if isinstance(x, dict):
                clean_dict = {}
                for key, coeff in x.items():
                    clean_key = self._validate_basis_key(key)
                    if clean_key is None:
                        continue
                    clean_dict[clean_key] = clean_dict.get(clean_key, 0) + coeff
                return super()._element_constructor_(clean_dict)

            if isinstance(x, (tuple, int)):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                return self.term(clean_key)

            return super()._element_constructor_(x)

        def arity(self) -> int:
            return self._arity

        def degree_on_basis(self, tree) -> int:
            """Compute the degree of a tree in Ω(C).

            The degree is sum over all vertices v of:
                deg_C(decoration(v)) - (arity(v) - 1)

            For the unit (leaf 1 in arity 1), degree is 0.
            """
            if is_leaf(tree):
                return 0
            return subtree_degree_cobar(tree, self._cooperad_cls, self.base_ring())

        def _repr_term(self, basis_element) -> str:
            """String representation of one cobar basis tree."""
            if is_leaf(basis_element):
                return "id"
            return tree_to_string(basis_element, self.factory.cooperad_cls.name)

        def _latex_term(self, basis_element) -> str:
            """LaTeX representation of one cobar basis tree."""
            if is_leaf(basis_element):
                return "\\mathrm{id}"
            return tree_to_latex(basis_element, self.factory.cooperad_cls.name)

        def _boundary_on_basis(self, tree) -> "CobarConstruction.Element":
            """Compute the cobar differential d = d_1 + d_2 on a tree.

            d_1: internal differential, applies C.boundary to each vertex
            d_2: structural differential, expands vertices using cocomposition
            """
            if is_leaf(tree):
                return self.zero()
            return self._d1_on_basis(tree) + self._d2_on_basis(tree)

        def _d1_on_basis(self, tree) -> "CobarConstruction.Element":
            """Internal differential: apply cooperad boundary to each vertex."""
            if is_leaf(tree):
                return self.zero()

            result = self.zero()
            verts = vertices_dfs(tree)
            base_ring = self.base_ring()

            cumulative_degree = 0

            for j, vertex in enumerate(verts):
                v_arity = vertex_arity(vertex)
                dec = decoration(vertex)
                cooperad_parent = self._cooperad_cls(v_arity, base_ring)

                # Degree of this vertex in s⁻¹C̄
                vertex_sinv_degree = cooperad_parent.degree_on_basis(dec) - 1

                sign = sign_from_exponent(cumulative_degree)

                # Apply boundary to this vertex's decoration
                dec_elem = cooperad_parent.term(dec)
                bdry = cooperad_parent.boundary(dec_elem)

                for new_dec, coeff in bdry:
                    new_tree = self._replace_vertex_decoration_by_index(
                        tree, verts, j, new_dec
                    )
                    result += sign * coeff * self.term(new_tree)

                cumulative_degree += vertex_sinv_degree

            return result

        def _d2_on_basis(self, tree) -> "CobarConstruction.Element":
            """Structural differential: expand vertices using cocomposition."""
            if is_leaf(tree):
                return self.zero()

            result = self.zero()
            base_ring = self.base_ring()
            verts = vertices_dfs(tree)

            for curr_vertex in verts:
                curr_arity = vertex_arity(curr_vertex)
                curr_dec = decoration(curr_vertex)
                curr_parent = self._cooperad_cls(curr_arity, base_ring)
                curr_elem = curr_parent(curr_dec)

                # Cumulative s^{-1}C-degree of DFS vertices before current
                global_accum = 0
                for vertex in verts:
                    if vertex is curr_vertex:
                        break
                    v_arity = vertex_arity(vertex)
                    v_deg = self._cooperad_cls(v_arity, base_ring).degree_on_basis(
                        decoration(vertex)
                    )
                    global_accum += v_deg - 1

                for m in range(2, curr_arity):
                    n = curr_arity - m + 1
                    for i in range(1, m + 1):
                        cocomp = curr_parent.infinitesimal_cocompose(curr_elem, i, m, n)
                        for (dec_left, dec_right), coeff in cocomp:
                            right_parent = self._cooperad_cls(n, base_ring)
                            right_sinv_deg = right_parent.degree_on_basis(dec_right) - 1

                            left_parent = self._cooperad_cls(m, base_ring)
                            left_degree = left_parent.degree_on_basis(dec_left)

                            # Cobar-degree of subtrees at positions 1, ..., i-1
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
                                tree, curr_vertex, i, dec_left, dec_right, m, n
                            )
                            result += total_sign * coeff * self.term(new_tree)
            return result

        def _replace_vertex_decoration_by_index(
            self, tree, vertices: list, index: int, new_decoration: tuple
        ) -> tuple:
            """Replace the decoration of the index-th vertex (DFS order)."""
            target_vertex = vertices[index]
            return self._replace_decoration_rec(tree, target_vertex, new_decoration)

        def _replace_decoration_rec(self, node, target: tuple, new_decoration: tuple):
            """Recursively replace decoration of target vertex."""
            if is_leaf(node):
                return node
            if node is target:
                return (new_decoration,) + children(node)
            new_children = tuple(
                self._replace_decoration_rec(c, target, new_decoration)
                for c in children(node)
            )
            return (decoration(node),) + new_children

        @staticmethod
        def unit() -> "CobarConstruction.Element":
            """The operadic unit is handled by the factory."""
            raise NotImplementedError("Use factory.unit() instead")

        def compose(
            self, x: "CobarConstruction.Element", i: int, y: "CobarConstruction.Element"
        ) -> "CobarConstruction.Element":
            """Delegate to factory compose."""
            return self.factory.compose(x, i, y)

    class Element(CombinatorialFreeModule.Element):
        """Element of a cobar construction operad component."""

        def pretty(self) -> str:
            """Return a readable linear-combination string for this element."""
            if not self:
                return "0"

            pieces = []
            for basis, coeff in self:
                term = self.parent()._repr_term(basis)
                if coeff == 1:
                    pieces.append(term)
                elif coeff == -1:
                    pieces.append(f"-{term}")
                else:
                    pieces.append(f"{coeff}*{term}")
            return " + ".join(pieces).replace("+ -", "- ")

        def pretty_latex(self) -> str:
            """Return a LaTeX linear-combination string for this element."""
            if not self:
                return "0"

            pieces = []
            for basis, coeff in self:
                term = self.parent()._latex_term(basis)
                if coeff == 1:
                    pieces.append(term)
                elif coeff == -1:
                    pieces.append(f"-{term}")
                else:
                    pieces.append(f"{coeff} \\left({term}\\right)")
            return " + ".join(pieces).replace("+ -", "- ")

        def _repr_(self) -> str:
            """Display this cobar element as a formatted linear combination."""
            return self.pretty()

        def _latex_(self) -> str:
            """LaTeX display for this cobar element."""
            return self.pretty_latex()

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "CobarConstruction.Element":
            return self.parent().boundary(self)

        def d1(self) -> "CobarConstruction.Element":
            """Internal differential: applies cooperad boundary to vertex decorations."""
            return self.parent()._d1(self)

        def d2(self) -> "CobarConstruction.Element":
            """Structural differential: expands internal edges."""
            return self.parent()._d2(self)

        def permute(self, sigma) -> "CobarConstruction.Element":
            """Permute leaf labels by sigma (no sign, just relabeling)."""
            parent = self.parent()
            n = parent.arity()

            if isinstance(sigma, (list, tuple)):
                sigma_values = list(sigma)
            else:
                sigma_values = list(sigma.tuple())

            relabel_map = {j: sigma_values[j - 1] for j in range(1, n + 1)}

            result = parent.zero()
            for tree, coeff in self:
                if is_leaf(tree):
                    new_tree = relabel_map.get(tree, tree)
                else:
                    new_tree = relabel_leaves(tree, relabel_map)
                result += coeff * parent.term(new_tree)
            return result


CobarConstruction.Component.Element = CobarConstruction.Element
