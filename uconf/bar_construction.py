"""Bar construction for connected dg-operads.

The bar construction B(P) of a connected augmented dg-operad P is the
cofree conilpotent cooperad on the suspension of the augmentation ideal:

    B(P) = (T^c(sP̄), d_1 + d_2)

where:
- P̄ is the augmentation ideal (P̄(1) = 0 for connected operads)
- sP̄ denotes the suspension (degree shift by +1 per arity-1)
- T^c denotes the cofree conilpotent cooperad (decorated rooted trees)
- d_1 is the internal differential from P
- d_2 is the structural differential from edge contractions

Reference: Loday-Vallette "Algebraic Operads", Chapter 6.
"""

from __future__ import annotations

from typing import ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    QQ,
    SymmetricGroup,
    tensor,
)

from .signs import (
    shifted_boundary_sign,
    shifted_operadic_compose_sign,
)
from .trees import (
    children,
    decoration,
    internal_edges,
    is_internal,
    is_leaf,
    leaves,
    relabel_leaves,
    split_at_vertex,
    subtree_degree,
    tree_arity,
    validate_tree,
    vertex_arity,
    vertices_dfs,
    weight,
    contract_edge,
)


class BarConstruction:
    """Factory for bar construction components of a connected dg-operad.

    Args:
        operad_cls: Base operad class (e.g., ``Lie``, ``Surjection``).
        max_weight: Maximum tree weight for enumeration helpers (default 3).

    The bar construction B(P) is a dg-cooperad whose arity-n component has
    basis elements given by rooted trees with n leaves, where internal
    vertices are decorated by elements of P̄ (the augmentation ideal).

    For connected operads, P̄(1) = 0, so all internal vertices have arity >= 2.
    """

    def __init__(self, operad_cls, max_weight: int = 3):
        self.operad_cls = operad_cls
        self.max_weight = int(max_weight)
        self.name = f"B({operad_cls.name})"

    def __call__(self, n: int, base_ring=QQ) -> "BarConstruction.Component":
        return BarConstruction.Component(self, n, base_ring)

    class Component(CombinatorialFreeModule):
        """A fixed-arity component of the bar construction cooperad."""

        name: ClassVar[str] = "B"

        def __init__(self, factory: "BarConstruction", n: int, base_ring=QQ):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            self.factory = factory
            self._arity = int(n)
            self._operad_cls = factory.operad_cls
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

        def _validate_basis_key(self, basis_key):
            """Validate a tree basis key."""
            return validate_tree(
                basis_key, self._arity, self._operad_cls, self.base_ring()
            )

        def _element_constructor_(self, x):
            """Build elements from tree basis keys or sparse dictionaries."""
            if isinstance(x, BarConstruction.Element):
                if x.parent().factory is self.factory:
                    return self.sum_of_terms((basis, coeff) for basis, coeff in x)
                raise TypeError("Element from different bar construction")

            if isinstance(x, dict):
                clean_dict = {}
                for key, coeff in x.items():
                    clean_key = self._validate_basis_key(key)
                    if clean_key is None:
                        continue
                    clean_dict[clean_key] = clean_dict.get(clean_key, 0) + coeff
                return super()._element_constructor_(clean_dict)

            if isinstance(x, tuple):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                return self.term(clean_key)

            return super()._element_constructor_(x)

        def arity(self) -> int:
            return self._arity

        def degree_on_basis(self, tree) -> int:
            """Compute the degree of a tree in B(P).

            The degree is sum over all vertices v of:
                deg_P(decoration(v)) + (arity(v) - 1)

            This equals sum_v deg_P(decoration(v)) + (n - 1) where n is tree arity,
            since sum_v (arity(v) - 1) = n - 1 for any tree.
            """
            return subtree_degree(tree, self._operad_cls, self.base_ring())

        def _boundary_on_basis(self, tree) -> "BarConstruction.Element":
            """Compute the bar differential d = d_1 + d_2 on a tree basis element.

            d_1: internal differential, applies P.boundary to each vertex decoration
            d_2: structural differential, contracts each internal edge
            """
            return self._d1_on_basis(tree) + self._d2_on_basis(tree)

        def _d1_on_basis(self, tree) -> "BarConstruction.Element":
            """Internal differential: apply operad boundary to each vertex.

            For each vertex v_j in DFS order, the sign is:
                (-1)^{1 + sum_{l < j} |s p_l|}

            where |s p_l| = deg_P(p_l) + (arity(v_l) - 1) is the suspended degree.
            The (-1)^1 comes from shifted_boundary_sign(1).
            """
            if is_leaf(tree):
                return self.zero()

            result = self.zero()
            verts = vertices_dfs(tree)
            base_ring = self.base_ring()

            # Compute cumulative degrees for Koszul signs
            cumulative_degree = 0

            for j, vertex in enumerate(verts):
                v_arity = vertex_arity(vertex)
                dec = decoration(vertex)
                operad_parent = self._operad_cls(v_arity, base_ring)

                # Degree of this vertex in sP̄
                vertex_sp_degree = operad_parent.degree_on_basis(dec) + (v_arity - 1)

                # Sign: (-1)^{shift_degree + cumulative}
                # shift_degree = 1 for the suspension
                sign = shifted_boundary_sign(1)
                if cumulative_degree % 2 == 1:
                    sign = -sign

                # Apply boundary to this vertex's decoration
                dec_elem = operad_parent.term(dec)
                bdry = operad_parent.boundary(dec_elem)

                # For each term in the boundary, build a new tree
                for new_dec, coeff in bdry:
                    # Replace decoration of this vertex
                    new_tree = self._replace_vertex_decoration_by_index(
                        tree, verts, j, new_dec
                    )
                    result += sign * coeff * self.term(new_tree)

                cumulative_degree += vertex_sp_degree

            return result

        def _d2_on_basis(self, tree) -> "BarConstruction.Element":
            """Structural differential: contract each internal edge.

            For each internal edge from parent vertex p to its l-th child c:
            - Composition sign: shifted_operadic_compose_sign(1, l, arity(p), arity(c), deg_P(c))
            - Koszul sign: (-1)^{|sc| * S_left} where S_left is the total sP̄-degree
              of siblings to the left of c
            """
            if is_leaf(tree):
                return self.zero()

            edges = internal_edges(tree)
            if not edges:
                return self.zero()

            result = self.zero()
            base_ring = self.base_ring()

            for parent_vertex, child_pos, child_vertex in edges:
                p_arity = vertex_arity(parent_vertex)
                c_arity = vertex_arity(child_vertex)
                p_dec = decoration(parent_vertex)
                c_dec = decoration(child_vertex)

                p_parent = self._operad_cls(p_arity, base_ring)
                c_parent = self._operad_cls(c_arity, base_ring)

                # Degree of child decoration in P
                c_deg_P = c_parent.degree_on_basis(c_dec)
                # Degree of child in sP̄
                c_sp_deg = c_deg_P + (c_arity - 1)

                # Compute S_left: total sP̄-degree of subtrees of parent before child_pos
                s_left = 0
                for i, sibling in enumerate(children(parent_vertex), start=1):
                    if i < child_pos:
                        s_left += subtree_degree(sibling, self._operad_cls, base_ring)

                # Signs
                compose_sign = shifted_operadic_compose_sign(
                    1, child_pos, p_arity, c_arity, c_deg_P
                )
                koszul_sign = (-1) ** ((c_sp_deg * s_left) % 2)
                total_sign = compose_sign * koszul_sign

                # Compute composition p ∘_l c
                p_elem = p_parent.term(p_dec)
                c_elem = c_parent.term(c_dec)
                composed = self._operad_cls.compose(p_elem, child_pos, c_elem)

                # For each term in the composition, build the contracted tree
                for new_dec, coeff in composed:
                    new_tree = contract_edge(tree, parent_vertex, child_pos, new_dec)
                    result += total_sign * coeff * self.term(new_tree)

            return result

        def _replace_vertex_decoration_by_index(
            self, tree, vertices: list, index: int, new_decoration: tuple
        ) -> tuple:
            """Replace the decoration of the index-th vertex (DFS order) in tree."""
            target_vertex = vertices[index]
            return self._replace_decoration_rec(tree, target_vertex, new_decoration)

        def _replace_decoration_rec(
            self, node, target: tuple, new_decoration: tuple
        ) -> tuple:
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
        def counit(x: "BarConstruction.Element"):
            """Cooperadic counit: extracts coefficient of the 'unit' tree.

            For connected operads, B(P)(1) is trivial (no trees), so counit
            is always 0 for arity >= 1.
            """
            # For arity 1 with connected operads, there are no valid trees
            # (all vertices must have arity >= 2, but a tree with one leaf
            # can only have one vertex of arity 1, which violates connectedness)
            return x.parent().base_ring().zero()

        @staticmethod
        def reduced(x: "BarConstruction.Element") -> "BarConstruction.Element":
            """Project to reduced part (kills counit component)."""
            # For connected operads, counit is always 0, so reduced = identity
            return x

        def infinitesimal_cocompose(
            self, x: "BarConstruction.Element", i: int, m: int, n: int
        ):
            """Partial cocomposition dual to free operad composition.

            Splits trees at internal edges where the lower subtree has leaves
            {i, i+1, ..., i+n-1}.
            """
            if m <= 0 or n <= 0:
                raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
            if not (1 <= i <= m):
                raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")
            if x.arity() != m + n - 1:
                raise ValueError(
                    f"Expected element in arity {m + n - 1}, got arity {x.arity()}."
                )

            left_parent = self.factory(m, self.base_ring())
            right_parent = self.factory(n, self.base_ring())
            target = tensor([left_parent, right_parent])

            def _on_basis(tree):
                """Cocompose a single tree basis element."""
                result = target.zero()
                target_leaves = set(range(i, i + n))

                # Find all internal vertices whose subtree leaves are exactly target_leaves
                for vertex in vertices_dfs(tree):
                    if leaves(vertex) == target_leaves:
                        split = split_at_vertex(tree, vertex)
                        if split is None:
                            continue

                        tree_top, placeholder, tree_bot = split

                        # Relabel tree_top: leaves {1,...,i-1, placeholder, i+n,...,m+n-1}
                        # should become {1,...,m}
                        top_relabel = {}
                        for leaf in leaves(tree_top):
                            if leaf < i:
                                top_relabel[leaf] = leaf
                            elif leaf == placeholder:
                                top_relabel[leaf] = i
                            else:
                                top_relabel[leaf] = leaf - n + 1

                        # Relabel tree_bot: leaves {i,...,i+n-1} should become {1,...,n}
                        bot_relabel = {leaf: leaf - i + 1 for leaf in target_leaves}

                        relabeled_top = relabel_leaves(tree_top, top_relabel)
                        relabeled_bot = relabel_leaves(tree_bot, bot_relabel)

                        # Validate
                        if left_parent._validate_basis_key(relabeled_top) is None:
                            continue
                        if right_parent._validate_basis_key(relabeled_bot) is None:
                            continue

                        # Add to result (sign = 1 for cofree cooperad)
                        result += left_parent.term(relabeled_top).tensor(
                            right_parent.term(relabeled_bot)
                        )

                return result

            total = target.zero()
            for tree, coeff in x:
                total += coeff * _on_basis(tree)
            return total

    class Element(CombinatorialFreeModule.Element):
        """Element of a bar construction cooperad component."""

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "BarConstruction.Element":
            return self.parent().boundary(self)

        def permute(self, sigma) -> "BarConstruction.Element":
            """Permute leaf labels by sigma (no sign, just relabeling)."""
            parent = self.parent()
            n = parent.arity()

            if isinstance(sigma, (list, tuple)):
                sigma_values = list(sigma)
            else:
                sigma_values = list(sigma.tuple())

            # Build relabeling: leaf j -> sigma(j)
            relabel_map = {j: sigma_values[j - 1] for j in range(1, n + 1)}

            result = parent.zero()
            for tree, coeff in self:
                new_tree = relabel_leaves(tree, relabel_map)
                result += coeff * parent.term(new_tree)
            return result

        def counit(self):
            return BarConstruction.Component.counit(self)

        def reduced(self) -> "BarConstruction.Element":
            return BarConstruction.Component.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            return self.parent().infinitesimal_cocompose(self, i, m, n)


BarConstruction.Component.Element = BarConstruction.Element
