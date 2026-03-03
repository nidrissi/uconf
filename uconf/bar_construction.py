"""Bar construction for connected dg-operads.

The bar construction B(P) of a connected augmented dg-operad P is the
cofree conilpotent cooperad on the suspension of the augmentation ideal:

    B(P) = (T^c(sP̄), d_1 + d_2)

where:
- P̄ is the augmentation ideal (P̄(1) = 0 for connected operads)
- sP̄ denotes the suspension used here (degree shift by +1 per internal vertex)
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
    sign_from_exponent,
)
from .trees import (
    children,
    decoration,
    internal_edges_dfs,
    is_internal,
    is_leaf,
    is_shuffle_tree,
    leaves,
    relabel_leaves,
    split_at_vertex,
    subtree_degree,
    to_shuffle_tree_bar,
    tree_arity,
    tree_to_latex,
    tree_to_string,
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

    def counit_element(self, base_ring=QQ) -> "BarConstruction.Element":
        """Return the counit element (single leaf tree in arity 1).

        The bar construction B(P) has a canonical counit ε: B(P)(1) → k,
        which is non-trivial on the single-leaf tree with no internal vertices.
        This method returns that generator.
        """
        component = self(1, base_ring)
        # The single-leaf tree "1" (no internal vertices)
        return component.term(1)

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
            self._d1 = self.module_morphism(
                on_basis=self._d1_on_basis,
                codomain=self,
            )
            self._d2 = self.module_morphism(
                on_basis=self._d2_on_basis,
                codomain=self,
            )

        def _validate_basis_key(self, basis_key):
            """Validate a tree basis key."""
            return validate_tree(
                basis_key, self._arity, self._operad_cls, self.base_ring()
            )

        def _normalize_to_shuffle(self, tree):
            """Normalize a tree to shuffle form for the bar construction.

            Returns ``(shuffle_tree, sign)`` where ``shuffle_tree`` is the
            normalized tree and ``sign`` is the accumulated Koszul and operad
            action sign.

            A shuffle tree has children at each vertex sorted by min leaf label.
            This implements the standard basis for the bar construction on a
            symmetric operad (cf. Bremner-Dotsenko, Loday-Vallette).
            """
            if is_leaf(tree):
                return tree, 1
            return to_shuffle_tree_bar(tree, self._operad_cls, self.base_ring())

        def _element_constructor_(self, x):
            """Build elements from tree basis keys or sparse dictionaries.

            Trees are automatically normalized to shuffle form. The shuffle tree
            basis identifies trees that differ only by permutations of children
            at vertices with symmetric decorations.
            """
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
                    # Normalize to shuffle form
                    shuffle_key, sign = self._normalize_to_shuffle(clean_key)
                    clean_dict[shuffle_key] = (
                        clean_dict.get(shuffle_key, 0) + coeff * sign
                    )
                return super()._element_constructor_(clean_dict)

            if isinstance(x, tuple):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                # Normalize to shuffle form
                shuffle_key, sign = self._normalize_to_shuffle(clean_key)
                if sign == 1:
                    return self.term(shuffle_key)
                else:
                    return sign * self.term(shuffle_key)

            return super()._element_constructor_(x)

        def arity(self) -> int:
            return self._arity

        def degree_on_basis(self, tree) -> int:
            """Compute the degree of a tree in B(P).

            The degree is sum over all vertices v of:
                deg_P(decoration(v)) + 1

            This is the grading convention implemented by ``subtree_degree``
            and used consistently in the bar differential sign exponents.
            """
            return subtree_degree(tree, self._operad_cls, self.base_ring())

        def _repr_term(self, basis_element) -> str:
            """String representation of one bar basis tree."""
            return tree_to_string(basis_element, self.factory.operad_cls.name)

        def _latex_term(self, basis_element) -> str:
            """LaTeX representation of one bar basis tree."""
            return tree_to_latex(basis_element, self.factory.operad_cls.name)

        def _boundary_on_basis(self, tree) -> "BarConstruction.Element":
            """Compute the bar differential d = d_1 + d_2 on a tree basis element.

            - ``d_1`` applies ``P.boundary`` to each vertex decoration.
            - ``d_2`` contracts each internal edge via partial composition.
            """
            return self._d1_on_basis(tree) + self._d2_on_basis(tree)

        def _d1_on_basis(self, tree) -> "BarConstruction.Element":
            """Internal differential: apply operad boundary to each vertex.

            For each vertex v_j in DFS order (pre-order), the sign is:
                (-1)^{sum_{l < j} (deg_P(p_l) + 1)}

            where deg_P(p_l) + 1 is the uniformly suspended degree of vertex l.

            This is the coderivation extending the internal differential on
            vertex decorations.
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
                vertex_sp_degree = operad_parent.degree_on_basis(dec) + 1

                # Sign: (-1)^{cumulative}
                sign = sign_from_exponent(cumulative_degree)

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

            For each internal edge (parent p, position l, child c), the sign is:
                (-1)^{global_accum(p) + deg_P(p) + |sc| * deg_bar_before(p, l)}

            where:
            - global_accum(p) = sum_{v before p in DFS} (deg_P(v) - 1)
                is the cumulative shifted degree used by the implementation
            - deg_P(p) is the P-degree of the parent decoration
            - |sc| = deg_P(c) - 1 is the shifted degree of the child
            - deg_bar_before(p, l) = sum_{j < l} subtree_degree(child_j of p)
                is the total bar-degree of the subtrees rooted at siblings of c
                occupying positions 1, ..., l-1
            """
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

                # Global cumulative: sP̄-degree of all DFS vertices before parent
                global_accum = 0
                for v in verts:
                    if v is parent_vertex:
                        break
                    v_arity = vertex_arity(v)
                    v_deg = self._operad_cls(v_arity, base_ring).degree_on_basis(
                        decoration(v)
                    )
                    global_accum += v_deg - 1

                # Koszul sign: sP̄-degree of child times bar-degree before position l
                c_sp_deg = c_deg_P - 1
                before_deg = sum(
                    subtree_degree(ch, self._operad_cls, base_ring)
                    for i, ch in enumerate(children(parent_vertex), start=1)
                    if i < child_pos
                )
                koszul_exp = c_sp_deg * before_deg

                total_sign = sign_from_exponent(global_accum + p_deg_P + koszul_exp)

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

            The counit ε: B(P)(1) → k is non-trivial only on the single-leaf
            tree (basis key = 1) and returns its coefficient.
            For arity ≠ 1, the counit is always 0.
            """
            if x.arity() != 1:
                return x.parent().base_ring().zero()
            # The single-leaf tree has basis key = 1 (an integer, not a tuple)
            # Get the coefficient from the element using dict-style access
            return x[1] if 1 in x.support() else x.parent().base_ring().zero()

        @staticmethod
        def reduced(x: "BarConstruction.Element") -> "BarConstruction.Element":
            """Project to reduced part (kills counit component).

            For arity 1, removes the coefficient of the single-leaf tree.
            For other arities, returns x unchanged.
            """
            if x.arity() != 1:
                return x
            # Remove the single-leaf (basis key = 1) component
            if 1 not in x.support():
                return x
            coeff = x[1]
            if coeff == 0:
                return x
            return x - coeff * x.parent().term(1)

        def infinitesimal_cocompose(
            self, x: "BarConstruction.Element", i: int, m: int, n: int
        ):
            """Partial cocomposition dual to free operad composition.

            Splits trees at internal edges where the lower subtree has leaves
            ``{i, i+1, ..., i+n-1}``.
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
            """Display this bar element as a formatted linear combination."""
            return self.pretty()

        def _latex_(self) -> str:
            """LaTeX display for this bar element."""
            return self.pretty_latex()

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "BarConstruction.Element":
            """Apply the bar differential (d_1 + d_2) to this element."""
            return self.parent().boundary(self)

        def d1(self) -> "BarConstruction.Element":
            """Internal differential: applies operad boundary to vertex decorations."""
            return self.parent()._d1(self)

        def d2(self) -> "BarConstruction.Element":
            """Structural differential: contracts internal edges."""
            return self.parent()._d2(self)

        def permute(self, sigma) -> "BarConstruction.Element":
            """Permute leaf labels by ``sigma`` (no extra sign)."""
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
            """Evaluate the cooperadic counit on this element."""
            return BarConstruction.Component.counit(self)

        def reduced(self) -> "BarConstruction.Element":
            """Project this element to the reduced bar cooperad."""
            return BarConstruction.Component.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Apply infinitesimal cocomposition ``Δ^{i;m,n}`` to this element."""
            return self.parent().infinitesimal_cocompose(self, i, m, n)


BarConstruction.Component.Element = BarConstruction.Element
