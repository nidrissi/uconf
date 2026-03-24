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

.. note::
   This module requires a **connected** operad input.  Connectedness means
   P(0) = 0 and P(1) = k (unit only), so every internal tree vertex has
   arity >= 2.  For a tree with n leaves this bounds the number of internal
   vertices by n - 1, making the basis in every (arity, degree) finite.

Reference: Loday-Vallette "Algebraic Operads", Chapter 6.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    UniqueRepresentation,
    tensor,
    Family,
    cached_method,
)

from uconf.core.signs import (
    sign_from_exponent,
)
from uconf.core.display import latex_linear_combination
from uconf.core.operad import OperadLike
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.quasi_planar import QuasiPlanarMixin
from uconf.core.trees import (
    children,
    decoration,
    enumerate_planar_trees_in_degree,
    enumerate_shuffle_trees_in_degree,
    internal_edges_dfs,
    is_leaf,
    leaves,
    relabel_leaves,
    split_at_vertex,
    subtree_degree,
    to_shuffle_tree_bar,
    tree_to_latex,
    tree_to_string,
    tree_to_svg,
    validate_tree,
    vertex_arity,
    vertices_dfs,
    contract_edge,
)


class BarConstruction(UniqueRepresentation):
    """Factory for bar construction components of a connected dg-operad.

    Args:
        operad_cls: Base operad provider (class or wrapper instance).
            Must be a **connected** operad (P(0) = 0, P(1) = k·unit).

    The bar construction B(P) is a dg-cooperad whose arity-n component has
    basis elements given by rooted trees with n leaves, where internal
    vertices are decorated by elements of P̄ (the augmentation ideal).

    For connected operads, P̄(1) = 0, so all internal vertices have arity >= 2.
    This bounds the number of internal vertices in arity n by n - 1, making
    every (arity, degree) basis finite without requiring an external weight cap.

    """

    def __init__(
        self,
        operad_cls: OperadLike,
    ):
        self.operad_cls = operad_cls
        self.name = f"B({operad_cls.name})"

    def _repr_(self) -> str:
        return self.name

    def _repr_latex_(self) -> str:
        base = getattr(self.operad_cls, "name", "P")
        return f"B({base})"

    @property
    def connectivity(self) -> int:
        """Connectivity of the bar construction.

        For an operad P with connectivity k (degrees >= k*(n-1)), the minimum
        bar degree of a single-vertex tree in arity n is k*(n-1) + 1.  The
        connectivity of B(P) as a cooperad is therefore k + 1 in the sense that
        B(P)(n) is concentrated in degrees >= (k+1)*(n-1) + 1 (for k >= 0).
        We store k here as a reference value derived from the underlying operad.
        """
        return getattr(self.operad_cls, "connectivity", 0)

    def __call__(self, n: int, base_ring) -> "BarConstruction.Component":
        return BarConstruction.Component(self, n, base_ring)

    @staticmethod
    def counit(x: "BarConstruction.Element"):
        """Cooperadic counit at the factory level."""
        return BarConstruction.Component.counit(x)

    @staticmethod
    def reduced(x: "BarConstruction.Element") -> "BarConstruction.Element":
        """Reduced projection at the factory level."""
        return BarConstruction.Component.reduced(x)

    @staticmethod
    def infinitesimal_cocompose(x: "BarConstruction.Element", i: int, m: int, n: int):
        """Infinitesimal cocomposition at the factory level."""
        return x.infinitesimal_cocompose(i, m, n)

    def counit_element(self, base_ring) -> "BarConstruction.Element":
        """Return the counit element (single leaf tree in arity 1).

        The bar construction B(P) has a canonical counit ε: B(P)(1) → k,
        which is non-trivial on the single-leaf tree with no internal vertices.
        This method returns that generator.
        """
        component = self(1, base_ring)
        # The single-leaf tree "1" (no internal vertices)
        return component.term(1)

    def unit_key(self) -> int:
        """Return the basis key of the counit generator in arity ``1``.

        In the bar construction, arity-1 elements are the single-leaf tree,
        whose basis key is the integer ``1``.
        """
        return 1

    class Component(QuasiPlanarMixin, CombinatorialFreeModule):
        """A fixed-arity component of the bar construction cooperad."""

        name: ClassVar[str] = "B"

        def __init__(self, factory: "BarConstruction", n: int, base_ring):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            self.factory = factory
            self._arity = int(n)
            self._operad_cls = factory.operad_cls
            # The maximum number of internal vertices in arity n is n - 1,
            # because every internal vertex has arity >= 2 (connected operad)
            # and sum(arity_v - 1) = n - 1 for a tree with n leaves.
            self._max_weight = max(0, self._arity - 1)

            name = f"{factory.name}{n}"
            super().__init__(
                base_ring,
                tuple,
                prefix=name,
                category=GradedModulesWithBasis(base_ring),
            )
            self.rename(name)
            self._symmetric_group = SymmetricGroup(n)
            self._symmetric_group_algebra = SymmetricGroupAlgebra(base_ring, n)

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

            # Set up planarize if the base operad supports it
            if self._arity > 0 and self._operad_has_planarize():
                self.planarize = self.module_morphism(
                    on_basis=self._planarize_on_basis,
                    codomain=tensor([self, self._symmetric_group_algebra]),
                )

        def _operad_has_planarize(self) -> bool:
            """Check if the base operad components have ``planarize``."""
            test = self._operad_cls(2, self.base_ring())
            return hasattr(test, "planarize")

        def _planarize_on_basis(self, tree) -> Any:
            """Decompose a bar tree into planar part ⊗ global permutation.

            For each internal vertex ``v`` the base operad's ``planarize``
            gives a (possibly multi-term) decomposition into planar
            decorations and vertex permutations.  The planarized tree is
            assembled by:

            1. Replacing each vertex decoration with its planar form.
            2. Reordering the children of ``v``: new child at position ``j``
               is old child at position ``σ_v(j)`` (1-indexed).
            3. Relabeling the leaves of the resulting tree so they run
               ``1, …, n`` in the new left-to-right order.  The result is
               automatically a shuffle tree.

            The global permutation ``σ ∈ S_n`` satisfies ``σ(j)`` = the
            original leaf label at canonical position ``j``.

            Returns an element of ``self ⊗ k[S_n]``.

            .. note::
               For quasi-planar operads (Surjection, BarrattEccles) each
               vertex yields a single term.  For operads like
               ``HadamardProduct(sLie, Surjection)`` the Lie factor's
               ``permute`` may expand into multiple planar-basis elements,
               and every term is propagated here.
            """
            sym_alg = self._symmetric_group_algebra
            base_ring = self.base_ring()

            if is_leaf(tree):
                identity = self._symmetric_group.identity()
                return self.term(tree).tensor(sym_alg.term(identity))

            def _planarize_subtree(node):
                """Return list of ``(coeff, planar_node, leaf_order)`` triples.

                Each triple represents one term in the planar decomposition.
                ``planar_node`` has planar vertex decorations with the
                *original* leaf labels (not yet renumbered).
                ``leaf_order[i]`` is the original leaf label that will sit at
                canonical position ``i+1`` after renumbering.
                """
                if is_leaf(node):
                    return [(1, node, [node])]

                k = vertex_arity(node)
                dec = decoration(node)
                op_parent = self._operad_cls(k, base_ring)

                dec_elem = op_parent.term(dec)
                planarized = op_parent.planarize(dec_elem)

                old_ch = children(node)
                results = []
                for (planar_dec_key, sigma_key), dec_coeff in planarized:
                    sigma_v_tuple = SymmetricGroup(k)(sigma_key).tuple()
                    new_ch = tuple(old_ch[sigma_v_tuple[j] - 1] for j in range(k))

                    # Recursively planarize each (reordered) child, combining
                    # all term combinations from children.
                    child_term_lists = []
                    for ch in new_ch:
                        child_term_lists.append(_planarize_subtree(ch))

                    from itertools import product as iter_product

                    for combo in iter_product(*child_term_lists):
                        total_child_coeff = 1
                        new_ch_planarized = []
                        leaf_order = []
                        for ch_coeff, p_ch, lo_ch in combo:
                            total_child_coeff *= ch_coeff
                            new_ch_planarized.append(p_ch)
                            leaf_order.extend(lo_ch)

                        total_coeff = dec_coeff * total_child_coeff
                        node_result = (planar_dec_key,) + tuple(new_ch_planarized)
                        results.append((total_coeff, node_result, leaf_order))

                return results

            target = tensor([self, sym_alg])
            result = target.zero()
            for total_coeff, planar_with_orig, leaf_order in _planarize_subtree(tree):
                sigma_global_inv = {l: pos for pos, l in enumerate(leaf_order, start=1)}
                canonical_tree = relabel_leaves(planar_with_orig, sigma_global_inv)
                sigma_global = sym_alg.term(self._symmetric_group(list(leaf_order)))
                result += total_coeff * self.term(canonical_tree).tensor(sigma_global)

            return result

        def _is_planar_tree(self, tree) -> bool:
            """Return ``True`` if every vertex decoration is planar.

            A decoration is planar when the base operad's ``planarize``
            returns ``dec ⊗ id`` (i.e. the vertex permutation is the
            identity).
            """
            if is_leaf(tree):
                return True
            base_ring = self.base_ring()
            for v in vertices_dfs(tree):
                k = vertex_arity(v)
                dec = decoration(v)
                op_parent = self._operad_cls(k, base_ring)
                if not hasattr(op_parent, "planarize"):
                    return False
                planarized = op_parent.planarize(op_parent.term(dec))
                Sk = SymmetricGroup(k)
                identity_k = Sk.identity()
                for (_pl_dec, sigma_key), _coeff in planarized:
                    if Sk(sigma_key) != identity_k:
                        return False
            return True

        def planar_basis_it(self, d: int) -> Iterator["BarConstruction.Element"]:
            """Iterate over planar basis elements of degree ``d``.

            A tree is *planar* when every vertex decoration is a planar
            element of the base operad and the global leaf permutation is the
            identity (children occupy consecutive leaf ranges).

            Trees are enumerated with the connectivity-derived weight bound
            ``n - 1`` (every internal vertex has arity ≥ 2 in a connected
            operad, so a tree with n leaves has at most n − 1 vertices).

            Requires the base operad to implement ``planarize`` and
            ``planar_basis_it``; raises :exc:`NotImplementedError` otherwise.
            Use :meth:`basis_iter` for a full shuffle-tree basis that works with
            any connected operad.
            """
            if not self._operad_has_planarize():
                raise NotImplementedError(
                    f"planar_basis_it requires {self._operad_cls.name!r} to implement "
                    "planarize and planar_basis_it (quasi-planar operad). "
                    "Use basis_iter() for the full shuffle-tree basis instead."
                )

            n = self._arity
            base_ring = self.base_ring()

            if n < 2:
                if n == 1 and d == 0:
                    yield self.term(1)
                return

            for tree in enumerate_planar_trees_in_degree(
                n, self._max_weight, self._operad_cls, base_ring, d
            ):
                yield self.term(tree)

        def basis_iter(self, d: int) -> Iterator["BarConstruction.Element"]:
            """Iterate over shuffle-tree basis elements of degree ``d``.

            Works for **any** connected operad, not just quasi-planar ones.
            Unlike :meth:`planar_basis_it`, this does not require the base
            operad to implement ``planarize``.

            The basis consists of all rooted trees with leaves
            ``{1, ..., n}`` whose children are sorted by minimum leaf
            label at every vertex (*shuffle trees*), decorated by any
            basis element of the underlying operad.  For quasi-planar
            operads the shuffle basis and the planar basis are related by
            the isomorphism ``B(P)_pl ⊗ k[S_n] ≅ B(P)``; for
            non-quasi-planar operads (e.g. ``Commutative``) the shuffle
            trees provide the canonical vector-space basis.
            """
            n = self._arity
            base_ring = self.base_ring()

            if n == 1:
                if d == 0:
                    yield self.term(1)
                return

            for tree in enumerate_shuffle_trees_in_degree(
                n, self._max_weight, self._operad_cls, base_ring, d
            ):
                yield self.term(tree)

        def _validate_basis_key(self, basis_key):
            """Validate a tree basis key."""
            return validate_tree(basis_key, self._arity, self._operad_cls, self.base_ring())

        @cached_method
        def graded_basis(self, d: int) -> Family:
            """Return the ``Family`` of all basis elements in degree ``d``."""
            return Family(self.basis_iter(d))

        @cached_method
        def graded_planar_basis(self, d: int) -> Family:
            """Return the ``Family`` of planar basis elements in degree ``d``."""
            return Family(self.planar_basis_it(d))

        def _normalize_to_shuffle(self, tree):
            """Normalize a tree to shuffle form for the bar construction.

            Returns a list of ``(shuffle_tree, coeff)`` pairs representing
            a (possibly multi-term) linear combination.

            A shuffle tree has children at each vertex sorted by min leaf label.
            This implements the standard basis for the bar construction on a
            symmetric operad (cf. Bremner-Dotsenko, Loday-Vallette).
            """
            if is_leaf(tree):
                return [(tree, 1)]
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
                    # Normalize to shuffle form (may produce multiple terms)
                    for shuffle_key, shuffle_coeff in self._normalize_to_shuffle(clean_key):
                        clean_dict[shuffle_key] = (
                            clean_dict.get(shuffle_key, 0) + coeff * shuffle_coeff
                        )
                return super()._element_constructor_(clean_dict)

            if isinstance(x, tuple):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                # Normalize to shuffle form (may produce multiple terms)
                return self.sum_of_terms(self._normalize_to_shuffle(clean_key))

            if isinstance(x, int) and self._arity == 1 and x == 1:
                # Special case: allow integer 1 to represent the single-leaf tree in arity 1
                return self.term(1)

            return super()._element_constructor_(x)

        def arity(self) -> int:
            return self._arity

        @property
        def connectivity(self) -> int:
            """Connectivity inherited from the underlying operad."""
            return getattr(self._operad_cls, "connectivity", 0)

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

            def _dec_fmt(dec, arity):
                parent = self.factory.operad_cls(arity, self.base_ring())
                repr_term = getattr(parent, "_repr_term", None)
                if callable(repr_term):
                    return repr_term(dec)
                return f"{self.factory.operad_cls.name}{dec}"

            return tree_to_string(
                basis_element,
                self.factory.operad_cls.name,
                decoration_formatter=_dec_fmt,
            )

        def _latex_term(self, basis_element) -> str:
            """LaTeX representation of one bar basis tree."""

            def _dec_fmt(dec, arity):
                parent = self.factory.operad_cls(arity, self.base_ring())
                latex_term = getattr(parent, "_latex_term", None)
                if callable(latex_term):
                    return latex_term(dec)
                return f"\\operatorname{{{self.factory.operad_cls.name}}}_{{{dec}}}"

            return tree_to_latex(
                basis_element,
                self.factory.operad_cls.name,
                decoration_formatter=_dec_fmt,
            )

        def _svg_term(self, basis_element) -> str:
            """SVG representation of one bar basis tree."""

            def _dec_fmt(dec, arity):
                parent = self.factory.operad_cls(arity, self.base_ring())
                repr_term = getattr(parent, "_repr_term", None)
                if callable(repr_term):
                    return repr_term(dec)
                return f"{self.factory.operad_cls.name}{dec}"

            return tree_to_svg(
                basis_element,
                operad_name=self.factory.operad_cls.name,
                decoration_formatter=_dec_fmt,
            )

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
                    new_tree = self._replace_vertex_decoration_by_index(tree, verts, j, new_dec)
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
                    v_deg = self._operad_cls(v_arity, base_ring).degree_on_basis(decoration(v))
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

        def _replace_decoration_rec(self, node, target: tuple, new_decoration: tuple) -> tuple:
            """Recursively replace decoration of target vertex."""
            if is_leaf(node):
                return node
            if node is target:
                return (new_decoration,) + children(node)
            new_children = tuple(
                self._replace_decoration_rec(c, target, new_decoration) for c in children(node)
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

        @staticmethod
        def unit_key() -> int:
            """Return the basis key of the counit generator in arity ``1``.

            In the bar construction, the arity-1 generator is the single-leaf
            tree, whose basis key is the integer ``1``.
            """
            return 1

        def infinitesimal_cocompose(self, x: "BarConstruction.Element", i: int, m: int, n: int):
            """Partial cocomposition dual to free operad composition.

            Splits trees at internal edges where the lower subtree has leaves
            ``{i, i+1, ..., i+n-1}``.
            """
            if m <= 0 or n <= 0:
                raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
            if not (1 <= i <= m):
                raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")
            if x.arity() != m + n - 1:
                raise ValueError(f"Expected element in arity {m + n - 1}, got arity {x.arity()}.")

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

    class Element(
        ParentedElementMixin["BarConstruction.Component"],
        CombinatorialFreeModule.Element,
    ):
        """Element of a bar construction cooperad component."""

        def _repr_latex_(self) -> str:
            """Return a LaTeX linear-combination string for this element."""
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def arity(self) -> int:
            return self.parent().arity()

        def _repr_svg_(self) -> str:
            """Return SVG markup for Sage display of a monomial bar tree."""
            if not self:
                raise ValueError("Cannot render SVG for the zero element.")
            if len(self.support()) != 1:
                raise ValueError(
                    "SVG rendering currently supports only monomials with one basis term."
                )
            basis = next(iter(self.support()))
            return self.parent()._svg_term(basis)

        def to_svg(self) -> str:
            """Compatibility alias for :meth:`_repr_svg_`."""
            return self._repr_svg_()

        def boundary(self) -> "BarConstruction.Element":
            """Apply the bar differential (d_1 + d_2) to this element."""
            parent = self.parent()
            return parent.boundary(self)

        def d1(self) -> "BarConstruction.Element":
            """Internal differential: applies operad boundary to vertex decorations."""
            parent = self.parent()
            return parent._d1(self)

        def d2(self) -> "BarConstruction.Element":
            """Structural differential: contracts internal edges."""
            parent = self.parent()
            return parent._d2(self)

        def planarize(self):
            """Decompose into planar representative ⊗ global permutation.

            Returns an element of ``B(P)(n) ⊗ k[S_n]``.
            Requires the base operad to implement ``planarize``.
            """
            parent = self.parent()
            return parent.planarize(self)

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
                # Use parent(new_tree) rather than parent.term(new_tree) so that
                # the shuffle normalization (and its Koszul sign) is applied: the
                # relabeled tree may no longer be in shuffle order.
                result += coeff * parent(new_tree)
            return result

        def counit(self):
            """Evaluate the cooperadic counit on this element."""
            return BarConstruction.Component.counit(self)

        def reduced(self) -> "BarConstruction.Element":
            """Project this element to the reduced bar cooperad."""
            return BarConstruction.Component.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Apply infinitesimal cocomposition ``Δ^{i;m,n}`` to this element."""
            parent = self.parent()
            return parent.infinitesimal_cocompose(self, i, m, n)


BarConstruction.Component.Element = BarConstruction.Element
