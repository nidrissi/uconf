"""Cobar construction for connected dg-cooperads.

The cobar construction Ω(C) of a connected coaugmented dg-cooperad C is the
free operad on the desuspension of the coaugmentation coideal:

    Ω(C) = (T(s⁻¹C̄), d_1 + d_2)

where:
- C̄ is the coaugmentation coideal (C̄(1) = 0 for connected cooperads)
- s⁻¹C̄ denotes the desuspension used here (degree shift by -1 per internal vertex)
- T denotes the free operad (decorated rooted trees)
- d_1 is the internal differential from C
- d_2 is the structural differential from vertex expansions

.. note::
   This module requires a **connected** cooperad input.  Connectedness means
   C(0) = 0 and C(1) = k (counit only), so every internal tree vertex has
   arity >= 2.  For a tree with n leaves this bounds the number of internal
   vertices by n - 1, making the basis in every (arity, degree) finite.

Reference: Loday-Vallette "Algebraic Operads", Chapter 6.
"""

from __future__ import annotations

from typing import ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    UniqueRepresentation,
    cached_method,
    tensor,
)

from uconf.core.cooperad import CooperadLike
from uconf.core.display import latex_linear_combination
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    decoration,
    enumerate_planar_trees_generic_in_degree,
    enumerate_shuffle_trees_cobar_in_degree,
    expand_vertex,
    graft,
    is_leaf,
    relabel_leaves,
    subtree_degree_cobar,
    to_shuffle_tree_cobar,
    tree_to_latex,
    tree_to_string,
    validate_tree,
    vertex_arity,
    vertices_dfs,
)


class CobarConstruction(UniqueRepresentation):
    """Factory for cobar construction components of a connected dg-cooperad.

    Args:
        cooperad_cls: Base cooperad provider (class or wrapper instance).
            Must be a **connected** cooperad (C(0) = 0, C(1) = k·counit).

    The cobar construction Ω(C) is a dg-operad whose arity-n component has
    basis elements given by rooted trees with n leaves, where internal
    vertices are decorated by elements of C̄ (the coaugmentation coideal).

    For connected cooperads, C̄(1) = 0, so all internal vertices have arity >= 2.
    This bounds the number of internal vertices in arity n by n - 1, making
    every (arity, degree) basis finite without requiring an external weight cap.

    Note:
        Trees are automatically normalized to shuffle form via
        ``to_shuffle_tree_cobar`` in the element constructor, analogous
        to the bar construction.

    """

    def __init__(self, cooperad_cls: CooperadLike):
        self.cooperad_cls = cooperad_cls
        self.name = f"Ω({cooperad_cls.name})"

    def _repr_(self) -> str:
        return self.name

    def _repr_latex_(self) -> str:
        base = getattr(self.cooperad_cls, "name", "C")
        return f"\\Omega({base})"

    @property
    def connectivity(self) -> int:
        """Connectivity inherited from the underlying cooperad."""
        return int(getattr(self.cooperad_cls, "connectivity", 0))

    def __call__(self, n: int, base_ring) -> "CobarConstruction.Component":
        return CobarConstruction.Component(self, n, base_ring)

    def unit(self, base_ring) -> "CobarConstruction.Element":
        """Return the unit element (identity in arity 1).

        For the free operad, the unit is represented by a single leaf.
        """
        component = self(1, base_ring)
        # Unit is the single leaf "1" (a trivial tree with no internal vertices)
        return component.term(1)

    def unit_key(self) -> int:
        """Return the basis key of the unit element in arity ``1``.

        In the cobar construction, the arity-1 unit is the single-leaf tree,
        whose basis key is the integer ``1``.
        """
        return 1

    def compose(
        self, x: "CobarConstruction.Element", i: int, y: "CobarConstruction.Element"
    ) -> "CobarConstruction.Element":
        """Free operad composition: graft y onto leaf i of x.

        In the free operad ``T(s⁻¹C̄)``, composition is tree grafting; no
        additional composition sign is introduced here.
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

        def __init__(self, factory: "CobarConstruction", n: int, base_ring):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            self.factory = factory
            self._arity = int(n)
            self._cooperad_cls = factory.cooperad_cls
            self._max_weight = max(0, self._arity - 1)

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

            # Set up planarize if the base cooperad supports it.
            # Ω(C) is quasi-planar when C is quasi-planar: the S_n-action on
            # cobar trees is free (by permuting leaf labels), so the global
            # permutation can always be extracted from the tree structure.
            if self._arity > 0 and self._cooperad_has_planarize():
                self._symmetric_group_algebra = SymmetricGroupAlgebra(base_ring, n)
                self.planarize = self.module_morphism(
                    on_basis=self._planarize_on_basis,
                    codomain=tensor([self, self._symmetric_group_algebra]),
                )

        def _cooperad_has_planarize(self) -> bool:
            """Check if the base cooperad components have ``planarize``."""
            test = self._cooperad_cls(2, self.base_ring())
            return callable(getattr(test, "planarize", None))

        @cached_method
        def _planarize_on_basis(self, tree):
            """Decompose a cobar tree into planar part ⊗ global permutation.

            Mirrors ``BarConstruction.Component._planarize_on_basis``: for each
            internal vertex ``v`` the base cooperad's ``planarize`` gives a
            (possibly multi-term) decomposition into planar decorations and
            vertex permutations.  The planarized tree is assembled by:

            1. Replacing each vertex decoration with its planar form.
            2. Reordering the children of ``v``: new child at position ``j``
               is old child at position ``σ_v(j)`` (1-indexed).
            3. Relabeling the leaves of the resulting tree so they run
               ``1, …, n`` in the new left-to-right order.

            The global permutation ``σ ∈ S_n`` satisfies ``σ(j)`` = the
            original leaf label at canonical position ``j``.

            Returns an element of ``self ⊗ k[S_n]``.
            """
            sym_alg = self._symmetric_group_algebra
            base_ring = self.base_ring()

            if is_leaf(tree):
                identity = self._symmetric_group.identity()
                return self.term(tree).tensor(sym_alg.term(identity))

            def _planarize_subtree(node):
                """Return list of ``(coeff, planar_node, leaf_order)`` triples."""
                if is_leaf(node):
                    return [(1, node, [node])]

                k = vertex_arity(node)
                dec = decoration(node)
                coop_parent = self._cooperad_cls(k, base_ring)

                dec_elem = coop_parent.term(dec)
                planarized = coop_parent.planarize(dec_elem)

                old_ch = children(node)
                results = []
                for (planar_dec_key, sigma_key), dec_coeff in planarized:
                    sigma_v_tuple = SymmetricGroup(k)(sigma_key).tuple()
                    new_ch = tuple(old_ch[sigma_v_tuple[j] - 1] for j in range(k))

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

        def planar_basis_iter(self, d: int) -> "Iterator[CobarConstruction.Element]":
            """Iterate over planar cobar basis elements of degree ``d``.

            A tree is *planar* when every vertex decoration is a planar element
            of the base cooperad and the global leaf permutation is the identity
            (children occupy consecutive leaf ranges).

            Requires the base cooperad to implement ``planarize`` and
            ``planar_basis_iter``; raises :exc:`NotImplementedError` otherwise.
            Use :meth:`basis_iter` for the full shuffle-tree basis instead.
            """
            if not self._cooperad_has_planarize():
                raise NotImplementedError(
                    f"planar_basis_iter requires {self._cooperad_cls.name!r} to implement "
                    "planarize and planar_basis_iter (quasi-planar cooperad). "
                    "Use basis_iter() for the full shuffle-tree basis instead."
                )

            n = self._arity
            base_ring = self.base_ring()

            if n < 2:
                if n == 1 and d == 0:
                    yield self.term(1)
                return

            for tree in enumerate_planar_trees_generic_in_degree(
                n,
                self._max_weight,
                self._cooperad_cls,
                base_ring,
                d,
                vertex_offset=-1,
                use_planar_decs=True,
            ):
                yield self.term(tree)

        @cached_method
        def graded_planar_basis(self, d: int) -> Family:
            """Return the ``Family`` of planar basis elements in degree ``d``."""
            return Family(self.planar_basis_iter(d))

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

            return validate_tree(basis_key, self._arity, self._cooperad_cls, self.base_ring())

        @cached_method
        def _normalize_to_shuffle(self, tree):
            """Normalize *tree* to shuffle form.

            Returns a tuple of ``(shuffle_tree, coeff)`` pairs representing
            a (possibly multi-term) linear combination.  Cached so that the
            same contracted tree is only normalised once.
            """
            if is_leaf(tree):
                return ((tree, 1),)
            return tuple(to_shuffle_tree_cobar(tree, self._cooperad_cls, self.base_ring()))

        def _element_constructor_(self, x):
            """Build elements from tree basis keys or sparse dictionaries.

            Trees are automatically normalized to shuffle form.
            """
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
                    for shuffle_key, shuffle_coeff in self._normalize_to_shuffle(clean_key):
                        clean_dict[shuffle_key] = (
                            clean_dict.get(shuffle_key, 0) + coeff * shuffle_coeff
                        )
                return super()._element_constructor_(clean_dict)

            if isinstance(x, (tuple, int)):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                return self.sum_of_terms(self._normalize_to_shuffle(clean_key))

            return super()._element_constructor_(x)

        def arity(self) -> int:
            return self._arity

        @property
        def connectivity(self) -> int:
            """Connectivity inherited from the underlying cooperad."""
            return getattr(self._cooperad_cls, "connectivity", 0)

        def degree_on_basis(self, tree) -> int:
            """Compute the degree of a tree in Ω(C).

            The degree is sum over all vertices v of:
                deg_C(decoration(v)) - 1

            For the unit (leaf 1 in arity 1), degree is 0.
            """
            if is_leaf(tree):
                return 0
            return subtree_degree_cobar(tree, self._cooperad_cls, self.base_ring())

        def basis_iter(self, d: int) -> Iterator["CobarConstruction.Element"]:
            """Iterate over shuffle-tree basis elements of degree *d*.

            Works for **any** connected cooperad.  For the cobar construction
            the degree of a tree is ``Σ_v (deg_C(dec(v)) - 1)``, which may be
            negative when the cooperad has degree-0 elements (e.g.
            ``CoAssociative``).

            Args:
                d: Cobar degree to enumerate.

            Yields:
                Elements of this cobar component with cobar degree ``d``.
            """
            n = self._arity
            base_ring = self.base_ring()

            if n == 1:
                if d == 0:
                    yield self.term(1)
                return

            for tree in enumerate_shuffle_trees_cobar_in_degree(
                n, self._max_weight, self._cooperad_cls, base_ring, d
            ):
                yield self.term(tree)

        def _repr_term(self, basis_element) -> str:
            """String representation of one cobar basis tree."""
            if is_leaf(basis_element):
                return "id"

            def _dec_fmt(dec, arity):
                parent = self.factory.cooperad_cls(arity, self.base_ring())
                repr_term = getattr(parent, "_repr_term", None)
                if callable(repr_term):
                    return repr_term(dec)
                return f"{self.factory.cooperad_cls.name}{dec}"

            return tree_to_string(
                basis_element,
                decoration_formatter=_dec_fmt,
            )

        @cached_method
        def graded_basis(self, d: int) -> Family:
            """Return the ``Family`` of all basis elements in degree ``d``."""
            return Family(self.basis_iter(d))

        def _latex_term(self, basis_element) -> str:
            """LaTeX representation of one cobar basis tree."""
            if is_leaf(basis_element):
                return f"\\eta_{{{self.factory.cooperad_cls.name}}}"

            def _dec_fmt(dec, arity):
                parent = self.factory.cooperad_cls(arity, self.base_ring())
                latex_term = getattr(parent, "_latex_term", None)
                if callable(latex_term):
                    return latex_term(dec)
                return f"\\operatorname{{{self.factory.cooperad_cls.name}}}({{{dec}}})"

            return tree_to_latex(
                basis_element,
                decoration_formatter=_dec_fmt,
            )

        @cached_method
        def _boundary_on_basis(self, tree) -> "CobarConstruction.Element":
            """Compute the cobar differential d = d_1 + d_2 on a tree.

            - ``d_1`` applies ``C.boundary`` to each vertex decoration.
            - ``d_2`` expands vertices via infinitesimal cocomposition.
            """
            if is_leaf(tree):
                return self.zero()
            return self._d1_on_basis(tree) + self._d2_on_basis(tree)

        @cached_method
        def _d1_on_basis(self, tree) -> "CobarConstruction.Element":
            """Internal differential: apply cooperad boundary to each vertex.

            For vertices in DFS order, the sign at vertex ``v_j`` is

                ``(-1)^{\\sum_{l < j} (deg_C(v_l) - 1)}``.
            """
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

                # Apply boundary to this vertex's decoration.
                # Use __call__ instead of term() to normalise through the
                # cooperad's element constructor.
                dec_elem = cooperad_parent(dec)
                bdry = cooperad_parent.boundary(dec_elem)

                for new_dec, coeff in bdry:
                    new_tree = self._replace_vertex_decoration_by_index(tree, verts, j, new_dec)
                    result += sign * coeff * self(new_tree)

                cumulative_degree += vertex_sinv_degree

            return result

        @cached_method
        def _d2_on_basis(self, tree) -> "CobarConstruction.Element":
            """Structural differential: expand vertices using cocomposition.

            For each expanded vertex ``c`` and split ``Δ^{i;m,n}(c) = Σ c_L ⊗ c_R``,
            the exponent used here is:

                global_accum + deg_C(c_L) + (deg_C(c_R) - 1) * before_deg

            where ``global_accum`` sums ``deg_C(v) - 1`` over DFS-preceding vertices,
            and ``before_deg`` is the total cobar degree of child subtrees in slots
            ``1, ..., i-1``.
            """
            if is_leaf(tree):
                return self.zero()

            result = self.zero()
            base_ring = self.base_ring()
            verts = vertices_dfs(tree)

            for curr_vertex in verts:
                curr_arity = vertex_arity(curr_vertex)
                curr_dec = decoration(curr_vertex)
                curr_parent = self._cooperad_cls(curr_arity, base_ring)
                curr_elem = curr_parent.term(curr_dec)

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
                            total_sign = sign_from_exponent(global_accum + left_degree + koszul_exp)

                            new_tree = expand_vertex(
                                tree, curr_vertex, i, dec_left, dec_right, m, n
                            )
                            # Use _element_constructor_ (via __call__) so that
                            # the expanded tree is normalised to shuffle form.
                            result += total_sign * coeff * self(new_tree)
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
                self._replace_decoration_rec(c, target, new_decoration) for c in children(node)
            )
            return (decoration(node),) + new_children

        @staticmethod
        def unit() -> "CobarConstruction.Element":
            """The operadic unit is handled by the factory."""
            raise NotImplementedError("Use factory.unit() instead")

        @staticmethod
        def unit_key() -> int:
            """Return the basis key of the unit element in arity ``1``.

            In the cobar construction, the arity-1 unit is the single-leaf
            tree, whose basis key is the integer ``1``.
            """
            return 1

        def compose(
            self, x: "CobarConstruction.Element", i: int, y: "CobarConstruction.Element"
        ) -> "CobarConstruction.Element":
            """Delegate to factory compose."""
            return self.factory.compose(x, i, y)

    class Element(
        ParentedElementMixin["CobarConstruction.Component"],
        CombinatorialFreeModule.Element,
    ):
        """Element of a cobar construction operad component."""

        def _repr_latex_(self) -> str:
            """Return a LaTeX linear-combination string for this element."""
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "CobarConstruction.Element":
            """Apply the cobar differential ``d = d_1 + d_2``."""
            parent = self.parent()
            return parent.boundary(self)

        def d1(self) -> "CobarConstruction.Element":
            """Internal differential: applies cooperad boundary to vertex decorations."""
            parent = self.parent()
            return parent._d1(self)

        def d2(self) -> "CobarConstruction.Element":
            """Structural differential: expands internal edges."""
            parent = self.parent()
            return parent._d2(self)

        def permute(self, sigma) -> "CobarConstruction.Element":
            """Permute leaf labels by ``sigma`` (no extra sign)."""
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
                # Use parent(new_tree) so that shuffle normalization
                # is applied — the relabeled tree may be out of order.
                result += coeff * parent(new_tree)
            return result


CobarConstruction.Component.Element = CobarConstruction.Element
