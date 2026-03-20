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
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    decoration,
    enumerate_shuffle_trees_cobar_in_degree,
    enumerate_shuffle_trees_generic_in_degree,
    expand_vertex,
    graft,
    is_leaf,
    relabel_leaves,
    subtree_degree_cobar,
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
        Unlike ``BarConstruction``, this module does not automatically normalize
        trees to shuffle form.  Utilities ``to_shuffle_tree_cobar`` and
        ``is_shuffle_tree`` in ``uconf.core.trees`` can be used explicitly when
        needed.

    """

    def __init__(self, cooperad_cls: CooperadLike):
        self.cooperad_cls = cooperad_cls
        self.name = f"Ω({cooperad_cls.name})"

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

        def _planarize_on_basis(self, tree):
            """Decompose a cobar tree into planar part ⊗ global permutation.

            Mirrors ``BarConstruction.Component._planarize_on_basis``: for each
            internal vertex ``v`` the base cooperad's ``planarize`` gives a
            planar decoration ``dec_pl`` and a vertex permutation ``σ_v ∈ S_{k_v}``.
            The planarized tree is assembled by:

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
                """Return ``(coeff, planar_node, leaf_order)``."""
                if is_leaf(node):
                    return 1, node, [node]

                k = vertex_arity(node)
                dec = decoration(node)
                coop_parent = self._cooperad_cls(k, base_ring)

                dec_elem = coop_parent.term(dec)
                planarized = coop_parent.planarize(dec_elem)

                planar_dec = dec
                sigma_v_tuple = tuple(range(1, k + 1))
                dec_coeff = 1
                for (planar_dec_key, sigma_key), c in planarized:
                    planar_dec = planar_dec_key
                    sigma_v_tuple = SymmetricGroup(k)(sigma_key).tuple()
                    dec_coeff = c
                    break  # single term expected

                old_ch = children(node)
                new_ch = tuple(old_ch[sigma_v_tuple[j] - 1] for j in range(k))

                new_ch_planarized = []
                leaf_order = []
                total_child_coeff = 1
                for ch in new_ch:
                    ch_coeff, p_ch, lo_ch = _planarize_subtree(ch)
                    total_child_coeff *= ch_coeff
                    new_ch_planarized.append(p_ch)
                    leaf_order.extend(lo_ch)

                total_coeff = dec_coeff * total_child_coeff
                return total_coeff, (planar_dec,) + tuple(new_ch_planarized), leaf_order

            total_coeff, planar_with_orig, leaf_order = _planarize_subtree(tree)

            sigma_global_inv = {l: pos for pos, l in enumerate(leaf_order, start=1)}
            canonical_tree = relabel_leaves(planar_with_orig, sigma_global_inv)

            sigma_global = sym_alg.term(self._symmetric_group(list(leaf_order)))

            return total_coeff * self.term(canonical_tree).tensor(sigma_global)

        def planar_basis_it(self, d: int) -> "Iterator[CobarConstruction.Element]":
            """Iterate over planar cobar basis elements of degree ``d``.

            A tree is *planar* when every vertex decoration is a planar element
            of the base cooperad and the global leaf permutation is the identity
            (children occupy consecutive leaf ranges).

            Requires the base cooperad to implement ``planarize`` and
            ``planar_basis_it``; raises :exc:`NotImplementedError` otherwise.
            Use :meth:`basis_it` for the full shuffle-tree basis instead.
            """
            if not self._cooperad_has_planarize():
                raise NotImplementedError(
                    f"planar_basis_it requires {self._cooperad_cls.name!r} to implement "
                    "planarize and planar_basis_it (quasi-planar cooperad). "
                    "Use basis_it() for the full shuffle-tree basis instead."
                )

            n = self._arity
            base_ring = self.base_ring()

            if n < 2:
                if n == 1 and d == 0:
                    yield self.term(1)
                return

            for tree in enumerate_shuffle_trees_generic_in_degree(
                n, self._max_weight, self._cooperad_cls, base_ring, d,
                vertex_offset=-1, use_planar_decs=True
            ):
                yield self.term(tree)

        @cached_method
        def graded_planar_basis(self, d: int) -> Family:
            """Return the ``Family`` of planar basis elements in degree ``d``."""
            return Family(self.planar_basis_it(d))

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

        def basis_it(self, d: int) -> Iterator["CobarConstruction.Element"]:
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
            return tree_to_string(basis_element, self.factory.cooperad_cls.name)

        @cached_method
        def graded_basis(self, d: int) -> Family:
            """Return the ``Family`` of all basis elements in degree ``d``."""
            return Family(self.basis_it(d))

        def _latex_term(self, basis_element) -> str:
            """LaTeX representation of one cobar basis tree."""
            if is_leaf(basis_element):
                return "\\mathrm{id}"
            return tree_to_latex(basis_element, self.factory.cooperad_cls.name)

        def _boundary_on_basis(self, tree) -> "CobarConstruction.Element":
            """Compute the cobar differential d = d_1 + d_2 on a tree.

            - ``d_1`` applies ``C.boundary`` to each vertex decoration.
            - ``d_2`` expands vertices via infinitesimal cocomposition.
            """
            if is_leaf(tree):
                return self.zero()
            return self._d1_on_basis(tree) + self._d2_on_basis(tree)

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

                # Apply boundary to this vertex's decoration
                dec_elem = cooperad_parent.term(dec)
                bdry = cooperad_parent.boundary(dec_elem)

                for new_dec, coeff in bdry:
                    new_tree = self._replace_vertex_decoration_by_index(tree, verts, j, new_dec)
                    result += sign * coeff * self.term(new_tree)

                cumulative_degree += vertex_sinv_degree

            return result

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

        def pretty(self) -> str:
            """Return a readable linear-combination string for this element."""
            if not self:
                return "0"

            pieces = []
            parent = self.parent()
            for basis, coeff in self:
                term = parent._repr_term(basis)
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
            parent = self.parent()
            for basis, coeff in self:
                term = parent._latex_term(basis)
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
                result += coeff * parent.term(new_tree)
            return result


CobarConstruction.Component.Element = CobarConstruction.Element
