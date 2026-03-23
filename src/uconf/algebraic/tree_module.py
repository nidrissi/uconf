"""Shared tree-decorated composite module over a symmetric sequence and module.

This module implements the common dg-module machinery for composites of the
form

    S ∘ M

where S is an arity-indexed symmetric sequence (operad-like or cooperad-like
for our use-cases) and M is an inner dg-module.
"""

from __future__ import annotations

from typing import Any, Iterator

from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    Family,
    SymmetricGroup,
    cached_method,
)

from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    _koszul_sign_of_permutation,
    children,
    decoration,
    enumerate_shuffle_trees_generic_in_degree,
    is_leaf,
    relabel_leaves,
    tree_arity,
    vertex_arity,
    vertices_dfs,
)
from uconf.core.vertex_decoration import VertexDecorationLike


def _dfs_all_iter(tree):
    """DFS pre-order traversal of all nodes (internal vertices and leaves)."""
    if is_leaf(tree):
        yield (tree, tree - 1)
        return
    yield (tree, None)
    for child in children(tree):
        yield from _dfs_all_iter(child)


def _leaves_dfs(tree) -> list[int]:
    """Return leaf labels in DFS pre-order."""
    if is_leaf(tree):
        return [tree]
    result: list[int] = []
    for child in children(tree):
        result.extend(_leaves_dfs(child))
    return result


def _module_basis_keys_in_degree(module, d: int) -> Iterator:
    """Yield all basis keys of ``module`` in degree ``d``."""
    basis_it_fn = getattr(module, "basis_iter", None)
    if basis_it_fn is not None:
        for elem in basis_it_fn(d):
            yield from elem.support()
        return
    for key in module.basis():
        if module.degree_on_basis(key) == d:
            yield key


def _tuples_in_degree(keys_by_deg: dict, n: int, d: int) -> Iterator[tuple]:
    """Yield all ``n``-tuples of keys whose total degree is ``d``."""
    if n == 0:
        if d == 0:
            yield ()
        return
    for d_first in range(d + 1):
        first_keys = keys_by_deg.get(d_first, [])
        for first_key in first_keys:
            for rest in _tuples_in_degree(keys_by_deg, n - 1, d - d_first):
                yield (first_key,) + rest


class TreeModule(CombinatorialFreeModule):
    """Common dg-module for tree-decorated composites ``S ∘ M``.

    Basis keys are ``(tree, m_tuple)`` where internal vertices are decorated by
    elements of ``S`` and leaves by basis keys of ``M``.
    """

    def __init__(
        self,
        symmetric_sequence_cls: VertexDecorationLike,
        inner_module: CombinatorialFreeModule,
        *,
        vertex_degree_shift: int = 0,
        name: str,
    ):
        self._symmetric_sequence_cls = symmetric_sequence_cls
        self._inner_module = inner_module
        self._vertex_degree_shift = vertex_degree_shift
        self._max_arity: int | None = None
        base_ring = inner_module.base_ring()

        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    def _validate_basis_key(self, key):
        """Validate and normalize a ``(tree, m_tuple)`` basis key."""
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        tree, m_tuple = key[0], key[1]
        if not isinstance(m_tuple, (tuple, list)):
            return None

        if is_leaf(tree):
            if tree != 1 or len(m_tuple) != 1:
                return None
            m_key = self._validate_m_key(m_tuple[0])
            if m_key is None:
                return None
            return (1, (m_key,))

        n = tree_arity(tree)
        if len(m_tuple) != n:
            return None
        new_m = []
        for m_key in m_tuple:
            vk = self._validate_m_key(m_key)
            if vk is None:
                return None
            new_m.append(vk)
        return (tree, tuple(new_m))

    def _validate_m_key(self, m_key):
        """Validate one inner-module basis key."""
        if hasattr(self._inner_module, "_validate_basis_key"):
            return self._inner_module._validate_basis_key(m_key)
        return m_key

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return self.sum_of_terms(clean.items())

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)

        return super()._element_constructor_(x)

    @property
    def connectivity(self) -> int:
        """Minimum degree of any basis element.

        Leaf-only elements (arity 1) have degree ``connectivity(M)``.
        Tree elements (arity ≥ 2) have degree
        ``≥ vertex_degree_shift + connectivity(S) + 2 · connectivity(M)``.
        """
        m_conn = int(getattr(self._inner_module, "connectivity", 0))
        s_conn = int(getattr(self._symmetric_sequence_cls, "connectivity", 0))
        tree_min = self._vertex_degree_shift + s_conn + 2 * m_conn
        return min(m_conn, tree_min)

    def degree_on_basis(self, key) -> int:
        """Degree = vertex contribution plus leaf-module contribution."""
        tree, m_tuple = key
        shift = self._vertex_degree_shift
        v_deg = (
            0
            if is_leaf(tree)
            else sum(
                self._symmetric_sequence_cls(vertex_arity(v), self.base_ring()).degree_on_basis(
                    decoration(v)
                )
                + shift
                for v in vertices_dfs(tree)
            )
        )
        m_deg = sum(self._inner_module.degree_on_basis(m) for m in m_tuple)
        return v_deg + m_deg

    def basis_iter(self, d: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d``.

        Requires the symmetric sequence to support ``planar_basis_it`` on its
        components (quasi-planar operads such as ``Associative``, ``Surjection``,
        ``BarrattEccles``, or ``ShiftedOperad`` wrapping any of these).  Only
        planar vertex decorations are enumerated, yielding one representative per
        ``S_n``-orbit via the isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}``.

        Raises:
            NotImplementedError: when the symmetric sequence does not expose
                ``planar_basis_it`` (e.g. ``Commutative``, ``Lie``).
            ValueError: when the arity is unbounded in the requested degree
                (both S and M admit degree-0 generators).
        """
        M = self._inner_module
        S = self._symmetric_sequence_cls
        R = self.base_ring()

        connectivity = int(getattr(S, "connectivity", 0))

        # When the symmetric sequence has negative-degree elements, the tree
        # contribution can be negative.  A corolla at arity n has minimum tree
        # degree = connectivity*(n-1) + vertex_degree_shift.  The inner-module
        # keys must therefore be collected up to degree d − min_tree_deg, which
        # can exceed d.
        min_tree_deg = min(self._vertex_degree_shift + connectivity, 0)
        max_m_deg = d - min_tree_deg  # could exceed d

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(max_m_deg + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        for m_key in m_keys_by_deg.get(d, []):
            yield self.term((1, (m_key,)))

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        min_tree_deg_n2 = self._vertex_degree_shift + connectivity

        if d < min_tree_deg_n2:
            return

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = (d - self._vertex_degree_shift) // connectivity + 1
        elif self._max_arity is not None:
            max_n = self._max_arity
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_iter(d): both the inner module "
                "and symmetric sequence admit degree-0 generators (min_deg=0, connectivity=0), "
                "so arity is unbounded in fixed degree.  "
                "Pass n_factors to chain_complex() to restrict to a finite subcomplex, "
                "or call set_max_arity() on this module directly."
            )

        # Require quasi-planar support: basis_iter() is only correct when the
        # symmetric sequence exposes planar_basis_it on its components, so that
        # we can pick one representative per S_n-orbit (matching the isomorphism
        # P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}).  For non-quasi-planar
        # operads (e.g. Commutative, Lie) the tensor-over-S_n quotient cannot
        # be represented by a naive product of full bases, and falling back to
        # the full basis would silently produce an overcomplete set.
        _use_planar = hasattr(S(2, R), "planar_basis_it")

        if not _use_planar:
            raise NotImplementedError(
                f"basis_iter() requires the symmetric sequence {S.name!r} to support "
                "planar_basis_it() on its arity-2 component.  "
                "Supported quasi-planar sequences include Associative, Surjection, "
                "BarrattEccles, and ShiftedOperad wrapping any of these.  "
                "For non-quasi-planar operads (e.g. Commutative, Lie) the basis "
                "of the composite product P ∘ M cannot be enumerated this way."
            )

        for n in range(2, max_n + 1):
            max_weight = n - 1
            for d_M in range(max_m_deg + 1):
                d_tree = d - d_M
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_M))
                if not m_tuples:
                    continue
                for tree in enumerate_shuffle_trees_generic_in_degree(
                    n,
                    max_weight,
                    S,
                    R,
                    d_tree,
                    self._vertex_degree_shift,
                    use_planar_decs=True,
                ):
                    for m_tuple in m_tuples:
                        yield self.term((tree, m_tuple))

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    def set_max_arity(self, max_arity: int | None) -> None:
        """Set the maximum leaf-arity for basis enumeration.

        When the tree degree and inner-module connectivity are both ≤ 0,
        the arity is unbounded.  Setting a finite ``max_arity`` truncates
        the enumeration so that only trees with at most ``max_arity`` leaves
        are generated.  This also clears the cached ``graded_basis`` results.
        """
        self._max_arity = max_arity
        self.graded_basis.clear_cache()

    def _boundary_on_basis(self, key) -> Any:
        """Differential using interleaved DFS Koszul sign rule."""
        tree, m_tuple = key
        result = self.zero()
        base_ring = self.base_ring()
        shift = self._vertex_degree_shift
        cumulative = 0

        for node, leaf_0idx in _dfs_all_iter(tree):
            sign = sign_from_exponent(cumulative)

            if leaf_0idx is not None:
                m_key = m_tuple[leaf_0idx]
                m_elem = self._inner_module.term(m_key)
                bdry = self._inner_module.boundary(m_elem)
                for new_m_key, coeff in bdry:
                    new_m = m_tuple[:leaf_0idx] + (new_m_key,) + m_tuple[leaf_0idx + 1 :]
                    result += sign * coeff * self.term((tree, new_m))
                cumulative += self._inner_module.degree_on_basis(m_key)
            else:
                v_arity = vertex_arity(node)
                dec = decoration(node)
                seq_parent = self._symmetric_sequence_cls(v_arity, base_ring)
                seq_elem: Any = seq_parent.term(dec)
                bdry = seq_parent.boundary(seq_elem)
                for new_dec, coeff in bdry:
                    new_tree = self._replace_dec(tree, node, new_dec)
                    result += sign * coeff * self.term((new_tree, m_tuple))
                cumulative += seq_parent.degree_on_basis(dec) + shift

        return result

    def _replace_dec(self, tree, target_vertex, new_dec):
        """Replace decoration of ``target_vertex`` in ``tree``."""
        if is_leaf(tree):
            return tree
        if tree is target_vertex:
            return (new_dec,) + children(tree)
        new_children = tuple(self._replace_dec(c, target_vertex, new_dec) for c in children(tree))
        return (decoration(tree),) + new_children

    # ------------------------------------------------------------------
    # Planar normalisation
    # ------------------------------------------------------------------

    def normalize_to_planar(self, elem: "TreeModule.Element") -> "TreeModule.Element":
        """Rewrite *elem* so every vertex decoration is planar.

        For each basis key ``(tree, m_tuple)`` whose tree contains a non-planar
        vertex decoration, ``planarize`` is applied to obtain a planar
        representative ``(planar_dec, σ)``.  The children of the vertex are
        permuted by ``σ⁻¹`` (matching the graded ``S_n``-coinvariant relation),
        leaves are relabeled to canonical DFS order, ``m_tuple`` is permuted to
        match, and a Koszul sign ``ε(σ⁻¹; degrees)`` is included to account
        for the permutation of graded leaf-module elements.
        """
        result = self.zero()
        for key, coeff in elem:
            for norm_coeff, norm_key in self._normalize_key(key):
                result += coeff * norm_coeff * self.term(norm_key)
        return result

    def _normalize_key(self, key):
        """Normalize a single ``(tree, m_tuple)`` key to use planar decorations.

        Returns a list of ``(coeff, key)`` pairs.  Includes the Koszul sign
        for permuting graded leaf-module elements when children are reordered.
        """
        tree, m_tuple = key
        if is_leaf(tree):
            return [(self.base_ring().one(), key)]

        base_ring = self.base_ring()

        for v in vertices_dfs(tree):
            k = vertex_arity(v)
            dec = decoration(v)
            seq_parent = self._symmetric_sequence_cls(k, base_ring)

            if not hasattr(seq_parent, "planarize"):
                continue

            planarized = seq_parent.planarize(seq_parent.term(dec))
            terms = list(planarized)

            # Check if already planar: single term with identity permutation
            S_k = SymmetricGroup(k)
            identity = S_k.identity()
            if len(terms) == 1:
                (pl_key, sigma_key), pl_coeff = terms[0]
                if S_k(sigma_key) == identity and pl_key == dec:
                    continue

            # This vertex needs normalisation
            result: list[tuple] = []
            for (pl_dec, sigma_key), pl_coeff in terms:
                sigma = S_k(sigma_key)
                sigma_inv = sigma.inverse()

                # Permute children of v by σ⁻¹
                old_kids = children(v)
                new_kids = tuple(old_kids[sigma_inv(j) - 1] for j in range(1, k + 1))
                new_v = (pl_dec,) + new_kids

                # Replace v in the tree
                new_tree = self._replace_subtree(tree, v, new_v)

                # Compute DFS leaf order, relabel to canonical 1..n, permute m_tuple
                new_leaf_order = _leaves_dfs(new_tree)
                relabel_map = {old: i + 1 for i, old in enumerate(new_leaf_order)}
                canonical_tree = relabel_leaves(new_tree, relabel_map)
                new_m = tuple(m_tuple[old - 1] for old in new_leaf_order)

                # Koszul sign: the permutation from old leaf order (1..n) to
                # new_leaf_order induces a sign from permuting graded elements.
                # perm[i] = new_leaf_order[i] - 1 gives the 0-indexed source
                # position for each new position.
                n_leaves = len(m_tuple)
                if n_leaves > 1:
                    degrees = [
                        self._inner_module.degree_on_basis(m_tuple[i]) for i in range(n_leaves)
                    ]
                    # new_leaf_order[i] is the old 1-indexed leaf at new position i
                    perm_0idx = [old - 1 for old in new_leaf_order]
                    koszul = _koszul_sign_of_permutation(perm_0idx, degrees)
                else:
                    koszul = 1

                new_key = (canonical_tree, new_m)

                # Recursively normalise remaining vertices
                for sub_coeff, sub_key in self._normalize_key(new_key):
                    result.append((pl_coeff * koszul * sub_coeff, sub_key))

            return result

        # All decorations already planar
        return [(self.base_ring().one(), key)]

    @staticmethod
    def _replace_subtree(tree, target, replacement):
        """Replace *target* (by identity) with *replacement* in *tree*."""
        if is_leaf(tree):
            return tree
        if tree is target:
            return replacement
        new_children = tuple(
            TreeModule._replace_subtree(c, target, replacement) for c in children(tree)
        )
        return (decoration(tree),) + new_children

    class Element(CombinatorialFreeModule.Element):
        """An element of the free S-algebra module ``P ∘ M``."""

        def boundary(self) -> "TreeModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)
