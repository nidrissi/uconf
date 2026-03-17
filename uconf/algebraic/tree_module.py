"""Shared tree-decorated composite module over a symmetric sequence and module.

This module implements the common dg-module machinery for composites of the
form

    S ∘ M

where S is an arity-indexed symmetric sequence (operad-like or cooperad-like
for our use-cases) and M is an inner dg-module.
"""

from __future__ import annotations

from typing import Any, Iterator

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    decoration,
    enumerate_shuffle_trees_generic_in_degree,
    is_leaf,
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


def _module_basis_keys_in_degree(module, d: int) -> Iterator:
    """Yield all basis keys of ``module`` in degree ``d``."""
    basis_it_fn = getattr(module, "basis_it", None)
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
        base_ring,
        *,
        vertex_degree_shift: int = 0,
        name: str,
    ):
        self._symmetric_sequence_cls = symmetric_sequence_cls
        self._inner_module = inner_module
        self._vertex_degree_shift = vertex_degree_shift

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

    def basis_it(self, d: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d``.

        When the symmetric sequence supports ``planar_basis_it`` (quasi-planar
        operads such as ``Associative``, ``Surjection``, ``BarrattEccles``),
        only planar vertex decorations are enumerated.  This yields one
        representative per ``S_n``-orbit, matching the isomorphism
        ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}`` for the composite
        product ``P ∘ M``.
        """
        M = self._inner_module
        S = self._symmetric_sequence_cls
        R = self.base_ring()

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(d + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        for m_key in m_keys_by_deg.get(d, []):
            yield self.term((1, (m_key,)))

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        connectivity = getattr(S, "connectivity", 0)
        min_tree_deg_n2 = self._vertex_degree_shift + connectivity

        if d < min_tree_deg_n2:
            return

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = (d - self._vertex_degree_shift) // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_it(d): both the inner module "
                "and symmetric sequence admit degree-0 generators (min_deg=0, connectivity=0), "
                "so arity is unbounded in fixed degree."
            )

        # Detect quasi-planar support: use planar vertex decorations when the
        # symmetric sequence exposes planar_basis_it on its components.
        # We probe with arity 2 (the minimal connected arity); if S(2, R) cannot
        # be constructed or does not implement planar_basis_it, fall back to the
        # full basis.  Common failure modes (unsupported arity, missing base_ring
        # argument) raise TypeError, ValueError, or NotImplementedError.
        try:
            _use_planar = hasattr(S(2, R), "planar_basis_it")
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            _use_planar = False

        for n in range(2, max_n + 1):
            max_weight = n - 1
            for d_M in range(d + 1):
                d_tree = d - d_M
                if d_tree < 0:
                    continue
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_M))
                if not m_tuples:
                    continue
                for tree in enumerate_shuffle_trees_generic_in_degree(
                    n, max_weight, S, R, d_tree, self._vertex_degree_shift,
                    use_planar_decs=_use_planar,
                ):
                    for m_tuple in m_tuples:
                        yield self.term((tree, m_tuple))

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

    class Element(CombinatorialFreeModule.Element):
        """An element of the free S-algebra module ``P ∘ M``."""

        def boundary(self) -> "TreeModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)
