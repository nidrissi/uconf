"""Free P-algebra on a dg-module M.

The free P-algebra on a dg-module M is the composite product

    Free_P(M) = P ∘ M = ⊕_{n≥1} P(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(tree, m_tuple) = Σ_{v internal} deg_P(dec(v)) + Σ_i deg_M(m_i)
  (no suspension factor; compare with the bar construction which uses deg_P + 1).
- Differential d = d_P + d_M from the Koszul sign rule on the interleaved
  DFS pre-order of vertices and leaves.
- P-algebra structure γ: P(k) ⊗ Free_P(M)^⊗k → Free_P(M) given by grafting
  a new root vertex over k subtrees.

The basis keys are pairs ``(tree, m_tuple)`` where:

- ``tree`` is an integer leaf (= 1 in arity 1) or a tuple representing a
  decorated rooted tree with leaves labeled ``1, ..., n``.
- ``m_tuple`` is a tuple of ``n`` basis keys of the inner module M.

The inclusion η: M → Free_P(M) sends a basis key m to (1, (m,)).

Reference: Loday-Vallette "Algebraic Operads", Section 5.2.
"""

from __future__ import annotations

import itertools
from typing import ClassVar, Iterator

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.operad import OperadLike
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    children,
    decoration,
    enumerate_shuffle_trees_free_in_degree,
    is_leaf,
    relabel_leaves,
    tree_arity,
    vertex_arity,
    vertices_dfs,
)


def _dfs_all_iter(tree):
    """DFS pre-order traversal of ALL nodes (both internal vertices and leaves).

    Yields ``(node, is_leaf_flag, leaf_0based_index)`` where:

    - For internal vertices: ``(vertex_tuple, None)``
    - For leaves: ``(leaf_int, leaf_int - 1)``
    """
    if is_leaf(tree):
        yield (tree, tree - 1)
        return
    yield (tree, None)
    for child in children(tree):
        yield from _dfs_all_iter(child)


def _module_basis_keys_in_degree(module, d: int) -> Iterator:
    """Yield all basis keys of *module* in degree *d*.

    Tries ``module.basis_it(d)`` first (degree-indexed, more efficient); falls
    back to iterating ``module.basis()`` and filtering by ``degree_on_basis``.
    """
    basis_it_fn = getattr(module, "basis_it", None)
    if basis_it_fn is not None:
        for elem in basis_it_fn(d):
            yield from elem.support()
        return
    for key in module.basis():
        if module.degree_on_basis(key) == d:
            yield key


def _tuples_in_degree(module, n: int, d: int) -> Iterator[tuple]:
    """Yield all ``n``-tuples of *module* basis keys whose total degree equals *d*.

    Assumes *module* has finitely many basis keys in each degree.

    Args:
        module: A :class:`CombinatorialFreeModule` with ``degree_on_basis``.
        n: Tuple length (≥ 0).
        d: Exact total degree.

    Yields:
        ``n``-tuples of basis keys.
    """
    if n == 0:
        if d == 0:
            yield ()
        return
    for d_first in range(d + 1):
        first_keys = list(_module_basis_keys_in_degree(module, d_first))
        for first_key in first_keys:
            for rest in _tuples_in_degree(module, n - 1, d - d_first):
                yield (first_key,) + rest


def _tuples_in_degree_precomputed(keys_by_deg: dict, n: int, d: int) -> Iterator[tuple]:
    """Yield all ``n``-tuples of keys (from *keys_by_deg*) whose total degree equals *d*.

    Args:
        keys_by_deg: Mapping from ``int`` degree to ``list`` of basis keys.
        n: Tuple length (≥ 0).
        d: Exact total degree.

    Yields:
        ``n``-tuples of basis keys.
    """
    if n == 0:
        if d == 0:
            yield ()
        return
    for d_first in range(d + 1):
        first_keys = keys_by_deg.get(d_first, [])
        for first_key in first_keys:
            for rest in _tuples_in_degree_precomputed(keys_by_deg, n - 1, d - d_first):
                yield (first_key,) + rest


class FreeAlgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the free P-algebra ``P ∘ M``.

    Basis keys are ``(tree, m_tuple)`` pairs.  The differential is
    ``d = d_P + d_M`` using the DFS-interleaved Koszul sign rule.

    This class is normally not instantiated directly; use
    :class:`FreeOperadAlgebra` instead.
    """

    name: ClassVar[str] = "Free"

    def __init__(self, operad_cls: OperadLike, inner_module, base_ring):
        """Initialize the free P-algebra module ``P ∘ M``.

        Args:
            operad_cls: Operad provider P (class or wrapper instance).
            inner_module: Generating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring.

        """
        self._operad_cls = operad_cls
        self._inner_module = inner_module

        name = f"{operad_cls.name} ∘ {inner_module}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)

        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    # -----------------------------------------------------------------------
    # Validation and element construction
    # -----------------------------------------------------------------------

    def _validate_basis_key(self, key):
        """Validate a ``(tree, m_tuple)`` basis key.

        Returns the normalized key as a ``(tree, tuple(m_tuple))`` pair, or
        ``None`` if the key is structurally invalid or contains invalid
        inner-module keys.
        """
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
        """Validate one inner-module basis key.

        Delegates to ``inner_module._validate_basis_key`` if available;
        otherwise returns ``m_key`` unchanged.
        """
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

    # -----------------------------------------------------------------------
    # Degree
    # -----------------------------------------------------------------------

    def degree_on_basis(self, key) -> int:
        """Degree = Σ_{v internal} deg_P(dec(v)) + Σ_i deg_M(m_i).

        No suspension: vertices contribute their P-degree directly, not
        deg_P + 1 as in the bar construction.
        """
        tree, m_tuple = key
        v_deg = (
            0
            if is_leaf(tree)
            else sum(
                self._operad_cls(vertex_arity(v), self.base_ring()).degree_on_basis(decoration(v))
                for v in vertices_dfs(tree)
            )
        )
        m_deg = sum(self._inner_module.degree_on_basis(m) for m in m_tuple)
        return v_deg + m_deg

    # -----------------------------------------------------------------------
    # Basis iteration
    # -----------------------------------------------------------------------

    def basis_it(self, d: int) -> Iterator["FreeAlgebraModule.Element"]:
        """Iterate over basis elements of degree *d*.

        Yields all ``(tree, m_tuple)`` pairs with total degree ``d``, where
        ``tree`` is a shuffle tree and ``m_tuple`` is a tuple of basis keys of
        the inner module *M*.

        The iteration covers all arities ``n ≥ 1``.  The arity is bounded
        automatically by the degree structure of *M* and the connectivity of
        *P*:

        - If *M* has elements only in strictly-positive degrees (``min_deg ≥ 1``),
          then ``n ≤ d / min_deg``, giving a finite range.
                - If *P* has ``connectivity ≥ 1``, the tree degree provides an
                    additional bound ``n ≤ d / connectivity + 1``.
                - If both *M* and *P* have degree-0 generators (``min_deg = 0`` and
                    ``connectivity = 0``), arity is not bounded from degree data alone,
                    so exhaustive degree-``d`` enumeration is not guaranteed.  In this
                    case the method raises ``ValueError`` instead of returning a partial
                    list.

        Requires *M* to implement ``basis_it(d)`` or have a finite basis
        accessible through ``basis()`` for each relevant degree.

        Args:
            d: Homological degree to enumerate.

        Yields:
            Elements of this module with degree ``d``.
        """
        M = self._inner_module
        P = self._operad_cls
        R = self.base_ring()

        # Pre-collect M-keys by degree from 0 to d (avoids iterating M.basis()).
        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(d + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        # Arity 1: single leaf (no internal vertices, tree degree = 0)
        for m_key in m_keys_by_deg.get(d, []):
            yield self.term((1, (m_key,)))

        if not m_keys_by_deg:
            return  # M has no basis elements in degrees 0..d

        # Arity n ≥ 2: determine upper arity bound
        min_m_deg = min(m_keys_by_deg.keys())
        connectivity = getattr(P, "connectivity", 0)

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = d // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_it(d): both the inner module "
                "and operad admit degree-0 generators (min_deg=0, connectivity=0), "
                "so arity is unbounded in fixed degree."
            )

        for n in range(2, max_n + 1):
            max_weight = n - 1  # connected operad: weight ≤ n - 1
            for d_M in range(d + 1):
                d_tree = d - d_M
                if d_tree < 0:
                    continue
                m_tuples = list(_tuples_in_degree_precomputed(m_keys_by_deg, n, d_M))
                if not m_tuples:
                    continue
                for tree in enumerate_shuffle_trees_free_in_degree(n, max_weight, P, R, d_tree):
                    for m_tuple in m_tuples:
                        yield self.term((tree, m_tuple))

    def _boundary_on_basis(self, key) -> "FreeAlgebraModule.Element":
        """Differential d = d_P + d_M using the interleaved DFS sign rule.

        For each node in DFS pre-order (vertices and leaves interleaved),
        the sign when applying ∂ at that node is

            ``(-1)^{Σ_{l before this node in DFS all order} deg(x_l)}``

        where ``deg(x_l) = deg_P(dec(v))`` for a vertex and ``deg_M(m)`` for a leaf.
        """
        tree, m_tuple = key
        result = self.zero()
        base_ring = self.base_ring()
        cumulative = 0

        for node, leaf_0idx in _dfs_all_iter(tree):
            sign = sign_from_exponent(cumulative)

            if leaf_0idx is not None:
                # d_M: apply ∂_M to this leaf
                m_key = m_tuple[leaf_0idx]
                m_elem = self._inner_module.term(m_key)
                bdry = self._inner_module.boundary(m_elem)
                for new_m_key, coeff in bdry:
                    new_m = m_tuple[:leaf_0idx] + (new_m_key,) + m_tuple[leaf_0idx + 1 :]
                    result += sign * coeff * self.term((tree, new_m))
                cumulative += self._inner_module.degree_on_basis(m_key)
            else:
                # d_P: apply ∂_P to this vertex
                v_arity = vertex_arity(node)
                dec = decoration(node)
                op_parent = self._operad_cls(v_arity, base_ring)
                bdry = op_parent.boundary(op_parent.term(dec))
                for new_dec, coeff in bdry:
                    new_tree = self._replace_dec(tree, node, new_dec)
                    result += sign * coeff * self.term((new_tree, m_tuple))
                cumulative += op_parent.degree_on_basis(dec)

        return result

    # -----------------------------------------------------------------------
    # Tree manipulation helpers
    # -----------------------------------------------------------------------

    def _replace_dec(self, tree, target_vertex, new_dec):
        """Replace the decoration of ``target_vertex`` in ``tree``.

        Returns a new tree identical to ``tree`` except that the decoration
        tuple of ``target_vertex`` is replaced by ``new_dec``.
        """
        if is_leaf(tree):
            return tree
        if tree is target_vertex:
            return (new_dec,) + children(tree)
        new_children = tuple(self._replace_dec(c, target_vertex, new_dec) for c in children(tree))
        return (decoration(tree),) + new_children

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the free P-algebra module ``P ∘ M``."""

        def boundary(self) -> "FreeAlgebraModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)

        def arity(self) -> int:
            """Return the arity of the underlying tree."""
            # Not a fixed-arity module; no single arity
            raise NotImplementedError(
                "FreeAlgebraModule elements have varying arity; use tree_arity on the basis key."
            )


FreeAlgebraModule.Element = FreeAlgebraModule.Element


class FreeOperadAlgebra(OperadAlgebra):
    """Free P-algebra on a dg-module M.

    Constructs the composite product ``P ∘ M`` as a
    :class:`FreeAlgebraModule` and equips it with the canonical P-algebra
    structure given by grafting (operadic composition on tree decorations).

    Args:
        operad_cls: Operad provider P (class or wrapper instance).
        inner_module: The generating dg-module M.
        base_ring: Coefficient ring.

    The inclusion ``η: M → Free_P(M)`` is::

        m_key  ↦  free_algebra.module.term((1, (m_key,)))

    Examples::

        free_lie = FreeOperadAlgebra(Lie, module_M, R)
        # Apply the Lie bracket to two generators:
        bracket = free_lie.act(Lie(2, R).term((1,)), [x, y])

    """

    def __init__(self, operad_cls: OperadLike, inner_module: CombinatorialFreeModule, base_ring):
        free_module = FreeAlgebraModule(operad_cls, inner_module, base_ring)
        super().__init__(free_module, operad_cls, self._act_impl)
        self._inner_module = inner_module

    def _act_impl(self, p_element, algebra_elements):
        """P-algebra action γ(p; x_1, ..., x_k) by grafting.

        Grafts the k input subtrees ``x_j ∈ Free_P(M)`` under a new root
        vertex decorated by ``p ∈ P(k)``.  Leaf labels of each ``x_j`` are
        shifted by ``Σ_{l<j} |x_l|`` and the M-tuples are concatenated.

        Args:
            p_element: An element of ``operad_cls(k)`` for some arity ``k``.
            algebra_elements: A list of ``k`` elements of ``free_module``.

        Returns:
            An element of ``free_module``.

        """
        k = p_element.arity()
        inputs = list(algebra_elements)
        if len(inputs) != k:
            raise ValueError(f"Expected {k} inputs for P({k}) action, got {len(inputs)}.")

        result = self.module.zero()
        input_term_lists = [list(x) for x in inputs]

        for p_key, p_coeff in p_element:
            for term_combo in itertools.product(*input_term_lists):
                input_keys = [bk for (bk, _) in term_combo]
                coeff = p_coeff
                for _, c in term_combo:
                    coeff = coeff * c
                new_tree, new_m_tuple = self._graft(p_key, input_keys)
                result += coeff * self.module.term((new_tree, new_m_tuple))

        return result

    @staticmethod
    def _graft(p_key, input_keys):
        """Build ``(new_tree, new_m_tuple)`` by grafting ``input_keys`` under ``p_key``.

        Each ``input_keys[j]`` is a ``(tree_j, m_j)`` basis key of
        ``FreeAlgebraModule``.  Leaves of ``tree_j`` are relabeled by the
        offset ``Σ_{l<j} n_l`` so that the full tree has leaves
        ``{1, ..., Σ n_j}``.

        Returns:
            A ``(new_tree, new_m_tuple)`` pair that is a valid
            ``FreeAlgebraModule`` basis key.

        """
        n_list = [len(k[1]) for k in input_keys]
        offsets = []
        running = 0
        for n in n_list:
            offsets.append(running)
            running += n

        new_children = []
        new_m: list = []
        for j, (tree_j, m_j) in enumerate(input_keys):
            off = offsets[j]
            if is_leaf(tree_j):
                new_children.append(off + 1)
            else:
                relabel = {leaf: leaf + off for leaf in range(1, n_list[j] + 1)}
                new_children.append(relabel_leaves(tree_j, relabel))
            new_m.extend(m_j)

        new_tree = (p_key,) + tuple(new_children)
        return new_tree, tuple(new_m)

    def include(self, m_key):
        """Return the image of basis key ``m_key`` under the inclusion η: M → P ∘ M.

        Args:
            m_key: A basis key of the inner module M.

        Returns:
            The element ``module.term((1, (m_key,)))``.

        """
        return self.module.term((1, (m_key,)))
