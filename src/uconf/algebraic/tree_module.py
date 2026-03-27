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

from uconf.core.display import latex_linear_combination
from uconf.core.signs import sign_from_exponent
from uconf.core.trees import (
    _koszul_sign_of_permutation,
    children,
    decoration,
    enumerate_planar_trees_generic_in_degree,
    is_leaf,
    relabel_leaves,
    tree_to_latex,
    tree_to_string,
    tree_arity,
    vertex_arity,
    vertices_dfs,
)
from uconf.core.vertex_decoration import VertexDecorationLike


def _min_tree_degree(connectivity: int, vertex_shift: int, max_arity: int) -> int:
    """Lower bound on the minimum tree degree across all arities and tree shapes.

    For a tree with *n* leaves in a composite ``S ∘ M`` where the symmetric
    sequence *S* has per-arity minimum degree ``connectivity * (n - 1)``
    (the standard convention for connected symmetric sequences), the tree
    degree of an *r*-vertex tree is at least::

        connectivity * (n - 1) + r * vertex_shift

    When ``vertex_shift ≥ 0`` the corolla (``r = 1``) minimises the
    expression; otherwise the fully binary tree (``r = n - 1``) does.
    """
    if max_arity < 2:
        return 0
    if vertex_shift >= 0:
        # Corolla at highest arity is worst case.
        return min(connectivity * (max_arity - 1) + vertex_shift, 0)
    # Binary tree at highest arity is worst case.
    return min((max_arity - 1) * (connectivity + vertex_shift), 0)


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
    else:
        for elem in module.basis():
            if module.degree_on_basis(elem) == d:
                yield from elem.support() if hasattr(elem, "support") else [elem]


def _inner_weight_on_key(module, m_key) -> int:
    """Return the weight of a single basis key of ``module``.

    If ``module`` exposes ``_weight_on_basis``, delegate to it.
    Otherwise every key is assumed to have weight 1 (the default for
    plain dg-modules without an explicit weight notion).
    """
    w_fn = getattr(module, "_weight_on_basis", None)
    if w_fn is not None:
        return w_fn(m_key)
    return 1


def _module_basis_keys_in_weight_and_degree(module, d: int, w: int) -> Iterator:
    """Yield all basis keys of ``module`` in degree ``d`` and weight ``w``.

    If ``module`` exposes ``basis_weight_iter``, it is used directly.
    Otherwise every key is assigned weight 1 and only ``w == 1`` returns
    any keys (those in degree ``d``).
    """
    basis_w_fn = getattr(module, "basis_weight_iter", None)
    if basis_w_fn is not None:
        for elem in basis_w_fn(d, w):
            yield from elem.support()
        return
    # Default: every generator has weight 1.
    if w == 1:
        yield from _module_basis_keys_in_degree(module, d)


def _tuples_in_degree_and_weight(keys_by_dw: dict, n: int, d: int, w: int) -> Iterator[tuple]:
    """Yield all ``n``-tuples of keys with total degree ``d`` and total weight ``w``."""
    if n == 0:
        if d == 0 and w == 0:
            yield ()
        return
    for d_first in range(d + 1):
        for w_first in range(1, w + 1):
            first_keys = keys_by_dw.get((d_first, w_first), [])
            for first_key in first_keys:
                for rest in _tuples_in_degree_and_weight(
                    keys_by_dw, n - 1, d - d_first, w - w_first
                ):
                    yield (first_key,) + rest


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
        base_ring = inner_module.base_ring()

        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    def _repr_(self) -> str:
        return str(getattr(self, "_prefix", self.__class__.__name__))

    def _repr_latex_(self) -> str:
        return self._repr_()

    @staticmethod
    def _short(text: str, max_len: int = 56) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "..."

    def _repr_vertex_decoration(self, dec_key, arity: int) -> str:
        comp = self._symmetric_sequence_cls(arity, self.base_ring())
        term_repr = getattr(comp, "_repr_term", None)
        if callable(term_repr):
            return self._short(str(term_repr(dec_key)))
        return self._short(str(dec_key))

    def _latex_vertex_decoration(self, dec_key, arity: int) -> str:
        comp = self._symmetric_sequence_cls(arity, self.base_ring())
        latex_repr = getattr(comp, "_latex_term", None)
        if callable(latex_repr):
            return str(latex_repr(dec_key))
        return str(dec_key)

    def _repr_inner_key(self, m_key) -> str:
        term_repr = getattr(self._inner_module, "_repr_term", None)
        if callable(term_repr):
            return self._short(str(term_repr(m_key)), max_len=36)
        return self._short(str(m_key), max_len=36)

    def _latex_inner_key(self, m_key) -> str:
        latex_repr = getattr(self._inner_module, "_latex_term", None)
        if callable(latex_repr):
            return str(latex_repr(m_key))
        return str(m_key)

    def _repr_term(self, basis_element) -> str:
        """Readable representation for one basis key ``(tree, m_tuple)``."""
        tree, m_tuple = basis_element

        tree_str = tree_to_string(
            tree,
            decoration_formatter=lambda dec, ar: self._repr_vertex_decoration(dec, ar),
        )
        leaves_str = " ⊗ ".join(self._repr_inner_key(mk) for mk in m_tuple)
        return f"{tree_str} ▷ ({leaves_str})"

    def _latex_term(self, basis_element) -> str:
        """LaTeX representation for one basis key ``(tree, m_tuple)``."""
        tree, m_tuple = basis_element

        tree_ltx = tree_to_latex(
            tree,
            decoration_formatter=lambda dec, ar: self._latex_vertex_decoration(dec, ar),
        )
        leaves_ltx = " \\otimes ".join(self._latex_inner_key(mk) for mk in m_tuple)
        return f"{tree_ltx} \\triangleright \\left({leaves_ltx}\\right)"

    def _validate_basis_key(self, key):
        """Validate and normalize a ``(tree, m_tuple)`` basis key."""
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise TypeError(
                f"Basis key must be a tuple/list of length 2: (tree, m_tuple). Got {key!r}."
            )
        tree, m_tuple = key[0], key[1]
        if not isinstance(m_tuple, (tuple, list)):
            raise TypeError(f"m_tuple must be a tuple/list of leaf keys. Got {m_tuple!r}.")

        if is_leaf(tree):
            if tree != 1 or len(m_tuple) != 1:
                raise ValueError(
                    f"Leaf trees must have the form (1, (m_key,)). Got tree={tree}, m_tuple={m_tuple!r}."
                )
            m_key = self._validate_m_key(m_tuple[0])
            if m_key is None:
                return None
            return (1, (m_key,))

        n = tree_arity(tree)
        if len(m_tuple) != n:
            return None
        clean_m = []
        for m_key in m_tuple:
            vk = self._validate_m_key(m_key)
            if vk is None:
                return None
            clean_m.append(vk)
        return (tree, tuple(clean_m))

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

        Requires the symmetric sequence to support ``planar_basis_iter`` on its
        components (quasi-planar operads such as ``Associative``, ``Surjection``,
        ``BarrattEccles``, or ``ShiftedOperad`` wrapping any of these).  Only
        planar vertex decorations are enumerated, yielding one representative per
        ``S_n``-orbit via the isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}``.

        Raises:
            NotImplementedError: when the symmetric sequence does not expose
                ``planar_basis_iter`` (e.g. ``Commutative``, ``Lie``).
            ValueError: when the arity is unbounded in the requested degree
                (both S and M admit degree-0 generators).
        """
        M = self._inner_module
        S = self._symmetric_sequence_cls
        R = self.base_ring()

        connectivity = int(getattr(S, "connectivity", 0))

        # When the symmetric sequence has negative-degree elements, the tree
        # contribution can be negative.  A corolla at arity n has minimum tree
        # degree connectivity*(n-1) + vertex_degree_shift, which can decrease
        # with arity when connectivity < 0.  We first compute a provisional
        # bound (for arity 2), then refine after max_n is known.
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

        # Early exit: the smallest tree for n≥2 is a single arity-2 vertex
        # with degree connectivity + vertex_shift.  If d is below this even
        # with m_tuple degree 0, no n≥2 elements exist.
        if d < self._vertex_degree_shift + connectivity:
            return

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = (d - self._vertex_degree_shift) // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_iter(d): both the inner module "
                "and symmetric sequence admit degree-0 generators (min_deg=0, connectivity=0), "
                "so arity is unbounded in fixed degree.  "
                "Use basis_weight_iter(d, w) to enumerate elements of a fixed weight."
            )

        # Refine min_tree_deg now that max_n is known.
        # For connected symmetric sequences, S(n) has minimum degree
        # connectivity*(n-1).  A tree with n leaves and r internal vertices
        # contributes at least connectivity*(n-1) + r*vertex_shift.  When
        # vertex_shift >= 0 the corolla (r=1) is worst; otherwise the
        # binary tree (r=n-1) is worst.
        min_tree_deg = _min_tree_degree(connectivity, self._vertex_degree_shift, max_n)
        max_m_deg = d - min_tree_deg

        # Collect any additional m_keys in the extended range.
        for d_m in range(max(m_keys_by_deg.keys()) + 1, max_m_deg + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        if d < min_tree_deg:
            return

        # Require quasi-planar support: basis_iter() is only correct when the
        # symmetric sequence exposes planar_basis_iter on its components, so that
        # we can pick one representative per S_n-orbit (matching the isomorphism
        # P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}).  For non-quasi-planar
        # operads (e.g. Commutative, Lie) the tensor-over-S_n quotient cannot
        # be represented by a naive product of full bases, and falling back to
        # the full basis would silently produce an overcomplete set.
        _use_planar = hasattr(S(2, R), "planar_basis_iter")

        if not _use_planar:
            raise NotImplementedError(
                f"basis_iter() requires the symmetric sequence {S.name!r} to support "
                "planar_basis_iter() on its arity-2 component.  "
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
                for tree in enumerate_planar_trees_generic_in_degree(
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

    # ------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------

    def _weight_on_basis(self, key) -> int:
        """Weight of a single basis key ``(tree, m_tuple)``.

        The weight is the sum of weights of the leaf-module elements in
        ``m_tuple``.  Each leaf element's weight is obtained from the inner
        module's ``_weight_on_basis``; modules without this attribute
        default to weight 1 per key.
        """
        _tree, m_tuple = key
        return sum(_inner_weight_on_key(self._inner_module, m) for m in m_tuple)

    def basis_weight_iter(self, d: int, w: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d`` and weight ``w``.

        Weight is additive: the weight of ``(tree, m_tuple)`` is the sum of
        the weights of its leaf-module elements.  The weight of an
        inner-module key is given by the inner module's
        :meth:`_weight_on_basis`; plain dg-modules without this attribute
        default to weight 1 per key.

        Unlike :meth:`basis_iter`, this method is always finite (``w``
        bounds the arity).

        Raises:
            NotImplementedError: when the symmetric sequence does not expose
                ``planar_basis_iter`` (e.g. ``Commutative``, ``Lie``).
        """
        if w < 1:
            return

        M = self._inner_module
        S = self._symmetric_sequence_cls
        R = self.base_ring()

        connectivity = int(getattr(S, "connectivity", 0))

        # Max arity: each leaf has weight >= 1, so n <= w.
        max_n = w

        # Minimum tree degree considering all arities from 2 to max_n.
        # For connected symmetric sequences, S(n) has minimum degree
        # connectivity*(n-1).  See _min_tree_degree for the derivation.
        min_tree_deg = _min_tree_degree(connectivity, self._vertex_degree_shift, max_n)
        max_m_deg = d - min_tree_deg

        # Collect inner-module keys by (degree, weight)
        keys_by_dw: dict[tuple, list] = {}
        for d_m in range(max_m_deg + 1):
            for w_m in range(1, w + 1):
                keys = list(_module_basis_keys_in_weight_and_degree(M, d_m, w_m))
                if keys:
                    keys_by_dw[(d_m, w_m)] = keys

        # Yield leaf elements: n=1, total weight=w, total degree=d
        for m_key in keys_by_dw.get((d, w), []):
            yield self.term((1, (m_key,)))

        if not keys_by_dw:
            return

        if d < min_tree_deg:
            return

        # Require quasi-planar support
        _use_planar = hasattr(S(2, R), "planar_basis_iter")
        if not _use_planar:
            raise NotImplementedError(
                f"basis_weight_iter() requires the symmetric sequence {S.name!r} to support "
                "planar_basis_iter() on its arity-2 component."
            )

        for n in range(2, max_n + 1):
            max_tree_weight = n - 1
            for d_M in range(max_m_deg + 1):
                d_tree = d - d_M
                m_tuples = list(_tuples_in_degree_and_weight(keys_by_dw, n, d_M, w))
                if not m_tuples:
                    continue
                for tree in enumerate_planar_trees_generic_in_degree(
                    n,
                    max_tree_weight,
                    S,
                    R,
                    d_tree,
                    self._vertex_degree_shift,
                    use_planar_decs=True,
                ):
                    for m_tuple in m_tuples:
                        yield self.term((tree, m_tuple))

    @cached_method
    def graded_basis_by_weight(self, d: int, w: int):
        """Cached family of basis elements of degree ``d`` and weight ``w``."""
        return Family(self.basis_weight_iter(d, w))

    def _boundary_on_basis(self, key) -> Any:
        """Differential using interleaved DFS Koszul sign rule.

        Inner-module boundary terms are normalised to planar form (when
        the inner module exposes ``normalize_to_planar``) so that the
        output lives in the same basis as the structure-map outputs
        produced by ``_dalpha_on_basis`` and similar methods.
        """
        tree, m_tuple = key
        result = self.zero()
        base_ring = self.base_ring()
        shift = self._vertex_degree_shift
        cumulative = 0
        inner_normalize = getattr(self._inner_module, "normalize_to_planar", None)

        for node, leaf_0idx in _dfs_all_iter(tree):
            sign = sign_from_exponent(cumulative)

            if leaf_0idx is not None:
                m_key = m_tuple[leaf_0idx]
                m_elem = self._inner_module.term(m_key)
                bdry = self._inner_module.boundary(m_elem)
                if inner_normalize is not None:
                    bdry = inner_normalize(bdry)
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

        If the inner module also exposes ``normalize_to_planar``, it is applied
        recursively to each leaf value so that nested tree-decorated structures
        (e.g. ``FreeAlgebraModule`` keys used as inner-module values) are fully
        normalised.
        """
        result = self.zero()
        for key, coeff in elem:
            for norm_coeff, norm_key in self._normalize_key(key):
                result += coeff * norm_coeff * self.term(norm_key)
        # Recursively normalise inner-module keys if possible.
        inner_normalize = getattr(self._inner_module, "normalize_to_planar", None)
        if inner_normalize is not None:
            result = self._normalize_inner_keys(result, inner_normalize)
        return result

    def _normalize_inner_keys(self, elem, inner_normalize):
        """Apply inner-module normalisation to every leaf value."""
        result = self.zero()
        for (tree, m_tuple), coeff in elem:
            # Build a temporary inner-module element for each leaf key,
            # normalise it, and rebuild the outer key.
            new_parts: list[list[tuple]] = []
            for mk in m_tuple:
                normed = inner_normalize(self._inner_module.term(mk))
                new_parts.append(list(normed))
            # Form all combinations (usually just 1 term per leaf)
            from itertools import product as iprod

            for combo in iprod(*new_parts):
                inner_coeff = coeff
                new_m_keys = []
                for new_mk, mk_c in combo:
                    inner_coeff *= mk_c
                    new_m_keys.append(new_mk)
                result += inner_coeff * self.term((tree, tuple(new_m_keys)))
        return result

    def _normalize_key(self, key):
        """Normalize a single ``(tree, m_tuple)`` key to canonical form.

        The canonical form has planar vertex decorations with leaves labeled
        ``1, …, n`` in DFS pre-order (consecutive-interval tree shape).

        The normalization uses a **bottom-up, all-at-once** strategy
        (Loday–Vallette, *Algebraic Operads*, §6.1 / §8.2):

        1. **Phase 1** – planarize every vertex decoration in a single
           bottom-up (post-order) pass.  At each internal vertex *v*,
           ``planarize(dec_v)`` yields ``(dec_pl, σ_v)`` and the children of
           *v* are reordered by ``σ_v⁻¹``.  Because ``planarize`` depends
           only on the abstract element ``dec_v ∈ S(k)`` and not on leaf
           labels, each vertex's planarization is independent of every
           other vertex's, so the order among siblings does not matter and
           no intermediate relabeling is needed.

        2. **Phase 2** – read the DFS leaf order of the fully-planarized
           tree, relabel leaves to ``1, …, n``, permute ``m_tuple`` to
           match, and include a single Koszul sign for the graded
           permutation of leaf-module elements.

        Returns a list of ``(coeff, key)`` pairs.
        """
        tree, m_tuple = key
        if is_leaf(tree):
            return [(self.base_ring().one(), key)]

        base_ring = self.base_ring()

        # Phase 1: bottom-up planarization of all vertices.
        planarized_trees = self._planarize_tree_bottom_up(tree, base_ring)

        # Phase 2: DFS relabeling + m_tuple permutation + Koszul sign.
        n_leaves = len(m_tuple)
        result: list[tuple] = []
        for tree_coeff, p_tree in planarized_trees:
            new_leaf_order = _leaves_dfs(p_tree)

            if new_leaf_order == list(range(1, n_leaves + 1)):
                result.append((tree_coeff, (p_tree, m_tuple)))
                continue

            relabel_map = {old: i + 1 for i, old in enumerate(new_leaf_order)}
            canonical_tree = relabel_leaves(p_tree, relabel_map)
            new_m = tuple(m_tuple[old - 1] for old in new_leaf_order)

            if n_leaves > 1:
                degrees = [self._inner_module.degree_on_basis(m_tuple[i]) for i in range(n_leaves)]
                perm_0idx = [old - 1 for old in new_leaf_order]
                koszul = _koszul_sign_of_permutation(perm_0idx, degrees)
            else:
                koszul = 1

            result.append((tree_coeff * koszul, (canonical_tree, new_m)))

        return result

    def _planarize_tree_bottom_up(self, tree, base_ring):
        """Planarize all vertex decorations bottom-up (post-order).

        Returns a list of ``(coeff, new_tree)`` pairs where every vertex
        decoration is planar.  Leaf labels are **not** relabeled – that
        is left to the caller.

        The bottom-up traversal guarantees that each vertex is processed
        exactly once, and the planarization at each vertex is independent
        of all others (it depends only on the decoration, which is an
        abstract element of the symmetric sequence component and is
        unaffected by leaf labels or sibling order).
        """
        if is_leaf(tree):
            return [(1, tree)]

        k = vertex_arity(tree)
        dec = decoration(tree)
        kids = children(tree)

        # Step 1: recursively planarize all children (bottom-up).
        child_options: list[list[tuple]] = []
        for ch in kids:
            if is_leaf(ch):
                child_options.append([(1, ch)])
            else:
                child_options.append(self._planarize_tree_bottom_up(ch, base_ring))

        # Step 2: form all combinations of planarized children.
        # For quasi-planar sequences each child yields a single term,
        # but we handle the general case for correctness.
        from itertools import product as iter_product

        results: list[tuple] = []
        for combo in iter_product(*child_options):
            child_coeff = 1
            new_kids: list = []
            for c, t in combo:
                child_coeff *= c
                new_kids.append(t)

            # Step 3: planarize the current vertex's decoration.
            seq_parent = self._symmetric_sequence_cls(k, base_ring)
            if not hasattr(seq_parent, "planarize"):
                results.append((child_coeff, (dec,) + tuple(new_kids)))
                continue

            planarized = seq_parent.planarize(seq_parent.term(dec))
            terms = list(planarized)

            S_k = SymmetricGroup(k)
            identity = S_k.identity()

            for (pl_dec, sigma_key), pl_coeff in terms:
                sigma = S_k(sigma_key)
                if sigma == identity:
                    results.append((child_coeff * pl_coeff, (pl_dec,) + tuple(new_kids)))
                else:
                    sigma_inv = sigma.inverse()
                    reordered = tuple(new_kids[sigma_inv(j) - 1] for j in range(1, k + 1))
                    results.append((child_coeff * pl_coeff, (pl_dec,) + reordered))

        return results

    class Element(CombinatorialFreeModule.Element):
        """An element of the free S-algebra module ``P ∘ M``."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def boundary(self) -> "TreeModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)
