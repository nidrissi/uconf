r"""Simplicial torus cochain model with its Surjection-algebra structure.

This module implements the normalized cochains of the *simplicial* torus
``T² = (Δ¹/∂Δ¹) × (Δ¹/∂Δ¹)`` (product of simplicial sets) as a rank-6
module, equipped with the canonical Berger--Fresse interval-cut action of
the ``Surjection`` operad.

**Why this model exists.**
The rank-4 tensor-square model of :mod:`uconf.algebraic.torus` is a
``BarrattEccles``-algebra whose action does *not* factor through the
surjection operad: there are elements of ``ker(TR) ⊆ E(3)`` acting
nontrivially (e.g. ``x - section(TR(x))`` for
``x = ((1,2,3),(1,3,2),(2,1,3))`` acts by ``-[γ]`` on ``([β],[γ],[α])``,
with coefficient ``±1``, hence in every characteristic).  Moreover no
*strict* quasi-isomorphism of Barratt--Eccles algebras exists between the
rank-4 model and (the table-reduction pullback of) this one, in either
direction, over ``QQ``, ``GF(2)`` or ``GF(3)`` — any strict morphism kills
the degree-2 generators.  The two models are connected by the canonical
zigzag of E_∞-quasi-isomorphisms.  By contrast, the present model is a
genuine ``Surjection``-algebra by naturality of the interval-cut action,
which makes the table-reduction (Euclidean-style) configuration pipeline
available for the torus.

**Simplicial combinatorics.**
An ``m``-simplex of ``S¹ = Δ¹/∂Δ¹`` is encoded by ``j ∈ {0, …, m}``:
``j = 0`` is the (totally degenerate) basepoint and ``j ≥ 1`` is the
monotone map ``[m] → [1]`` jumping from ``0`` to ``1`` at position ``j``.
An ``m``-simplex of ``T²`` is a pair ``(j_x, j_y)``; it is nondegenerate
iff no slot ``i ∈ {0, …, m-1}`` is collapsible in both coordinates, i.e.
iff for every ``i`` either ``j_x = i+1`` or ``j_y = i+1``.  The
nondegenerate simplices are

- ``v = (0,0)``    (the vertex, degree 0),
- ``a = (1,0)``    (the edge ``a × *``, degree -1),
- ``b = (0,1)``    (the edge ``* × a``, degree -1),
- ``g = (1,1)``    (the diagonal edge, degree -1),
- ``t1 = (1,2)``, ``t2 = (2,1)``  (the two shuffle triangles, degree -2),

with simplicial boundary ``∂t1 = ∂t2 = a + b - g`` (and ``∂ = 0`` on
edges, in normalized chains).  Dualizing with the convention of
:class:`~uconf.models.simplicial.SimplicialCochains`
(``(δf)(σ) = -(-1)^{|f|} f(∂σ)``), the cochain differential is

    ``δ[a] = δ[b] = [t1] + [t2]``,  ``δ[g] = -[t1] - [t2]``,

and zero on ``[v]``, ``[t1]``, ``[t2]``.  In cohomology ``[a]+[g]`` and
``[b]+[g]`` generate ``H^{-1}`` and ``[t1]`` (``= -[t2]``) generates
``H^{-2}``.

**The action.**
The structure map is the Berger--Fresse interval-cut action, *defined by
naturality*: for a surjection ``u``, input cochains ``z̲`` and a
nondegenerate ``m``-simplex ``x`` of ``T²``,

    ``⟨θ_u(z̲), x⟩ = ⟨θ_u(x*(z̲)), ι_m⟩``

where ``x*: N*(T²) → N*(Δ^m)`` is restriction along the classifying map
of ``x`` (``m ≤ 2``), and the right-hand side is computed by
:func:`uconf.algebraic.simplicial.surjection_cochain_action`.  This is
exactly the BF action on ``N*(T²)`` because the chain-level action is
natural in the simplicial set.  This defining formula is implemented as
:meth:`SurjectionTorusSimplicialCochainAlgebra._reference_single_action`.

**Closed form.**
Because ``m ≤ 2``, the interval-cut sum collapses to an explicit finite
enumeration, implemented in
:meth:`SurjectionTorusSimplicialCochainAlgebra._single_action`.  Write
``N = n + d`` for the length of ``u ∈ X(n)_d`` and
``m = Σ dim(z_i) - d``.  A *cut* of ``[0..m]`` into ``N`` consecutive
intervals ``I_1, …, I_N`` (overlapping in endpoints) is determined by its
jump intervals:

- ``m = 0``: every ``I_i = [0]``;
- ``m = 1``: ``I_p = [01]`` for one ``p``, vertices elsewhere;
- ``m = 2``: either ``I_p = [01]`` and ``I_q = [12]`` with ``p < q``, or
  ``I_p = [012]`` for one ``p``, vertices elsewhere.

A cut is *admissible* for ``(u, z̲, x)`` if for every value
``k ∈ {1..n}`` the concatenation ``y_k`` of the intervals at the
``u``-preimage of ``k`` is strictly increasing and ``x ∘ y_k`` is the
nondegenerate simplex of ``z_k``, where the restrictions are

- any vertex of any ``x`` is ``v``;
- for the edges: ``x ∘ (0,1) = x``;
- for ``x = t1``: ``(0,1) ↦ a``, ``(1,2) ↦ b``, ``(0,2) ↦ g``,
  ``(0,1,2) ↦ t1``;
- for ``x = t2``: ``(0,1) ↦ b``, ``(1,2) ↦ a``, ``(0,2) ↦ g``,
  ``(0,1,2) ↦ t2``.

Then ``⟨θ_u(z̲), x⟩ = Σ_{admissible cuts} (-1)^{pos + ord + dual}`` with

- ``pos = Σ_i max(I_i)`` over the *inner* intervals ``I_i`` (positions
  ``i`` that are not the last occurrence of their value ``u_i``),
- ``ord = Σ_{i<j, u_i > u_j} ℓ_i ℓ_j`` where ``ℓ_i = dim I_i`` for final
  occurrences and ``dim I_i + 1`` for inner ones,
- ``dual = d·Σ_i dim(z_i) + Σ_{i<j} dim(z_i) dim(z_j)``
  (the cochain duality sign of
  :func:`~uconf.algebraic.simplicial.surjection_cochain_action`).

In particular ``μ_u(…, [v] at slot i, …)`` vanishes unless ``i`` occurs
exactly once in ``u`` and then equals the action of ``u`` with ``i``
deleted (no sign), and inputs ``t1, t2`` only produce the matching
output triangle.

.. note::
   The closed form was verified against the naturality reference on
   *all* of ``X(n)_d`` for ``n ≤ 3``, ``d ≤ 4``, with all ``6^n`` input
   tuples, over ``QQ`` (and sampled over ``GF(2)``, ``GF(3)``), and on
   2500 random samples at ``n = 4``, ``d ≤ 3``.  In that entire range
   each nonzero coefficient arises from a *unique* admissible cut (in
   particular every structure constant is ``0`` or ``±1``).
"""

from __future__ import annotations

import itertools

from sage.all import CombinatorialFreeModule, Family, GradedModulesWithBasis, cached_method

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.simplicial import surjection_cochain_action
from uconf.core.display import latex_linear_combination
from uconf.models.simplicial import SimplicialCochains
from uconf.models.surjection import Surjection

# Nondegenerate simplices of T² = (Δ¹/∂Δ¹)²: label -> (dimension, (j_x, j_y)).
_SIMPLICES: dict[str, tuple[int, tuple[int, int]]] = {
    "v": (0, (0, 0)),
    "a": (1, (1, 0)),
    "b": (1, (0, 1)),
    "g": (1, (1, 1)),
    "t1": (2, (1, 2)),
    "t2": (2, (2, 1)),
}
_DIM = {label: dim for label, (dim, _) in _SIMPLICES.items()}
_LABEL_OF = {(dim, pair): label for label, (dim, pair) in _SIMPLICES.items()}
_GENERATORS = tuple(_SIMPLICES)


def _restrict_jump(j: int, sigma: tuple[int, ...]) -> int:
    """Restrict the ``S¹``-simplex ``j`` along the vertex subset ``sigma``.

    ``sigma`` is a strictly increasing tuple of vertices of ``Δ^m``; the
    result is the encoded ``len(sigma)-1``-simplex of ``S¹`` (``0`` for
    the basepoint, i.e. when the restriction is a constant map).
    """
    if j == 0:
        return 0
    t = next((idx for idx, v in enumerate(sigma) if v >= j), None)
    # t is None: constant 0; t == 0: constant 1.  Both are the basepoint.
    return 0 if t is None or t == 0 else t


def _simplex_restrict(label: str, sigma: tuple[int, ...]) -> str | None:
    """Return the label of ``x∘σ`` for ``x`` nondegenerate, or ``None``.

    ``None`` means the restricted simplex is degenerate (it then pairs to
    zero with any normalized cochain).
    """
    _, (jx, jy) = _SIMPLICES[label]
    k = len(sigma) - 1
    rx, ry = _restrict_jump(jx, sigma), _restrict_jump(jy, sigma)
    if not all((rx == i + 1) or (ry == i + 1) for i in range(k)):
        return None
    return _LABEL_OF[(k, (rx, ry))]


def _pullback_dual(label: str, x_label: str, cochains: SimplicialCochains):
    """The restriction ``x*([label])`` as an element of ``N*(Δ^m)``."""
    m = _DIM[x_label]
    keys = []
    for r in range(m + 1):
        for sigma in itertools.combinations(range(m + 1), r + 1):
            if _simplex_restrict(x_label, sigma) == label:
                keys.append(sigma)
    R = cochains.base_ring()
    return cochains.sum_of_terms((key, R.one()) for key in keys)


def _cut_intervals(N: int, jumps: dict[int, tuple[int, int]]) -> list[tuple[int, int]]:
    """The ``N`` intervals of the cut with the given jump intervals.

    ``jumps`` maps slot indices to ``(lo, hi)`` jump intervals; every
    other slot is the single vertex at the current level.
    """
    out: list[tuple[int, int]] = []
    level = 0
    for i in range(N):
        if i in jumps:
            lo, hi = jumps[i]
            out.append((lo, hi))
            level = hi
        else:
            out.append((level, level))
    return out


def _join_vertices(
    intervals: list[tuple[int, int]], occurrences: list[int]
) -> tuple[int, ...] | None:
    """Concatenate the intervals at ``occurrences``; ``None`` if degenerate."""
    vertices: list[int] = []
    for o in occurrences:
        lo, hi = intervals[o]
        vertices.extend(range(lo, hi + 1))
    if any(x >= y for x, y in zip(vertices, vertices[1:])):
        return None
    return tuple(vertices)


def _bf_sign_exponent(u_key: tuple[int, ...], intervals: list[tuple[int, int]]) -> int:
    """Berger--Fresse sign exponent of one cut (``pos + ord`` terms).

    Matches ``_compute_bf_sign`` of
    :func:`~uconf.algebraic.simplicial.surjection_chain_action`
    specialized to cuts of a standard simplex.
    """
    N = len(u_key)
    final: dict[int, int] = {}
    for i in reversed(range(N)):
        if u_key[i] not in final:
            final[u_key[i]] = i
    pos_exp = 0
    lengths: list[int] = []
    for i in range(N):
        lo, hi = intervals[i]
        dim = hi - lo
        if i == final[u_key[i]]:
            lengths.append(dim)
        else:
            lengths.append(dim + 1)
            pos_exp += hi
    order_exp = sum(
        lengths[i] * lengths[j] for i in range(N) for j in range(i + 1, N) if u_key[i] > u_key[j]
    )
    return pos_exp + order_exp


class TorusSimplicialCochains(CombinatorialFreeModule):
    r"""Normalized cochains of the simplicial torus as a rank-6 module.

    Basis elements ``[v], [a], [b], [g], [t1], [t2]`` in homological
    degrees ``0, -1, -1, -1, -2, -2``; differential as in the module
    docstring.
    """

    def __init__(self, base_ring):
        super().__init__(
            base_ring,
            list(_GENERATORS),
            prefix="N*T²Δ",
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename("N*T²Δ")
        self._generators = {gen: self(gen) for gen in _GENERATORS}
        self.connectivity = -2
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    def _boundary_on_basis(self, label: str):
        """Cochain differential ``(δf)(σ) = -(-1)^{|f|} f(∂σ)``."""
        R = self.base_ring()
        if label in ("a", "b"):
            return self.sum_of_terms([("t1", R.one()), ("t2", R.one())])
        if label == "g":
            return self.sum_of_terms([("t1", -R.one()), ("t2", -R.one())])
        return self.zero()

    def degree_on_basis(self, label: str) -> int:
        """Homological degree of a basis element (``-dimension``)."""
        return -_DIM[label]

    def _weight_on_basis(self, _) -> int:
        return 0

    def generator(self, name: str):
        """Return the generator with the given label."""
        return self._generators[name]

    def basis_iter(self, d: int):
        """Iterate over basis elements of homological degree ``d``."""
        for gen, dim in _DIM.items():
            if dim == -d:
                yield self(gen)

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    def basis_weight_iter(self, d: int, w: int):
        if w == 0:
            yield from self.basis_iter(d)

    @cached_method
    def graded_weighted_basis(self, d: int, w: int):
        return Family(self.basis_weight_iter(d, w))

    def _repr_term(self, label: str) -> str:
        return label

    def _latex_term(self, label: str) -> str:
        latex_map = {
            "v": "[v]",
            "a": "[a]",
            "b": "[b]",
            "g": "[g]",
            "t1": "[t_1]",
            "t2": "[t_2]",
        }
        return latex_map[label]

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


class SurjectionTorusSimplicialCochainAlgebra(OperadAlgebra):
    r"""The Berger--Fresse ``Surjection``-algebra on the simplicial torus.

    The structure map is the interval-cut action computed by naturality —
    see the module docstring.  Being a ``Surjection``-algebra, its
    pullback along table reduction is a ``BarrattEccles``-algebra whose
    action factors through the surjection operad by construction (which
    the rank-4 model of :mod:`uconf.algebraic.torus` does not).
    """

    def __init__(self, base_ring):
        module = TorusSimplicialCochains(base_ring=base_ring)
        super().__init__(
            module=module,
            operad_cls=Surjection,
            structure_map=self._act_impl,
        )

    @cached_method
    def _single_action(
        self, u_key: tuple[int, ...], labels: tuple[str, ...]
    ) -> tuple[tuple[str, object], ...]:
        r"""Action of one surjection basis key on basis labels (closed form).

        Returns the result ``θ_u([z_1], …, [z_n])`` as a tuple of
        ``(output_label, coefficient)`` pairs, via the interval-cut
        enumeration of the module docstring.  Verified against
        :meth:`_reference_single_action` — see the module docstring for
        the verified range.
        """
        n = len(labels)
        N = len(u_key)
        d = N - n
        R = self.module.base_ring()
        # Homologically |θ_u(z̲)| = d + Σ|z_i|, so the output pairs with
        # simplices of dimension m = Σ dim(z_i) - d, and m ∈ {0, 1, 2}.
        dims = [_DIM[lab] for lab in labels]
        m = sum(dims) - d
        if m < 0 or m > 2:
            return ()
        occurrences = {k: [i for i, w in enumerate(u_key) if w == k] for k in range(1, n + 1)}
        dual_exp = d * sum(dims) + sum(dims[i] * dims[j] for i in range(n) for j in range(i + 1, n))
        if m == 0:
            patterns: list[dict[int, tuple[int, int]]] = [{}]
        elif m == 1:
            patterns = [{p: (0, 1)} for p in range(N)]
        else:
            patterns = [{p: (0, 1), q: (1, 2)} for p in range(N) for q in range(p + 1, N)]
            patterns += [{p: (0, 2)} for p in range(N)]

        result = []
        for x_label, dim in _DIM.items():
            if dim != m:
                continue
            total = 0
            for jumps in patterns:
                intervals = _cut_intervals(N, jumps)
                admissible = True
                for k in range(1, n + 1):
                    y = _join_vertices(intervals, occurrences[k])
                    if y is None or _simplex_restrict(x_label, y) != labels[k - 1]:
                        admissible = False
                        break
                if admissible:
                    exponent = _bf_sign_exponent(u_key, intervals) + dual_exp
                    total += -1 if exponent % 2 else 1
            if total:
                result.append((x_label, R(total)))
        return tuple(result)

    @cached_method
    def _reference_single_action(
        self, u_key: tuple[int, ...], labels: tuple[str, ...]
    ) -> tuple[tuple[str, object], ...]:
        r"""First-principles action of one surjection basis key.

        Same output format as :meth:`_single_action`, computed by
        naturality: the coefficient on ``x`` is ``⟨θ_u(x*(z̲)), ι_m⟩``
        evaluated via
        :func:`~uconf.algebraic.simplicial.surjection_cochain_action`
        on ``Δ^m``.  Kept as the canonical reference for tests.
        """
        n = len(labels)
        d = len(u_key) - n
        R = self.module.base_ring()
        m = sum(_DIM[lab] for lab in labels) - d
        if m < 0 or m > 2:
            return ()
        surjection = Surjection(n, R)(u_key)
        result = []
        for x_label, dim in _DIM.items():
            if dim != m:
                continue
            cochains_dm = SimplicialCochains(N=m, base_ring=R)
            pulled = tuple(_pullback_dual(lab, x_label, cochains_dm) for lab in labels)
            if any(not c for c in pulled):
                continue
            value = surjection_cochain_action(surjection, pulled)[tuple(range(m + 1))]
            if value:
                result.append((x_label, value))
        return tuple(result)

    def _act_impl(self, p_element: Surjection.Element, algebra_elements):
        r"""Implement ``θ_u(z_1, …, z_n)`` on the simplicial torus cochains."""
        if p_element.arity() == 0:
            return self.module.zero()

        R = self.module.base_ring()
        arg_terms = [list(self.module(z)) for z in algebra_elements]
        if any(len(terms) == 0 for terms in arg_terms):
            return self.module.zero()

        result: dict[str, object] = {}
        for u_key, u_coeff in p_element:
            for selected in itertools.product(*arg_terms):
                labels = tuple(lab for lab, _ in selected)
                scalar = u_coeff
                for _, coeff in selected:
                    scalar *= coeff
                if scalar == 0:
                    continue
                for out_label, value in self._single_action(u_key, labels):
                    contribution = R(scalar * value)
                    if out_label in result:
                        result[out_label] += contribution
                    else:
                        result[out_label] = contribution

        return self.module.sum_of_terms((key, c) for key, c in result.items() if c != 0)
