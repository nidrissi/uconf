r"""Torus cochain model and explicit Barratt--Eccles action.

This module implements the reduced cochain complex of the torus
``T² = S¹ × S¹`` as a rank-4 module and equips it with the explicit
``BarrattEccles``-algebra structure.

**Basis.**
Writing ``[*]`` for the basepoint class and ``[a_1]`` for the fundamental
class of ``C*(S¹)``, the four basis elements are

- ``[0] = [*] ⊗ [*]``    (degree 0),
- ``[α] = [a_1] ⊗ [*]``  (degree 1),
- ``[β] = [*] ⊗ [a_1]``  (degree 1),
- ``[γ] = [a_1] ⊗ [a_1]``(degree 2).

(Internally we follow the homological/negative grading convention of
:mod:`uconf.algebraic.spherical`, so the degrees are ``0, -1, -1, -2``.)

**Algebra structure.**
For ``σ̲ = (σ_0, …, σ_{s-1}) ∈ E(n)`` (an ``s``-tuple of permutations of
``{1,…,n}``, of degree ``s-1``) and the combinatorial function
``ψ_{I_S}`` defined in :func:`_psi`, the action is determined by the
following operations (``article.tex`` Proposition, §"E_∞-algebra of the
torus"):

1. ``μ_σ̲([0]^{⊗n}) = [0]`` when ``σ̲`` has a single permutation;
2. ``μ_σ̲([α]^{⊗a}, [0]^{⊗d}) = (-1)^{a(a-1)/2} ψ_{I_A}(σ̲) [α]``
   when ``s = a``;
3. ``μ_σ̲([β]^{⊗b}, [0]^{⊗d}) = (-1)^{b(b-1)/2} ψ_{I_B}(σ̲) [β]``
   when ``s = b``;
4. for the mixed case producing ``[γ]`` (``c`` copies of ``[γ]``, ``a`` of
   ``[α]``, ``b`` of ``[β]``, ``d`` of ``[0]``), when ``s = 2c+a+b-1``,
   ``μ_σ̲(…) = (-1)^{ε} ψ_{I_{C⊔A}}(σ̲_1) ψ_{I_{C⊔B}}(σ̲_2) [γ]`` with
   ``ε = c(c-1)/2 + a(a-1)/2 + b(b-1)/2 + ab + c + b - 1``;
5. every other product is zero.

When the inputs are not in the article's normal order ``[γ]…[α]…[β]…[0]``
the result is multiplied by the Koszul reordering sign (see
:func:`_koszul_reordering_sign`).

.. warning::
   **Status.** The resulting chain complex satisfies ``d² = 0`` over
   ``GF(2)`` (verified on thousands of basis elements), which is the
   characteristic relevant to this project.  Over ``GF(3)``/``QQ`` the
   Case-4 (``[γ]``-involving) sign is **not yet correct** — see the halt
   note in :meth:`BarrattEcclesTorusCochainAlgebra._single_action`.  The
   open conventions are flagged as OQ1/OQ2/OQ3 and must be resolved with
   the author before relying on this model over ``ℚ``.

Open questions:

- OQ1 — meaning of k per case. The Proposition writes σ̲ = (σ_0,…,σ_{k-1}) (so
#perms = k, degree k-1), yet Case 1 says "if k = 0" (which would be degree −1,
impossible) while Cases 2–4 use k as the number of permutations (e.g. Case 2 "if
k = a" needs #perms = a to match ψ on a size-a subset). My reading: Case 1 fires
when σ̲ has a single permutation (degree 0); Cases 2/3 need #perms = a / = b; Case
4 needs #perms = 2c+a+b-1. Please confirm.
- OQ2 — length mismatch in Case 4. σ̲_2 = (σ_{c+a},…,σ_{2c+a+b-2}) has length
c+b-1, but ψ_{I_{C⊔B}} requires an argument of length |I_{C⊔B}| = c+b. This is an
off-by-one (possibly an intended overlap, i.e. σ̲_2 should start at σ_{c+a-1}, or a
typo). It must be resolved before the Case-4 sign/ψ is correct.
- OQ3 — input ordering. The article states each case for inputs in the normal order
[γ]…[α]…[β]…[0]. In the operad action inputs arrive at arbitrary positions; the ψ
factors already encode positions via the index-subsets, but please confirm no extra
Koszul reordering sign on the (graded) inputs is required beyond what ψ + the
count-based prefactor capture.
"""

from __future__ import annotations

import itertools

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, cached_method, Family

from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.display import latex_linear_combination
from uconf.models.barratt_eccles import BarrattEccles

# Basis labels and their homological (negative) degrees, matching the
# convention of :mod:`uconf.algebraic.spherical`.
_DEGREES = {"0": 0, "α": -1, "β": -1, "γ": -2}


def _permutation_sign(perm: tuple[int, ...]) -> int:
    """Return the sign of a permutation given in one-line notation."""
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _psi(perms: tuple[tuple[int, ...], ...], index_subset: tuple[int, ...]) -> int:
    r"""Compute the combinatorial sign function ``ψ_{I_S}(σ̲)``.

    Given an ``s``-tuple of permutations ``σ̲ = (σ_0, …, σ_{s-1})`` in
    one-line notation and a subset ``I_S ⊆ {1,…,n}`` of size ``s``, let
    ``σ_j(I_S)`` be the first entry of ``σ_j`` lying in ``I_S``.  If the
    resulting tuple ``(σ_0(I_S), …, σ_{s-1}(I_S))`` consists of distinct
    elements of ``I_S`` — i.e. it is a permutation of ``I_S`` under the
    order-preserving identification of ``I_S`` with ``{1,…,s}`` — then
    ``ψ_{I_S}(σ̲)`` is the sign of that permutation; otherwise it is ``0``.

    See ``article.tex`` for the definition.
    """
    if not index_subset:
        return 0
    # ψ is only defined when there are exactly |I_S| permutations.
    if len(perms) != len(index_subset):
        return 0

    subset = set(index_subset)
    first_occurrences: list[int] = []
    for perm in perms:
        found = None
        for val in perm:
            if val in subset:
                found = val
                break
        if found is None:
            return 0
        first_occurrences.append(found)

    if len(set(first_occurrences)) != len(first_occurrences):
        return 0

    mapping = {val: i + 1 for i, val in enumerate(sorted(index_subset))}
    image = tuple(mapping[val] for val in first_occurrences)
    return _permutation_sign(image)


def _sign(exponent: int) -> int:
    """Return ``(-1)**exponent`` as ``±1``."""
    return -1 if exponent % 2 else 1


# Parity (degree mod 2) and normal-order key of each basis label. The article
# states each product for inputs in the normal order ``[γ] … [α] … [β] … [0]``;
# only ``[α]`` and ``[β]`` are odd, so the only Koszul sign incurred when the
# actual inputs are in a different order comes from ``[β]``-before-``[α]`` pairs.
_PARITY = {"0": 0, "α": 1, "β": 1, "γ": 0}
_NORMAL_ORDER_KEY = {"γ": 0, "α": 1, "β": 2, "0": 3}


def _koszul_reordering_sign(labels: tuple[str, ...]) -> int:
    r"""Koszul sign of stably sorting ``labels`` into the normal order.

    The normal order is ``γ, α, β, 0`` (see :data:`_NORMAL_ORDER_KEY`). The
    sign is ``(-1)^{Σ}`` over pairs ``i < j`` that are out of normal order,
    weighted by the product of their parities. Since only ``α``/``β`` are odd
    and ``key(α) < key(β)``, this equals ``(-1)`` to the number of pairs
    ``i < j`` with ``label_i = β`` and ``label_j = α``; it is ``+1`` whenever
    at most one of ``α``/``β`` is present.
    """
    exponent = 0
    for i in range(len(labels)):
        ki = _NORMAL_ORDER_KEY[labels[i]]
        pi = _PARITY[labels[i]]
        if pi == 0:
            continue
        for j in range(i + 1, len(labels)):
            if ki > _NORMAL_ORDER_KEY[labels[j]]:
                exponent += pi * _PARITY[labels[j]]
    return _sign(exponent)


class ReducedTorusCochains(CombinatorialFreeModule):
    r"""Reduced cochains of ``T² = S¹ × S¹`` as a rank-4 graded module.

    Basis elements ``[0], [α], [β], [γ]`` in (homological) degrees
    ``0, -1, -1, -2`` respectively.  The boundary is zero.
    """

    def __init__(self, base_ring):
        generators = ["0", "α", "β", "γ"]
        super().__init__(
            base_ring,
            generators,
            prefix="N*T²",
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename("N*T²")
        self._generators = {gen: self(gen) for gen in generators}
        self.connectivity = -2
        self.boundary = self.module_morphism(on_basis=lambda _: self.zero(), codomain=self)

    def degree_on_basis(self, element: str) -> int:
        """Return the homological degree of a basis element."""
        return _DEGREES[element]

    def _weight_on_basis(self, _) -> int:
        return 0

    def generator(self, name: str):
        """Return the generator with the given label (``"0"/"α"/"β"/"γ"``)."""
        return self._generators[name]

    def basis_iter(self, d: int):
        """Iterate over basis elements of homological degree ``d``."""
        for gen, deg in _DEGREES.items():
            if deg == d:
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

    def _repr_term(self, element: str) -> str:
        return element

    def _latex_term(self, element: str) -> str:
        latex_map = {"0": "[0]", "α": "[\\alpha]", "β": "[\\beta]", "γ": "[\\gamma]"}
        return latex_map[element]

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


class BarrattEcclesTorusCochainAlgebra(OperadAlgebra):
    r"""Explicit ``BarrattEccles``-algebra structure on the torus cochains.

    Implements the operations of the Proposition in ``article.tex``,
    §"The E_∞-algebra structure of C*(S¹) ⊗ C*(S¹)", on
    :class:`ReducedTorusCochains`.
    """

    def __init__(self, base_ring):
        module = ReducedTorusCochains(base_ring=base_ring)
        super().__init__(
            module=module,
            operad_cls=BarrattEccles,
            structure_map=self._act_impl,
        )

    def _single_action(
        self, perms: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
    ) -> tuple[str | None, int]:
        r"""Return ``(output_label, ε)`` for one Barratt--Eccles basis key.

        ``perms`` is the basis key ``σ̲`` as a tuple of one-line
        permutations; ``labels`` are the ``n`` input basis labels in order.
        ``ε`` is the resulting scalar (``0`` when the product vanishes).
        """
        s = len(perms)
        I_A = tuple(i + 1 for i, lab in enumerate(labels) if lab == "α")
        I_B = tuple(i + 1 for i, lab in enumerate(labels) if lab == "β")
        I_C = tuple(i + 1 for i, lab in enumerate(labels) if lab == "γ")
        a, b, c = len(I_A), len(I_B), len(I_C)

        # The article states each product for inputs in the normal order
        # [γ]…[α]…[β]…[0]; recover arbitrary orderings via the Koszul sign of
        # sorting the (graded) inputs into that order. This is +1 for Cases
        # 1–3 and only bites in Case 4 (interleaved [α]/[β]).
        koszul = _koszul_reordering_sign(labels)

        # Case 1: all inputs are [0].  The basepoint class is the algebra
        # unit, so a single (degree-0) permutation acts as the augmentation
        # and higher-degree elements act as zero.
        if a == 0 and b == 0 and c == 0:
            return ("0", 1 if s == 1 else 0)

        # Case 2: copies of [α] and [0] only.
        if c == 0 and b == 0:
            if s != a:
                return ("α", 0)
            return ("α", koszul * _sign(a * (a - 1) // 2) * _psi(perms, I_A))

        # Case 3: copies of [β] and [0] only.
        if c == 0 and a == 0:
            if s != b:
                return ("β", 0)
            return ("β", koszul * _sign(b * (b - 1) // 2) * _psi(perms, I_B))

        # Case 4: mixed case producing [γ] (γ present, or both α and β).
        #
        # HALT / TODO (OQ2 + Case-4 sign — needs author confirmation):
        # The σ̲-split below is the well-typed "overlap" reading of the
        # article: σ̲_1 (length c+a) pairs with I_{C⊔A} and σ̲_2 (length c+b)
        # with I_{C⊔B}, the two sharing σ_{c+a-1} so the total length is
        # 2c+a+b-1 = s.  (article.tex literally writes σ̲_2 = (σ_{c+a}, …,
        # σ_{2c+a+b-2}) of length c+b-1, which is one short of |I_{C⊔B}| = c+b
        # and makes ψ ill-typed — see OQ2.)  This *structure* is correct, but
        # the *sign* below is NOT yet right: with it, d² = 0 holds over GF(2)
        # but FAILS over GF(3)/QQ on γ-involving products (residual ±2[γ]).
        # The discrepancy is an ordering/σ-dependent sign that no count-only
        # correction f(a,b,c) can fix (verified by search); swapping the
        # ψ-pairing *does* give d²=0 but only by wrongly zeroing μ([γ],[α]),
        # so it is rejected.  Per the project halt protocol, the Case-4 sign
        # convention (OQ1/OQ2/OQ3) must be resolved with the author against
        # Berger–Fresse / Roca i Lucio before this is trusted over ℚ.
        if s != 2 * c + a + b - 1:
            return ("γ", 0)
        I_CA = tuple(sorted(I_C + I_A))
        I_CB = tuple(sorted(I_C + I_B))
        sigma_1 = perms[: c + a]
        sigma_2 = perms[c + a - 1 :]
        exponent = c * (c - 1) // 2 + a * (a - 1) // 2 + b * (b - 1) // 2 + a * b + c + b - 1
        return ("γ", koszul * _sign(exponent) * _psi(sigma_1, I_CA) * _psi(sigma_2, I_CB))

    def _act_impl(self, p_element: BarrattEccles.Element, algebra_elements):
        r"""Implement ``μ_σ̲(x_1, …, x_n)`` on the torus cochains.

        Inputs are identified by position; the article's normal-order
        presentation is recovered through the index subsets ``I_S`` passed
        to ``ψ`` together with the Koszul reordering sign of
        :func:`_koszul_reordering_sign` (OQ3: an extra Koszul sign on the
        graded inputs *is* required — without it ``[α]·[β]`` and ``[β]·[α]``
        would coincide instead of being negatives, breaking ``d² = 0``).
        See the Case-4 halt note in :meth:`_single_action` for the remaining
        γ-involving sign that is not yet settled.
        """
        if p_element.arity() == 0:
            return self.module.zero()

        R = self.module.base_ring()

        # Pre-expand each input into its (basis_label, coeff) terms.
        arg_terms = [list(self.module(a)) for a in algebra_elements]
        if any(len(terms) == 0 for terms in arg_terms):
            return self.module.zero()

        result: dict[str, object] = {}
        for perms_key, p_coeff in p_element:
            perms = tuple(tuple(sigma.tuple()) for sigma in perms_key)
            for selected in itertools.product(*arg_terms):
                labels = tuple(lab for lab, _ in selected)
                scalar = p_coeff
                for _, coeff in selected:
                    scalar *= coeff
                if scalar == 0:
                    continue

                out_label, epsilon = self._single_action(perms, labels)
                if epsilon == 0 or out_label is None:
                    continue

                contribution = R(scalar * epsilon)
                if out_label in result:
                    result[out_label] += contribution
                else:
                    result[out_label] = contribution

        return self.module.sum_of_terms((key, c) for key, c in result.items() if c != 0)
