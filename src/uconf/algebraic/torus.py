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
The structure is *defined* as the tensor square of the circle algebra: the
Barratt--Eccles operad ``E`` is a Hopf operad via the levelwise
Alexander--Whitney diagonal ``Δ_E(σ_0,…,σ_k) = Σ_i (σ_0,…,σ_i) ⊗
(σ_i,…,σ_k)``, so ``C*(S¹) ⊗ C*(S¹)`` is an ``E``-algebra by

    ``μ_σ̲ = (μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E(σ̲)``

with the Hadamard Koszul convention of
:mod:`uconf.algebraic.hadamard_algebra`.  Here ``μ^{S¹}`` is the circle
action ``μ_σ̲([*],…,[a_1],…) = (-1)^{s(s-1)/2} ψ_{I_S}(σ̲) [a_1]`` (with
``s`` permutations and ``s`` inputs equal to ``[a_1]`` at positions
``I_S``), i.e. the table-reduction pullback of the Berger--Fresse
interval-cut action on ``N*(Δ¹)`` restricted along ``N*(S¹) ↪ N*(Δ¹)``.

Evaluating the composite in closed form: for
``σ̲ = (σ_0, …, σ_{s-1}) ∈ E(n)`` of degree ``s-1`` and inputs with ``c``
copies of ``[γ]``, ``a`` of ``[α]``, ``b`` of ``[β]`` (rest ``[0]``) at
positions ``I_C, I_A, I_B``, writing ``u = c+a`` and ``v = c+b``:

1. ``μ_σ̲([0],…,[0]) = [0]`` when ``s = 1``;
2. ``μ_σ̲(…) = (-1)^{a(a-1)/2} ψ_{I_A}(σ̲) [α]`` when ``b = c = 0`` and
   ``s = a``;
3. ``μ_σ̲(…) = (-1)^{b(b-1)/2} ψ_{I_B}(σ̲) [β]`` when ``a = c = 0`` and
   ``s = b``;
4. ``μ_σ̲(…) = (-1)^{ε} ψ_{I_{C⊔A}}(σ̲_1) ψ_{I_{C⊔B}}(σ̲_2) [γ]`` when
   ``u ≥ 1``, ``v ≥ 1`` and ``s = u + v - 1``, where

   - ``σ̲_1 = (σ_0,…,σ_{u-1})`` and ``σ̲_2 = (σ_{u-1},…,σ_{s-1})``
     (the two halves of the AW diagonal *overlap* in ``σ_{u-1}``), and
   - ``ε = u(u-1)/2 + v(v-1)/2 + u(v+1) + T`` with ``T`` the bi-graded
     interleaving exponent of :func:`_interleaving_sign_exponent`,
     ``T = #{p < q : z_p ∈ {γ,β}, z_q ∈ {γ,α}}``;

5. every other product is zero.

For inputs in the normal order ``[γ]…[α]…[β]…[0]`` one has
``T = c(c-1)/2 + ca`` and the Case-4 exponent reduces to
``ε ≡ c(c-1)/2 + a(a-1)/2 + b(b-1)/2 + ab + ca + a (mod 2)``.

.. note::
   This corrects the Proposition in ``article.tex`` (as of 2026-06), whose
   Case-4 exponent ``… + ab + c + b - 1`` and σ̲_2 starting at ``σ_{c+a}``
   do not define a chain map: the stated sign differs from the one above
   by ``(-1)^{ca + a + c + b + 1}`` in normal order (a discrepancy
   containing the cross-term ``ca``, hence not removable by rescaling
   basis elements), and the position dependence is *not* the naive Koszul
   sign on total degrees — ``[γ]`` is even but both of its tensor factors
   are odd, so transposing ``[γ]`` past ``[α]`` or ``[β]`` flips the sign.
   The formula implemented here was verified over ``QQ`` against the
   first-principles composite ``(μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E`` (with
   ``μ^{S¹}`` computed by table reduction + the interval-cut action of
   :mod:`uconf.algebraic.simplicial`) on all basis elements of ``E(n)_d``
   for ``n ≤ 3``, ``d ≤ 3`` (``d ≤ 4`` for ``n = 2``) and all inputs, and
   satisfies the chain-map identity ``μ_{∂σ̲} = 0`` for ``n ≤ 3``,
   ``d ≤ 5``.
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


# Tensor bi-grading of each basis label: writing ``z = x ⊗ y`` with
# ``x, y ∈ {[*], [a_1]}``, ``_X_PARITY`` (resp. ``_Y_PARITY``) is the degree
# parity of the first (resp. second) circle factor.
_X_PARITY = {"0": 0, "α": 1, "β": 0, "γ": 1}
_Y_PARITY = {"0": 0, "α": 0, "β": 1, "γ": 1}


def _interleaving_sign_exponent(labels: tuple[str, ...]) -> int:
    r"""Bi-graded Koszul exponent ``T = Σ_{p<q} υ(z_p)·ξ(z_q)``.

    Writing each input as a tensor ``z_p = x_p ⊗ y_p`` of circle factors,
    this is the Koszul exponent of the unshuffle
    ``(x_1, y_1, …, x_n, y_n) ↦ (x_1, …, x_n, y_1, …, y_n)``, i.e. the
    number of pairs ``p < q`` with ``z_p ∈ {γ, β}`` and ``z_q ∈ {γ, α}``.

    This is *not* a function of the total degrees: ``[γ]`` is even, but
    both of its tensor factors are odd, so transposing ``[γ]`` past
    ``[α]`` or ``[β]`` flips the sign while transposing it past ``[0]``
    or another ``[γ]`` does not.
    """
    exponent = 0
    odd_y_seen = 0
    for lab in labels:
        if _X_PARITY[lab]:
            exponent += odd_y_seen
        odd_y_seen += _Y_PARITY[lab]
    return exponent


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

    Implements the closed form of ``(μ^{S¹} ⊗ μ^{S¹}) ∘ Δ_E`` on
    :class:`ReducedTorusCochains` — see the module docstring.  This is the
    Proposition in ``article.tex``, §"The E_∞-algebra structure of
    C*(S¹) ⊗ C*(S¹)", with the Case-4 sign corrected.
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

        # Case 1: all inputs are [0].  The basepoint class is the algebra
        # unit, so a single (degree-0) permutation acts as the augmentation
        # and higher-degree elements act as zero.
        if a == 0 and b == 0 and c == 0:
            return ("0", 1 if s == 1 else 0)

        # Case 2: copies of [α] and [0] only — the circle action on the
        # first tensor factor (the second factor contributes the unit).
        if c == 0 and b == 0:
            if s != a:
                return ("α", 0)
            return ("α", _sign(a * (a - 1) // 2) * _psi(perms, I_A))

        # Case 3: copies of [β] and [0] only.
        if c == 0 and a == 0:
            if s != b:
                return ("β", 0)
            return ("β", _sign(b * (b - 1) // 2) * _psi(perms, I_B))

        # Case 4: γ-output ([γ] present, or both [α] and [β]).  Exactly one
        # term of the AW diagonal Δ_E(σ̲) = Σ_i (σ_0,…,σ_i) ⊗ (σ_i,…,σ_{s-1})
        # survives, the one whose front half has u = c+a permutations (the
        # number of odd inputs seen by the first circle factor); the two
        # halves overlap in σ_{u-1}.  The exponent collects the two circle
        # signs C(u,2) and C(v,2), the Hadamard interchange sign
        # |σ̲_2|·u = (v-1)·u (folded with C(u,2) into u(v+1) mod 2), and the
        # bi-graded input-unshuffle sign T.  See the module docstring.
        u, v = c + a, c + b
        if s != u + v - 1:
            return ("γ", 0)
        I_CA = tuple(sorted(I_C + I_A))
        I_CB = tuple(sorted(I_C + I_B))
        sigma_1 = perms[:u]
        sigma_2 = perms[u - 1 :]
        exponent = u * (u - 1) // 2 + v * (v - 1) // 2 + u * (v + 1)
        exponent += _interleaving_sign_exponent(labels)
        return ("γ", _sign(exponent) * _psi(sigma_1, I_CA) * _psi(sigma_2, I_CB))

    def _act_impl(self, p_element: BarrattEccles.Element, algebra_elements):
        r"""Implement ``μ_σ̲(x_1, …, x_n)`` on the torus cochains.

        Inputs are identified by position; positions enter through the
        index subsets ``I_S`` passed to ``ψ`` and through the bi-graded
        interleaving sign of :func:`_interleaving_sign_exponent`.
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
