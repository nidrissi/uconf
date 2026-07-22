r"""Cobar construction for a C-coalgebra: Ω_α(V).

Given a twisting morphism α: C → P and a C-coalgebra (V, δ), the cobar
construction is the free P-algebra

    Ω_α(V) = (T_P(V),  d_{T_P} + d_α)

where:

- ``T_P(V) = ⊕_{n≥1} P(n) ⊗_{S_n} V^{⊗n}`` is the free P-algebra on V
  (see :mod:`uconf.algebraic.free_algebra`).
- ``d_{T_P} = d_P + d_V`` is the Leibniz differential on the free algebra
  (Koszul sign rule).
- ``d_α`` is the extra twisting differential.

The twisting differential d_α acts on each leaf of a corolla::

    d_α(p ⊗ v_1 ⊗ … ⊗ v_n) = Σ_{i=1}^{n} Σ_{k≥2}
        (-1)^{|p| + |v_1| + … + |v_{i-1}|}
        · (p ∘_i α(c_k)) ⊗ v_1 ⊗ … ⊗ v'_1 ⊗ … ⊗ v'_k ⊗ … ⊗ v_n

where δ_k(v_i) = Σ c_k ⊗ v'_1 ⊗ … ⊗ v'_k is the C-coalgebra coaction and
``p ∘_i α(c_k) ∈ P(n + k - 1)`` is the operadic composition.

Maurer-Cartan equation ∂α + α ⋆ α = 0 ensures d² = 0.

Ω_α(V) is a dg-P-algebra.

Reference: Loday-Vallette "Algebraic Operads", Section 11.4.
"""

from __future__ import annotations

import itertools
from typing import ClassVar


from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.free_algebra import FreeAlgebraModule
from uconf.core.display import latex_linear_combination
from uconf.core.signs import sign_from_exponent
from uconf.core.twisting import TwistingMorphism


class CobarCoalgebraModule(FreeAlgebraModule):
    r"""Underlying dg-module of the cobar construction Ω_α(V).

    Extends :class:`FreeAlgebraModule` with the twisting differential d_α.
    Basis keys are ``(p_key, m_tuple)`` pairs inherited from the free algebra,
    where ``p_key ∈ P(n)_planar`` and ``m_tuple`` has length ``n``.

    The total differential is ``d = d_{T_P} + d_α`` where ``d_{T_P}`` is
    the free-algebra Leibniz differential and ``d_α`` inserts new vertices
    via the coalgebra coaction and twisting morphism.
    """

    name: ClassVar[str] = "Ω_α"

    def __init__(
        self,
        alpha: TwistingMorphism,
        coalgebra: CooperadCoalgebra,
    ):
        self._alpha = alpha
        self._coalgebra = coalgebra
        operad_cls = alpha.operad
        inner_module = coalgebra.module

        super().__init__(
            operad_cls,
            inner_module,
            name=f"Ω_{{{alpha.name}}}({inner_module})",
        )

        # Override the boundary with the twisted differential
        self._d_free = self.module_morphism(
            on_basis=lambda key: FreeAlgebraModule._boundary_on_basis(self, key),
            codomain=self,
        )
        self._d_alpha = self.module_morphism(
            on_basis=self._dalpha_on_basis,
            codomain=self,
        )
        self.boundary = self.module_morphism(
            on_basis=self._twisted_boundary_on_basis,
            codomain=self,
        )

    # ------------------------------------------------------------------
    # Twisted differential
    # ------------------------------------------------------------------

    def _twisted_boundary_on_basis(self, key):
        """Total differential d = d_{T_P} + d_α."""
        return FreeAlgebraModule._boundary_on_basis(self, key) + self._dalpha_on_basis(key)

    def _dalpha_on_basis(self, key):
        r"""Twisting differential d_α.

        For a basis key ``(p_key, (v_1, …, v_n))``:

            d_α(p ⊗ v_1 ⊗ … ⊗ v_n) = Σ_{i=1}^{n} Σ_{k≥2}
                ε_i · (p ∘_i α(c)) ⊗ v_1 ⊗ … ⊗ v'_1 ⊗ … ⊗ v'_k ⊗ … ⊗ v_n

        where:
        - δ_k(v_i) = Σ c ⊗ v'_1 ⊗ … ⊗ v'_k is the C-coalgebra coaction
        - α(c) ∈ P(k) is the twisting morphism applied to c
        - p ∘_i α(c) is the operadic composition at position i
        - ε_i = (-1)^{|p| + |v_1| + … + |v_{i-1}|} is the Koszul sign

        Each coaction term yields a new basis element of arity n + k - 1:
        ``(composed_key, v_1, …, v_{i-1}, v'_1, …, v'_k, v_{i+1}, …, v_n)``.
        """
        p_key, m_tuple = key
        n = len(m_tuple)
        base_ring = self.base_ring()
        P = self._operad_cls
        C = self._alpha.cooperad
        M = self._inner_module

        result = self.zero()

        # Compute cumulative signs once
        p_deg = P(n, base_ring).degree_on_basis(p_key)
        cumulative = p_deg
        leaf_degs = [M.degree_on_basis(mk) for mk in m_tuple]

        for i in range(n):  # 0-indexed leaf position
            sign = sign_from_exponent(cumulative)
            v_key = m_tuple[i]
            v_elem = M(v_key)

            # Try all coaction arities k ≥ 2
            for k in range(2, n + 2):
                coaction = self._coalgebra.coact(v_elem, k)

                for coact_key, coact_coeff in coaction:
                    # coact_key is a tensor product key:
                    # (c_key, v'_1_key, …, v'_k_key) or similar
                    # The exact format depends on the coaction implementation
                    c_key, new_v_keys = self._extract_coaction_components(coact_key, k)
                    if c_key is None:
                        continue

                    # Apply α to get P-element
                    c_comp = C(k, base_ring)
                    c_elem = c_comp(c_key)
                    alpha_c = self._alpha(c_elem)

                    if not alpha_c:
                        continue

                    # Compose p ∘_{i+1} α(c) (1-indexed position)
                    p_comp = P(n, base_ring)
                    p_elem = p_comp(p_key)
                    composed = P.compose(p_elem, i + 1, alpha_c)

                    # Build new m_tuple: replace v_i with v'_1, …, v'_k
                    new_m = m_tuple[:i] + tuple(new_v_keys) + m_tuple[i + 1 :]

                    # Add to result via _normalized_corolla_sum for planarization
                    result += sign * coact_coeff * self._normalized_corolla_sum(composed, new_m)

            cumulative += leaf_degs[i]

        return result

    def _extract_coaction_components(self, coact_key, k):
        """Extract cooperad key and leaf keys from a coaction tensor key.

        The coaction returns elements in ``C(k) ⊗ V^{⊗k}``, whose basis keys
        are tensor-product tuples.  This method extracts the cooperad key and
        the k leaf keys from such a tuple.

        Args:
            coact_key: A basis key from the coaction output.
            k: Expected arity (number of V-factors).

        Returns:
            ``(c_key, (v'_1, …, v'_k))`` or ``(None, None)`` if the format
            is not recognised.
        """
        if isinstance(coact_key, (tuple, list)):
            if len(coact_key) == k + 1:
                # Format: (c_key, v'_1, …, v'_k)
                return coact_key[0], coact_key[1:]
            elif len(coact_key) == 2:
                # Format: (c_key, (v'_1, …, v'_k)) — from cofree coalgebra
                c_key, v_part = coact_key
                if isinstance(v_part, (tuple, list)) and len(v_part) == k:
                    return c_key, tuple(v_part)
                # Format: c_key ⊗ (single tensor of V^⊗k)
                # Need to further decompose...
                return c_key, (v_part,) if k == 1 else (None, None)
        return None, None

    # ------------------------------------------------------------------
    # Expose component differentials for debugging
    # ------------------------------------------------------------------

    def d_free(self, elem):
        """Apply only the free algebra differential d_{T_P}."""
        return self._d_free(elem)

    def d_alpha(self, elem):
        """Apply only the twisting differential d_α."""
        return self._d_alpha(elem)

    # Representation
    def _repr_term(self, basis_element) -> str:
        return super()._repr_term(basis_element) + "_Ω"

    def _latex_term(self, basis_element) -> str:
        return super()._latex_term(basis_element) + "_Ω"

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(FreeAlgebraModule.Element):
        """An element of the cobar construction Ω_α(V)."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def boundary(self) -> "CobarCoalgebraModule.Element":
            """Apply the full twisted cobar differential d = d_{T_P} + d_α."""
            return self.parent().boundary(self)

        def d_free(self) -> "CobarCoalgebraModule.Element":
            """Apply only the free algebra differential."""
            return self.parent().d_free(self)

        def d_alpha(self) -> "CobarCoalgebraModule.Element":
            """Apply only the twisting differential."""
            return self.parent().d_alpha(self)

        # Backward-compatible aliases
        dalpha = d_alpha
        dcoact = d_alpha


class CobarCoalgebra(OperadAlgebra):
    """Cobar construction Ω_α(V) as a P-algebra.

    Wraps :class:`CobarCoalgebraModule` with the canonical P-algebra
    structure inherited from the free algebra.

    Args:
        alpha: A twisting morphism α: C → P.
        coalgebra: A C-coalgebra (V, δ).
    """

    def __init__(self, alpha: TwistingMorphism, coalgebra: CooperadCoalgebra):
        cobar_module = CobarCoalgebraModule(alpha, coalgebra)
        self._alpha = alpha
        self._coalgebra = coalgebra

        def _act_impl(p_element, algebra_elements):
            """P-algebra action via full operad substitution (same as FreeOperadAlgebra)."""
            k = p_element.arity()
            inputs = list(algebra_elements)
            if len(inputs) != k:
                raise ValueError(f"Expected {k} inputs for P({k}) action, got {len(inputs)}.")

            P = alpha.operad
            R = cobar_module.base_ring()
            result = cobar_module.zero()
            input_term_lists = [list(x) for x in inputs]

            for q_key, q_coeff in p_element:
                for term_combo in itertools.product(*input_term_lists):
                    input_keys = [bk for (bk, _) in term_combo]
                    coeff = q_coeff
                    for _, c in term_combo:
                        coeff = coeff * c

                    n_list = [len(ik[1]) for ik in input_keys]
                    composed_elem = P(k, R)(q_key)

                    for j in range(k - 1, -1, -1):
                        p_j_key, m_j_tuple = input_keys[j]
                        n_j = n_list[j]
                        p_j_elem = P(n_j, R)(p_j_key)
                        pos = j + 1
                        composed_elem = P.compose(composed_elem, pos, p_j_elem)

                    m_concat = tuple(mk for ik in input_keys for mk in ik[1])
                    result += coeff * cobar_module._normalized_corolla_sum(composed_elem, m_concat)

            return result

        super().__init__(cobar_module, alpha.operad, _act_impl)

    def include(self, v_key):
        """Return the image of v_key under the inclusion η: V → Ω_α(V).

        Args:
            v_key: A basis key of the coalgebra module V.

        Returns:
            The element (id_P, (v_key,)) in Ω_α(V).
        """
        id_key = self._alpha.operad.unit_key()
        return self.module((id_key, (v_key,)))
