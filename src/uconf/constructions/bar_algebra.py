r"""Bar construction for a P-algebra: B_α(A).

Given a twisting morphism α: C → P and a P-algebra (A, γ), the bar
construction is the cofree conilpotent C-coalgebra

    B_α(A) = (T^c_C(A),  d_{T^c} + d_α)

where:

- ``T^c_C(A) = ⊕_{n≥1} C(n) ⊗_{S_n} A^{⊗n}`` is the cofree conilpotent
  C-coalgebra on A (see :mod:`uconf.algebraic.cofree_coalgebra`).
- ``d_{T^c} = d_C + d_A`` is the Leibniz differential on the cofree
  coalgebra (Koszul sign rule).
- ``d_α`` is the extra twisting differential.

The twisting differential d_α acts only on **corollas** (basis keys
``(c, (a_1, …, a_n))`` where n ≥ 2):

    d_α(c ⊗ a_1 ⊗ … ⊗ a_n) = γ(α(c); a_1, …, a_n)

where γ is the P-algebra structure map and α(c) ∈ P(n) is the image of
c under the twisting morphism.  The result is a single-leaf element
``(id_C, (γ(α(c); a_1,…,a_n),))``.

The sign convention is: d_α carries a Koszul sign (-1)^{|c|} since α has
degree -1 and passes past c of degree |c| in the bar ordering.

Maurer-Cartan equation ∂α + α ⋆ α = 0 ensures d² = 0.

B_α(A) is a dg-C-coalgebra.

Reference: Loday-Vallette "Algebraic Operads", Section 11.2.
"""

from __future__ import annotations

from typing import ClassVar


from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.core.display import latex_linear_combination
from uconf.core.signs import sign_from_exponent
from uconf.core.twisting import TwistingMorphism


class BarAlgebraModule(CofreeCoalgebraModule):
    r"""Underlying dg-module of the bar construction B_α(A).

    Extends :class:`CofreeCoalgebraModule` with the twisting differential
    d_α.  Basis keys are ``(c_key, m_tuple)`` pairs inherited from the cofree
    coalgebra, where ``c_key ∈ C(n)_planar`` and ``m_tuple`` has length ``n``.

    The total differential is ``d = d_{T^c} + d_α`` where ``d_{T^c}`` is
    the cofree-coalgebra Leibniz differential and ``d_α`` contracts corollas
    via the algebra action.
    """

    name: ClassVar[str] = "B_α"

    def __init__(
        self,
        alpha: TwistingMorphism,
        algebra: OperadAlgebra,
    ):
        self._alpha = alpha
        self._algebra = algebra
        cooperad_cls = alpha.cooperad
        inner_module = algebra.module

        super().__init__(
            cooperad_cls,
            inner_module,
            name=f"B_{{{alpha.name}}}({inner_module})",
        )

        # Override the boundary with the twisted differential
        self._d_cofree = self.module_morphism(
            on_basis=lambda key: CofreeCoalgebraModule._boundary_on_basis(self, key),
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
        """Total differential d = d_{T^c} + d_α."""
        return CofreeCoalgebraModule._boundary_on_basis(self, key) + self._dalpha_on_basis(key)

    def _dalpha_on_basis(self, key):
        r"""Twisting differential d_α (coderivation).

        For a basis key ``(c_key, (a_1, …, a_n))`` with n = m + n_r - 1:

            d_α(c ⊗ a_1 ⊗ … ⊗ a_n) = Σ_{i,m,n_r} (-1)^{|c_L|}
                (c_L ⊗ a_1 ⊗ … ⊗ γ(α(c_R); a_i,…,a_{i+n_r-1}) ⊗ … ⊗ a_n)

        where Δ^{i;m,n_r}(c) = Σ c_L ⊗ c_R is the cooperad's infinitesimal
        cocomposition, α(c_R) ∈ P(n_r) is the twisting morphism, and
        γ is the algebra action.

        The Koszul sign ``(-1)^{|c_L|}`` comes from commuting d_α (which has
        degree -1) past the cooperad element c_L in the cofree coalgebra
        decomposition.  The algebra elements a_i do *not* contribute to the
        sign because the cofree coalgebra decomposition Δ_{(1)} preserves the
        left-to-right ordering of the A-factors.

        Reference: Loday-Vallette, Section 6.3 and 11.2.
        """
        c_key, m_tuple = key
        n = len(m_tuple)
        if n <= 1:
            return self.zero()

        base_ring = self.base_ring()
        C = self._cooperad_cls
        M = self._inner_module
        result = self.zero()

        c_comp = C(n, base_ring)
        c_elem = c_comp(c_key)

        for n_r in range(2, n + 1):
            m = n - n_r + 1
            if m < 1:
                continue
            for i in range(1, m + 1):
                # Infinitesimal cocomposition Δ^{i;m,n_r}
                cocomp = C.infinitesimal_cocompose(c_elem, i, m, n_r)

                for (c_L_key, c_R_key), coeff in cocomp:
                    # Apply α to c_R
                    c_R_comp = C(n_r, base_ring)
                    c_R_elem = c_R_comp(c_R_key)
                    alpha_c_R = self._alpha(c_R_elem)

                    if not alpha_c_R:
                        continue

                    # Apply the algebra action γ(α(c_R); a_i, …, a_{i+n_r-1})
                    a_slice = [M(m_tuple[j]) for j in range(i - 1, i + n_r - 1)]
                    action_result = self._algebra.act(alpha_c_R, a_slice)

                    if not action_result:
                        continue

                    # Koszul sign: (-1)^{|c_L|}
                    c_L_comp = C(m, base_ring)
                    c_L_deg = c_L_comp.degree_on_basis(c_L_key)
                    sign = sign_from_exponent(c_L_deg)

                    # Build the new m_tuple: replace a_i,...,a_{i+n_r-1}
                    # with the single action result.
                    # Keep raw cooperad keys — do NOT planarize here.
                    # Planarization inside the boundary breaks d²=0.
                    for a_new_key, a_coeff in action_result:
                        new_m = m_tuple[: i - 1] + (a_new_key,) + m_tuple[i + n_r - 1 :]
                        result += sign * coeff * a_coeff * self((c_L_key, new_m))

        return result

    # ------------------------------------------------------------------
    # Expose component differentials for debugging
    # ------------------------------------------------------------------

    def d_cofree(self, elem):
        """Apply only the cofree coalgebra differential d_{T^c}."""
        return self._d_cofree(elem)

    def d_alpha(self, elem):
        """Apply only the twisting differential d_α."""
        return self._d_alpha(elem)

    # Representation
    def _repr_term(self, basis_element) -> str:
        return super()._repr_term(basis_element) + "_B"

    def _latex_term(self, basis_element) -> str:
        return super()._latex_term(basis_element) + "_B"

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(CofreeCoalgebraModule.Element):
        """An element of the bar construction B_α(A)."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def boundary(self) -> "BarAlgebraModule.Element":
            """Apply the full twisted bar differential d = d_{T^c} + d_α."""
            return self.parent().boundary(self)

        def d_cofree(self) -> "BarAlgebraModule.Element":
            """Apply only the cofree coalgebra differential."""
            return self.parent().d_cofree(self)

        def d_alpha(self) -> "BarAlgebraModule.Element":
            """Apply only the twisting differential."""
            return self.parent().d_alpha(self)

        # Backward-compatible aliases
        dalpha = d_alpha
        dact = d_alpha
        dtwist = d_alpha


class BarAlgebra(CooperadCoalgebra):
    """Bar construction B_α(A) as a C-coalgebra.

    Wraps :class:`BarAlgebraModule` with the canonical C-coalgebra structure
    inherited from the cofree coalgebra.

    Args:
        alpha: A twisting morphism α: C → P.
        algebra: A P-algebra (A, γ).
    """

    def __init__(self, alpha: TwistingMorphism, algebra: OperadAlgebra):
        bar_module = BarAlgebraModule(alpha, algebra)
        self._alpha = alpha
        self._algebra = algebra
        super().__init__(bar_module, alpha.cooperad, self._coact_impl)

    def _coact_impl(self, v_element, n: int):
        """C-coalgebra coaction δ_n on B_α(A).

        Same as the cofree coalgebra coaction: for each basis element
        ``(c_key, (a_1, …, a_k))`` with k = n, returns
        ``c_key ⊗ (id, (a_1,)) ⊗ … ⊗ (id, (a_n,))``.
        """
        from sage.all import tensor

        base_ring = self.module.base_ring()
        C = self.cooperad_cls
        coop_parent = C(n, base_ring)
        cofree_mod = self.module
        id_key = C.unit_key()

        right_factors = [cofree_mod] * n
        target = tensor([coop_parent] + right_factors)
        result = target.zero()

        for (c_key, m_tuple), v_coeff in v_element:
            k = len(m_tuple)
            if k != n:
                continue
            coop_elem = coop_parent(c_key)
            leaf_elems = [cofree_mod((id_key, (mk,))) for mk in m_tuple]
            term = tensor([coop_elem] + leaf_elems)
            result += v_coeff * term

        return result

    def project(self, x):
        """Coprojection π: B_α(A) → A, onto weight-1 generators.

        Returns the image in the algebra module A.  Non-zero only for
        ``(id_key, (a,))`` terms.
        """
        inner = self._algebra.module
        id_key = self.cooperad_cls.unit_key()
        result = inner.zero()
        for (c_key, m_tuple), coeff in x:
            if len(m_tuple) == 1 and c_key == id_key:
                result += coeff * inner(m_tuple[0])
        return result
