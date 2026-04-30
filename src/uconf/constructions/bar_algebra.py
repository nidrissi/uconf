r"""Bar construction for a P-algebra: B_α(A).

Given a twisting morphism α: C → P and a P-algebra (A, γ), the bar
construction is the cofree conilpotent C-coalgebra

    B_α(A) = (T^c_C(A),  d_{T^c} + d_α)

where:

- ``T^c_C(A) = ⊕_{n≥1} C(n) ⊗ A^{⊗n}`` is the cofree conilpotent
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

from sage.all import cached_method

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.core.display import latex_linear_combination
from uconf.core.signs import koszul_sign_of_permutation, sign_from_exponent
from uconf.core.twisting import TwistingMorphism


class BarAlgebraModule(CofreeCoalgebraModule):
    r"""Underlying dg-module of the bar construction B_α(A).

    Extends :class:`CofreeCoalgebraModule` with the twisting differential
    d_α.  Basis keys are ``(c_key, m_tuple)`` pairs inherited from the cofree
    coalgebra, where ``c_key ∈ C(n)`` and ``m_tuple`` has length ``n``.

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

    @cached_method
    def _twisted_boundary_on_basis(self, key):
        """Total differential d = d_{T^c} + d_α."""
        return CofreeCoalgebraModule._boundary_on_basis(self, key) + self._dalpha_on_basis(key)

    @cached_method
    def _dalpha_on_basis(self, key):
        r"""Twisting differential d_α (coderivation).

        For a basis key ``(c_key, (a_1, …, a_n))`` with n = m + n_r - 1:

            d_α(c ⊗ a_1 ⊗ … ⊗ a_n) = Σ_{i,m,n_r}
                (-1)^{|c_L| + |α(c_R)|(|a_1| + … + |a_{i-1}|)}
                (c_L ⊗ a_1 ⊗ … ⊗ γ(α(c_R); a_i,…,a_{i+n_r-1}) ⊗ … ⊗ a_n)

        where Δ^{i;m,n_r}(c) = Σ c_L ⊗ c_R is the cooperad's infinitesimal
        cocomposition, α(c_R) ∈ P(n_r) is the twisting morphism, and
        γ is the algebra action.

        The first factor ``(-1)^{|c_L|}`` comes from commuting d_α (which has
        degree -1) past the cooperad element c_L in the cofree coalgebra
        decomposition.  The second factor records the further Koszul sign from
        moving the inserted operad element α(c_R) past the preceding algebra
        inputs ``a_1, …, a_{i-1}`` before applying γ to the contracted block.

        When the cooperad is a bar/cofree cooperad whose elements may have
        non-contiguous leaf orderings, this method uses ``_iter_all_splits``
        to capture all internal-edge decompositions—including those with
        non-contiguous leaf subsets—with the correct algebra slicing and
        Koszul gathering sign.

        Reference: Loday-Vallette, Section 6.3 and 11.2.
        """
        c_key, m_tuple = key
        n = len(m_tuple)
        if n <= 1:
            return self.zero()

        base_ring = self.base_ring()
        C = self._cooperad_cls
        c_comp = C(n, base_ring)

        # Use _iter_all_splits when available (bar construction cooperads)
        # to handle non-contiguous leaf subsets correctly.
        if hasattr(c_comp, "_iter_all_splits"):
            return self._dalpha_all_splits(c_key, m_tuple, n, base_ring, c_comp)

        return self._dalpha_contiguous(c_key, m_tuple, n, base_ring, c_comp)

    @cached_method
    def _alpha_on_basis(self, arity, c_key):
        """Return ``α(c_key)`` for a cooperad basis key."""
        c_comp = self._cooperad_cls(arity, self.base_ring())
        return self._alpha(c_comp.term(c_key))

    @cached_method
    def _normalized_left_corolla(self, arity, c_left_key, new_m):
        """Return the normalized corolla for a cached left cooperad factor."""
        c_left_comp = self._cooperad_cls(arity, self.base_ring())
        return self._normalized_corolla_sum(c_left_comp.term(c_left_key), new_m)

    def _dalpha_contiguous(self, c_key, m_tuple, n, base_ring, c_comp):
        """d_α via contiguous partial cocompositions (original path)."""
        C = self._cooperad_cls
        M = self._inner_module
        result_dict: dict = {}
        zero = base_ring.zero()
        c_elem = c_comp.term(c_key)
        m_terms = [M.term(m_key) for m_key in m_tuple]
        m_prefix_degrees = [0]
        for m_key in m_tuple:
            m_prefix_degrees.append(m_prefix_degrees[-1] + M.degree_on_basis(m_key))
        action_cache: dict = {}

        for n_r in range(1, n + 1):
            m = n - n_r + 1
            for i in range(1, m + 1):
                # Infinitesimal cocomposition Δ^{i;m,n_r}
                cocomp = C.infinitesimal_cocompose(c_elem, i, m, n_r)

                for (c_L_key, c_R_key), coeff in cocomp:
                    # Apply α to c_R
                    alpha_c_R = self._alpha_on_basis(n_r, c_R_key)

                    if not alpha_c_R:
                        continue

                    # Apply the algebra action γ(α(c_R); a_i, …, a_{i+n_r-1})
                    action_cache_key = (c_R_key, i)
                    action_result = action_cache.get(action_cache_key)
                    if action_result is None:
                        a_slice = m_terms[i - 1 : i + n_r - 1]
                        action_result = self._algebra.act(alpha_c_R, a_slice)
                        action_cache[action_cache_key] = action_result

                    if not action_result:
                        continue

                    # Koszul sign from moving the twisting coderivation
                    # past the left cooperad factor, and from moving the
                    # inserted operad element α(c_R) past the leaf factors
                    # that occur before the contracted block.
                    c_L_comp = C(m, base_ring)
                    c_L_deg = c_L_comp.degree_on_basis(c_L_key)
                    prefix_deg = m_prefix_degrees[i - 1]
                    alpha_deg = alpha_c_R.degree()
                    sign = sign_from_exponent(c_L_deg + alpha_deg * prefix_deg)

                    for a_new_key, a_coeff in action_result:
                        new_m = m_tuple[: i - 1] + (a_new_key,) + m_tuple[i + n_r - 1 :]
                        normalized = self._normalized_left_corolla(m, c_L_key, new_m)
                        scale = sign * coeff * a_coeff
                        for out_key, out_coeff in normalized:
                            # Sage's free-module constructors do not always
                            # coerce products of ring elements automatically.
                            combined = base_ring(scale * out_coeff)
                            result_dict[out_key] = result_dict.get(out_key, zero) + combined

        return self._from_dict(result_dict, remove_zeros=True)

    def _dalpha_all_splits(self, c_key, m_tuple, n, base_ring, c_comp):
        """d_α via _iter_all_splits (handles non-contiguous leaf subsets).

        ``_iter_all_splits`` yields both the reduced cocompositions
        (splitting at internal edges) and the identity summands
        (counit axioms):

        - ``Δ^{i;n,1}(c) = c ⊗ η`` (right counit): these vanish
          automatically because twisting morphisms send the counit
          to zero (α(η) = 0).
        - ``Δ^{1;1,n}(c) = η ⊗ c`` (left counit): α is applied to
          the whole element *c*.  The sign is +1 because *η* has
          bar degree 0.
        """
        C = self._cooperad_cls
        M = self._inner_module
        result_dict: dict = {}
        zero = base_ring.zero()
        m_terms = [M.term(m_key) for m_key in m_tuple]
        m_degrees = [M.degree_on_basis(m_key) for m_key in m_tuple]
        action_cache: dict = {}
        split_cache: dict = {}

        for child_positions, c_L_key, c_R_key, coop_sign in c_comp._iter_all_splits(c_key):
            n_r = len(child_positions)
            m = n - n_r + 1

            # Apply α to c_R
            alpha_c_R = self._alpha_on_basis(n_r, c_R_key)

            if not alpha_c_R:
                continue

            # Algebra slice at the ACTUAL (possibly non-contiguous) positions
            action_cache_key = (child_positions, c_R_key)
            action_result = action_cache.get(action_cache_key)
            if action_result is None:
                a_slice = [m_terms[s - 1] for s in child_positions]
                action_result = self._algebra.act(alpha_c_R, a_slice)
                action_cache[action_cache_key] = action_result

            if not action_result:
                continue

            # Koszul sign from moving the twisting coderivation past
            # the left cooperad factor, and from moving the inserted
            # operad element α(c_R) past the leaf factors that stay
            # before the contracted block.
            c_L_comp = C(m, base_ring)
            c_L_deg = c_L_comp.degree_on_basis(c_L_key)

            # Build new m_tuple: order-preserving mapping from
            # original positions to the m-element result tuple.
            # Placeholder position gets the action result; other
            # positions keep their original algebra elements.
            split_data = split_cache.get(child_positions)
            if split_data is None:
                selected = set(child_positions)
                min_S = child_positions[0]
                top_positions = tuple(
                    pos for pos in range(1, n + 1) if pos not in selected or pos == min_S
                )
                prefix_deg = sum(
                    m_degrees[pos - 1] for pos in range(1, min_S) if pos not in selected
                )
                gathering_sign = 1
                if child_positions != tuple(range(min_S, min_S + n_r)):
                    t_before = [pos for pos in range(1, min_S) if pos not in selected]
                    t_after = [pos for pos in range(min_S + 1, n + 1) if pos not in selected]
                    gathered = t_before + list(child_positions) + t_after
                    perm_0idx = [g - 1 for g in gathered]
                    gathering_sign = koszul_sign_of_permutation(perm_0idx, m_degrees)
                split_data = (top_positions, prefix_deg, gathering_sign, min_S)
                split_cache[child_positions] = split_data
            top_positions, prefix_deg, gathering_sign, min_S = split_data

            alpha_deg = alpha_c_R.degree()
            sign = sign_from_exponent(c_L_deg + alpha_deg * prefix_deg)

            for a_new_key, a_coeff in action_result:
                new_m = tuple(
                    a_new_key if pos == min_S else m_tuple[pos - 1] for pos in top_positions
                )
                normalized = self._normalized_left_corolla(m, c_L_key, new_m)
                scale = sign * coop_sign * gathering_sign * a_coeff
                for out_key, out_coeff in normalized:
                    # Sage's free-module constructors do not always coerce
                    # accumulated coefficients on their own.
                    combined = base_ring(scale * out_coeff)
                    result_dict[out_key] = result_dict.get(out_key, zero) + combined

        return self._from_dict(result_dict, remove_zeros=True)

    # ------------------------------------------------------------------
    # Expose component differentials for debugging
    # ------------------------------------------------------------------

    def d_cofree(self, elem):
        """Apply only the cofree coalgebra differential d_{T^c}."""
        return self._d_cofree(elem)

    def d_alpha(self, elem):
        """Apply only the twisting differential d_α."""
        return self._d_alpha(elem)

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
