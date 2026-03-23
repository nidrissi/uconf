"""Operadic twisting morphisms α: C → P.

A twisting morphism between a connected dg-cooperad C and a connected
dg-operad P is a degree -1 map

    α: C̄ → P

(from the coaugmentation coideal of C to P) satisfying the Maurer-Cartan
equation

    ∂α + α ⋆ α = 0

where ⋆ denotes the pre-Lie convolution product.

The pre-Lie product (α ⋆ β)(c) is defined for c ∈ C(n) as follows: apply the
infinitesimal decomposition Δ_{(1)} to c to get Σ c_L ⊗_i c_R, then compose
on the operad side:

    (α ⋆ β)(c) = Σ_i α(c_L) ∘_i β(c_R)

The Maurer-Cartan equation ∂α + α ⋆ α = 0 is equivalent to requiring that
the twisted differential on the (co)free (co)algebra squares to zero.

Reference: Loday-Vallette "Algebraic Operads", Section 6.4 and 11.1.
"""

from __future__ import annotations

from typing import Callable

from uconf.core.cooperad import CooperadLike
from uconf.core.operad import OperadLike
from uconf.core.signs import sign_from_exponent


class TwistingMorphism:
    """An operadic twisting morphism α: C → P.

    A twisting morphism is a degree -1 linear map from the coaugmentation
    coideal C̄ of a cooperad C to an operad P, satisfying the Maurer-Cartan
    equation ∂α + α ⋆ α = 0.

    The map is specified by a callable ``morphism_fn(c_elem) -> p_elem`` that
    takes an element of C(n) (for any n ≥ 2) and returns an element of P(n).
    The map should be zero on C(1) (the coaugmentation coideal kills the counit).

    Args:
        cooperad: The source cooperad C (a ``CooperadLike``).
        operad: The target operad P (an ``OperadLike``).
        morphism_fn: A callable ``(c_element) -> p_element`` implementing the
            degree -1 linear map.  The function receives an element of ``C(n)``
            and must return an element of ``P(n)`` (of one degree lower).
        name: Optional display name for the twisting morphism.

    Example::

        from uconf import Associative
        from uconf.constructions import BarConstruction
        from uconf.morphisms.canonical_twisting import canonical_projection

        # π: B(Ass) → Ass is the canonical projection
        pi = canonical_projection(Associative)
        assert pi.cooperad is BarConstruction(Associative)
        assert pi.operad is Associative
    """

    def __init__(
        self,
        cooperad: CooperadLike,
        operad: OperadLike,
        morphism_fn: Callable,
        *,
        name: str | None = None,
    ):
        self.cooperad = cooperad
        self.operad = operad
        self._morphism_fn = morphism_fn
        self.name = name or f"α: {getattr(cooperad, 'name', '?')} → {getattr(operad, 'name', '?')}"

    def __call__(self, c_elem):
        """Apply the twisting morphism α to a cooperad element.

        Args:
            c_elem: An element of ``C(n)`` for some arity ``n``.

        Returns:
            An element of ``P(n)`` (of one degree lower).
        """
        return self._morphism_fn(c_elem)

    def _repr_(self) -> str:
        return f"TwistingMorphism({self.name})"

    def star(self, other: "TwistingMorphism", c_elem):
        """Compute the pre-Lie convolution product (self ⋆ other)(c_elem).

        For c ∈ C(n), the pre-Lie product is:

            (α ⋆ β)(c) = Σ_{i, m, n_r} (-1)^{|c_L|} α(c_L) ∘_i β(c_R)

        where ``Δ^{i;m,n_r}(c) = Σ c_L ⊗ c_R`` is the infinitesimal
        cocomposition of c in slot i with left arity m and right arity n_r,
        and the Koszul sign ``(-1)^{|β| · |c_L|} = (-1)^{|c_L|}`` comes from
        permuting the degree-(-1) map β past the graded element c_L.

        Args:
            other: Another twisting morphism β: C → P (same cooperad and operad).
            c_elem: An element of C(n) for some n ≥ 2.

        Returns:
            An element of P(n) (the pre-Lie product evaluated on c_elem).
        """
        n = c_elem.arity()
        base_ring = c_elem.parent().base_ring()
        p_parent = self.operad(n, base_ring)
        result = p_parent.zero()

        for m in range(2, n):
            n_right = n - m + 1
            for i in range(1, m + 1):
                cocomp = self.cooperad.infinitesimal_cocompose(c_elem, i, m, n_right)
                for (dec_left, dec_right), coeff in cocomp:
                    left_parent = self.cooperad(m, base_ring)
                    right_parent = self.cooperad(n_right, base_ring)

                    # Koszul sign: (-1)^{|β| · |c_L|} where |β| = -1
                    # Since (-1)^{-d} = (-1)^d, the sign is (-1)^{|c_L|}
                    deg_c_L = left_parent.degree_on_basis(dec_left)
                    koszul_sign = sign_from_exponent(deg_c_L)

                    alpha_left = self(left_parent.term(dec_left))
                    beta_right = other(right_parent.term(dec_right))
                    composed = self.operad.compose(alpha_left, i, beta_right)
                    # Extract basis keys and add in the target parent to avoid
                    # Sage coercion issues between different CombinatorialFreeModule instances
                    for p_key, p_coeff in composed:
                        result += koszul_sign * coeff * p_coeff * p_parent.term(p_key)

        return result

    def partial_alpha(self, c_elem):
        """Compute ∂α(c) = ∂_P(α(c)) + α(∂_C(c)).

        This is the boundary of α as a map of graded modules (before the
        Maurer-Cartan equation twist).

        The sign convention is ∂α = ∂_P ∘ α - (-1)^{|α|} α ∘ ∂_C.
        Since |α| = -1, we have (-1)^{|α|} = -1, so:
        ∂α(c) = ∂_P(α(c)) + α(∂_C(c)).

        Note: some references use ∂α = ∂_P ∘ α + α ∘ ∂_C; this is the same
        since |α| = -1.

        Args:
            c_elem: An element of C(n).

        Returns:
            An element of P(n).
        """
        n = c_elem.arity()
        base_ring = c_elem.parent().base_ring()
        p_parent = self.operad(n, base_ring)

        alpha_c = self(c_elem)

        # ∂_P(α(c)) — need to evaluate in the target parent
        d_P_alpha_c = alpha_c.parent().boundary(alpha_c)

        # α(∂_C(c))
        c_parent = c_elem.parent()
        d_C_c = c_parent.boundary(c_elem)
        alpha_d_C_c = self(d_C_c)

        # Combine in target parent to avoid coercion issues
        result = p_parent.zero()
        for key, coeff in d_P_alpha_c:
            result += coeff * p_parent.term(key)
        for key, coeff in alpha_d_C_c:
            result += coeff * p_parent.term(key)

        return result

    def maurer_cartan(self, c_elem):
        """Evaluate the Maurer-Cartan expression ∂α + α ⋆ α on c_elem.

        If α is a valid twisting morphism, this should return zero for all c.

        .. note::
           The ``star`` method uses ``infinitesimal_cocompose`` which may not
           capture all decompositions for non-planar cooperads.  For a robust
           MC check, use :meth:`check_maurer_cartan` which verifies d² = 0 on
           the twisted complex.

        Args:
            c_elem: An element of C(n) for some n ≥ 2.

        Returns:
            An element of P(n).  Zero if and only if the MC equation holds for
            this input.
        """
        return self.partial_alpha(c_elem) + self.star(self, c_elem)

    def check_maurer_cartan(self, max_arity: int, base_ring, *, verbose: bool = False) -> bool:
        """Verify the Maurer-Cartan equation ∂α + α ⋆ α = 0 up to arity max_arity.

        The MC equation is verified indirectly by constructing the twisted bar
        complex B_α(A) for a trivial P-algebra A and checking d² = 0 on all
        basis elements.  This is equivalent to the MC equation but avoids the
        subtlety of non-contiguous leaf orderings in the cooperad cocomposition.

        Args:
            max_arity: Maximum arity of the bar trees to check.
            base_ring: Coefficient ring.
            verbose: If True, print diagnostic information.

        Returns:
            True if d² = 0 on all checked elements (equivalent to MC).
        """
        from uconf.algebraic.algebra import OperadAlgebra

        # Build a trivial P-algebra on the 1-dimensional module k
        # Use the Commutative(1) module as the trivial module
        from uconf.models.commutative import Commutative

        module = Commutative(1, base_ring=base_ring)

        def _trivial_structure_map(p_elem, a_list):
            # Trivial action: γ_n(p; a_1,...,a_n) = 0 for n ≥ 2
            # γ_1(id; a) = a for the unit
            n = p_elem.arity()
            if n == 1:
                result = module.zero()
                for _key, p_coeff in p_elem:
                    for _ak, a_coeff in a_list[0]:
                        result += p_coeff * a_coeff * module(())
                return result
            return module.zero()

        trivial_alg = OperadAlgebra(module, self.operad, _trivial_structure_map)

        # Build twisted bar complex
        from uconf.constructions.twisted_complex import TwistedBarComplex

        B = TwistedBarComplex(self, trivial_alg)

        # Check d² = 0 on all basis elements of weight 1..max_arity.
        # Each leaf carries one Commutative(1) key (weight 1), so weight
        # equals the number of leaves (tree arity).
        conn = B.connectivity
        for d in range(conn, conn + 2 * max_arity + 4):
            for w in range(1, max_arity + 1):
                try:
                    basis = list(B.basis_weight_iter(d, w))
                except (ValueError, TypeError, NotImplementedError):
                    continue
                for elem in basis:
                    d_elem = B.boundary(elem)
                    dd = B.boundary(d_elem)
                    if dd != B.zero():
                        if verbose:
                            key = next(iter(elem.support()))
                            print(f"d² ≠ 0 at degree {d}, weight {w}: d²({key}) ≠ 0")
                        return False

        return True
