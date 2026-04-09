"""Quasi-planar structure for operads and cooperads.

A quasi-planar operad/cooperad is free as a symmetric module::

    P(n) = P_pl(n) ⊗ k[S_n]

with ``planarize`` decomposing any element into its planar part and symmetric
group factor.  The differential then splits as::

    d(x ⊗ id) = Σ_σ d_σ(x) ⊗ σ

where each ``d_σ`` is a degree -1 map ``P_pl(n) → P_pl(n)``.

This module provides:

- :class:`QuasiPlanarMixin` — adds a generic ``d_sigma`` method to any
  graded module that already has ``planarize`` and ``boundary``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable
from sage.all import Family

from uconf.core.component import ComponentProtocol


@runtime_checkable
class QuasiPlanarProtocol(ComponentProtocol, Protocol):
    """Structural contract for a quasi-planar operad/cooperad component.

    A class satisfies this protocol when it exposes ``planarize`` and
    ``boundary`` (both linear maps) and ``arity()``.
    """

    def planar_basis_iter(self, d: int) -> Iterable:
        """Returns an iterator over the planar basis keys of this component."""
        ...

    def graded_planar_basis(self, d: int) -> Family:
        """Returns the planar basis of this component in degree d."""
        ...

    def planarize(self, x: QuasiPlanarProtocol.Element) -> Any: ...


# We use a conditional base class because otherwise we have a conflict with Protocol and the SageMath metaclass.
if TYPE_CHECKING:
    __quasi_planar_base = QuasiPlanarProtocol
else:
    __quasi_planar_base = object


class QuasiPlanarMixin(__quasi_planar_base):
    """Mixin providing ``d_sigma`` from ``boundary`` and ``planarize``.

    Mix this into any ``CombinatorialFreeModule`` component that already
    implements ``planarize`` (element → C_pl ⊗ k[S_n]) and ``boundary``.

    The ``d_sigma`` method computes the σ-component of the differential
    restricted to planar generators::

        d_σ(x) = π_σ(d(x))

    where ``π_σ`` is the projection onto the σ-factor of the symmetric
    decomposition ``P(n) = P_pl(n) ⊗ k[S_n]``.
    """

    def is_planar_key(self, key: Any) -> bool:
        """Return ``True`` if ``key`` is a planar basis key.

        A key is planar when ``planarize(term(key))`` yields only the
        identity permutation factor.
        """
        from sage.all import SymmetricGroup

        n = self.arity()
        S_n = SymmetricGroup(n)
        planarized = self.planarize(self.term(key))
        for (_c_key, sigma_key), coeff in planarized:
            if coeff != 0 and S_n(sigma_key) != S_n.identity():
                return False
        return True

    def d_sigma(self, x: Any, sigma: Any) -> Any:
        """Return the ``sigma``-component of ``boundary(x)``.

        Given a planar element ``x ∈ P_pl(n)`` (or any element), compute
        ``boundary(x)`` and project onto the ``sigma``-factor::

            d_σ(x) = Σ_{basis b in d(x)} coeff(b) · planar(b)
                      where planarize(b) has σ-component

        Parameters
        ----------
        x : element of this component
        sigma : permutation in ``S_n`` (Sage element, list, or tuple)

        Returns
        -------
        The planar element ``d_σ(x) ∈ P_pl(n)`` such that the σ-component
        of ``d(x)`` is ``d_σ(x) ⊗ σ``.

        """
        from sage.all import SymmetricGroup

        n = self.arity()
        S_n = SymmetricGroup(n)

        if isinstance(sigma, (list, tuple)):
            sigma = S_n(sigma)

        bdry = self.boundary(x)
        planarized = self.planarize(bdry)

        # planarized lives in self ⊗ k[S_n].
        # Extract the coefficient of sigma in the k[S_n] factor.
        result = self.zero()
        for tensor_basis, coeff in planarized:
            # tensor_basis is a pair (planar_key, group_element_index)
            # In Sage's tensor product of CFMs, basis keys are tuples
            planar_key, group_key = tensor_basis
            # group_key is a basis key of SymmetricGroupAlgebra,
            # which is a permutation
            if S_n(group_key) == sigma:
                result += coeff * self(planar_key)

        return result

    def d_sigma_decompose(self, x: Any) -> dict[Any, Any]:
        """Decompose ``boundary(x)`` into all σ-indexed components at once.

        Returns a dictionary ``{σ: d_σ(x)}`` containing only the non-zero
        components.  This is equivalent to calling ``d_sigma(x, σ)`` for
        every ``σ ∈ S_n``, but computes ``boundary`` and ``planarize``
        only **once** instead of once per permutation query.

        Parameters
        ----------
        x : element of this component

        Returns
        -------
        dict
            Mapping from permutation σ to the non-zero planar element
            ``d_σ(x) ∈ P_pl(n)``.
        """
        # Bypass morphism overhead for single-term elements: call on_basis
        # directly instead of going through morphism.__call__ + linear_combination.
        bdry_on_basis = getattr(self.boundary, "on_basis", None)
        if bdry_on_basis is not None:
            bdry_on_basis = bdry_on_basis()
            bdry = self.zero()
            for key, coeff in x:
                bdry += coeff * bdry_on_basis(key)
        else:
            bdry = self.boundary(x)

        if not bdry:
            return {}

        # Similarly bypass planarize morphism overhead
        planarize_on_basis = getattr(self, "_planarize_on_basis", None)
        if planarize_on_basis is not None:
            planarized_terms = []
            for key, coeff in bdry:
                p_result = planarize_on_basis(key)
                for pk, pc in p_result:
                    planarized_terms.append((pk, coeff * pc))
        else:
            planarized = self.planarize(bdry)
            planarized_terms = list(planarized)

        # Accumulate as {sigma_key: {planar_key: coeff}} for faster grouping,
        # then build elements at the end.
        result_dicts: dict[Any, dict] = {}
        for (planar_key, group_key), coeff in planarized_terms:
            if group_key in result_dicts:
                d = result_dicts[group_key]
                if planar_key in d:
                    d[planar_key] += coeff
                else:
                    d[planar_key] = coeff
            else:
                result_dicts[group_key] = {planar_key: coeff}

        # Convert to Sage elements and filter zeros.
        # We need the sigma as a SymmetricGroup element for the caller.
        from sage.all import SymmetricGroup
        n = self.arity()
        S_n = SymmetricGroup(n)

        result: dict[Any, Any] = {}
        for group_key, coeff_dict in result_dicts.items():
            terms = [(k, v) for k, v in coeff_dict.items() if v]
            if terms:
                result[S_n(group_key)] = self.sum_of_terms(terms)

        return result

    def d_sigma_iterate(self, x: Any, sigmas: Iterable[Any]) -> Any:
        """Apply ``d_sigma`` iteratively for a sequence of permutations.

        Computes ``d_{σ₁} ∘ d_{σ₂} ∘ ··· ∘ d_{σₖ}(x)`` where
        ``sigmas = (σ₁, σ₂, ..., σₖ)``.

        Parameters
        ----------
        x : element of this component
        sigmas : iterable of permutations

        Returns
        -------
        The result of applying d_{σₖ}, then d_{σ_{k-1}}, ..., then d_{σ₁}.

        """
        sigmas = list(sigmas)
        result = x
        for sigma in reversed(sigmas):
            result = self.d_sigma(result, sigma)
            if not result:  # early exit on zero
                return self.zero()
        return result
