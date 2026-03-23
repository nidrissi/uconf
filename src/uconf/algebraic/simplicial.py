"""Simplicial chain/cochain structures as operad-(co)algebra objects.

This module provides:

- :func:`surjection_chain_action` – the :class:`~uconf.models.surjection.Surjection`
  action on normalized simplicial chains via the Berger–Fresse formula.
- :func:`surjection_cochain_action` – the dual action on simplicial cochains.
- :class:`SurjectionSimplicialCochainAlgebra` – the resulting *P*-algebra.
- :class:`SurjectionSimplicialChainCoalgebra` – the resulting *C*-coalgebra.

**Reference**: C. Berger, B. Fresse, "Combinatorial operad actions on cochains",
Math. Proc. Cambridge Philos. Soc. **137** (2004), 135–174.  All sign
conventions in this module follow that paper (hereafter *[BF04]*).
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations, pairwise
from typing import TYPE_CHECKING

from sage.all import tensor

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.models.simplicial import SimplicialChains, SimplicialCochains
from uconf.models.surjection import Surjection
from uconf.models.surjection_dual import SurjectionDual

if TYPE_CHECKING:
    pass


def surjection_chain_action(
    surj: "Surjection.Element",
    chain: "SimplicialChains.Element",
):
    r"""Action of a surjection element on normalized simplicial chains.

    Implements the Berger–Fresse formula [BF04, §1.2] for the
    ``Surjection``-module structure on normalized simplicial chains.

    **Algorithm** (one basis term at a time):

    Let `u = (u_0, \ldots, u_{r+d-1}) \in S(r)` be a surjection of arity
    `r` and degree `d`, and let `x \in C_n(\Delta^\infty)` be a chain.

    1. **Iterated Alexander–Whitney diagonal.**
       Apply the AW diagonal `r + d - 1` times to `x`, producing an
       element of `C^{\otimes(r+d)}`.  Each basis term is an
       `(r+d)`-tuple of simplices `(x_0, \ldots, x_{r+d-1})` with
       `\sum \dim(x_i) = n`.

    2. **Group factors by the surjection.**
       For each output slot `k \in \{1, \ldots, r\}` collect all simplex
       factors `x_i` whose index satisfies `u_i = k`:
       `y_k = x_{i_1} * \cdots * x_{i_m}` (concatenation in the simplex
       sense, i.e.  taking the appropriate face).  Discard the term if any
       `y_k` is degenerate (has a repeated vertex).

    3. **Berger–Fresse sign.**
       Multiply by the sign `\varepsilon(u; x_0, \ldots, x_{r+d-1})`
       computed by :func:`_compute_bf_sign`.  This sign has two
       contributions:

       - *Ordering sign*: Koszul sign of the permutation that rearranges
         `(x_0, \ldots, x_{r+d-1})` into value-sorted order, computed
         using the graded-commutativity rule (transposing two factors of
         degrees `a` and `b` costs `(-1)^{ab}`).
       - *Action sign*: additional sign arising from adjacent positions of
         equal ``u``-value in sorted order; see :func:`_compute_bf_sign`.

    4. **Accumulate.**
       Sum all contributions `\varepsilon \cdot (y_1 \otimes \cdots \otimes y_r)`.

    Parameters
    ----------
    surj : Surjection.Element
        Homogeneous element of the surjection operad of arity ``r``.
    chain : SimplicialChains.Element
        Element of :class:`~uconf.models.simplicial.SimplicialChains`.

    Returns
    -------
    Element of ``tensor([SimplicialChains(R)]*r)`` (native Sage tensor)
    if ``r >= 2``; element of :class:`SimplicialChains` if ``r == 1``.

    Raises
    ------
    TypeError
        If ``surj`` and ``chain`` have different base rings.

    """
    if surj.parent().base_ring() != chain.parent().base_ring():
        raise TypeError("Surjection and chain must have the same base ring.")

    r = surj.arity()
    SC = chain.parent()

    # Target: tensor([SC]*r) for r >= 2, or SC itself for r == 1.
    target = SC if r == 1 else tensor([SC] * r)

    if not surj or not chain:
        return target.zero()

    surj_support = list(surj.support())
    degrees = {surj.parent().degree_on_basis(k) for k in surj_support}
    assert len(degrees) == 1, "Surjection must be homogeneous in degree."
    d = degrees.pop()

    # AW diagonal applied r+d-1 times gives r+d tensor factors.
    times = r + d - 1
    pre_diag = chain.iterated_diagonal(times=times) if times > 0 else chain

    def _compute_bf_sign(
        surj_tuple: tuple[int, ...],
        simplex_factors: tuple[tuple[int, ...], ...],
        positions: dict[int, int],
    ):
        r"""
        Compute the Berger–Fresse sign for a given surjection and simplex factors.

        This function calculates the sign ε(u; x_0, …, x_{r+d-1}) according to the
        Berger–Fresse construction, which is used in operadic homology computations.

        Parameters
        ----------
        surj_tuple : tuple[int, ...]
            A tuple representing a surjection u, where each element indicates an index
            in the target set. Elements may repeat, with repetitions indicating which
            intervals are "inner" versus "final".

        simplex_factors : tuple[tuple[int, ...], ...]
            A tuple of tuples representing the simplex factors (vertices) corresponding
            to each position in surj_tuple. Each inner tuple contains vertex indices.

        positions : dict[int, int]
            A mapping from each vertex label (an integer appearing in the simplex factors)
            to its position in the ambient simplex.
        Returns
        -------
        int
            Returns 1 or -1, the computed Berger–Fresse sign.

        Notes
        -----
        The sign is computed as (-1)^(ordering_sign_exp + position_exp), where:

        - **Final intervals**: Positions i where surj_tuple[i] appears for the last time
        - **Inner intervals**: All other positions
        - **position_exp**: Sum of the positions of the last vertices in all inner interval
          simplex factors
        - **ordering_sign_exp**: Koszul sign computed from inversions when sorting by
          u-values, weighted by lengths of simplex factors
        - **lengths**: For final intervals, the length is len(simplex_factor) - 1;
          for inner intervals, the length is len(simplex_factor)

        Raises
        ------
        AssertionError
            If the lengths of surj_tuple and simplex_factors do not match.

        """
        assert len(surj_tuple) == len(simplex_factors), "Length mismatch in BF sign computation."

        # Find the indices of "final" intervals, i.e., the values i such that u_i is the last
        # occurrence of that value in surj_tuple. The other intervals are called "inner" intervals.
        final_intervals: dict[int, int] = dict()
        for i in reversed(range(len(surj_tuple))):
            u = surj_tuple[i]
            if u not in final_intervals:
                final_intervals[u] = i

        # The position exponent is the sum of the positions of the last vertices in each
        # *inner* interval.
        position_exp = 0
        for i in range(len(surj_tuple)):
            u = surj_tuple[i]
            if i != final_intervals[u]:
                # last vertex of the simplex factor
                position_exp += positions[simplex_factors[i][-1]]

        # The length of a final interval is its number of elements minus one; the length of
        # inner intervals is their number of elements.
        lengths = []
        for i, u in enumerate(surj_tuple):
            if i == final_intervals[u]:
                lengths.append(len(simplex_factors[i]) - 1)
            else:
                lengths.append(len(simplex_factors[i]))

        # Sort indices stably by their u-value to get the ordering permutation π.
        # inv_ordering[new_pos] = original index (i.e. π^{-1}).
        indexed = list(enumerate(surj_tuple))
        sorted_indexed = sorted(indexed, key=lambda pair: pair[1])
        inv_ordering = [pair[0] for pair in sorted_indexed]

        # ordering_perm[i] = new position of original index i (i.e. π(i)).
        ordering_sign_exp = 0
        ordering_perm = [0] * len(inv_ordering)
        for new_pos, old_pos in enumerate(inv_ordering):
            ordering_perm[old_pos] = new_pos
        # Koszul sign: sum deg(x_i)*deg(x_j) over inversions (i < j, π(i) > π(j)).
        for i in range(len(ordering_perm)):
            for j in range(i + 1, len(ordering_perm)):
                if ordering_perm[i] > ordering_perm[j]:
                    ordering_sign_exp += lengths[i] * lengths[j]

        total_sign_exp = (ordering_sign_exp + position_exp) % 2
        return (-1) ** total_sign_exp

    def _join_simplices(simplex_list):
        """Concatenate a list of overlapping simplex faces into one simplex.

        The AW diagonal splits `[v_0, \\ldots, v_n]` into consecutive sub-faces.
        This function reassembles the full (concatenated) simplex from such a list.
        Returns ``None`` if the result would be degenerate (i.e. has a repeated
        vertex or is not strictly increasing).

        Parameters
        ----------
        simplex_list : list of tuple
            Non-empty list of simplex tuples to concatenate.

        Returns
        -------
        tuple or None

        """
        if not simplex_list:
            return None
        result = reduce(lambda x, y: x + y, simplex_list)
        if any(a >= b for a, b in pairwise(result)):
            return None
        return result

    def term_generator():
        for surj_tuple, surj_coeff in surj:
            for diag_key, diag_coeff in pre_diag:
                # diag_key is a tuple of simplex tuples (len = r+d for times>0,
                # or just a single simplex for times==0 when r==1, d==0).
                if times == 0:
                    # Single-factor case: the diagonal was a no-op.
                    diag_key = (diag_key,)  # wrap into 1-tuple for uniform handling

                # --- Step 2: group factors by u-value and concatenate ---
                # For output slot k, collect factors x_i where u_i = k and join them.
                new_factors = []
                zero_term = False
                for i in range(1, r + 1):
                    to_join = [
                        diag_key[idx] for idx in range(len(surj_tuple)) if surj_tuple[idx] == i
                    ]
                    joined = _join_simplices(to_join)
                    if joined is None:
                        # Degenerate factor → this AW term contributes 0.
                        zero_term = True
                        break
                    new_factors.append(joined)
                if zero_term:
                    continue

                # --- Step 3: Berger–Fresse sign ---
                curr_vertices: list[int] = sorted(set(v for simplex in diag_key for v in simplex))
                vertex_to_index = {v: idx for idx, v in enumerate(curr_vertices)}
                sign = _compute_bf_sign(surj_tuple, diag_key, vertex_to_index)
                coeff = sign * surj_coeff * diag_coeff

                # Validate each output factor individually.
                if any(SC._validate_basis_key(f) is None for f in new_factors):
                    continue

                if r == 1:
                    yield (new_factors[0], coeff)
                else:
                    yield (tuple(new_factors), coeff)

    return target.sum_of_terms(term_generator())


def surjection_cochain_action(
    surj: "Surjection.Element",
    cochains: tuple["SimplicialCochains.Element", ...],
) -> "SimplicialCochains.Element":
    r"""Dual surjection action on simplicial cochains.

    For `u \in S(r)` of degree `d` and cochains
    `f_1, \dots, f_r \in C^*(\Delta^N)`, the result
    `\mu_u(f_1 \otimes \dots \otimes f_r) \in C^*(\Delta^N)` is
    defined by the duality

    .. math::
        \langle \mu_u(f_1 \otimes \dots \otimes f_r),\, x \rangle
        = \langle f_1 \otimes \dots \otimes f_r,\, \theta_u(x) \rangle

    for all chains `x`, where `\theta_u` is the chain action of
    :func:`surjection_chain_action`.

    **Degree bookkeeping.**
    Cochains are graded homologically (``degree_on_basis`` returns
    `-\dim(\sigma)` for a simplex `\sigma`).  If `f_i` has homological
    degree `-n_i` (i.e.  is an `n_i`-cochain) and `u` has degree `d`,
    then `\mu_u(f_1 \otimes \dots \otimes f_r)` is an `n`-cochain where
    `n = n_1 + \dots + n_r + d`.  Equivalently, the chain `x` that can
    pair non-trivially with `\mu_u(f_1 \otimes \dots \otimes f_r)` has
    dimension

    .. math::
        \dim(x) = n_1 + \dots + n_r + d
        = \bigl(-\deg(f_1)\bigr) + \dots + \bigl(-\deg(f_r)\bigr) + d.

    **Implementation.**
    Iterate over all non-degenerate simplices `\sigma` of the correct
    dimension in `\Delta^N`, evaluate the chain action
    `\theta_u(\sigma)`, then contract with `f_1 \otimes \dots \otimes f_r`
    via the Kronecker pairing to read off the coefficient.

    Parameters
    ----------
    surj : Surjection.Element
        Homogeneous element of the surjection operad of arity ``r``.
    cochains : tuple of SimplicialCochains.Element
        Exactly ``r`` cochains, all on the same ``Δ^N``.

    Raises
    ------
    TypeError
        If ``surj`` and the cochains have different base rings.

    """
    r = surj.arity()
    assert len(cochains) == r, f"Expected {r} cochains, got {len(cochains)}."
    if r == 0:
        raise ValueError("Coaction requires surjection arity at least 1.")

    cochain_base_ring = cochains[0].parent().base_ring()
    if surj.parent().base_ring() != cochain_base_ring:
        raise TypeError("Surjection and cochains must have the same base ring.")

    N = cochains[0].parent().simplex_dim()
    for c in cochains:
        if c.parent().base_ring() != cochain_base_ring:
            raise TypeError("All cochains must have the same base ring.")
        if c.parent().simplex_dim() != N:
            raise ValueError("All cochains must be on the same simplex dimension.")
    target = SimplicialCochains(N=N, base_ring=cochain_base_ring)

    # If any input cochain is zero, the result is zero.
    for c in cochains:
        if c == c.parent().zero():
            return target.zero()

    d = surj.degree()
    # Degree bookkeeping: cochains use the homological convention
    # degree_on_basis(σ) = -dim(σ), so deg(f_i) = -n_i.
    # The output cochain μ_u(f_1,...,f_r) has homological degree
    # deg(f_1)+...+deg(f_r)-d = -(n_1+...+n_r+d).
    # Equivalently, the dual chain dimension is n_1+...+n_r+d.
    cochain_degrees = [c.degree() for c in cochains]
    cochain_total_degree = sum(cochain_degrees)
    chain_deg = -cochain_total_degree - d

    if chain_deg < 0:
        # No simplex of negative dimension exists; the action is zero.
        return target.zero()

    SC = SimplicialChains(base_ring=cochain_base_ring)

    result_dict: dict[tuple, int] = {}
    # Iterate over all dim-chain_deg simplices of Δ^N and apply the pairing.
    for simplex in combinations(range(N + 1), chain_deg + 1):
        x = SC.term(simplex)
        # θ_u(x): element of tensor([SC]*r) (or SC for r=1).
        theta = surjection_chain_action(surj, x)
        value = 0
        for basis_key, coeff in theta:
            # For r >= 2: basis_key = (simplex_1, ..., simplex_r).
            # For r == 1: basis_key is a single simplex tuple.
            if r == 1:
                factor_keys = (basis_key,)
            else:
                factor_keys = basis_key
            # Kronecker pairing: ⟨f_1⊗...⊗f_r, y_1⊗...⊗y_r⟩ = ∏ ⟨f_i, y_i⟩.
            contrib = coeff
            for slot in range(r):
                f = cochains[slot]
                fk = factor_keys[slot]
                factor_coeff = 0
                for f_key, f_coeff in f:
                    if f_key == fk:
                        factor_coeff = f_coeff
                        break
                contrib *= factor_coeff
                if contrib == 0:
                    break
            value += contrib
        if value != 0:
            duality_sign_exp = d * cochain_total_degree
            for i in range(0, r):
                duality_sign_exp += cochain_degrees[i] * sum(cochain_degrees[i + 1 :])
            duality_sign = -1 if duality_sign_exp % 2 == 1 else 1
            value *= duality_sign
            result_dict[simplex] = result_dict.get(simplex, 0) + value

    result_dict = {k: v for k, v in result_dict.items() if v != 0}
    if not result_dict:
        return target.zero()
    return target.sum_of_terms(result_dict.items())


class SurjectionSimplicialCochainAlgebra(OperadAlgebra):
    r""":class:`~uconf.models.surjection.Surjection`-algebra on simplicial cochains.

    Equips :class:`~uconf.models.simplicial.SimplicialCochains` with the
    ``Surjection``-algebra structure given by :func:`surjection_cochain_action`.

    This is a direct implementation of the Berger–Fresse cochain algebra
    [BF04]: for `u \in S(r)` and cochains `f_1, \ldots, f_r \in C^*(\Delta^N)`,

    .. math::
        \mu_u(f_1 \otimes \cdots \otimes f_r)(\sigma)
        = (f_1 \otimes \cdots \otimes f_r)(\theta_u(\sigma))

    where `\theta_u` is the chain action of :func:`surjection_chain_action`.
    """

    def __init__(self, N: int, base_ring):
        super().__init__(
            module=SimplicialCochains(N=N, base_ring=base_ring),
            operad_cls=Surjection,
            structure_map=self._act_impl,
        )

    def _act_impl(
        self,
        p_element: Surjection.Element,
        algebra_elements,
    ):
        return surjection_cochain_action(p_element, tuple(algebra_elements))


class SurjectionSimplicialChainCoalgebra(CooperadCoalgebra):
    r""":class:`~uconf.models.surjection_dual.SurjectionDual`-coalgebra on simplicial chains.

    The dual of the surjection chain action equips
    :class:`~uconf.models.simplicial.SimplicialChains` with a
    ``SurjectionDual``-coalgebra structure.

    The coaction `\delta_n : C \to \mathrm{SD}(n) \otimes C^{\otimes n}` is
    defined by `\delta_n(x) = \sum_u u^* \otimes \theta_u(x)`, where the
    sum runs over all surjections `u \in S(n)` of appropriate degree and
    `\theta_u` is the chain action of :func:`surjection_chain_action`.
    """

    def __init__(self, base_ring):

        super().__init__(
            module=SimplicialChains(base_ring=base_ring),
            cooperad_cls=SurjectionDual,
            coaction_map=self._coact_impl,
        )

    def _coact_impl(self, v_element: "SimplicialChains.Element", n: int):
        """C-coalgebra coaction δ_n via the surjection chain action.

        Returns an element of ``tensor([SurjectionDual(n)] + [SC]*n)``
        (a flat ``(n+1)``-fold tensor product).
        """
        if n <= 0:
            raise ValueError(f"Coaction arity must be positive, got {n}.")

        base_ring = self.module.base_ring()
        if v_element.parent().base_ring() != base_ring:
            raise TypeError("Chain element base ring does not match module base ring.")

        SC = self.module  # SimplicialChains instance
        left_parent = SurjectionDual(n, base_ring=base_ring)
        # Flat (n+1)-way tensor: SurjectionDual(n) ⊗ SC ⊗ ... ⊗ SC
        target = tensor([left_parent, SC] if n == 1 else [left_parent] + [SC] * n)

        if not v_element:
            return target.zero()

        min_v_degree = min(v_element.parent().degree_on_basis(key) for key in v_element.support())
        max_v_dim = max(len(key) - 1 for key in v_element.support())
        surj_parent = Surjection(n, base_ring=base_ring)

        result = target.zero()
        for degree in range(n * max_v_dim + min_v_degree + 1):
            for u in surj_parent.basis_it(degree):
                # surjection_chain_action returns:
                #   SC element  (if n == 1)
                #   tensor([SC]*n) element  (if n >= 2)
                right_elem = surjection_chain_action(u, v_element)
                if not right_elem:
                    continue
                for u_basis, u_coeff in u:
                    for right_key, right_coeff in right_elem:
                        # right_key is either:
                        #   a single simplex tuple  (n == 1)
                        #   an n-tuple of simplex tuples  (n >= 2)
                        if n == 1:
                            flat_key = (u_basis, right_key)
                        else:
                            flat_key = (u_basis,) + right_key
                        result += u_coeff * right_coeff * target.term(flat_key)
        return result
