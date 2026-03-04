"""Simplicial chain/cochain structures as operad-(co)algebra objects."""

from __future__ import annotations

from functools import reduce
from itertools import combinations, pairwise
from typing import TYPE_CHECKING

from sage.all import QQ, tensor

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.models.simplicial import SimplicialChains, SimplicialCochains

if TYPE_CHECKING:
    from uconf.models.surjection import Surjection


def surjection_chain_action(
    surj: "Surjection.Element",
    chain: SimplicialChains.Element,
    coord: int = 1,
) -> SimplicialChains.Element:
    r"""Action of a surjection element on normalized simplicial chains."""
    if surj.parent().base_ring() != chain.parent().base_ring():
        raise TypeError("Surjection and chain must have the same base ring.")

    r = surj.arity()
    t = chain.arity()
    assert 1 <= coord <= t, f"coord={coord} out of range [1, {t}]"
    target = SimplicialChains(r=t + r - 1, base_ring=surj.parent().base_ring())

    if not surj or not chain:
        return target.zero()

    surj_support = list(surj.support())
    degrees = {surj.parent().degree_on_basis(k) for k in surj_support}
    assert len(degrees) == 1, "Surjection must be homogeneous in degree."
    d = degrees.pop()

    times = r + d - 1
    pre_diag = chain.iterated_diagonal(times=times, coord=coord)

    def _compute_bf_sign(surj_tuple, simplex_factors):
        weights = [len(s) - 1 for s in simplex_factors]

        indexed = list(enumerate(surj_tuple))
        sorted_indexed = sorted(indexed, key=lambda pair: pair[1])
        inv_ordering = [pair[0] for pair in sorted_indexed]

        ordering_sign_exp = 0
        ordering_perm = [0] * len(inv_ordering)
        for new_pos, old_pos in enumerate(inv_ordering):
            ordering_perm[old_pos] = new_pos
        for i in range(len(ordering_perm)):
            for j in range(i + 1, len(ordering_perm)):
                if ordering_perm[i] > ordering_perm[j]:
                    ordering_sign_exp += weights[i] * weights[j]

        sorted_weights = [weights[i] for i in inv_ordering]
        sorted_surj = [surj_tuple[i] for i in inv_ordering]

        action_sign_exp = 0
        for idx in range(len(sorted_surj) - 1):
            if sorted_surj[idx] == sorted_surj[idx + 1]:
                action_sign_exp += sum(sorted_weights[: idx + 1])

        total_sign_exp = (ordering_sign_exp + action_sign_exp) % 2
        return (-1) ** total_sign_exp

    def _join_simplices(simplex_list):
        if not simplex_list:
            return None
        result = reduce(lambda x, y: x + y, simplex_list)
        if any(a >= b for a, b in pairwise(result)):
            return None
        return result

    def term_generator():
        for surj_tuple, surj_coeff in surj:
            for diag_key, diag_coeff in pre_diag:
                left_idx = coord - 1
                right_idx = left_idx + len(surj_tuple)
                left = diag_key[:left_idx]
                middle = diag_key[left_idx:right_idx]
                right = diag_key[right_idx:]

                new_factors = []
                zero_term = False
                for i in range(1, r + 1):
                    to_join = [
                        middle[idx]
                        for idx in range(len(surj_tuple))
                        if surj_tuple[idx] == i
                    ]
                    joined = _join_simplices(to_join)
                    if joined is None:
                        zero_term = True
                        break
                    new_factors.append(joined)
                if zero_term:
                    continue
                new_key = left + tuple(new_factors) + right
                validated = SimplicialChains._validate_basis_key(new_key)
                if validated is None:
                    continue

                sign = _compute_bf_sign(surj_tuple, middle)
                deg_left = sum(len(spx) - 1 for spx in left)
                sign *= (-1) ** (deg_left * d)
                yield (validated, sign * surj_coeff * diag_coeff)

    out = target.sum_of_terms(term_generator())
    _ = out.to_native_tensor()
    return out


def surjection_cochain_action(
    surj: "Surjection.Element",
    cochains: tuple[SimplicialCochains.Element, ...],
) -> SimplicialCochains.Element:
    r"""Dual surjection action on simplicial cochains."""
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

    if surj.parent().base_ring() == QQ:
        target = SimplicialCochains(N=N, r=1)
    else:
        target = SimplicialCochains(N=N, r=1, base_ring=surj.parent().base_ring())

    result_dict: dict[tuple, int] = {}
    cochain_degrees = [c.degree() for c in cochains]
    total_deg = sum(cochain_degrees)

    surj_deg = list({surj.parent().degree_on_basis(k) for k in surj.support()})
    assert len(surj_deg) == 1
    d = surj_deg[0]
    chain_deg = total_deg + d

    for simplex_tuple in combinations(range(N + 1), chain_deg + 1):
        if surj.parent().base_ring() == QQ:
            chain_parent = SimplicialChains(r=1)
        else:
            chain_parent = SimplicialChains(r=1, base_ring=surj.parent().base_ring())
        x = chain_parent((simplex_tuple,))
        theta = surjection_chain_action(surj, x)
        value = 0
        for basis_key, coeff in theta:
            contrib = coeff
            for slot in range(r):
                f = cochains[slot]
                simplex_key = (basis_key[slot],)
                matched = False
                for f_key, f_coeff in f:
                    if f_key == simplex_key:
                        contrib *= f_coeff
                        matched = True
                        break
                if not matched:
                    contrib = 0
                    break
            value += contrib
        if value != 0:
            result_dict[(simplex_tuple,)] = result_dict.get((simplex_tuple,), 0) + value

    result_dict = {k: v for k, v in result_dict.items() if v != 0}
    if not result_dict:
        return target.zero()
    return target(result_dict)


class SurjectionSimplicialCochainAlgebra(OperadAlgebra):
    """`Surjection`-algebra structure on arity-1 simplicial cochains."""

    def __init__(self, module: SimplicialCochains):
        if module.arity() != 1:
            raise ValueError(
                f"Expected arity-1 simplicial cochains module, got arity={module.arity()}."
            )
        from uconf.models.surjection import Surjection

        super().__init__(
            module=module,
            operad_cls=Surjection,
            structure_map=lambda p_element, a_list: surjection_cochain_action(
                p_element, tuple(a_list)
            ),
        )

    def coact(
        self,
        surj: "Surjection.Element",
        cochains: tuple[SimplicialCochains.Element, ...],
    ) -> SimplicialCochains.Element:
        """Compatibility alias for the operad action on cochains."""
        return self.act(surj, list(cochains))


class SurjectionSimplicialChainCoalgebra(CooperadCoalgebra):
    """`SurjectionDual`-coalgebra structure on arity-1 simplicial chains."""

    def __init__(self, module: SimplicialChains):
        if module.arity() != 1:
            raise ValueError(
                f"Expected arity-1 simplicial chains module, got arity={module.arity()}."
            )
        from uconf.models.surjection_dual import SurjectionDual

        super().__init__(
            module=module,
            cooperad_cls=SurjectionDual,
            costructure_map=self._costructure_map,
        )

    def _costructure_map(self, v_element: SimplicialChains.Element, n: int):
        if n <= 0:
            raise ValueError(f"Coaction arity must be positive, got {n}.")

        from uconf.models.surjection import Surjection
        from uconf.models.surjection_dual import SurjectionDual

        base_ring = self.module.base_ring()
        if v_element.parent().base_ring() != base_ring:
            raise TypeError("Chain element base ring does not match module base ring.")

        left_parent = SurjectionDual(n, base_ring=base_ring)
        factor = SimplicialChains(r=1, base_ring=base_ring)
        if n == 1:
            right_parent = factor
        else:
            right_parent = tensor([factor] * n)
        target = tensor([left_parent, right_parent])

        if not v_element:
            return target.zero()

        max_degree = max(
            v_element.parent().degree_on_basis(key) for key in v_element.support()
        )
        surj_parent = Surjection(n, base_ring=base_ring)

        result = target.zero()
        for degree in range(max_degree + 1):
            for u in surj_parent.basis_it(degree):
                theta = surjection_chain_action(u, v_element)
                if not theta:
                    continue
                right_elem = theta.to_native_tensor()
                right_elem = right_parent(right_elem)
                for u_basis, u_coeff in u:
                    left_elem = left_parent.term(u_basis)
                    result += u_coeff * left_elem.tensor(right_elem)
        return result

    def act(
        self,
        surj: "Surjection.Element",
        chain: SimplicialChains.Element,
        coord: int = 1,
    ) -> SimplicialChains.Element:
        """Convenience wrapper for the induced chain action."""
        return surjection_chain_action(surj, chain, coord=coord)
