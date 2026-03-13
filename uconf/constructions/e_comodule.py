"""E_ν-comodule structure on cobar constructions of quasi-planar cooperads.

Implements the comodule map from Proposition~\\ref{prop:comodule structure}
in the article:

.. math::

    \\Delta: \\Omega(\\mathcal{C}) \\to \\mathcal{E}_\\nu \\otimes \\Omega(\\mathcal{C})

defined on planar generators
:math:`s^{-1}x \\otimes \\{\\mathrm{id}\\} \\in s^{-1}\\mathcal{C}_\\mathrm{pl}(n)`
by:

.. math::

    \\Delta(s^{-1}x \\otimes \\{\\mathrm{id}\\})
      = \\sum_{k \\ge 0} \\sum_{\\underline\\sigma \\in S_n^k}
          \\rho(\\underline\\sigma) \\otimes D_{\\underline\\sigma}(s^{-1}x \\otimes \\{\\mathrm{id}\\})

where:

- :math:`\\rho(\\underline\\sigma) = (\\mathrm{id},\\, \\sigma_k,\\,
  \\sigma_{k-1}\\sigma_k,\\, \\ldots,\\, \\sigma_1 \\cdots \\sigma_k)
  \\in \\mathcal{E}_\\mathrm{pl}(n)`,
- :math:`D_{\\underline\\sigma}(s^{-1}x \\otimes \\{\\mathrm{id}\\})
  = d_{\\underline\\sigma}(x) \\otimes \\{\\sigma_1 \\cdots \\sigma_k\\}`,
- :math:`d_{\\underline\\sigma} = d_{\\sigma_1} \\circ \\cdots \\circ d_{\\sigma_k}`
  is the iterated *d_sigma* on :math:`\\mathcal{C}`.

The sum terminates at :math:`k = \\deg_{\\mathcal{C}}(x)` since each
:math:`d_\\sigma` lowers the degree by 1, so all terms with
:math:`k > \\deg(x)` are zero.

The cooperad :math:`\\mathcal{C}` must be quasi-planar: its arity-n components
must inherit from :class:`~uconf.core.quasi_planar.QuasiPlanarMixin` and expose
``planarize``, ``boundary``, and ``d_sigma``.  The canonical case is
:class:`~uconf.constructions.bar_construction.BarConstruction` applied to a
Hadamard product :math:`P \\otimes \\mathcal{E}` or :math:`P \\otimes \\Surj`.
"""

from __future__ import annotations

from typing import Any, Optional

from sage.all import SymmetricGroup, tensor

from uconf.models.barratt_eccles import BarrattEccles


def e_comodule_on_generator(
    dec_elem: Any,
    cooperad_component: Any,
    cobar_component: Any,
    be_component: Optional[Any] = None,
) -> Any:
    """Compute the :math:`\\mathcal{E}_\\nu`-comodule map on a planar generator.

    Given a planar element *dec_elem* :math:`\\in \\mathcal{C}_\\mathrm{pl}(n)`,
    returns

    .. math::

        \\Delta(s^{-1}\\,\\text{dec\\_elem})
          = \\sum_{k \\ge 0}\\;
            \\sum_{\\underline\\sigma \\in (S_n \\setminus \\{\\mathrm{id}\\})^k}
            \\rho(\\underline\\sigma) \\otimes
            \\operatorname{cobar}(d_{\\underline\\sigma}(\\text{dec\\_elem}))
            \\cdot \\sigma_1 \\cdots \\sigma_k

    as an element of :math:`\\mathcal{E}(n) \\otimes \\Omega(\\mathcal{C})(n)`.

    Parameters
    ----------
    dec_elem :
        Planar element of *cooperad_component*.  The caller is responsible
        for ensuring planarity.
    cooperad_component :
        Arity-n component of a quasi-planar cooperad.  Must expose
        ``planarize``, ``boundary``, ``degree_on_basis``, and inherit from
        :class:`~uconf.core.quasi_planar.QuasiPlanarMixin` (hence has
        ``d_sigma``).
    cobar_component :
        Arity-n component of the cobar construction :math:`\\Omega(\\mathcal{C})`.
    be_component :
        Optional :class:`~uconf.models.barratt_eccles.BarrattEccles` component
        of the same arity and base ring.  Created automatically if omitted.

    Returns
    -------
    Element of ``tensor([be_component, cobar_component])``.

    Notes
    -----
    Sequences :math:`\\underline\\sigma` that contain the identity permutation
    give :math:`\\rho(\\underline\\sigma) = 0` (degenerate Barratt–Eccles
    element), so they are skipped.  Zero branches of :math:`d_\\sigma` are
    pruned early.  The recursion depth is bounded by
    :math:`\\deg_{\\mathcal{C}}(\\text{dec\\_elem})`.
    """
    n = cooperad_component.arity()
    base_ring = cooperad_component.base_ring()
    S_n = SymmetricGroup(n)
    identity_n = S_n.identity()

    if be_component is None:
        be_component = BarrattEccles(n, base_ring)

    target = tensor([be_component, cobar_component])

    if not dec_elem:
        return target.zero()

    # Maximum recursion depth = degree of the input element
    max_deg = max(cooperad_component.degree_on_basis(key) for key in dec_elem.support())

    result = target.zero()

    # -----------------------------------------------------------------------
    # Local helpers
    # -----------------------------------------------------------------------

    def make_cobar_generator(planar_elem):
        """Build the weight-1 cobar tree element for *planar_elem* ∈ C(n)."""
        acc = cobar_component.zero()
        for dec_key, coeff in planar_elem:
            tree = (dec_key,) + tuple(range(1, n + 1))
            acc += coeff * cobar_component.term(tree)
        return acc

    def recurse(current_d_elem, sigma_bar):
        """Accumulate contributions, pruning zero branches.

        Parameters
        ----------
        current_d_elem :
            Element :math:`d_{\\underline\\sigma}(\\text{dec\\_elem})` computed
            so far (a planar element of *cooperad_component*).
        sigma_bar :
            The list :math:`[\\sigma_1, \\ldots, \\sigma_k]` accumulated so far.
        """
        nonlocal result

        if not current_d_elem:
            return

        k = len(sigma_bar)

        # --- k = 0 term: ρ(()) = (id,), no permutation --------------------
        if k == 0:
            be_elem = be_component.term((identity_n,))
            cobar_gen = make_cobar_generator(current_d_elem)
            result += be_elem.tensor(cobar_gen)
        else:
            # ρ(σ̄) = BE_n.rho([σ_k, σ_{k-1}, ..., σ_1])
            be_elem = be_component.rho(list(reversed(sigma_bar)))

            if be_elem:
                # σ_prod = σ_1 · σ_2 · ... · σ_k
                sigma_prod = sigma_bar[0]
                for s in sigma_bar[1:]:
                    sigma_prod = sigma_prod * s

                cobar_gen = make_cobar_generator(current_d_elem)
                if sigma_prod != identity_n:
                    cobar_gen = cobar_gen.permute(sigma_prod)
                result += be_elem.tensor(cobar_gen)

        # --- Degree truncation: stop extending when depth = max_deg --------
        if k >= max_deg:
            return

        # --- Recurse: extend sigma_bar by one more non-identity perm -------
        for sigma in S_n:
            if sigma == identity_n:
                continue  # identity always makes ρ = 0
            next_d = cooperad_component.d_sigma(current_d_elem, sigma)
            if next_d:  # prune zero branches
                recurse(next_d, sigma_bar + [sigma])

    recurse(dec_elem, [])
    return result
