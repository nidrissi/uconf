r"""Torus configuration model via operadic bar construction.

This module implements the *configuration model* for computing the homology
of the ordered/unordered configuration spaces of the torus T^2 = S^1 × S^1,
following the construction in `article.tex`.

The pipeline is identical to the Euclidean case in `configuration.py`, but
uses a torus cochain model instead of the sphere cochains.  Two manifold
models are supported, distinguished by their operad:

- :class:`~uconf.algebraic.torus.BarrattEcclesTorusCochainAlgebra` — the
  rank-4 tensor-square model.  Its E_∞-structure lives over the (Hopf)
  Barratt--Eccles operad and provably does not factor through the
  surjection operad, so the comodule morphism keeps its Barratt--Eccles
  left factor directly (no table reduction).
- :class:`~uconf.algebraic.torus_simplicial.SurjectionTorusSimplicialCochainAlgebra`
  — the rank-6 simplicial model, a genuine ``Surjection``-algebra.  The
  pipeline then mirrors the Euclidean case: the comodule morphism is
  table-reduced to ``Surjection ⊙ Ω(C)``, which keeps the operad factor
  small.

Overview
--------

.. code-block:: text

    Layer 0: Basic operads
      sLie = ShiftedOperad(Lie, -1)
      XsLie = HadamardProduct(sLie, Surjection)

    Layer 1: Bar/cobar cooperad/operad
      BXsLie = BarConstruction(XsLie)
      OBXsLie = CobarConstruction(BXsLie)

    Layer 2: Torus model and free algebra
      manifold_model = BarrattEcclesTorusCochainAlgebra(R)
                       # or SurjectionTorusSimplicialCochainAlgebra(R)
      coefficients = TrivialModule(2, R)     # Torus has dimension 2
      free_alg = FreeOperadAlgebra(OBXsLie, coefficients)

    Layer 3: Tensor and pullback
      tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)
      comodule_morphism: OBXsLie → BarrattEccles ⊙ OBXsLie
                         # or OBXsLie → Surjection ⊙ OBXsLie
      pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)

    Layer 4: Final bar algebra
      iota = canonical_inclusion(BXsLie)
      bar = BarAlgebra(iota, pulled_back)

The result ``bar`` is a B(sLie ⊙ Surjection)-coalgebra whose homology
computes the factorization homology of the torus.

Key classes used
~~~~~~~~~~~~~~~~

- :class:`~uconf.algebraic.torus.BarrattEcclesTorusCochainAlgebra` — Barratt--Eccles-algebra on torus.
- :class:`~uconf.wrappers.shifted_operad.ShiftedOperad` — degree-shifted operad.
- :class:`~uconf.wrappers.hadamard_operad.HadamardProduct` — arity-wise tensor product of operads.
- :class:`~uconf.constructions.bar_construction.BarConstruction` — bar construction (cooperad).
- :class:`~uconf.constructions.cobar_construction.CobarConstruction` — cobar construction (operad).
- :class:`~uconf.algebraic.free_algebra.FreeOperadAlgebra` — free P-algebra.
- :class:`~uconf.algebraic.hadamard_algebra.HadamardTensorAlgebra` — tensor product algebra.
- :class:`~uconf.algebraic.pullback_algebra.PullbackAlgebra` — pullback along operad morphism.
- :class:`~uconf.constructions.bar_algebra.BarAlgebra` — twisted bar construction.
"""

from dataclasses import dataclass

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.torus import BarrattEcclesTorusCochainAlgebra
from uconf.algebraic.torus_simplicial import SurjectionTorusSimplicialCochainAlgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.display import latex_linear_combination
from uconf.core.morphism import OperadMorphism
from uconf.core.twisting import TwistingMorphism
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.lie import Lie
from uconf.models.surjection import Surjection
from uconf.morphisms.canonical_twisting import canonical_inclusion
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.morphisms.surjection_comodule import make_surjection_comodule_morphism
from uconf.wrappers.hadamard_operad import HadamardProduct
from uconf.wrappers.shifted_operad import ShiftedOperad


@dataclass(frozen=True, slots=True)
class TorusConfigurationLayers:
    """Typed container for intermediate torus configuration-model layers."""

    manifold_model: OperadAlgebra
    coefficients: CombinatorialFreeModule
    sLie: ShiftedOperad
    XsLie: HadamardProduct
    BXsLie: BarConstruction
    OBXsLie: CobarConstruction
    free_alg: FreeOperadAlgebra
    tensor_alg: HadamardTensorAlgebra
    comodule_morphism: OperadMorphism
    pulled_back: PullbackAlgebra
    pi: TwistingMorphism
    bar: BarAlgebra


def labelled_torus_configuration_model(
    manifold_model: OperadAlgebra, coefficients: CombinatorialFreeModule
):
    r"""Build the labelled configuration model for the torus.

    Given a torus manifold model (encoding the torus's cohomology with its
    E_∞-algebra structure) and a coefficient module, this constructs the
    bar algebra whose homology computes the factorization homology.

    Parameters
    ----------
    manifold_model:
        Either a Barratt--Eccles-algebra
        (:class:`~uconf.algebraic.torus.BarrattEcclesTorusCochainAlgebra`)
        or a Surjection-algebra
        (:class:`~uconf.algebraic.torus_simplicial.SurjectionTorusSimplicialCochainAlgebra`).
    coefficients:
        A graded module with boundary and connectivity, used as the label
        space for the free algebra.

    Returns
    -------
    BarAlgebra
        The bar construction ``B_π(pulled_back)``.  Its ``.module`` gives
        the underlying dg-module for homology computation.
    """
    return _build_torus_labelled_layers(manifold_model, coefficients).bar


def _torus_comodule_morphism(manifold_model: OperadAlgebra, BXsLie) -> OperadMorphism:
    """Comodule morphism matching the manifold model's operad.

    Barratt--Eccles models keep the E-comodule morphism
    Δ: Ω(C) → E ⊙ Ω(C); Surjection models table-reduce its left factor to
    Δ': Ω(C) → S ⊙ Ω(C), as in the Euclidean pipeline.
    """
    if manifold_model.operad_cls == BarrattEccles:
        return make_e_comodule_morphism(BXsLie)
    if manifold_model.operad_cls == Surjection:
        return make_surjection_comodule_morphism(BXsLie)
    raise TypeError(
        "Torus manifold model must be a Barratt--Eccles or Surjection algebra, "
        f"got operad {manifold_model.operad_cls!r}."
    )


def _build_torus_labelled_layers(
    manifold_model: OperadAlgebra, coefficients: CombinatorialFreeModule
) -> TorusConfigurationLayers:
    """Build all intermediate layers for the labelled torus configuration model."""
    base_ring = coefficients.base_ring()
    assert manifold_model.module.base_ring() == base_ring, (
        "Coefficient module must have the same base ring as the manifold model."
    )

    sLie = ShiftedOperad(Lie, -1)
    XsLie = HadamardProduct(sLie, Surjection)
    BXsLie = BarConstruction(XsLie)
    OBXsLie = CobarConstruction(BXsLie)

    free_alg = FreeOperadAlgebra(OBXsLie, coefficients)

    tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)
    comodule_morphism = _torus_comodule_morphism(manifold_model, BXsLie)
    pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)
    iota = canonical_inclusion(BXsLie)
    bar = BarAlgebra(iota, pulled_back)

    return TorusConfigurationLayers(
        manifold_model=manifold_model,
        coefficients=coefficients,
        sLie=sLie,
        XsLie=XsLie,
        BXsLie=BXsLie,
        OBXsLie=OBXsLie,
        free_alg=free_alg,
        tensor_alg=tensor_alg,
        comodule_morphism=comodule_morphism,
        pulled_back=pulled_back,
        pi=iota,
        bar=bar,
    )


class TrivialModule(CombinatorialFreeModule):
    """Rank-1 graded module concentrated in a single degree.

    The module has a single basis element `β` with `degree = dimension`
    and trivial (zero) boundary.  Used as the coefficient module for
    configuration models.
    """

    def __init__(self, dimension: int, base_ring):
        super().__init__(base_ring, [f"β{dimension}"], category=GradedModulesWithBasis(base_ring))
        self._dimension = dimension
        self.boundary = lambda _: self.zero()
        self.connectivity = 0
        self.rename(f"K[{dimension}]")

    def degree_on_basis(self, key):
        return self._dimension

    def _repr_term(self, basis_key):
        return f"β{self._dimension}"

    def _latex_term(self, basis_key):
        return f"\\beta_{{{self._dimension}}}"

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            """Return a LaTeX linear-combination string for this element."""
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


def unordered_torus_configuration_model(base_ring):
    """Build the unordered configuration model for the torus T^2.

    Uses the rank-4 Barratt--Eccles manifold model; see
    :func:`surjection_torus_configuration_model` for the Surjection-based
    variant.

    Parameters
    ----------
    base_ring:
        Coefficient ring (e.g. ``QQ``, ``GF(2)``).

    Returns
    -------
    BarAlgebra
        The configuration model.  Use ``model.module`` for the
        underlying dg-module and :func:`~uconf.homology.compute_chain_complex`
        to build a chain complex for homology computation.

    Notes
    -----
    The torus has dimension 2, so the coefficient module is T^2[2].
    """
    return _build_unordered_torus_layers(base_ring).bar


def surjection_torus_configuration_model(base_ring):
    """Build the unordered torus configuration model over the Surjection operad.

    Same pipeline as :func:`unordered_torus_configuration_model`, but the
    manifold model is the rank-6 simplicial torus
    (:class:`~uconf.algebraic.torus_simplicial.SurjectionTorusSimplicialCochainAlgebra`)
    and the comodule morphism is table-reduced to ``Surjection ⊙ Ω(C)``,
    mirroring the Euclidean pipeline.
    """
    return _build_surjection_torus_layers(base_ring).bar


def _build_unordered_torus_layers(base_ring) -> TorusConfigurationLayers:
    """Build all intermediate layers for the unordered torus configuration model."""
    manifold_model = BarrattEcclesTorusCochainAlgebra(base_ring)
    coefficients = TrivialModule(2, base_ring)  # Torus dimension is 2
    return _build_torus_labelled_layers(manifold_model, coefficients)


def _build_surjection_torus_layers(base_ring) -> TorusConfigurationLayers:
    """Build all intermediate layers for the Surjection-based torus model."""
    manifold_model = SurjectionTorusSimplicialCochainAlgebra(base_ring)
    coefficients = TrivialModule(2, base_ring)  # Torus dimension is 2
    return _build_torus_labelled_layers(manifold_model, coefficients)


def _build_torus_layers(base_ring) -> TorusConfigurationLayers:
    """Build every intermediate object in the torus configuration pipeline.

    Returns a typed container mapping layer names to constructed objects.
    """
    return _build_unordered_torus_layers(base_ring)
