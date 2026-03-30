r"""Euclidean configuration model via operadic bar construction.

This module implements the *configuration model* for computing the real
homotopy type of the ordered/unordered configuration spaces of a smooth
closed manifold, following Idrissi's construction [Idrissi2022]_.

Overview
--------

The construction builds a chain complex whose homology computes the
factorization homology of a manifold model.  The pipeline is:

.. code-block:: text

    Layer 0: Basic operads
      sLie = ShiftedOperad(Lie, -1)    # s⁻¹Lie, degree shift -(n-1) at arity n
      XsLie = HadamardProduct(sLie, Surjection)   # H = s⁻¹Lie ⊙ Surjection

    Layer 1: Bar/cobar cooperad/operad
      BXsLie = BarConstruction(XsLie)   # cooperad C = B(H), connectivity -1
      OBXsLie = CobarConstruction(BXsLie)  # operad P = Ω(C), quasi-free resolution

    Layer 2: Manifold model and free algebra
      manifold_model = SurjectionSphereCochainAlgebra(dim, R)   # Surjection-algebra
      coefficients = TrivialModule(dim, R)     # k[dim], rank-1 module
      free_alg = FreeOperadAlgebra(OBXsLie, coefficients)  # free P-algebra

    Layer 3: Tensor and pullback
      tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)
         # (Surj ⊙ P)-algebra on (cochains ⊗ free_alg)
      comodule_morphism: P → Surj ⊙ P   # via E-comodule + table reduction
      pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)
         # P-algebra via pullback along comodule_morphism

    Layer 4: Final bar algebra
      pi = canonical_projection(P)   # twisting morphism π: B(P) → P
      bar = BarAlgebra(pi, pulled_back)   # B_π(pulled_back)

The result ``bar`` is a C-coalgebra whose underlying dg-module
``bar.module`` can be truncated by weight and degree to compute homology
via :func:`~uconf.homology.compute_chain_complex`.

Key classes used
~~~~~~~~~~~~~~~~

- :class:`~uconf.wrappers.shifted_operad.ShiftedOperad` — degree-shifted operad.
- :class:`~uconf.wrappers.hadamard_operad.HadamardProduct` — arity-wise tensor product of operads.
- :class:`~uconf.constructions.bar_construction.BarConstruction` — bar construction (cooperad).
- :class:`~uconf.constructions.cobar_construction.CobarConstruction` — cobar construction (operad).
- :class:`~uconf.algebraic.free_algebra.FreeOperadAlgebra` — free P-algebra.
- :class:`~uconf.algebraic.hadamard_algebra.HadamardTensorAlgebra` — tensor product algebra.
- :class:`~uconf.algebraic.pullback_algebra.PullbackAlgebra` — pullback along operad morphism.
- :class:`~uconf.constructions.bar_algebra.BarAlgebra` — twisted bar construction.

References
----------

.. [Idrissi2022] N. Idrissi, *Real Homotopy of Configuration Spaces*,
   Lecture Notes in Mathematics, vol. 2303, Springer, 2022.
"""

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.spherical import SurjectionSphereCochainAlgebra
from uconf.constructions.bar_algebra import BarAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.display import latex_linear_combination
from uconf.core.morphism import OperadMorphism
from uconf.models.lie import Lie
from uconf.models.surjection import Surjection
from uconf.morphisms.canonical_twisting import canonical_projection
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.wrappers.hadamard_operad import HadamardProduct
from uconf.wrappers.shifted_operad import ShiftedOperad


def _make_surjection_comodule_morphism(cooperad_cls) -> OperadMorphism:
    """Build the operad morphism Ω(C) → S ⊙ Ω(C).

    Composes the e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) (where
    E = BarrattEccles) with table reduction on the left factor, yielding
    a morphism into S ⊙ Ω(C) (where S = Surjection).
    """
    be_comodule = make_e_comodule_morphism(cooperad_cls)
    cobar = CobarConstruction(cooperad_cls)
    surj_target = HadamardProduct(Surjection, cobar)

    def _on_element(element):
        # Apply the BE-valued e-comodule morphism
        be_had_elem = be_comodule(element)

        # Apply table_reduction to the left (BarrattEccles) factor
        source_parent = be_had_elem.parent()
        n = source_parent.arity()
        base_ring = source_parent.base_ring()
        target_n = surj_target(n, base_ring)

        result = target_n.zero()
        be_parent = source_parent.left_parent()
        for (be_key, cobar_key), coeff in be_had_elem:
            surj_elem = be_parent(be_key).table_reduction()

            for surj_key, surj_coeff in surj_elem:
                result += coeff * surj_coeff * target_n((surj_key, cobar_key))

        return result

    return OperadMorphism(cobar, surj_target, _on_element)


def labelled_configuration_model(
    manifold_model: OperadAlgebra, coefficients: CombinatorialFreeModule
):
    r"""Build the labelled configuration model for a manifold model.

    Given a Surjection-algebra ``manifold_model`` (encoding the manifold's
    real homotopy type) and a coefficient module, this constructs the bar
    algebra whose homology computes the factorization homology.

    Parameters
    ----------
    manifold_model:
        A Surjection-algebra, e.g. from
        :class:`~uconf.algebraic.spherical.SurjectionSphereCochainAlgebra`.
    coefficients:
        A graded module with boundary and connectivity, used as the label
        space for the free algebra.

    Returns
    -------
    BarAlgebra
        The bar construction ``B_π(pulled_back)``.  Its ``.module`` gives
        the underlying dg-module for homology computation.
    """
    assert manifold_model.operad_cls == Surjection, "Manifold model must be a surjection algebra."

    base_ring = coefficients.base_ring()
    assert manifold_model.module.base_ring() == base_ring, (
        "Coefficient module must have the same base ring as the manifold model."
    )
    return _build_labelled_layers(manifold_model, coefficients)["bar"]


def _build_labelled_layers(manifold_model: OperadAlgebra, coefficients: CombinatorialFreeModule):
    """Build all intermediate layers for the labelled configuration model."""
    assert manifold_model.operad_cls == Surjection, "Manifold model must be a surjection algebra."

    base_ring = coefficients.base_ring()
    assert manifold_model.module.base_ring() == base_ring, (
        "Coefficient module must have the same base ring as the manifold model."
    )

    sLie = ShiftedOperad(Lie, -1)
    XsLie = HadamardProduct(sLie, Surjection)
    BXsLie = BarConstruction(XsLie)
    OBXsLie = CobarConstruction(BXsLie)

    free_alg = FreeOperadAlgebra(OBXsLie, coefficients)

    # The e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) is postcomposed with
    # table reduction on the left factor to obtain Δ': Ω(C) → S ⊙ Ω(C).
    # This way the tensor algebra is built over (S ⊙ Ω(C)) and the
    # manifold_model (a Surjection-algebra) is used directly.
    tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)

    comodule_morphism = _make_surjection_comodule_morphism(BXsLie)
    pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)
    pi = canonical_projection(pulled_back.operad_cls)
    bar = BarAlgebra(pi, pulled_back)

    return {
        "manifold_model": manifold_model,
        "coefficients": coefficients,
        "sLie": sLie,
        "XsLie": XsLie,
        "BXsLie": BXsLie,
        "OBXsLie": OBXsLie,
        "free_alg": free_alg,
        "tensor_alg": tensor_alg,
        "comodule_morphism": comodule_morphism,
        "pulled_back": pulled_back,
        "pi": pi,
        "bar": bar,
    }


class TrivialModule(CombinatorialFreeModule):
    """Rank-1 graded module concentrated in a single degree.

    The module has a single basis element ``'*'`` with ``degree = dimension``
    and trivial (zero) boundary.  Used as the coefficient module for
    unordered configuration models.
    """

    def __init__(self, dimension: int, base_ring):
        super().__init__(base_ring, ["*"], category=GradedModulesWithBasis(base_ring))
        self._dimension = dimension
        self.boundary = lambda _: self.zero()
        self.connectivity = 0
        self.rename(f"K[{dimension}]")

    def degree_on_basis(self, key):
        return self._dimension

    def _repr_term(self, basis_key):
        return f"*[{self._dimension}]"

    def _repr_latex(self, basis_key):
        return f"\\ast^{{{self._dimension}}}"

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            """Return a LaTeX linear-combination string for this element."""
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


def unordered_configuration_model(manifold_model: OperadAlgebra, dimension: int):
    """Build the unordered configuration model with a trivial coefficient module.

    Parameters
    ----------
    manifold_model:
        A Surjection-algebra encoding the manifold's real homotopy type.
    dimension:
        Dimension of the manifold.  Used to set the degree of the trivial
        coefficient module k[dimension].

    Returns
    -------
    BarAlgebra
        The bar construction ``B_π(pulled_back)`` with coefficients in k[d].
    """
    assert dimension >= 0, "Dimension must be non-negative."
    return _build_unordered_layers(manifold_model, dimension)["bar"]


def _build_unordered_layers(manifold_model: OperadAlgebra, dimension: int):
    """Build all intermediate layers for the unordered configuration model."""
    assert dimension >= 0, "Dimension must be non-negative."
    base_ring = manifold_model.module.base_ring()
    coefficients = TrivialModule(dimension, base_ring)
    return _build_labelled_layers(manifold_model, coefficients)


def euclidean_unordered_configuration_model(base_ring, dimension: int):
    """Build the configuration model for Euclidean space R^d.

    Uses the sphere cochain algebra as the manifold model (since
    S^d is the one-point compactification of R^d).

    Parameters
    ----------
    base_ring:
        Coefficient ring (e.g. ``QQ``, ``GF(2)``).
    dimension:
        Dimension d ≥ 0.

    Returns
    -------
    BarAlgebra
        The configuration model.  Use ``model.module`` for the
        underlying dg-module and :func:`~uconf.homology.compute_chain_complex`
        to build a chain complex for homology computation.
    """
    assert dimension >= 0, "Dimension must be non-negative."
    return _build_euclidean_layers(base_ring, dimension)["bar"]


def _build_euclidean_layers(base_ring, dimension: int):
    """Build all intermediate layers for the Euclidean unordered model."""
    assert dimension >= 0, "Dimension must be non-negative."
    manifold_model = SurjectionSphereCochainAlgebra(dimension, base_ring)
    return _build_unordered_layers(manifold_model, dimension)


def _build_layers(base_ring, dimension: int):
    """Build every intermediate object in the Euclidean configuration pipeline.

    Returns a dict mapping layer names to the constructed objects.
    """
    return _build_euclidean_layers(base_ring, dimension)
