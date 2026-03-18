"""Shared typing alias for tree-vertex decoration providers.

The shared tree-decorated base modules
:class:`uconf.algebraic.free_algebra.FreeAlgebraModule` and
:class:`uconf.algebraic.cofree_coalgebra.CofreeCoalgebraModule`
only require arity-indexed components with grading and differential.

In practice, both operad-like and cooperad-like providers are used.
This module defines a common alias for those accepted inputs.
"""

from __future__ import annotations

from typing import TypeAlias

from uconf.core.cooperad import CooperadLike
from uconf.core.operad import OperadLike


VertexDecorationLike: TypeAlias = OperadLike | CooperadLike
"""Accepted decoration providers for shared tree-decorated base modules."""

QuasiPlanarLike: TypeAlias = OperadLike | CooperadLike
"""Accepted decoration providers for free/cofree algebra composite modules.

A quasi-planar operad/cooperad is one whose arity-n component ``P(n)``
satisfies ``P(n) ≅ P_pl(n) ⊗ k[S_n]`` and exposes a ``planarize`` linear
map decomposing each element into its planar representative and symmetric-group
factor.

Supported examples: :class:`~uconf.models.associative.Associative`,
:class:`~uconf.models.surjection.Surjection`,
:class:`~uconf.models.barratt_eccles.BarrattEccles`, and wrappers thereof.

Non-quasi-planar operads (e.g. :class:`~uconf.models.commutative.Commutative`,
:class:`~uconf.models.lie.Lie`) are **not** accepted; they have non-trivial
coinvariants and cannot be enumerated as ``P_pl(n) ⊗ M^{⊗n}``.
"""
