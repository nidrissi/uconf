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


VertexDecoratedLike: TypeAlias = OperadLike | CooperadLike
"""Accepted decoration providers for shared tree-decorated base modules."""
