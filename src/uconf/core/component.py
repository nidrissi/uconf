"""Base classes shared by every operad and cooperad in this package.

An operad (or cooperad) is an *instance* of a :class:`SymmetricSequence`
subclass, e.g. ``Lie`` or ``ShiftedOperad(Lie, 1)``.  Calling it with an arity
and a base ring returns the arity-``n`` component, a Sage
``CombinatorialFreeModule`` deriving from :class:`ComponentBase`::

    Lie                    # LieOperad instance
    Lie(3, QQ)             # LieComponent, a CombinatorialFreeModule
    Lie(3, QQ)((1, 2))     # LieElement

Each concrete sequence ``XOperad`` comes with module-level classes
``XComponent`` and ``XElement``, exposed as ``XOperad.Component`` and
``XOperad.Element``; ``XComponent.Element = XElement`` tells Sage which class
to build elements with.  (The classes are not nested inside ``XOperad`` because
type checkers cannot resolve generic arguments that refer back into the
enclosing class.)

Static typing
-------------
``CombinatorialFreeModule`` inherits from Cython classes that ship without
type stubs, and much of its API is injected at runtime by Sage's category
framework, so type checkers cannot see it.  The ``if TYPE_CHECKING:`` blocks in
:class:`ComponentBase` and :class:`ElementBase` declare the part of that API
this package uses, with precise types.  They have no runtime effect: Sage's own
implementations are always the ones that run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    UniqueRepresentation,
    cached_method,
)

from uconf.core.display import latex_linear_combination


class SymmetricSequence[C: ComponentBase[Any]](UniqueRepresentation):
    """A sequence of graded modules ``M(n)``, ``n >= 0``, given aritywise.

    Subclasses set :attr:`name`, :attr:`connectivity`, :attr:`Component` (a
    :class:`ComponentBase` subclass) and :attr:`Element` (an
    :class:`ElementBase` subclass).  :class:`~uconf.core.operad.Operad` and
    :class:`~uconf.core.cooperad.Cooperad` add the structure maps.
    """

    name: str
    """Display name, also used as the prefix of component names."""

    connectivity: int
    """Constant ``k`` such that ``M(n)`` is concentrated in degrees ``>= k (n - 1)``."""

    Component: type[C]
    """Class of the fixed-arity components."""

    Element: type[ElementBase[Any]]
    """Class of the elements of the components."""

    def __call__(self, n: int, base_ring: Any) -> C:
        """Return the arity-``n`` component over ``base_ring``."""
        return self.Component(self, n, base_ring)

    @property
    def __name__(self) -> str:
        # Lets pytest ids and error messages print ``Lie`` rather than an index.
        return self.name

    def __repr__(self) -> str:
        return self.name

    def _repr_latex_(self) -> str:
        return self.name

    def unit_key(self) -> Any:
        """Return the basis key of the (co)unit generator in arity ``1``."""
        raise NotImplementedError


class ComponentBase[E: ElementBase[Any]](CombinatorialFreeModule):
    """Fixed-arity component ``M(n)`` of a :class:`SymmetricSequence`.

    The type parameter ``E`` is the element class.  Subclasses whose code
    relies on a specific owning sequence narrow :attr:`factory` with an
    ``if TYPE_CHECKING:`` property override.

    Subclasses must implement ``degree_on_basis``, ``basis_iter`` and
    ``_validate_basis_key``, and assign ``self.boundary`` (usually through
    ``self.module_morphism``) in their ``__init__``.
    """

    boundary: Callable[[E], E]
    """The differential, a linear map ``M(n) -> M(n)``."""

    def __init__(self, factory: SymmetricSequence[Any], n: int, base_ring: Any):
        assert n >= 0, f"Arity must be non-negative. Got {n}."
        self._factory = factory
        self._arity = int(n)
        name = f"{factory.name}{n}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._symmetric_group = SymmetricGroup(self._arity)

    @property
    def factory(self) -> SymmetricSequence[Any]:
        """The operad or cooperad this component belongs to."""
        return self._factory

    @property
    def name(self) -> str:
        """Name of the owning sequence."""
        return self.factory.name

    @property
    def connectivity(self) -> int:
        """Connectivity of the owning sequence."""
        return self.factory.connectivity

    def arity(self) -> int:
        """Return the arity of this component."""
        return self._arity

    def unit_key(self) -> Any:
        """Return the basis key of the (co)unit generator in arity ``1``."""
        return self.factory.unit_key()

    @cached_method
    def graded_basis(self, d: int) -> Family:
        """Return the ``Family`` of all basis elements in degree ``d``."""
        return Family(self.basis_iter(d))

    def _validate_basis_key(self, basis_key: Any, /) -> Any:
        """Validate and normalize a basis key; ``None`` means the key is zero."""
        return basis_key

    def _element_constructor_(self, x: Any) -> E:
        """Build an element from a basis key or a ``{basis_key: coeff}`` dict.

        Keys that validate to ``None`` (degenerate keys) contribute zero.
        """
        if isinstance(x, dict):
            R = self.base_ring()
            terms = []
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is not None:
                    terms.append((clean_key, R(coeff)))
            return self.sum_of_terms(terms)

        if isinstance(x, (tuple, list)):
            clean_key = self._validate_basis_key(x)
            if clean_key is None:
                return self.zero()
            return self.term(clean_key)

        raise TypeError(f"Expected dict or tuple/list; got {type(x).__name__}: {x!r}.")

    def _coerce_permutation(self, sigma: Any) -> Any:
        """Return ``sigma`` as an element of ``S_n``.

        Accepts one-line notation as a list or tuple, or an element of the
        symmetric group of this component.
        """
        if isinstance(sigma, (list, tuple)):
            return self._symmetric_group(sigma)
        if not (hasattr(sigma, "parent") and sigma.parent() == self._symmetric_group):
            raise TypeError(
                f"Permutation must be a list, tuple, or S_{self.arity()} element; "
                f"got {type(sigma).__name__}: {sigma!r}."
            )
        return sigma

    if TYPE_CHECKING:

        def __call__(self, x: Any = 0, *args: Any, **kwds: Any) -> E: ...
        def base_ring(self) -> Any: ...
        def basis(self) -> Any: ...
        def basis_iter(self, d: int) -> Iterator[E]: ...
        def degree_on_basis(self, basis_key: Any, /) -> int: ...
        def term(self, index: Any, coeff: Any = None) -> E: ...
        def monomial(self, index: Any) -> E: ...  # pyright: ignore[reportIncompatibleVariableOverride]
        def sum_of_terms(self, terms: Iterable[tuple[Any, Any]], distinct: bool = False) -> E: ...
        def zero(self) -> E: ...  # pyright: ignore[reportIncompatibleVariableOverride]
        def _from_dict(
            self, d: Mapping[Any, Any], coerce: bool = False, remove_zeros: bool = True
        ) -> E: ...
        def module_morphism(self, *args: Any, **kwds: Any) -> Any: ...
        def tensor(self, *others: Any, **kwds: Any) -> Any: ...
        def rename(self, x: str | None = None) -> None: ...
        def _repr_term(self, basis_key: Any, /) -> str: ...
        def _latex_term(self, basis_key: Any, /) -> str: ...


class ElementBase[P: ComponentBase[Any]](CombinatorialFreeModule.Element):
    """Element of a fixed-arity component ``P``.

    A sparse linear combination of basis keys, iterable over
    ``(basis_key, coefficient)`` pairs.  Subclasses implement :meth:`permute`.
    """

    def _repr_latex_(self) -> str:
        return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

    def arity(self) -> int:
        """Return the arity of this element."""
        return self.parent().arity()

    def boundary(self) -> Self:
        """Apply the differential."""
        return self.parent().boundary(self)

    def permute(self, sigma: Any) -> Self:
        """Return the right action of the permutation ``sigma`` on this element."""
        raise NotImplementedError

    if TYPE_CHECKING:

        def parent(self) -> P: ...
        def __iter__(self) -> Iterator[tuple[Any, Any]]: ...
        def __bool__(self) -> bool: ...
        def __add__(self, other: Self) -> Self: ...
        def __sub__(self, other: Self) -> Self: ...
        def __neg__(self) -> Self: ...
        def __mul__(self, other: Any) -> Self: ...
        def __rmul__(self, other: Any) -> Self: ...
        def support(self) -> list[Any]: ...
        def monomial_coefficients(self, copy: bool = True) -> dict[Any, Any]: ...
        def coefficient(self, m: Any) -> Any: ...
        def is_zero(self) -> bool: ...
        def tensor(self, *others: Any) -> Any: ...
        def leading_support(self, *args: Any, **kwds: Any) -> Any: ...
