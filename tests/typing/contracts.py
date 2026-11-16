"""Static regressions; checked by test_typing.py, never executed."""

# An unused ignore is a failure: invalid calls must actually be rejected.
# pyright: reportUnnecessaryTypeIgnoreComment=error

from typing import assert_type

from sage.all import QQ, cached_function, cached_method

from uconf import CoAssociative, Lie, ShiftedCooperad, ShiftedOperad
from uconf.core.cooperad import Cooperad
from uconf.core.operad import Operad
from uconf.models.lie import LieComponent, LieElement


def needs_operad(operad: Operad) -> None: ...
def needs_cooperad(cooperad: Cooperad) -> None: ...


needs_operad(Lie)
needs_cooperad(CoAssociative)
needs_operad(CoAssociative)  # pyright: ignore[reportArgumentType]
needs_cooperad(Lie)  # pyright: ignore[reportArgumentType]
ShiftedOperad(CoAssociative, 1)  # pyright: ignore[reportArgumentType]
ShiftedCooperad(Lie, 1)  # pyright: ignore[reportArgumentType]

component = Lie(2, QQ)
assert_type(component, LieComponent)
element = component.term((1,))
assert_type(element, LieElement)
assert_type(element.parent(), LieComponent)
assert_type(element.boundary(), LieElement)
element.missing_attribute()  # pyright: ignore[reportAttributeAccessIssue]
assert_type(component._basis_keys(), list[tuple[int, ...]])
component._basis_keys(1)  # pyright: ignore[reportCallIssue]
component._assoc_from_basis_key("wrong")  # pyright: ignore[reportArgumentType]


class CachedExample:
    @cached_method
    def render(self, count: int, *, prefix: str = "") -> str:
        return prefix + str(count)

    @cached_method
    def no_args(self) -> int:
        return 1

    @cached_method(key=lambda self, count: count, do_pickle=True)
    def configured(self, count: int = 0) -> str:
        return str(count)


example = CachedExample()
assert_type(example.render(1, prefix="x"), str)
assert_type(CachedExample.render(example, 1), str)
assert_type(example.no_args(), int)
assert_type(CachedExample.no_args(example), int)
assert_type(example.configured(), str)
assert_type(example.configured(count=2), str)
assert_type(example.render.is_in_cache(1), bool)
assert_type(example.render.cached(1), str)
example.render.set_cache("cached", 1)
example.render.clear_cache()
example.no_args.set_cache(2)
assert_type(example.no_args.is_in_cache(), bool)
example.render("wrong")  # pyright: ignore[reportArgumentType]
example.render(1, prefix=2)  # pyright: ignore[reportArgumentType]
example.render()  # pyright: ignore[reportCallIssue]
example.no_args(1)  # pyright: ignore[reportCallIssue]
example.configured("wrong")  # pyright: ignore[reportArgumentType]
example.render.is_in_cache("wrong")  # pyright: ignore[reportArgumentType]
example.render.set_cache(1, 1)  # pyright: ignore[reportArgumentType]
example.no_args.set_cache("wrong")  # pyright: ignore[reportArgumentType]
CachedExample.render(example, "wrong")  # pyright: ignore[reportArgumentType]
CachedExample.render(object(), 1)  # pyright: ignore[reportArgumentType]


@cached_function
def cached_length(value: str) -> int:
    return len(value)


assert_type(cached_length("value"), int)
cached_length(1)  # pyright: ignore[reportArgumentType]
