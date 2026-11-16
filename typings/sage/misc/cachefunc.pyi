"""Signature-preserving types for Sage's cache decorators and callers."""

from collections.abc import Callable, Iterable
from typing import Any, Concatenate, ParamSpec, TypeVar, overload

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")

def dict_key(o: Any) -> Any: ...
def cache_key(o: Any) -> Any: ...

class CachedFunction[**P, R]:
    def __init__(
        self,
        f: Callable[P, R],
        *,
        classmethod: bool = False,
        name: str | None = None,
        key: Callable[..., Any] | None = None,
        do_pickle: bool = False,
    ) -> None: ...
    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R: ...
    def cached(self, *args: P.args, **kwds: P.kwargs) -> R: ...
    def is_in_cache(self, *args: P.args, **kwds: P.kwargs) -> bool: ...
    def set_cache(self, value: R, /, *args: P.args, **kwds: P.kwargs) -> None: ...
    def get_key(self, *args: P.args, **kwds: P.kwargs) -> Any: ...
    def clear_cache(self) -> None: ...
    def precompute(self, arglist: Iterable[Any], num_processes: int = 1) -> None: ...
    cache: dict[Any, R]
    f: Callable[P, R]

class CachedMethodCaller[**P, R](CachedFunction[P, R]): ...

class CachedMethod[T, **P, R]:
    def __init__(
        self,
        f: Callable[Concatenate[T, P], R],
        name: str | None = None,
        key: Callable[..., Any] | None = None,
        do_pickle: bool = False,
    ) -> None: ...
    def __call__(self, inst: T, /, *args: P.args, **kwds: P.kwargs) -> R: ...
    @overload
    def __get__(self, inst: None, cls: type[T]) -> CachedMethod[T, P, R]: ...
    @overload
    def __get__(self, inst: T, cls: type[T] | None = None) -> CachedMethodCaller[P, R]: ...

@overload
def cached_method[T, **P, R](
    f: Callable[Concatenate[T, P], R],
    name: str | None = None,
    key: Callable[..., Any] | None = None,
    do_pickle: bool = False,
) -> CachedMethod[T, P, R]: ...
@overload
def cached_method(
    f: None = None,
    name: str | None = None,
    key: Callable[..., Any] | None = None,
    do_pickle: bool = False,
) -> Callable[[Callable[Concatenate[_T, _P], _R]], CachedMethod[_T, _P, _R]]: ...
@overload
def cached_function[**P, R](
    f: Callable[P, R],
    *,
    name: str | None = None,
    key: Callable[..., Any] | None = None,
    do_pickle: bool = False,
) -> CachedFunction[P, R]: ...
@overload
def cached_function(
    f: None = None,
    *,
    name: str | None = None,
    key: Callable[..., Any] | None = None,
    do_pickle: bool = False,
) -> Callable[[Callable[_P, _R]], CachedFunction[_P, _R]]: ...

# The remainder of Sage's cache module is outside this local stub's scope.
def __getattr__(name: str) -> Any: ...
