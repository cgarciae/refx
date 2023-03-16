import contextlib
import dataclasses
import functools
import threading
import typing as tp

import jax
import simple_pytree as spt

from refx import tracers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


class Nothing:
    pass


def _flatten_nothing(_: Nothing) -> tp.Tuple[tp.Tuple[()], None]:
    return (), None


def _unflatten_nothing(_: None, __: tp.Tuple[()]) -> Nothing:
    return Nothing()


jax.tree_util.register_pytree_node(Nothing, _flatten_nothing, _unflatten_nothing)


@dataclasses.dataclass(frozen=True)
class _RefContext:
    level: int


@dataclasses.dataclass
class _Context(threading.local):
    ref_context_stack: tp.List[_RefContext] = dataclasses.field(
        default_factory=lambda: [_RefContext(0)]
    )

    @property
    def current_ref_context(self) -> _RefContext:
        return self.ref_context_stack[-1]


_CONTEXT = _Context()


@contextlib.contextmanager
def incremented_ref():
    _CONTEXT.ref_context_stack.append(
        _RefContext(_CONTEXT.current_ref_context.level + 1)
    )
    try:
        yield
    finally:
        _CONTEXT.ref_context_stack.pop()


class Ref(tp.Generic[A]):
    def __init__(self, value: A):
        self._value = value
        self._ref_context = _CONTEXT.current_ref_context
        self._trace_level = tracers.current_trace_level()

    @property
    def value(self) -> A:
        return self._value

    @value.setter
    def value(self, value: A):
        if self._ref_context is not _CONTEXT.current_ref_context:
            raise ValueError("Cannot mutate ref from different context")
        if self._trace_level != tracers.current_trace_level():
            raise ValueError("Cannot mutate ref from different trace level")
        self._value = value

    def clone(self) -> "Ref[A]":
        if self._ref_context.level > _CONTEXT.current_ref_context.level:
            raise ValueError("Cannot clone ref from higher context level")
        if self._trace_level > tracers.current_trace_level():
            raise ValueError("Cannot clone ref from higher trace level")
        return Ref(self.value)


@dataclasses.dataclass
class ValueIndex(spt.Pytree):
    value: tp.Any
    index: int = spt.static_field()


def clone_references(pytree: tp.Any) -> tp.Any:
    cache: tp.Dict[Ref[tp.Any], Ref[tp.Any]] = {}

    def clone_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in cache:
                cache[ref] = ref.clone()
            return cache[ref]
        return ref

    return jax.tree_map(clone_fn, pytree)


def deref(pytree: A) -> A:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}

    def deref_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in ref_index:
                ref_index[ref] = len(ref_index)
                return ValueIndex(ref.value, index=ref_index[ref])
            else:
                return ValueIndex(Nothing(), index=ref_index[ref])
        return ref

    return jax.tree_map(deref_fn, pytree)


def reref(pytree: A) -> A:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}

    def reref_fn(value_index: tp.Any) -> tp.Any:
        if isinstance(value_index, ValueIndex):
            # NOTE: because pytree flatten and unflatten in a deterministic
            # order, this should probably never trigger
            if value_index.index not in index_ref:
                assert not isinstance(value_index.value, Nothing)
                index_ref[value_index.index] = Ref(value_index.value)
            elif not isinstance(value_index.value, Nothing):
                # index_ref[value_index.index].value = value_index.value
                raise RuntimeError(
                    "BUG: Got multiple values for the same index. This should never "
                    "happen, please report it."
                )
            ref = index_ref[value_index.index]
            return ref
        return value_index

    return jax.tree_map(reref_fn, pytree, is_leaf=lambda x: isinstance(x, ValueIndex))


def cross_barrier(decorator: F, *decorator_args, **decorator_kwargs) -> F:
    # decorator = e.g. jit
    # barrier = e.g. jit(inner_wrapper)
    # -----------
    # call order:
    # - - - - - -
    # outer_wrapper => barrier => inner_wrapper => f
    @functools.wraps(decorator)
    def decorator_wrapper(f: tp.Callable[..., B]) -> tp.Callable[..., B]:
        @functools.wraps(f)
        def inner_wrapper(*args, **kwargs):
            with incremented_ref():
                # rereference inputs
                args, kwargs = reref((args, kwargs))
                out = f(*args, **kwargs)
            # dereference the output so it can be returned through the barrier
            out = deref(out)
            return out

        barrier = decorator(inner_wrapper, *decorator_args, **decorator_kwargs)

        @functools.wraps(f)
        def outer_wrapper(*args, **kwargs) -> B:
            # dereference inputs so they can pass through the barrier
            args, kwargs = deref((args, kwargs))
            out = barrier(*args, **kwargs)
            # rereference the output
            out = reref(out)
            return out

        return outer_wrapper

    return decorator_wrapper
