import contextlib
import dataclasses
import functools
import threading
import typing as tp

import jax

from refx import ids, tracers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])
H = tp.TypeVar("H", bound=tp.Callable[..., tp.Any])


class Nothing:
    pass


NOTHING = Nothing()


@dataclasses.dataclass(frozen=True)
class _RefContext:
    level: int


@dataclasses.dataclass
class _Context(threading.local):
    ref_context_stack: tp.List[_RefContext] = dataclasses.field(
        default_factory=lambda: [_RefContext(0)]
    )
    barrier_pytree: tp.Any = NOTHING
    # is_crossing_barrier: bool = False
    # unflattening_ref_cache: tp.Optional[tp.Dict["Ref[tp.Any]", "Ref[tp.Any]"]] = None
    # flattening_value_cache: tp.Optional[tp.Dict["Ref[tp.Any]", tp.Any]] = None

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
    def __init__(self, value: A, *, id: tp.Optional[ids.Id] = None):
        self._value = value
        self._id = ids.uuid() if id is None else id
        self._ref_context = _CONTEXT.current_ref_context
        self._trace_level = tracers.current_trace_level()

    @property
    def id(self) -> ids.Id:
        return self._id

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


def clone_references(pytree: tp.Any) -> tp.Any:
    cache: tp.Dict[Ref[tp.Any], Ref[tp.Any]] = {}

    def clone_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in cache:
                cache[ref] = ref.clone()
            return cache[ref]
        return ref

    return jax.tree_map(clone_fn, pytree)


# @contextlib.contextmanager
# def crossing_barrier():
#     _CONTEXT.unflattening_ref_cache = {}
#     _CONTEXT.flattening_value_cache = {}
#     _CONTEXT.is_crossing_barrier = True
#     try:
#         yield
#     finally:
#         _CONTEXT.unflattening_ref_cache = None
#         _CONTEXT.flattening_value_cache = None
#         _CONTEXT.is_crossing_barrier = False


# def _update_ref_context(pytree: tp.Any) -> tp.Any:
#     for pytree in jax.tree_util.tree_leaves(
#         pytree, is_leaf=lambda x: isinstance(x, PytreeRef)
#     ):
#         if isinstance(pytree, PytreeRef):
#             pytree.ref._trace_level = tracers.current_trace_level()
#             pytree.ref._ref_context = _CONTEXT.current_ref_context


def deref(
    pytree: A, callback: tp.Optional[tp.Callable[[Ref[tp.Any]], None]] = None
) -> A:
    def deref_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if callback is not None:
                callback(ref)
            return ref.value
        return ref

    return jax.tree_map(deref_fn, pytree)


def reref(
    values: A,
    refs: A,
    callback: tp.Optional[tp.Callable[[Ref[tp.Any]], Ref[tp.Any]]] = None,
) -> A:
    ref_cache: tp.Dict[Ref[tp.Any], Ref[tp.Any]] = {}

    def reref_fn(value: tp.Any, ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            assert not isinstance(value, Ref)
            if ref not in ref_cache:
                ref_cache[ref] = Ref(value, id=ref.id)
            ref = ref_cache[ref]
            if callback is not None:
                ref = callback(ref)
            return ref
        return value

    return jax.tree_map(reref_fn, values, refs)


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
                args, kwargs = reref((args, kwargs), _CONTEXT.barrier_pytree)
                _CONTEXT.barrier_pytree = NOTHING
                out = f(*args, **kwargs)

            # add the refs from the output to the cache
            _CONTEXT.barrier_pytree = out
            # dereference the output so it can be returned
            out = deref(out)
            return out

        barrier = decorator(inner_wrapper, *decorator_args, **decorator_kwargs)

        @functools.wraps(f)
        def outer_wrapper(*args, **kwargs) -> B:
            # gather all input references and their values
            input_id_ref: tp.Dict[ids.Id, Ref[tp.Any]] = {}

            def add_refs(ref: Ref[tp.Any]):
                input_id_ref[ref.id] = ref

            # put original input references in the context
            _CONTEXT.barrier_pytree = (args, kwargs)
            # dereference inputs so they can pass through the barrier
            args, kwargs = deref((args, kwargs), callback=add_refs)
            try:
                out = barrier(*args, **kwargs)

                def update_inputs_refs(output_ref: Ref[A]) -> Ref[A]:
                    # if the output reference has the same id as an input reference
                    # update the input reference with the value from the output
                    # reference, and return the input reference to form part of
                    # the 'out' pytree
                    if output_ref.id in input_id_ref:
                        input_ref = input_id_ref[output_ref.id]
                        input_ref.value = output_ref.value
                        return input_ref
                    # otherwise, return the output reference
                    return output_ref

                out = reref(out, _CONTEXT.barrier_pytree, callback=update_inputs_refs)
            finally:
                _CONTEXT.barrier_pytree = NOTHING

            return out

        return outer_wrapper

    return decorator_wrapper


# class Base(tp.Generic[A]):
#     @abstractmethod
#     def __init__(self, value: A):
#         ...

#     @property
#     @abstractmethod
#     def value(self) -> A:
#         ...

#     @value.setter
#     @abstractmethod
#     def value(self, value: A):
#         ...


# class PytreeValue(Base[A]):
#     def __init__(self, value: A):
#         self._value = value

#     @property
#     def value(self) -> A:
#         return self._value

#     @value.setter
#     def value(self, value: A):
#         raise AttributeError(f"Cannot mutate PytreeValue '{type(self).__name__}'")


# def flatten_pytree_value(pytree: PytreeValue[A]) -> tp.Tuple[tp.Tuple[A], None]:
#     return (pytree.value,), None


# def unflatten_pytree_value(_, children: tp.Tuple[A]) -> PytreeValue[A]:
#     return PytreeValue(children[0])


# class PytreeRef(Base[A]):
#     def __init__(self, ref_or_value: tp.Union[Ref[A], A]):
#         if isinstance(ref_or_value, Ref):
#             self._ref = ref_or_value
#         else:
#             self._ref = Ref(ref_or_value)

#     @property
#     def id(self) -> ids.Id:
#         return self.ref.id

#     @property
#     def ref(self) -> Ref[A]:
#         return self._ref

#     @property
#     def value(self) -> A:
#         return self.ref.value

#     @value.setter
#     def value(self, value: A):
#         self.ref.value = value


# class _Missing:
#     pass


# MISSING = _Missing()


# def flatten_pytree_ref(
#     pytree: PytreeRef[A],
# ) -> tp.Tuple[tp.Tuple[tp.Union[A, Ref[A]]], tp.Optional[Ref[A]]]:
#     if _CONTEXT.is_crossing_barrier:
#         assert _CONTEXT.flattening_value_cache is not None
#         ref = pytree.ref
#         if ref.value is not MISSING:
#             _CONTEXT.flattening_value_cache[ref] = ref.value
#         value = _CONTEXT.flattening_value_cache[ref]
#         ref._value = MISSING
#         return (value,), ref
#     else:
#         return (pytree.ref,), None


# def unflatten_pytree_ref(
#     ref: tp.Optional[Ref[A]], children: tp.Tuple[tp.Union[A, Ref[A]]]
# ) -> PytreeRef[A]:
#     value = children[0]

#     if _CONTEXT.is_crossing_barrier:
#         assert _CONTEXT.unflattening_ref_cache is not None
#         assert not isinstance(value, Ref)
#         assert ref is not None
#         if ref not in _CONTEXT.unflattening_ref_cache:
#             _CONTEXT.unflattening_ref_cache[ref] = Ref(value, id=ref.id)
#         new_ref = _CONTEXT.unflattening_ref_cache[ref]
#         return PytreeRef(new_ref)
#     else:
#         assert isinstance(value, Ref)
#         return PytreeRef(value)


# jax.tree_util.register_pytree_node(PytreeRef, flatten_pytree_ref, unflatten_pytree_ref)
