from abc import abstractmethod
import contextlib
import dataclasses
import functools
import threading
import typing as tp

import jax

from refx import ids, tracers

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


@dataclasses.dataclass(frozen=True)
class _RefContext:
    level: int


@dataclasses.dataclass
class _Context(threading.local):
    ref_context_stack: tp.List[_RefContext] = dataclasses.field(
        default_factory=lambda: [_RefContext(0)]
    )
    is_crossing_barrier: bool = False
    unflattening_ref_cache: tp.Optional[tp.Dict["Ref[tp.Any]", "Ref[tp.Any]"]] = None
    flattening_value_cache: tp.Optional[tp.Dict["Ref[tp.Any]", tp.Any]] = None

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


def clone_references(pytree: tp.Any) -> tp.Any:
    cache: tp.Dict[Ref[tp.Any], Ref[tp.Any]] = {}

    def clone_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in cache:
                cache[ref] = ref.clone()
            return cache[ref]
        return ref

    return jax.tree_map(clone_fn, pytree)


@contextlib.contextmanager
def crossing_barrier():
    _CONTEXT.unflattening_ref_cache = {}
    _CONTEXT.flattening_value_cache = {}
    _CONTEXT.is_crossing_barrier = True
    try:
        yield
    finally:
        _CONTEXT.unflattening_ref_cache = None
        _CONTEXT.flattening_value_cache = None
        _CONTEXT.is_crossing_barrier = False


def _update_ref_context(pytree: tp.Any) -> tp.Any:
    for pytree in jax.tree_util.tree_leaves(
        pytree, is_leaf=lambda x: isinstance(x, PytreeRef)
    ):
        if isinstance(pytree, PytreeRef):
            pytree.ref._trace_level = tracers.current_trace_level()
            pytree.ref._ref_context = _CONTEXT.current_ref_context


def cross_barrier(
    decorator, *decorator_args, **decorator_kwargs
) -> tp.Callable[[F], F]:
    # decorator = e.g. jit
    # barrier = e.g. jit(inner_wrapper)
    # -----------
    # call order:
    # - - - - - -
    # outer_wrapper => barrier => inner_wrapper => f
    #
    @functools.wraps(decorator)
    def decorator_wrapper(f):
        @functools.wraps(f)
        def inner_wrapper(*args, **kwargs):
            _CONTEXT.is_crossing_barrier = False
            _CONTEXT.unflattening_ref_cache = None
            _CONTEXT.flattening_value_cache = None
            with incremented_ref():
                _update_ref_context((args, kwargs))
                out = f(*args, **kwargs)
            _CONTEXT.is_crossing_barrier = True
            _CONTEXT.unflattening_ref_cache = {}
            _CONTEXT.flattening_value_cache = {}
            return out

        barrier = decorator(inner_wrapper, *decorator_args, **decorator_kwargs)

        @functools.wraps(f)
        def outer_wrapper(*args, **kwargs):
            # gather all input references and their values
            input_id_ref_value = {
                ref.id: (ref, ref.value)
                for ref in jax.tree_util.tree_leaves((args, kwargs))
                if isinstance(ref, Ref)
            }
            output_ids_checklist = set(input_id_ref_value)

            try:
                with crossing_barrier():
                    out = barrier(*args, **kwargs)

                # Update input reference with the values from the output
                # reference and return the input reference.
                def update_inputs_refs(output_ref: tp.Any) -> tp.Any:
                    if (
                        isinstance(output_ref, Ref)
                        and output_ref.id in input_id_ref_value
                    ):
                        input_ref = input_id_ref_value[output_ref.id][0]
                        input_ref.value = output_ref.value
                        output_ids_checklist.discard(output_ref.id)
                        return input_ref

                    return output_ref

                out = jax.tree_map(update_inputs_refs, out)

                # All input references must have a corresponding output reference that
                # has the input reference as a parent.
                if output_ids_checklist:
                    raise ValueError(
                        f"Could not find references {output_ids_checklist} on the output."
                        "All input references must be outputs of the function."
                    )
            except:
                # Restore input references values in case of failure
                for ref, value in input_id_ref_value.values():
                    ref.value = value
                raise

            return out

        return outer_wrapper

    return decorator_wrapper


class Ref(tp.Generic[A]):
    def __init__(self, value: A, id: tp.Optional[ids.Id] = None):
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


class Base(tp.Generic[A]):
    @abstractmethod
    def __init__(self, value: A):
        ...

    @property
    @abstractmethod
    def value(self) -> A:
        ...

    @value.setter
    @abstractmethod
    def value(self, value: A):
        ...


class PytreeValue(Base[A]):
    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value

    @value.setter
    def value(self, value: A):
        raise AttributeError(f"Cannot mutate PytreeValue '{type(self).__name__}'")


def flatten_pytree_value(pytree: PytreeValue[A]) -> tp.Tuple[tp.Tuple[A], None]:
    return (pytree.value,), None


def unflatten_pytree_value(_, children: tp.Tuple[A]) -> PytreeValue[A]:
    return PytreeValue(children[0])


class PytreeRef(Base[A]):
    def __init__(self, ref_or_value: tp.Union[Ref[A], A]):
        if isinstance(ref_or_value, Ref):
            self._ref = ref_or_value
        else:
            self._ref = Ref(ref_or_value)

    @property
    def id(self) -> ids.Id:
        return self.ref.id

    @property
    def ref(self) -> Ref[A]:
        return self._ref

    @property
    def value(self) -> A:
        return self.ref.value

    @value.setter
    def value(self, value: A):
        self.ref.value = value


class _Missing:
    pass


MISSING = _Missing()


def flatten_pytree_ref(
    pytree: PytreeRef[A],
) -> tp.Tuple[tp.Tuple[tp.Union[A, Ref[A]]], tp.Optional[Ref[A]]]:
    if _CONTEXT.is_crossing_barrier:
        assert _CONTEXT.flattening_value_cache is not None
        ref = pytree.ref
        if ref.value is not MISSING:
            _CONTEXT.flattening_value_cache[ref] = ref.value
        value = _CONTEXT.flattening_value_cache[ref]
        ref._value = MISSING
        return (value,), ref
    else:
        return (pytree.ref,), None


def unflatten_pytree_ref(
    ref: tp.Optional[Ref[A]], children: tp.Tuple[tp.Union[A, Ref[A]]]
) -> PytreeRef[A]:
    value = children[0]

    if _CONTEXT.is_crossing_barrier:
        assert _CONTEXT.unflattening_ref_cache is not None
        assert not isinstance(value, Ref)
        assert ref is not None
        if ref not in _CONTEXT.unflattening_ref_cache:
            _CONTEXT.unflattening_ref_cache[ref] = Ref(value, id=ref.id)
        new_ref = _CONTEXT.unflattening_ref_cache[ref]
        return PytreeRef(new_ref)
    else:
        assert isinstance(value, Ref)
        return PytreeRef(value)


jax.tree_util.register_pytree_node(PytreeRef, flatten_pytree_ref, unflatten_pytree_ref)
