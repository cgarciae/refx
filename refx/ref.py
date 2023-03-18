import dataclasses
import typing as tp

import jax

from refx import tracers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jax.tree_util.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


@dataclasses.dataclass(frozen=True)
class ValueIndex:
    value: tp.Any
    index: int
    ref_type: type


def _value_index_flatten(
    x: ValueIndex,
) -> tp.Tuple[tp.Tuple[tp.Any], tp.Tuple[int, type]]:
    return (x.value,), (x.index, x.ref_type)


def _value_index_flatten_with_keys(
    x: ValueIndex,
) -> tp.Tuple[tp.Tuple[tp.Any], tp.Tuple[int, type]]:
    return ((jax.tree_util.GetAttrKey("value"), x.value),), (x.index, x.ref_type)


def _value_index_unflatten(aux_data: tp.Tuple[int, type], children: tp.Tuple[tp.Any]):
    return ValueIndex(children[0], *aux_data)


if hasattr(jax.tree_util, "register_pytree_with_keys"):
    jax.tree_util.register_pytree_with_keys(
        ValueIndex,
        _value_index_flatten_with_keys,
        _value_index_unflatten,
    )
else:
    jax.tree_util.register_pytree_node(
        ValueIndex,
        _value_index_flatten,
        _value_index_unflatten,
    )


class Ref(tp.Generic[A]):
    def __init__(self, value: A):
        self._value = value
        self._trace = tracers.current_trace()

    @property
    def value(self) -> A:
        return self._value

    @value.setter
    def value(self, value: A):
        if self._trace is not tracers.current_trace():
            raise ValueError("Cannot mutate ref from different trace level")
        self._value = value


def deref(pytree: A) -> A:
    if isinstance(pytree, Ref):
        return ValueIndex(pytree.value, index=0, ref_type=type(pytree))

    ref_index: tp.Dict[Ref[tp.Any], int] = {}

    def deref_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in ref_index:
                ref_index[ref] = len(ref_index)
                return ValueIndex(ref.value, index=ref_index[ref], ref_type=type(ref))
            else:
                return ValueIndex(NOTHING, index=ref_index[ref], ref_type=type(ref))
        return ref

    return jax.tree_map(deref_fn, pytree)


def reref(pytree: A) -> A:
    if isinstance(pytree, ValueIndex):
        return Ref(pytree.value)

    index_ref: tp.Dict[int, Ref[tp.Any]] = {}

    def reref_fn(value_index: tp.Any) -> tp.Any:
        if isinstance(value_index, ValueIndex):
            # NOTE: because pytree flatten and unflatten in a deterministic
            # order, this should probably never trigger
            if value_index.index not in index_ref:
                assert value_index.value is not NOTHING
                index_ref[value_index.index] = Ref(value_index.value)
            elif value_index.value is not NOTHING:
                # index_ref[value_index.index].value = value_index.value
                raise RuntimeError(
                    "BUG: Got multiple values for the same index. This should never "
                    "happen, please report it."
                )
            ref = index_ref[value_index.index]
            return ref
        return value_index

    return jax.tree_map(reref_fn, pytree, is_leaf=lambda x: isinstance(x, ValueIndex))
