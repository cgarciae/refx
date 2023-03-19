from abc import ABC, abstractmethod
import dataclasses
from functools import partial
import typing as tp

import jax

from refx import tracers

A = tp.TypeVar("A")
A_cov = tp.TypeVar("A_cov", covariant=True)
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
Leaf = tp.Any
Leaves = tp.List[Leaf]


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jax.tree_util.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class Ref(tp.Generic[A]):
    __slots__ = ("_value", "_trace")

    def __init__(self, value: A):
        self._value = value
        self._trace = tracers.current_trace()

    @property
    def value(self) -> A:
        if self._trace is not tracers.current_trace():
            raise ValueError("Cannot access ref from different trace level")
        return self._value

    @value.setter
    def value(self, value: A):
        if self._trace is not tracers.current_trace():
            raise ValueError("Cannot mutate ref from different trace level")
        self._value = value

    def __new__(cls, value: A):
        if cls is Ref:
            raise TypeError(
                "Cannot instantiate Ref directly, create a subclass instead"
            )
        return super().__new__(cls)


class AnyRef(Ref[A]):
    pass


@tp.runtime_checkable
class Deref(tp.Protocol, tp.Generic[A]):
    @property
    def index(self) -> int:
        ...

    @property
    def ref_type(self) -> tp.Type[Ref[A]]:
        ...


class Value(tp.Generic[A]):
    __slots__ = ("_value", "_index", "_ref_type")

    def __init__(self, value: A, index: int, ref_type: type):
        self._value = value
        self._index = index
        self._ref_type = ref_type

    @property
    def value(self) -> A:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    @property
    def ref_type(self) -> tp.Type[Ref[A]]:
        return self._ref_type


def _value_index_flatten(
    x: Value[A],
) -> tp.Tuple[tp.Tuple[A], tp.Tuple[int, type]]:
    return (x.value,), (x.index, x.ref_type)


def _value_index_flatten_with_keys(
    x: Value[A],
) -> tp.Tuple[tp.Tuple[tp.Tuple[tp.Any, A]], tp.Tuple[int, type]]:
    return ((jax.tree_util.GetAttrKey("value"), x.value),), (x.index, x.ref_type)


def _value_index_unflatten(aux_data: tp.Tuple[int, type], children: tp.Tuple[tp.Any]):
    return Value(children[0], *aux_data)


if hasattr(jax.tree_util, "register_pytree_with_keys"):
    jax.tree_util.register_pytree_with_keys(
        Value, _value_index_flatten_with_keys, _value_index_unflatten
    )
else:
    jax.tree_util.register_pytree_node(
        Value, _value_index_flatten, _value_index_unflatten
    )


class Index(tp.Generic[A]):
    __slots__ = ("_index", "_ref_type")

    def __init__(self, index: int, ref_type: type):
        self._index = index
        self._ref_type = ref_type

    @property
    def index(self) -> int:
        return self._index

    @property
    def ref_type(self) -> tp.Type[Ref[A]]:
        return self._ref_type


def _index_flatten(
    x: Index[tp.Any],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[int, type]]:
    return (), (x.index, x.ref_type)


def _index_unflatten(aux_data: tp.Tuple[int, type], children: tp.Tuple[()]):
    return Index(*aux_data)


jax.tree_util.register_pytree_node(Index, _index_flatten, _index_unflatten)


class Static(tp.Generic[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value


def _static_flatten(x: Static[A]) -> tp.Tuple[tp.Tuple[()], A]:
    return (), x.value


def _static_unflatten(aux_data: A, _: tp.Tuple[()]) -> Static[A]:
    return Static(aux_data)


jax.tree_util.register_pytree_node(Static, _static_flatten, _static_unflatten)


class Dag(tp.Generic[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value


def _dag_flatten(x: Dag[tp.Any]) -> tp.Tuple[Leaves, jax.tree_util.PyTreeDef]:
    return deref_flatten(x.value)


def _dag_unflatten(treedef: jax.tree_util.PyTreeDef, leaves: Leaves) -> Dag[tp.Any]:
    return Dag(reref_unflatten(treedef, leaves))


jax.tree_util.register_pytree_node(Dag, _dag_flatten, _dag_unflatten)


def deref_fn(ref_index: tp.Dict[Ref[tp.Any], int], x: tp.Any) -> tp.Any:
    if isinstance(x, Ref):
        if x not in ref_index:
            ref_index[x] = len(ref_index)
            return Value(x.value, index=ref_index[x], ref_type=type(x))
        else:
            return Index(ref_index[x], ref_type=type(x))
    elif isinstance(x, Deref) and ref_index:
        raise ValueError("Cannot 'deref' pytree with a mix of Refs and Derefs")
    else:
        return x


def deref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, jax.tree_util.PyTreeDef]:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    leaves, treedef = jax.tree_util.tree_flatten(
        pytree, is_leaf=lambda x: isinstance(x, Deref)
    )
    return list(map(partial(deref_fn, ref_index), leaves)), treedef


def deref_unflatten(treedef: jax.tree_util.PyTreeDef, leaves: Leaves) -> A:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    leaves = list(map(partial(deref_fn, ref_index), leaves))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def deref(pytree: A) -> A:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    return jax.tree_map(
        partial(deref_fn, ref_index), pytree, is_leaf=lambda x: isinstance(x, Deref)
    )


def reref_fn(index_ref: tp.Dict[int, Ref[tp.Any]], x: tp.Any) -> tp.Any:
    if isinstance(x, Ref):
        raise ValueError("Cannot 'reref' pytree with a mix of Refs and Derefs")
    elif isinstance(x, Value):
        if x.index in index_ref:
            raise ValueError("Value already exists")
        ref = x.ref_type(x.value)
        index_ref[x.index] = ref
        return ref
    elif isinstance(x, Index):
        if x.index not in index_ref:
            # NOTE: because pytree flatten and unflatten in a deterministic
            # order, this should never trigger
            raise RuntimeError(
                "BUG: Got multiple values for the same index. This should never "
                "happen, please report it."
            )
        return index_ref[x.index]
    else:
        return x


def reref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, jax.tree_util.PyTreeDef]:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    leaves, treedef = jax.tree_util.tree_flatten(
        pytree, is_leaf=lambda x: isinstance(x, Deref)
    )
    return list(map(partial(reref_fn, index_ref), leaves)), treedef


def reref_unflatten(treedef: jax.tree_util.PyTreeDef, leaves: Leaves) -> tp.Any:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    leaves = list(map(partial(reref_fn, index_ref), leaves))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def reref(pytree: A) -> A:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    return jax.tree_map(
        partial(reref_fn, index_ref), pytree, is_leaf=lambda x: isinstance(x, Deref)
    )
