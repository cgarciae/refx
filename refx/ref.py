from abc import abstractmethod
import dataclasses
import typing as tp

import jax

from refx import tracers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


class Deref(tp.Generic[A]):
    index: int
    ref_type: type

    @property
    @abstractmethod
    def value(self) -> A:
        ...


@dataclasses.dataclass(frozen=True)
class Value(Deref[A]):
    _value: A
    index: int
    ref_type: type

    @property
    def value(self) -> A:
        return self._value


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


@dataclasses.dataclass(frozen=True)
class Index(Deref[A]):
    index: int
    ref_type: type

    @property
    def value(self) -> A:
        raise ValueError("Cannot get value of Index")


def _index_flatten(
    x: Index[tp.Any],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[int, type]]:
    return (), (x.index, x.ref_type)


def _index_unflatten(aux_data: tp.Tuple[int, type], children: tp.Tuple[()]):
    return Index(*aux_data)


jax.tree_util.register_pytree_node(Index, _index_flatten, _index_unflatten)


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
        return Value(pytree.value, index=0, ref_type=type(pytree))  # type: ignore

    ref_index: tp.Dict[Ref[tp.Any], int] = {}

    def deref_fn(ref: tp.Any) -> tp.Any:
        if isinstance(ref, Ref):
            if ref not in ref_index:
                ref_index[ref] = len(ref_index)
                return Value(ref.value, index=ref_index[ref], ref_type=type(ref))
            else:
                return Index(ref_index[ref], ref_type=type(ref))
        elif isinstance(ref, Deref) and ref_index:
            raise ValueError("Cannot 'deref' pytree with a mix of Refs and Derefs")
        return ref

    return jax.tree_map(deref_fn, pytree, is_leaf=lambda x: isinstance(x, Deref))


def reref(pytree: A) -> A:
    if isinstance(pytree, Deref):
        if isinstance(pytree, Index):
            raise ValueError("Cannot reref Index")
        assert isinstance(pytree, Value)
        return pytree.ref_type(pytree.value)

    index_ref: tp.Dict[int, Ref[tp.Any]] = {}

    def reref_fn(x: tp.Any) -> tp.Any:
        if not isinstance(x, Deref):
            return x
        if isinstance(x, Value):
            if x.index in index_ref:
                raise ValueError("Value already exists")
            index_ref[x.index] = x.ref_type(x.value)
        elif isinstance(x, Index) and x.index not in index_ref:
            # NOTE: because pytree flatten and unflatten in a deterministic
            # order, this should never trigger
            raise RuntimeError(
                "BUG: Got multiple values for the same index. This should never "
                "happen, please report it."
            )
        ref = index_ref[x.index]
        return ref

    return jax.tree_map(reref_fn, pytree, is_leaf=lambda x: isinstance(x, Deref))


def update_from_deref(target: A, source: A) -> A:
    seen_refs = set()
    seen_indexes = set()

    target_leaves, treedef = jax.tree_util.tree_flatten(
        target, is_leaf=lambda x: isinstance(x, Deref)
    )
    source_leaves = treedef.flatten_up_to(source)

    for target_leaf, source_leaf in zip(target_leaves, source_leaves):
        if isinstance(source_leaf, Value):
            assert isinstance(target_leaf, Ref) and target_leaf not in seen_refs
            target_leaf.value = source_leaf.value
            seen_refs.add(target_leaf)
            seen_indexes.add(source_leaf.index)
        elif isinstance(source_leaf, Index):
            if not isinstance(target_leaf, Ref):
                raise ValueError("Index target must be aligned with Ref")
            if target_leaf not in seen_refs:
                raise ValueError("Unseen Ref aligned with Index")
            if source_leaf.index not in seen_indexes:
                raise ValueError("Index seen before Value")
        elif isinstance(target_leaf, Ref):
            raise ValueError("Ref target must be aligned with Value or Index")
