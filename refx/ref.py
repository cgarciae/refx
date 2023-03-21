from functools import partial
import typing as tp
import typing_extensions as tpe

import jax
import jax.tree_util as jtu

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

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


@tpe.final
class Ref(tp.Generic[A]):
    __slots__ = ("_collection", "_value", "_trace")

    def __init__(self, collection: str, value: A):
        self._collection = collection
        self._value = value
        self._trace = tracers.current_trace()

    @property
    def collection(self) -> str:
        return self._collection

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


class DerefMeta(type):
    def __subclasscheck__(self, __subclass: type) -> bool:
        return issubclass(__subclass, (Value, Index))

    def __instancecheck__(self, __instance: object) -> bool:
        return isinstance(__instance, (Value, Index))


class Deref(metaclass=DerefMeta):
    @property
    def collection(self) -> str:
        ...

    @property
    def index(self) -> int:
        ...


class Value(tp.Generic[A]):
    __slots__ = ("_collection", "_value", "_index")

    def __init__(self, collection: str, value: A, index: int):
        self._collection = collection
        self._value = value
        self._index = index

    @property
    def value(self) -> A:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    @property
    def collection(self) -> str:
        return self._collection


def _value_index_flatten_with_keys(
    x: Value[A],
) -> tp.Tuple[tp.Tuple[tp.Tuple[jtu.GetAttrKey, A]], tp.Tuple[str, int]]:
    return ((jtu.GetAttrKey("value"), x.value),), (x.collection, x.index)


def _value_index_unflatten(
    aux_data: tp.Tuple[str, int], children: tp.Tuple[A]
) -> Value[A]:
    collection, index = aux_data
    return Value(collection, children[0], index)


jtu.register_pytree_with_keys(
    Value, _value_index_flatten_with_keys, _value_index_unflatten
)


class Index(tp.Generic[A]):
    __slots__ = ("_collection", "_index")

    def __init__(self, collection: str, index: int):
        self._collection = collection
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def collection(self) -> str:
        return self._collection


def _index_flatten(
    x: Index[tp.Any],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[str, int]]:
    return (), (x.collection, x.index)


def _index_unflatten(aux_data: tp.Tuple[str, int], children: tp.Tuple[()]):
    return Index(*aux_data)


jtu.register_pytree_node(Index, _index_flatten, _index_unflatten)


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


jtu.register_pytree_node(Static, _static_flatten, _static_unflatten)


class Dag(tp.Generic[A]):
    __slots__ = ("_value",)

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value


def _dag_flatten_with_keys(
    x: Dag[tp.Any],
) -> tp.Tuple[tp.Tuple[tp.Tuple[jtu.GetAttrKey, Leaves]], jtu.PyTreeDef]:
    leaves, treedef = deref_flatten(x.value)
    key_node = (jtu.GetAttrKey("value"), leaves)
    return (key_node,), treedef


def _dag_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> Dag[tp.Any]:
    return Dag(reref_unflatten(treedef, leaves))


jtu.register_pytree_with_keys(Dag, _dag_flatten_with_keys, _dag_unflatten)


def deref_fn(ref_index: tp.Dict[Ref[tp.Any], int], x: tp.Any) -> tp.Any:
    if isinstance(x, Ref):
        if x not in ref_index:
            index = len(ref_index)
            ref_index[x] = index
            return Value(x.collection, x.value, index=index)
        else:
            return Index(x.collection, index=ref_index[x])
    elif isinstance(x, Deref) and ref_index:
        raise ValueError("Cannot 'deref' pytree with a mix of Refs and Derefs")
    else:
        return x


def deref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, jtu.PyTreeDef]:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    return list(map(partial(deref_fn, ref_index), leaves)), treedef


def deref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> A:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    leaves = list(map(partial(deref_fn, ref_index), leaves))
    return jtu.tree_unflatten(treedef, leaves)


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
        ref = Ref(x.collection, x.value)
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


def reref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, jtu.PyTreeDef]:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    return list(map(partial(reref_fn, index_ref), leaves)), treedef


def reref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> tp.Any:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    leaves = list(map(partial(reref_fn, index_ref), leaves))
    return jtu.tree_unflatten(treedef, leaves)


def reref(pytree: A) -> A:
    index_ref: tp.Dict[int, Ref[tp.Any]] = {}
    return jax.tree_map(
        partial(reref_fn, index_ref), pytree, is_leaf=lambda x: isinstance(x, Deref)
    )
