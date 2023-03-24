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
K = tp.TypeVar("K", bound=tp.Hashable)
Leaf = tp.Any
Leaves = tp.List[Leaf]
Key = tp.Hashable


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class ReferentialfMeta(type):
    def __subclasscheck__(self, __subclass: type) -> bool:
        return issubclass(__subclass, (Ref, Value, Index))

    def __instancecheck__(self, __instance: object) -> bool:
        return isinstance(__instance, (Ref, Value, Index))


class Referential(tp.Generic[K], metaclass=ReferentialfMeta):
    @property
    def index(self) -> int:
        ...

    @property
    def collection(self) -> K:
        ...


class DerefMeta(type):
    def __subclasscheck__(self, __subclass: type) -> bool:
        return issubclass(__subclass, (Value, Index))

    def __instancecheck__(self, __instance: object) -> bool:
        return isinstance(__instance, (Value, Index))


class Deref(tp.Generic[K], metaclass=DerefMeta):
    @property
    def index(self) -> int:
        ...

    @property
    def collection(self) -> K:
        ...


@tpe.final
class Ref(tp.Generic[A]):
    __slots__ = ("_value", "_collection", "_trace")

    def __init__(self, value: A, collection: tp.Hashable = None):
        self._value = value
        self._collection = collection
        self._trace = tracers.current_trace()

    @property
    def collection(self) -> tp.Hashable:
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

    @property
    def index(self) -> int:
        return -id(self)


@tpe.final
class Value(tp.Generic[A]):
    __slots__ = ("_value", "_index", "_collection")

    def __init__(self, value: A, index: int, collection: tp.Hashable):
        self._value = value
        self._index = index
        self._collection = collection

    @property
    def value(self) -> A:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    @property
    def collection(self) -> tp.Hashable:
        return self._collection


def _value_index_flatten_with_keys(
    x: Value[A],
) -> tp.Tuple[tp.Tuple[tp.Tuple[jtu.GetAttrKey, A]], tp.Tuple[int, tp.Hashable]]:
    return ((jtu.GetAttrKey("value"), x.value),), (x.index, x.collection)


def _value_index_unflatten(
    aux_data: tp.Tuple[int, tp.Hashable], children: tp.Tuple[A]
) -> Value[A]:
    return Value(children[0], *aux_data)


jtu.register_pytree_with_keys(
    Value, _value_index_flatten_with_keys, _value_index_unflatten
)


@tpe.final
class Index:
    __slots__ = ("_index", "_collection")

    def __init__(self, index: int, collection: tp.Hashable):
        self._index = index
        self._collection = collection

    @property
    def index(self) -> int:
        return self._index

    @property
    def collection(self) -> tp.Hashable:
        return self._collection


def _index_flatten(x: Index) -> tp.Tuple[tp.Tuple[()], tp.Tuple[int, tp.Hashable]]:
    return (), (x.index, x.collection)


def _index_unflatten(
    aux_data: tp.Tuple[int, tp.Hashable], children: tp.Tuple[()]
) -> Index:
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
            return Value(x.value, index, collection=x.collection)
        else:
            return Index(ref_index[x], x.collection)
    elif isinstance(x, Deref) and ref_index:
        raise ValueError("Cannot 'deref' pytree containing Derefs")
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
        raise ValueError("Cannot 'reref' pytree containing Refs")
    elif isinstance(x, Value):
        if x.index in index_ref:
            raise ValueError("Value already exists")
        ref = Ref(x.value, x.collection)
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


def update_from(target_tree: tp.Any, source_tree: tp.Any):
    target_leaves = jtu.tree_leaves(
        target_tree, is_leaf=lambda x: isinstance(x, Referential)
    )
    source_leaves = jtu.tree_leaves(
        source_tree, is_leaf=lambda x: isinstance(x, Referential)
    )

    if len(target_leaves) != len(source_leaves):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(target_leaves)} and {len(source_leaves)}"
        )

    ref_to_index: tp.Dict[Ref[tp.Any], int] = {}
    index_to_ref: tp.Dict[int, Ref[tp.Any]] = {}
    source_has_ref = False
    source_has_deref = False

    for i, (target_leaf, source_leaf) in enumerate(zip(target_leaves, source_leaves)):
        if isinstance(source_leaf, Deref):
            source_has_deref = True
        elif isinstance(source_leaf, Ref):
            source_has_ref = True
        if source_has_ref and source_has_deref:
            raise ValueError("Got source with mixed Ref and Deref instances")

        if isinstance(target_leaf, Ref):
            if target_leaf in ref_to_index:
                if isinstance(source_leaf, (Ref, Index)):
                    if ref_to_index[target_leaf] != source_leaf.index:
                        raise ValueError
                else:
                    raise ValueError
                continue
            elif isinstance(source_leaf, (Value, Ref)):
                target_leaf.value = source_leaf.value
                ref_to_index[target_leaf] = source_leaf.index
            elif isinstance(source_leaf, Index):
                raise ValueError(
                    f"Unseen Ref '{type(target_leaf).__name__}' at position [{i}] "
                    f"aligned with source 'Index(index={source_leaf.index})'"
                )
            else:
                raise ValueError(
                    f"Unexpected source type '{type(source_leaf).__name__}' "
                    f"at position [{i}]"
                )
        elif isinstance(target_leaf, Deref):
            raise ValueError(
                f"Target partition should not contain Deref instances, got "
                f"'{type(target_leaf).__name__}' at position [{i}]"
            )
        # elif target_leaf is not NOTHING:
        #     raise ValueError(
        #         f"Expected NOTHING target at position [{i}], "
        #         f"got '{type(target_leaf).__name__}'"
        #     )
        # elif source_leaf is not NOTHING:
        #     raise ValueError(
        #         f"Expected NOTHING source at position [{i}], "
        #         f"got '{type(source_leaf).__name__}'"
        #     )
