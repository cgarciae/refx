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


@tpe.final
class Ref(tp.Generic[A, K]):
    __slots__ = ("_value", "_key", "_trace")

    def __init__(self, value: A, key: K = None):
        self._value = value
        self._key = key
        self._trace = tracers.current_trace()

    @property
    def key(self) -> K:
        return self._key

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


class Deref(tp.Generic[K], metaclass=DerefMeta):
    @property
    def index(self) -> int:
        ...

    @property
    def key(self) -> K:
        ...


class Value(tp.Generic[A, K]):
    __slots__ = ("_value", "_index", "_key")

    def __init__(self, value: A, index: int, key: K):
        self._value = value
        self._index = index
        self._key = key

    @property
    def value(self) -> A:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    @property
    def key(self) -> K:
        return self._key


def _value_index_flatten_with_keys(
    x: Value[A, K],
) -> tp.Tuple[tp.Tuple[tp.Tuple[jtu.GetAttrKey, A]], tp.Tuple[int, K]]:
    return ((jtu.GetAttrKey("value"), x.value),), (x.index, x.key)


def _value_index_unflatten(
    aux_data: tp.Tuple[int, K], children: tp.Tuple[A]
) -> Value[A, K]:
    return Value(children[0], *aux_data)


jtu.register_pytree_with_keys(
    Value, _value_index_flatten_with_keys, _value_index_unflatten
)


class Index(tp.Generic[K]):
    __slots__ = ("_index", "_key")

    def __init__(self, index: int, key: K):
        self._index = index
        self._key = key

    @property
    def index(self) -> int:
        return self._index

    @property
    def key(self) -> K:
        return self._key


def _index_flatten(
    x: Index[K],
) -> tp.Tuple[tp.Tuple[()], tp.Tuple[int, K]]:
    return (), (x.index, x.key)


def _index_unflatten(aux_data: tp.Tuple[int, K], children: tp.Tuple[()]) -> Index[K]:
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


def deref_fn(ref_index: tp.Dict[Ref[tp.Any, tp.Hashable], int], x: tp.Any) -> tp.Any:
    if isinstance(x, Ref):
        if x not in ref_index:
            index = len(ref_index)
            ref_index[x] = index
            return Value(x.value, index, key=x.key)
        else:
            return Index(index=ref_index[x], key=x.key)
    elif isinstance(x, Deref) and ref_index:
        raise ValueError("Cannot 'deref' pytree with a mix of Refs and Derefs")
    else:
        return x


def deref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, jtu.PyTreeDef]:
    ref_index: tp.Dict[Ref[tp.Any, tp.Hashable], int] = {}
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    return list(map(partial(deref_fn, ref_index), leaves)), treedef


def deref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> A:
    ref_index: tp.Dict[Ref[tp.Any, tp.Hashable], int] = {}
    leaves = list(map(partial(deref_fn, ref_index), leaves))
    return jtu.tree_unflatten(treedef, leaves)


def deref(pytree: A) -> A:
    ref_index: tp.Dict[Ref[tp.Any, tp.Hashable], int] = {}
    return jax.tree_map(
        partial(deref_fn, ref_index), pytree, is_leaf=lambda x: isinstance(x, Deref)
    )


def reref_fn(index_ref: tp.Dict[int, Ref[tp.Any, tp.Hashable]], x: tp.Any) -> tp.Any:
    if isinstance(x, Ref):
        raise ValueError("Cannot 'reref' pytree with a mix of Refs and Derefs")
    elif isinstance(x, Value):
        if x.index in index_ref:
            raise ValueError("Value already exists")
        ref = Ref(x.value, x.key)
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
    index_ref: tp.Dict[int, Ref[tp.Any, tp.Hashable]] = {}
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    return list(map(partial(reref_fn, index_ref), leaves)), treedef


def reref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> tp.Any:
    index_ref: tp.Dict[int, Ref[tp.Any, tp.Hashable]] = {}
    leaves = list(map(partial(reref_fn, index_ref), leaves))
    return jtu.tree_unflatten(treedef, leaves)


def reref(pytree: A) -> A:
    index_ref: tp.Dict[int, Ref[tp.Any, tp.Hashable]] = {}
    return jax.tree_map(
        partial(reref_fn, index_ref), pytree, is_leaf=lambda x: isinstance(x, Deref)
    )


def update_from(target_tree: tp.Any, source_tree: tp.Any):
    target_leaves = jtu.tree_leaves(target_tree)
    source_leaves = jtu.tree_leaves(source_tree)

    if len(target_leaves) != len(source_leaves):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(target_leaves)} and {len(source_leaves)}"
        )

    seen_refs: tp.Set[Ref[tp.Any, tp.Hashable]] = set()
    seen_indexes: tp.Set[int] = set()
    source_has_ref = False
    source_has_deref = False

    for i, (target_leaf, source_leaf) in enumerate(zip(target_leaves, source_leaves)):
        if isinstance(target_leaf, Ref):
            if target_leaf in seen_refs:
                if not isinstance(source_leaf, Index):
                    raise ValueError(
                        f"Ref '{type(target_leaf).__name__}' at position [{i}] has "
                        f"already been updated, trying to update it with "
                        f"'{type(source_leaf).__name__}'"
                    )
                continue
            elif isinstance(source_leaf, Ref):
                # use a negative index to avoid collisions
                # with indexes from Deref instances
                index = -id(source_leaf)
                value = source_leaf.value
                source_has_ref = True
            elif isinstance(source_leaf, Value):
                index = source_leaf.index
                value = source_leaf.value
                source_has_deref = True
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
            if source_has_ref and source_has_deref:
                raise ValueError("Got source with mixed Ref and Deref instances")
            target_leaf.value = value
            seen_refs.add(target_leaf)
            seen_indexes.add(index)
        elif isinstance(target_leaf, Deref):
            raise ValueError(
                f"Target partition should not contain Deref instances, got "
                f"'{type(target_leaf).__name__}' at position [{i}]"
            )
        elif target_leaf is not NOTHING:
            raise ValueError(
                f"Expected NOTHING target at position [{i}], "
                f"got '{type(target_leaf).__name__}'"
            )
        elif source_leaf is not NOTHING:
            raise ValueError(
                f"Expected NOTHING source at position [{i}], "
                f"got '{type(source_leaf).__name__}'"
            )
