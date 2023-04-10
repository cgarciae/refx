import contextlib
import dataclasses
from functools import partial
import typing as tp

import jax
import jax.tree_util as jtu

from refx import tracers

A = tp.TypeVar("A")
A_cov = tp.TypeVar("A_cov", covariant=True)
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
K = tp.TypeVar("K", bound=tp.Hashable)
MutablePredicate = tp.Callable[[tp.Hashable], bool]
Leaf = tp.Any
Leaves = tp.List[Leaf]
Key = tp.Hashable


def all_mutable(_):
    return True


@dataclasses.dataclass
class RefContext:
    mutable_stack: tp.List[MutablePredicate] = dataclasses.field(
        default_factory=lambda: [all_mutable]
    )


_REF_CONTEXT = RefContext()


@contextlib.contextmanager
def mutable(mutable_fn: MutablePredicate):
    _REF_CONTEXT.mutable_stack.append(mutable_fn)
    try:
        yield
    finally:
        _REF_CONTEXT.mutable_stack.pop()


def is_mutable(collection: tp.Hashable) -> bool:
    return _REF_CONTEXT.mutable_stack[-1](collection)


def mutable_predicate() -> MutablePredicate:
    return _REF_CONTEXT.mutable_stack[-1]


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"  # pragma: no cover


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jtu.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


class Referential:
    __slots__ = ()

    @property
    def index(self) -> int:
        ...

    @property
    def collection(self) -> tp.Hashable:
        ...


class Deref(Referential):
    __slots__ = ()

    @property
    def index(self) -> int:
        ...

    @property
    def collection(self) -> tp.Hashable:
        ...


class Ref(Referential, tp.Generic[A]):
    __slots__ = ("_value", "_collection", "_jax_trace", "_refx_trace", "_trace_set")

    def __init__(self, value: A, collection: tp.Hashable = None):
        self._value = value
        self._collection = collection
        self._jax_trace = tracers.current_jax_trace()
        self._refx_trace = tracers.current_refx_trace()
        self._trace_set = frozenset((self._jax_trace, self._refx_trace))

    @property
    def collection(self) -> tp.Hashable:
        return self._collection

    @property
    def value(self) -> A:
        if (
            self._jax_trace is not tracers.current_jax_trace()
            or self._refx_trace is not tracers.current_refx_trace()
        ):
            raise ValueError("Cannot access ref from different trace level")
        return self._value

    @value.setter
    def value(self, value: A):
        if not is_mutable(self.collection):
            raise ValueError(f"Collection '{self.collection}' is not mutable")

        if (
            self._jax_trace is not tracers.current_jax_trace()
            or self._refx_trace is not tracers.current_refx_trace()
        ):
            raise ValueError("Cannot mutate ref from different trace level")

        invalid_traces = tracers.get_all_traces(value) - self._trace_set
        if invalid_traces:
            raise ValueError(
                "Cannot mutate ref with value that contains tracers from other "
                f"traces: {invalid_traces}"
            )

        self._value = value

    @property
    def index(self) -> int:
        return -id(self)

    def to_value(self, index: int) -> "Value[A]":
        return Value(self.value, index, self.collection)

    def to_index(self, index: int) -> "Index":
        return Index(index, self.collection)


class Value(Deref, tp.Generic[A]):
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

    def to_ref(self) -> "Ref[A]":
        return Ref(self.value, self.collection)

    def __repr__(self) -> str:
        return f"Value(index={self.index}, collection={repr(self.collection)})"


def _value_flatten_with_keys(
    x: Value[A],
) -> tp.Tuple[tp.Tuple[tp.Tuple[jtu.GetAttrKey, A]], tp.Tuple[int, tp.Hashable]]:
    return ((jtu.GetAttrKey("value"), x.value),), (x.index, x.collection)


def _value_unflatten(
    aux_data: tp.Tuple[int, tp.Hashable], children: tp.Tuple[A]
) -> Value[A]:
    return Value(children[0], *aux_data)


jtu.register_pytree_with_keys(Value, _value_flatten_with_keys, _value_unflatten)


class Index(Deref):
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

    def __repr__(self) -> str:
        return f"Index(index={self.index}, collection={self.collection})"


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
            return x.to_value(index)
        else:
            return x.to_index(ref_index[x])
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
        ref = x.to_ref()
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


def update_refs(target_tree: tp.Any, source_tree: tp.Any):
    target_leaves = jtu.tree_leaves(
        target_tree, is_leaf=lambda x: isinstance(x, Referential) or x is NOTHING
    )
    source_leaves = jtu.tree_leaves(
        source_tree, is_leaf=lambda x: isinstance(x, Referential) or x is NOTHING
    )

    if len(target_leaves) != len(source_leaves):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(target_leaves)} and {len(source_leaves)}"
        )

    ref_to_index: tp.Dict[Ref[tp.Any], int] = {}
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
