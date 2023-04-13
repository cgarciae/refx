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
    def value(self) -> tp.Any:
        ...

    @property
    def collection(self) -> tp.Hashable:
        ...


class Deref(Referential):
    __slots__ = ()

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

    def to_value(self) -> "Value[A]":
        return Value(self.value, self.collection)

    def to_index(self) -> "Index":
        return Index(self.collection)


class Value(Deref, tp.Generic[A]):
    __slots__ = ("_value", "_collection")

    def __init__(self, value: A, collection: tp.Hashable):
        self._value = value
        self._collection = collection

    @property
    def value(self) -> A:
        return self._value

    @property
    def collection(self) -> tp.Hashable:
        return self._collection

    def to_ref(self) -> "Ref[A]":
        return Ref(self._value, self.collection)

    def __repr__(self) -> str:
        return f"Value(collection={repr(self.collection)}, value={repr(self._value)})"


def _value_flatten(
    x: Value[A],
    *,
    with_keys: bool,
) -> tp.Tuple[tp.Tuple[tp.Any], tp.Hashable]:
    if with_keys:
        node = (jtu.GetAttrKey("value"), x._value)
    else:
        node = x._value

    return (node,), x.collection


def _value_unflatten(collection: tp.Hashable, children: tp.Tuple[A]) -> Value[A]:
    return Value(children[0], collection)


jtu.register_pytree_with_keys(
    Value,
    partial(_value_flatten, with_keys=True),
    _value_unflatten,
    flatten_func=partial(_value_flatten, with_keys=False),
)


class Index(Deref):
    __slots__ = "_collection"

    def __init__(self, collection: tp.Hashable):
        self._collection = collection

    @property
    def collection(self) -> tp.Hashable:
        return self._collection

    def __repr__(self) -> str:
        return f"Index(collection={self.collection})"


def _index_flatten(x: Index) -> tp.Tuple[tp.Tuple[()], tp.Hashable]:
    return (), x.collection


def _index_unflatten(colletion: tp.Hashable, children: tp.Tuple[()]) -> Index:
    return Index(colletion)


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
    __slots__ = "_value"

    def __init__(self, value: A):
        self._value = value

    @property
    def value(self) -> A:
        return self._value


def _dag_flatten(
    x: Dag[tp.Any],
    *,
    with_keys: bool,
) -> tp.Tuple[tp.Tuple[tp.Any], tp.Tuple["DagDef", jtu.PyTreeDef]]:
    leaves, dagdef, treedef = deref_flatten(x.value)
    if with_keys:
        node = (jtu.GetAttrKey("value"), leaves)
    else:
        node = leaves
    return (node,), (dagdef, treedef)


def _dag_unflatten(
    metadata: tp.Tuple["DagDef", jtu.PyTreeDef], leaves: Leaves
) -> Dag[tp.Any]:
    dagdef, treedef = metadata
    return Dag(reref_unflatten(treedef, leaves, dagdef))


jtu.register_pytree_with_keys(
    Dag,
    partial(_dag_flatten, with_keys=True),
    _dag_unflatten,
    flatten_func=partial(_dag_flatten, with_keys=False),
)

DagIndexes = tp.List[tp.List[int]]
DagDef = Static[DagIndexes]


def deref_leaves(
    ref_index: tp.Dict[Ref[tp.Any], int],
    indexes: DagIndexes,
    leaves: Leaves,
) -> tp.Iterator[tp.Any]:
    for leaf_idx, x in enumerate(leaves):
        if isinstance(x, Ref):
            if x not in ref_index:
                ref_index[x] = len(ref_index)
                indexes.append([leaf_idx])
                yield x.to_value()
            else:
                indexes[ref_index[x]].append(leaf_idx)
                yield x.to_index()
        elif isinstance(x, Deref):
            raise ValueError("Cannot 'deref' pytree containing Derefs")
        else:
            yield x


def deref_flatten(pytree: tp.Any) -> tp.Tuple[Leaves, DagDef, jtu.PyTreeDef]:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    indexes: DagIndexes = []
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    leaves = list(deref_leaves(ref_index, indexes, leaves))
    return leaves, Static(indexes), treedef


def deref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves) -> tp.Tuple[A, DagDef]:
    ref_index: tp.Dict[Ref[tp.Any], int] = {}
    indexes: DagIndexes = []
    leaves = list(deref_leaves(ref_index, indexes, leaves))
    return jtu.tree_unflatten(treedef, leaves), Static(indexes)


def deref(pytree: A) -> tp.Tuple[A, DagDef]:
    leaves, dag_def, treedef = deref_flatten(pytree)
    return jtu.tree_unflatten(treedef, leaves), dag_def


def _validate_reref(x: A) -> A:
    if isinstance(x, Ref):
        raise ValueError("Cannot 'reref' pytree containing Refs")

    return x


def reref_leaves(indexes: DagIndexes, leaves: Leaves) -> Leaves:
    leaves_out = list(map(_validate_reref, leaves))

    for leaf_indexes in indexes:
        leaf_index = leaf_indexes[0]
        value = leaves[leaf_index]

        if not isinstance(value, Value):
            raise ValueError(
                f"Expected 'Value' as first leaf, got {type(value).__name__}"
            )

        ref = value.to_ref()
        leaves_out[leaf_index] = ref

        for leaf_index in leaf_indexes[1:]:
            x = leaves[leaf_index]

            if not isinstance(x, Index):
                raise ValueError(f"Expected 'Index' as leaf, got {type(x).__name__}")

            leaves_out[leaf_index] = ref

    return leaves_out


def reref_flatten(pytree: tp.Any, dagdef: DagDef) -> tp.Tuple[Leaves, jtu.PyTreeDef]:
    indexes = dagdef.value
    leaves, treedef = jtu.tree_flatten(pytree, is_leaf=lambda x: isinstance(x, Deref))
    return reref_leaves(indexes, leaves), treedef


def reref_unflatten(treedef: jtu.PyTreeDef, leaves: Leaves, dagdef: DagDef) -> tp.Any:
    indexes = dagdef.value
    leaves = reref_leaves(indexes, leaves)
    return jtu.tree_unflatten(treedef, leaves)


def reref(pytree: A, dagdef: DagDef) -> A:
    leaves, treedef = reref_flatten(pytree, dagdef)
    return jtu.tree_unflatten(treedef, leaves)


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

    seen_target_refs: tp.Set[Ref[tp.Any]] = set()
    seen_source_refs: tp.Set[Ref[tp.Any]] = set()
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
            if target_leaf in seen_target_refs:
                if isinstance(source_leaf, Ref) and source_leaf not in seen_source_refs:
                    raise ValueError
                if not isinstance(source_leaf, (Index, Ref)):
                    raise ValueError
                continue
            elif isinstance(source_leaf, (Value, Ref)):
                target_leaf.value = source_leaf.value
                seen_target_refs.add(target_leaf)
                if isinstance(source_leaf, Ref):
                    seen_source_refs.add(source_leaf)
            elif isinstance(source_leaf, Index):
                raise ValueError(
                    f"Unseen Ref '{type(target_leaf).__name__}' at position [{i}] "
                    f"aligned with source '{source_leaf}'"
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
