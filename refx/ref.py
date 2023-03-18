from abc import abstractmethod
import dataclasses
import typing as tp

import jax

from refx import tracers

A = tp.TypeVar("A")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
TypeOrSeqType = tp.Union[tp.Type["Ref[tp.Any]"], tp.Sequence[tp.Type["Ref[tp.Any]"]]]
Leaf = tp.Any
Leaves = tp.List[Leaf]


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jax.tree_util.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


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


@tp.overload
def update_partition_from_derefed(
    refed_leaves: Leaves,
    derefed_leaves: Leaves,
):
    ...


@tp.overload
def update_partition_from_derefed(
    refed_leaves: tp.Any,
    derefed_leaves: Leaves,
    type_predicate: TypeOrSeqType,
):
    ...


def update_partition_from_derefed(
    refed_leaves: tp.Union[Leaves, tp.Any],
    derefed_leaves: Leaves,
    type_predicate: tp.Optional[TypeOrSeqType] = None,
):
    if type_predicate is not None:
        refed_leaves = get_partition(refed_leaves, type_predicate)

    assert isinstance(refed_leaves, list)

    if len(refed_leaves) != len(derefed_leaves):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(refed_leaves)} and {len(derefed_leaves)}"
        )

    seen_refs: tp.Set[Ref[tp.Any]] = set()
    seen_indexes: tp.Set[int] = set()

    for i, (refed_leaf, derefed_leaf) in enumerate(zip(refed_leaves, derefed_leaves)):
        if isinstance(derefed_leaf, Ref):
            raise ValueError(
                f"Ref instance '{type(derefed_leaf).__name__}' found on derefed source "
                f"at position [{i}]"
            )
        elif isinstance(derefed_leaf, Value):
            if not isinstance(refed_leaf, Ref):
                raise ValueError(
                    f"derefed 'Value(index={derefed_leaf.index})' source at position "
                    f"[{i}] must be aligned with a Ref, "
                    f"got '{type(refed_leaf).__name__}'."
                )
            if refed_leaf in seen_refs:
                raise ValueError(
                    f"Ref '{type(refed_leaf).__name__}' at position [{i}] has already "
                    f"been updated, trying to update it with "
                    f"'Value(index={derefed_leaf.index})'"
                )
            refed_leaf.value = derefed_leaf.value
            seen_refs.add(refed_leaf)
            seen_indexes.add(derefed_leaf.index)
        elif isinstance(derefed_leaf, Index):
            if not isinstance(refed_leaf, Ref):
                raise ValueError(
                    f"derefed 'Index(index={derefed_leaf.index})' source at position "
                    f"[{i}] must be aligned with Ref, got '{type(refed_leaf).__name__}'"
                )
            if refed_leaf not in seen_refs:
                raise ValueError(
                    f"Unseen Ref '{type(refed_leaf).__name__}' at position [{i}] "
                    f"aligned with derefed 'Index(index={derefed_leaf.index})'"
                )
            if derefed_leaf.index not in seen_indexes:
                raise ValueError(
                    f"derefed 'Index(index={derefed_leaf.index})' at position [{i}] "
                    "visited before its Value."
                )
        elif isinstance(refed_leaf, Ref):
            raise ValueError(
                f"Ref '{type(refed_leaf).__name__}' at position [{i}] "
                "must be aligned with Deref instance, "
                f"got '{type(derefed_leaf).__name__}'"
            )
        elif isinstance(refed_leaf, Deref):
            raise ValueError(
                f"Deref instance '{type(refed_leaf).__name__}' found on refed target "
                f"at position [{i}]"
            )


def update_from(refed: A, derefed: A):
    target_leaves, treedef = jax.tree_util.tree_flatten(
        refed, is_leaf=lambda x: isinstance(x, Deref)
    )
    source_leaves = treedef.flatten_up_to(derefed)

    update_partition_from_derefed(target_leaves, source_leaves)


def _standar_type_partition(
    type_partition: TypeOrSeqType,
) -> tp.Tuple[tp.Type[Ref[tp.Any]], ...]:
    if isinstance(type_partition, type):
        return (type_partition,)
    else:
        return tuple(type_partition)


def get_partition(pytree, type_predicate: TypeOrSeqType) -> Leaves:
    type_predicate = _standar_type_partition(type_predicate)
    (partition, _rest), _treedef = partition_tree(pytree, type_predicate)
    return partition


def partition_tree(pytree, *type_predicates: TypeOrSeqType):
    type_predicates = tuple(_standar_type_partition(p) for p in type_predicates)
    leaves, treedef = jax.tree_util.tree_flatten(pytree)

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partitions: tp.Tuple[tp.List[tp.Any]] = tuple(
        [NOTHING] * len(leaves) for _ in range(len(type_predicates) + 1)
    )
    for j, leaf in enumerate(leaves):
        for i, predicate in enumerate(type_predicates):
            if (isinstance(leaf, Ref) and isinstance(leaf, predicate)) or (
                isinstance(leaf, Deref) and issubclass(leaf.ref_type, predicate)
            ):
                partitions[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partitions[-1][j] = leaf

    return partitions, treedef


def merge_partitions(partitions, treedef):
    leaves = []
    for i, options in enumerate(zip(*partitions)):
        non_null = [option for option in options if option is not NOTHING]
        if len(non_null) == 0:
            raise ValueError(f"Expected at least one non-null value for position {i}")
        elif len(non_null) > 1:
            raise ValueError(f"Expected at most one non-null value for position {i}")
        leaves.append(non_null[0])

    return jax.tree_util.tree_unflatten(treedef, leaves)
