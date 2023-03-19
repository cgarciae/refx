import typing as tp
import jax
from refx.ref import Ref, Deref, Value, Index, NOTHING, Nothing, Static

A = tp.TypeVar("A")
RefTypeOrSeq = tp.Union[
    tp.Type[Ref[tp.Any]],
    tp.Sequence[tp.Type[Ref[tp.Any]]],
]
Leaf = tp.Any
Leaves = tp.List[Leaf]


@tp.overload
def update_partition(
    refed_leaves: Leaves,
    derefed_leaves: Leaves,
):
    ...


@tp.overload
def update_partition(
    refed_leaves: tp.Any,
    derefed_leaves: Leaves,
    type_predicate: RefTypeOrSeq,
):
    ...


def update_partition(
    refed_leaves: tp.Union[Leaves, tp.Any],
    derefed_leaves: Leaves,
    type_predicate: tp.Optional[RefTypeOrSeq] = None,
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

    update_partition(target_leaves, source_leaves)


def _standar_type_partition(
    type_partition: RefTypeOrSeq,
) -> tp.Tuple[tp.Type[Ref[tp.Any]], ...]:
    if isinstance(type_partition, type):
        return (type_partition,)
    else:
        return tuple(type_partition)


def get_partition(pytree, type_predicate: RefTypeOrSeq) -> Leaves:
    type_predicate = _standar_type_partition(type_predicate)
    (partition, _rest), _treedef = partition_tree(pytree, type_predicate)
    return partition


def partition_tree(
    pytree, *type_predicates: RefTypeOrSeq
) -> tp.Tuple[tp.Tuple[Leaves, ...], jax.tree_util.PyTreeDef]:
    type_predicates = tuple(_standar_type_partition(p) for p in type_predicates)
    leaves, treedef = jax.tree_util.tree_flatten(
        pytree, is_leaf=lambda x: isinstance(x, Deref) or x is NOTHING
    )

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


def _get_non_nothing(options: tp.Tuple[tp.Union[Leaf, Nothing], ...], position: int):
    non_null = [option for option in options if option is not NOTHING]
    if len(non_null) == 0:
        raise ValueError(
            f"Expected at least one non-null value for position {position}"
        )
    elif len(non_null) > 1:
        raise ValueError(f"Expected at most one non-null value for position {position}")
    return non_null[0]


def merge_partitions(partitions: tp.Sequence[Leaves], treedef: jax.tree_util.PyTreeDef):
    lenghts = [len(partition) for partition in partitions]

    if not all(length == lenghts[0] for length in lenghts):
        raise ValueError(
            "All partitions must have the same length, got "
            f"{', '.join(str(length) for length in lenghts)}"
        )

    leaves = [
        _get_non_nothing(options, i) for i, options in enumerate(zip(*partitions))
    ]

    return jax.tree_util.tree_unflatten(treedef, leaves)
