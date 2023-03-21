from dataclasses import fields
import dataclasses
import typing as tp
import jax
import jax.tree_util as jtu
from refx.ref import Ref, Deref, Value, Index, NOTHING, Nothing

A = tp.TypeVar("A")
CollectionFilter = tp.Union[
    str,
    tp.Sequence[str],
    tp.Callable[[str], bool],
]
Leaf = tp.Any
Leaves = tp.List[Leaf]
KeyPath = tp.Tuple[tp.Hashable, ...]


class StrPath(tp.Tuple[str, ...]):
    pass


tp.MappingView


class Partition(tp.Dict[tp.Tuple[str, ...], Leaf]):
    def __setitem__(self, key, value):
        raise TypeError("Partition is immutable")


def _partition_flatten_with_keys(
    x: Partition,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[StrPath, Leaf], ...], tp.Tuple[tp.Tuple[str, ...], ...]
]:
    children = tuple((StrPath(key), value) for key, value in x.items())
    return children, tuple(x.keys())


def _partition_unflatten(keys: tp.Tuple[StrPath, ...], leaves: tp.Tuple[Leaf, ...]):
    return Partition(zip(keys, leaves))


jax.tree_util.register_pytree_with_keys(
    Partition, _partition_flatten_with_keys, _partition_unflatten
)


class Not:
    def __init__(self, collection_filter: CollectionFilter):
        self.collection_filter = collection_filter
        self.predicate = _collection_filter_to_predicate(collection_filter)

    def __call__(self, collection: str) -> bool:
        return not self.predicate(collection)

    def __repr__(self) -> str:
        return f"Not({self.collection_filter})"


class Is:
    def __init__(self, collection: str):
        self.collection = collection

    def __call__(self, collection: str) -> bool:
        return collection == self.collection

    def __repr__(self) -> str:
        return f"Is({self.collection})"


class In:
    def __init__(self, *collections: str):
        self.collections = collections

    def __call__(self, collection: str) -> bool:
        return collection in self.collections

    def __repr__(self) -> str:
        return f"In({', '.join(self.collections)})"


class _All:
    def __call__(self, collection: str) -> bool:
        return True

    def __repr__(self) -> str:
        return "All()"


All = _All()


def _collection_filter_to_predicate(
    collection_filter: CollectionFilter,
) -> tp.Callable[[str], bool]:
    if isinstance(collection_filter, str):
        return Is(collection_filter)
    elif isinstance(collection_filter, (list, tuple)):
        return In(*collection_filter)
    elif callable(collection_filter):
        return collection_filter
    else:
        raise ValueError(
            "Expected a string, sequence of strings, or callable, "
            f"got {collection_filter}"
        )


dataclasses.Field


def _key_path_to_str_gen(key_path: KeyPath) -> tp.Generator[str, None, None]:
    for key_entry in key_path:
        if isinstance(key_entry, StrPath):
            yield from key_entry
        elif isinstance(key_entry, jtu.SequenceKey):
            yield str(key_entry.idx)
        elif isinstance(key_entry, jtu.DictKey):
            yield str(key_entry.key)
        elif isinstance(key_entry, jtu.GetAttrKey):
            yield str(key_entry.name)
        elif isinstance(key_entry, jtu.FlattenedIndexKey):
            yield str(key_entry.key)
        elif hasattr(key_entry, "__dict__") and len(key_entry.__dict__) == 1:
            yield str(next(iter(key_entry.__dict__.values())))
        else:
            yield str(key_entry)


def _key_path_to_str_path(key_path: KeyPath) -> StrPath:
    return StrPath(_key_path_to_str_gen(key_path))


def partition_tree(
    pytree, *collection_predicates: CollectionFilter
) -> tp.Tuple[tp.Tuple[Partition, ...], jax.tree_util.PyTreeDef]:
    collection_predicates = tuple(
        _collection_filter_to_predicate(p) for p in collection_predicates
    )
    paths_leaves: tp.List[tp.Tuple[KeyPath, Leaf]]
    paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        pytree, is_leaf=lambda x: isinstance(x, Deref) or x is NOTHING
    )

    leaves: tp.Tuple[Leaf, ...]
    paths, leaves = zip(*paths_leaves)
    paths = tuple(map(_key_path_to_str_path, paths))

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partition_leaves: tp.Tuple[Leaves, ...] = tuple(
        [NOTHING] * len(leaves) for _ in range(len(collection_predicates) + 1)
    )
    for j, leaf in enumerate(leaves):
        for i, predicate in enumerate(collection_predicates):
            if isinstance(leaf, (Ref, Deref)) and predicate(leaf.collection):
                partition_leaves[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partition_leaves[-1][j] = leaf

    partitions = tuple(
        Partition(zip(paths, partition)) for partition in partition_leaves
    )
    return partitions, treedef


def get_partition(pytree, collection_filter: CollectionFilter) -> Partition:
    collection_filter = _collection_filter_to_predicate(collection_filter)
    (partition, _rest), _treedef = partition_tree(pytree, collection_filter)
    return partition


def _get_non_nothing(
    paths: tp.Tuple[StrPath, ...],
    leaves: tp.Tuple[tp.Union[Leaf, Nothing], ...],
    position: int,
):
    # check that all paths are the same
    paths_set = set(paths)
    if len(paths_set) != 1:
        raise ValueError(
            "All partitions must have the same paths, "
            f" at position [{position}] got "
            "".join(f"\n- {path}" for path in paths_set)
        )
    non_null = [option for option in leaves if option is not NOTHING]
    if len(non_null) == 0:
        raise ValueError(
            f"Expected at least one non-null value for position [{position}]"
        )
    elif len(non_null) > 1:
        raise ValueError(
            f"Expected at most one non-null value for position [{position}]"
        )
    return non_null[0]


def merge_partitions(
    partitions: tp.Sequence[Partition], treedef: jax.tree_util.PyTreeDef
):
    lenghts = [len(partition) for partition in partitions]
    if not all(length == lenghts[0] for length in lenghts):
        raise ValueError(
            "All partitions must have the same length, got "
            f"{', '.join(str(length) for length in lenghts)}"
        )

    partition_paths = (list(partition.keys()) for partition in partitions)
    partition_leaves = (list(partition.values()) for partition in partitions)

    merged_leaves = [
        _get_non_nothing(paths, leaves, i)
        for i, (paths, leaves) in enumerate(
            zip(zip(*partition_paths), zip(*partition_leaves))
        )
    ]

    return jax.tree_util.tree_unflatten(treedef, merged_leaves)


@tp.overload
def update_partition(
    target_partition: Partition,
    source_partition: Partition,
):
    ...


@tp.overload
def update_partition(
    target_partition: tp.Any,
    source_partition: Partition,
    type_predicate: CollectionFilter,
):
    ...


def update_partition(
    target_partition: tp.Union[Partition, tp.Any],
    source_partition: Partition,
    type_predicate: tp.Optional[CollectionFilter] = None,
):
    if type_predicate is not None:
        target_partition = get_partition(target_partition, type_predicate)

    assert isinstance(target_partition, Partition)

    if len(target_partition) != len(source_partition):
        raise ValueError(
            f"Target and source leaves must have the same length, got "
            f"{len(target_partition)} and {len(source_partition)}"
        )

    seen_refs: tp.Set[Ref[tp.Any]] = set()
    seen_indexes: tp.Set[int] = set()
    source_has_ref = False
    source_has_deref = False

    for (target_path, target_leaf), (source_path, source_leaf) in zip(
        target_partition.items(), source_partition.items()
    ):
        if target_path != source_path:
            raise ValueError(
                f"Target and source paths must be the same, got "
                f"{target_path} and {source_path}"
            )
        if isinstance(target_leaf, Ref):
            if target_leaf in seen_refs:
                if not isinstance(source_leaf, Index):
                    raise ValueError(
                        f"Ref '{type(target_leaf).__name__}' at position {target_path} has already "
                        f"been updated, trying to update it with "
                        f"'{type(source_leaf).__name__}'"
                    )
                continue
            elif isinstance(source_leaf, Ref):
                index = -id(source_leaf)
                value = source_leaf.value
                source_has_ref = True
            elif isinstance(source_leaf, Value):
                index = source_leaf.index
                value = source_leaf.value
                source_has_deref = True
            elif isinstance(source_leaf, Index):
                raise ValueError(
                    f"Unseen Ref '{type(target_leaf).__name__}' at {target_path} "
                    f"aligned with source 'Index(index={source_leaf.index})'"
                )
            else:
                raise ValueError(
                    f"Unexpected source type '{type(source_leaf).__name__}' "
                    f"at {target_path}"
                )
            if source_has_ref and source_has_deref:
                raise ValueError("Got source with mixed Ref and Deref instances")
            target_leaf.value = value
            seen_refs.add(target_leaf)
            seen_indexes.add(index)
        elif isinstance(target_leaf, Deref):
            raise ValueError(
                f"Target partition should not contain Deref instances, got "
                f"'{type(target_leaf).__name__}' at {target_path}"
            )
        elif target_leaf is not NOTHING:
            raise ValueError(
                f"Expected NOTHING target at {target_path}, "
                f"got '{type(target_leaf).__name__}'"
            )
        elif source_leaf is not NOTHING:
            raise ValueError(
                f"Expected NOTHING source at {target_path}, "
                f"got '{type(source_leaf).__name__}'"
            )


def update_from(target: A, source: A):
    target_partition = get_partition(target, All)
    source_partition = get_partition(source, All)
    update_partition(target_partition, source_partition)
