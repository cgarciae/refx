import typing as tp
import jax
import hashlib

from refx import tracers
import jax.tree_util as jtu


def _stable_hash(x: str) -> int:
    _hash = hashlib.blake2s(x.encode())
    # & 0xFFFFFFFF is to make sure the hash is a 32-bit integer
    return int(_hash.hexdigest(), 16) & 0xFFFFFFFF


class RngStream:
    __slots__ = (
        "_key",
        "_count",
        "_count_path",
        "_trace",
    )

    def __init__(
        self,
        key: jax.random.KeyArray,
        count: int = 0,
        count_path: tp.Tuple[int, ...] = (),
    ):
        self._key = key
        self._count = count
        self._count_path = count_path
        self._trace = tracers.current_trace()

    def _validate_trace(self, tracer_seq=()):
        if self._trace is not tracers.current_trace(tracer_seq):
            raise ValueError("Rng used in a different trace")

    @property
    def key(self) -> jax.random.KeyArray:
        self._validate_trace()
        return self._key

    @property
    def count(self) -> int:
        return self._count

    @property
    def count_path(self) -> tp.Tuple[int, ...]:
        return self._count_path

    def next(self) -> jax.random.KeyArray:
        self._validate_trace()
        data = _stable_hash(str((self._count_path, self._count)))
        self._count += 1
        return jax.random.fold_in(self._key, data)

    def fork(self, tracers_seq=()) -> "RngStream":
        self._validate_trace(tracers_seq)
        count_path = self._count_path + (self._count,)
        self._count += 1
        return RngStream(self._key, count_path=count_path)


def _rng_flatten_with_keys(
    rng: RngStream,
) -> tp.Tuple[
    tp.Tuple[tp.Tuple[tp.Hashable, jax.random.KeyArray], ...],
    tp.Tuple[int, tp.Tuple[int, ...]],
]:
    return ((jtu.GetAttrKey("key"), rng.key),), (rng.count, rng.count_path)


def _rng_unflatten(
    aux_data: tp.Tuple[int, tp.Tuple[int, ...]],
    children: tp.Tuple[jax.random.KeyArray, ...],
) -> RngStream:
    count, count_path = aux_data
    key = children[0]
    return RngStream(key, count, count_path)


jax.tree_util.register_pytree_with_keys(
    RngStream, _rng_flatten_with_keys, _rng_unflatten
)
