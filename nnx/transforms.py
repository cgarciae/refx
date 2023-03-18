import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import jax
import jax.stages
import refx
from jax._src.interpreters import pxla
import simple_pytree as spt

from nnx.refs import Param

A = TypeVar("A")
F = TypeVar("F", bound=Callable[..., Any])
AxisName = Hashable


class Null:
    def __repr__(self) -> str:
        return "Null"


def _null_flatten(x):
    return (), None


def _null_unflatten(aux_data, children):
    return NULL


NULL = Null()

jax.tree_util.register_pytree_node(Null, _null_flatten, _null_unflatten)


class RefJIT(jax.stages.Wrapped):
    def __init__(self, fun, **jit_kwargs):
        @functools.partial(jax.jit, **jit_kwargs)
        def jitted_fn(*args, **kwargs):
            args, kwargs = refx.reref((args, kwargs))
            out = fun(*args, **kwargs)
            out = refx.deref(out)
            return out

        self.jitted_fn = jitted_fn

    def __call__(self, *args, **kwargs):
        args, kwargs = refx.deref((args, kwargs))
        out = self.jitted_fn(*args, **kwargs)
        out = refx.reref(out)
        return out

    def __repr__(self):
        return f"RefJIT({self.jitted_fn})"

    def lower(self, *args, **kwargs):
        return self.jitted_fn.lower(*args, **kwargs)


def jit(
    fun: Callable[..., Any],
    in_shardings: Any = pxla._UNSPECIFIED,
    out_shardings: Any = pxla._UNSPECIFIED,
    static_argnums: Union[int, Sequence[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    keep_unused: bool = False,
    device: Optional[jax.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> jax.stages.Wrapped:
    """JIT compile a function, dereferencing and rereferencing Refs."""
    ref_jit = RefJIT(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )
    functools.wraps(fun)(ref_jit)
    return ref_jit


def partition(pytree, *predicates: Callable[[Any], bool]):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    predicates = predicates

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partitions: Tuple[List[Any]] = tuple(
        [NULL] * len(leaves) for _ in range(len(predicates) + 1)
    )
    for j, leaf in enumerate(leaves):
        for i, predicate in enumerate(predicates):
            if predicate(leaf):
                partitions[i][j] = leaf
                break
        else:
            # if we didn't break, set leaf to last partition
            partitions[-1][j] = leaf

    return partitions, treedef


def merge_partitions(partitions, treedef):
    leaves = []
    for i, options in enumerate(zip(*partitions)):
        non_null = [option for option in options if option is not NULL]
        if len(non_null) == 0:
            raise ValueError(f"Expected at least one non-null value for position {i}")
        elif len(non_null) > 1:
            raise ValueError(f"Expected at most one non-null value for position {i}")
        leaves.append(non_null[0])

    return jax.tree_util.tree_unflatten(treedef, leaves)


class RefGrad:
    def __init__(self, fun, partition_fn, **grad_kwargs):
        @functools.partial(jax.grad, **grad_kwargs)
        def grad_fn(diff, non_diff, treedef, *args, **kwargs):
            diff, non_diff = refx.reref((diff, non_diff))
            pytree = merge_partitions((diff, non_diff), treedef)
            out = fun(pytree, *args, **kwargs)
            out = refx.deref(out)
            return out

        self.grad_fn = grad_fn
        self.partition_fn = partition_fn
        self.has_aux: bool = grad_kwargs["has_aux"]

    def __call__(self, pytree, *args, **kwargs):
        (diff, non_diff), treedef = partition(pytree, self.partition_fn)
        diff, non_diff = refx.deref((diff, non_diff))
        grads = self.grad_fn(diff, non_diff, treedef, *args, **kwargs)

        if self.has_aux:
            grad, aux = grads
            aux = refx.reref(aux)
            return grad, aux
        else:
            return grads

    def __repr__(self):
        return f"RefGrad({self.grad_fn})"


def is_param(x):
    return isinstance(x, Param)


PredicateOrType = Union[Callable[[Any], bool], type, List[type]]


def grad(
    fun: F,
    partition_fn: PredicateOrType = Param,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> F:
    # if its a type
    if isinstance(partition_fn, type):
        _partition_fn = lambda x: isinstance(x, partition_fn)
    # if its a list of types
    elif isinstance(partition_fn, list):
        _partition_fn = lambda x: isinstance(x, tuple(partition_fn))
    elif callable(partition_fn):
        _partition_fn = partition_fn
    else:
        raise ValueError(f"Invalid partition_fn: {partition_fn}")

    ref_grad = RefGrad(
        fun,
        _partition_fn,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    functools.wraps(fun)(ref_grad)
    return ref_grad
