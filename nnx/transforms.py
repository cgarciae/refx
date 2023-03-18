import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import jax
import jax.stages
import refx
from jax._src.interpreters import pxla

from nnx.refs import Param

A = TypeVar("A")
F = TypeVar("F", bound=Callable[..., Any])
AxisName = Hashable


class Nothing:
    def __repr__(self) -> str:
        return "Nothing"


def _nothing_flatten(x):
    return (), None


def _nothing_unflatten(aux_data, children):
    return NOTHING


NOTHING = Nothing()

jax.tree_util.register_pytree_node(Nothing, _nothing_flatten, _nothing_unflatten)


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


def partition(pytree, *type_predicates: Tuple[Type[refx.Ref[Any]], ...]):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)

    # we have n + 1 partitions, where n is the number of predicates
    # the last partition is for values that don't match any predicate
    partitions: Tuple[List[Any]] = tuple(
        [NOTHING] * len(leaves) for _ in range(len(type_predicates) + 1)
    )
    for j, leaf in enumerate(leaves):
        for i, predicate in enumerate(type_predicates):
            if (isinstance(leaf, refx.Ref) and isinstance(leaf, predicate)) or (
                isinstance(leaf, refx.Deref) and issubclass(leaf.ref_type, predicate)
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


class RefGrad:
    def __init__(
        self,
        fun: Callable[..., Any],
        type_predicate: Tuple[Type[refx.Ref[Any]], ...],
        **grad_kwargs,
    ):
        @functools.partial(jax.grad, **grad_kwargs)
        def grad_fn(diff, non_diff, treedef, *args, **kwargs):
            diff, non_diff = refx.reref((diff, non_diff))
            pytree = merge_partitions((diff, non_diff), treedef)
            out = fun(pytree, *args, **kwargs)
            out = refx.deref(out)
            return out

        self.grad_fn = grad_fn
        self.type_predicate = type_predicate
        self.has_aux: bool = grad_kwargs["has_aux"]

    def __call__(self, pytree, *args, **kwargs):
        (diff, non_diff), treedef = partition(pytree, self.type_predicate)
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


TypeOrSeqType = Union[Type[refx.Ref[Any]], Sequence[Type[refx.Ref[Any]]]]


@overload
def grad(
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[False] = False,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Any]:
    ...


@overload
def grad(
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    has_aux: Literal[True],
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Tuple[Any, Any]]:
    ...


def grad(
    fun: Callable[..., Any],
    type_predicate: TypeOrSeqType = Param,
    *,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Union[Tuple[Any, Any], Any]]:
    # if its a type
    if isinstance(type_predicate, type):
        type_predicate = (type_predicate,)

    if not isinstance(type_predicate, tuple):
        type_predicate = tuple(type_predicate)

    ref_grad = RefGrad(
        fun,
        type_predicate,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    functools.wraps(fun)(ref_grad)
    return ref_grad
