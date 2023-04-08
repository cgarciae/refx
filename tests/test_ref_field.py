import dataclasses
import typing as tp
import typing_extensions as tpe

import jax
import pytest
from simple_pytree import Pytree

import refx


def field(
    default: tp.Any = dataclasses.MISSING, *, collection: tp.Hashable = None, **kwargs
):
    metadata = {"pytree_node": True, "collection": collection}
    return refx.field(default=default, **kwargs, metadata=metadata)


@tpe.dataclass_transform(field_specifiers=(field,))
def dataclass(*args, **kwargs):
    return dataclasses.dataclass(*args, **kwargs)


class TestRefField:
    def test_ref_field_dataclass(self):
        @dataclass
        class Foo(Pytree):
            a: int = field()

        foo1 = Foo(a=1)
        foo1.a = 2
        assert foo1.a == 2

        def add_one(r):
            r.value += 1
            return r

        foo2 = jax.tree_map(add_one, foo1)

        assert foo1.a == 3
        assert foo2.a == 3

    def test_cannot_change_ref(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int = ref_field()

        foo1 = Foo(a=1)

        with pytest.raises(ValueError, match="Cannot change Ref"):
            foo1.a = refx.Ref(2)

    def test_ref_field_normal_class(self):
        class Foo(Pytree):
            a: int = ref_field()

            def __init__(self, a: int):
                pass

        foo1 = Foo(a=1)
        foo1.a = 2
        assert foo1.a == 2

        def add_one(r):
            r.value += 1
            return r

        foo2 = jax.tree_map(add_one, foo1)

        assert foo1.a == 3
        assert foo2.a == 3

    def test_unset_field(self):
        class Foo(Pytree):
            a = ref_field()

        foo1 = Foo()

        with pytest.raises(AttributeError, match="Attribute 'a' is not set"):
            foo1.a

    def test_barrier(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int = ref_field()

        foo1 = Foo(a=1)

        @jax.jit
        def g(foo2: Foo):
            foo2 = refx.reref(foo2)
            foo2.a += 1
            return refx.deref(foo2)

        foo2 = refx.reref(g(refx.deref(foo1)))
        assert foo1.a == 1
        assert foo2.a == 2
