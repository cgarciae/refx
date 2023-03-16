import dataclasses

import jax
import pytest
from simple_pytree import Pytree

import refx


class TestPytreeRef:
    def test_ref(self):
        r1 = refx.Ref(1)
        assert r1.value == 1

        def add_one(r):
            r.value += 1
            return r

        r2 = jax.tree_map(add_one, r1)

        assert r1.value == 2
        assert r2.value == 2
        assert r1 is r2

        r1.value = 3

        assert r1.value == 3
        assert r2.value == 3

    def test_ref_context(self):
        r1 = refx.Ref(1)
        r2 = jax.tree_map(lambda x: x, r1)  # copy
        assert r1.value == 1
        assert r2.value == 1
        r1.value = 2  # OK
        assert r2.value == 2

        with refx.incremented_ref():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different context"
            ):
                r1.value = 3

            r1, r2, _ = refx.clone_references((r1, r2, "foo"))
            assert r1.value == 2
            r2.value = 3  # OK
            assert r1.value == 3

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different context"
        ):
            r1.value = 4

        with pytest.raises(
            ValueError, match="Cannot clone ref from higher context level"
        ):
            r1, r2 = refx.clone_references((r1, r2))

    def test_clone_leakage_error(self):
        r1 = None

        @jax.jit
        def f():
            nonlocal r1
            x = jax.numpy.empty(1)
            r1 = refx.Ref(x)
            return x

        f()

        with pytest.raises(
            ValueError, match="Cannot clone ref from higher trace level"
        ):
            r2 = refx.clone_references(r1)

    def test_ref_trace_level(self):
        r1: refx.Ref[int] = refx.Ref(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @refx.cross_barrier(jax.jit)
        def g(r2: refx.Ref[int], r3: refx.Ref[int]):
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return r2

        r2 = g(r1, r1)

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = g(r1, r1)

        assert r3 is not r2
        assert r3.value == 2

        @jax.jit
        def h():
            r4 = refx.clone_references(r3)
            with jax.ensure_compile_time_eval():
                assert r4.value == 2
            assert r4 is not r3
            r4.value = 5
            assert r4.value == 5
            with jax.ensure_compile_time_eval():
                assert r3.value == 2

        h()

    def test_cross_barrier(self):
        r1: refx.Ref[int] = refx.Ref(1)

        @refx.cross_barrier(jax.jit)
        def g(r2: refx.Ref[int]):
            r2.value += 1
            assert r1 is not r2
            return r2

        r2 = g(r1)
        assert r1 is not r2
        assert r1.value == 1
        assert r2.value == 2

        r3 = g(r2)
        assert r1 is not r2
        assert r2 is not r3
        assert r1.value == 1
        assert r2.value == 2
        assert r3.value == 3

        # test passing a reference to a jitted function without refx.cross_barrier
        @jax.jit
        def f(r1):
            return None

        with pytest.raises(TypeError, match="Cannot interpret value of type"):
            f(r1)

        assert isinstance(r1.value, int)
        assert r1.value == 1

    def test_no_rejit(self):
        n = 0
        r1 = refx.Ref(1)
        r2 = refx.Ref(2)

        @refx.cross_barrier(jax.jit)
        def g(r3: refx.Ref[int], r4: refx.Ref[int], r5: refx.Ref[int]):
            nonlocal n
            n += 1
            assert r3 is r4
            assert r4 is not r5
            return r3

        r6 = g(r1, r1, r2)
        assert r6 is not r1
        assert r6.value == r1.value
        assert n == 1

        g(r1, r1, r2)
        assert n == 1

        g(r2, r2, r1)
        assert n == 1

        with pytest.raises(AssertionError):
            g(r1, r2, r1)

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = refx.Ref(1)
        r2 = refx.Ref(2)
        v1 = 3
        pytree = {
            "a": [r1, r2, v1],
            "b": {"c": r1, "d": r2},
        }
        assert len(jax.tree_util.tree_leaves(pytree)) == 5

        pytree = refx.deref(pytree)
        assert len(jax.tree_util.tree_leaves(pytree)) == 3

        pytree = refx.reref(pytree)
        assert len(jax.tree_util.tree_leaves(pytree)) == 5


class TestRefField:
    def test_ref_field_dataclass(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int = refx.field(type=refx.Ref[int])

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
            a: int = refx.field(type=refx.Ref[int])

        foo1 = Foo(a=1)

        with pytest.raises(ValueError, match="Cannot change Ref"):
            foo1.a = refx.Ref(2)

    def test_ref_field_normal_class(self):
        class Foo(Pytree):
            a = refx.field(type=refx.Ref[int])

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
            a = refx.field(type=refx.Ref[int])

        foo1 = Foo()

        with pytest.raises(AttributeError, match="Attribute 'a' is not set"):
            foo1.a

    def test_barrier(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int = refx.field(type=refx.Ref[int])

        foo1 = Foo(a=1)

        @refx.cross_barrier(jax.jit)
        def g(foo2: Foo):
            foo2.a += 1
            return foo2

        foo2 = g(foo1)
        assert foo1.a == 1
        assert foo2.a == 2

    def test_send_pytree_node_in_metdata(self):
        with pytest.raises(ValueError, match="'pytree_node' found in metadata"):

            @dataclasses.dataclass
            class Foo(Pytree):
                a: int = refx.field(type=refx.Ref[int], metadata={"pytree_node": False})
