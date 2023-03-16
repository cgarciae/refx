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

            r1, r2 = refx.clone_references((r1, r2))
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
            assert r1.id is r2.id
            return r2

        r2 = g(r1, r1)

        assert r1.value == 2
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 3
        assert r2.value == 3

        r3 = g(r1, r1)

        assert r3 is r2
        assert r3.value == 2

        @jax.jit
        def h():
            p4 = refx.clone_references(r3)
            with jax.ensure_compile_time_eval():
                assert p4.value == 2
            assert p4.ref is not r3.ref
            assert p4.id is not r3.id
            p4.value = 5
            assert p4.value == 5

        h()

    def test_cross_barrier(self):
        r1: refx.Ref[int] = refx.Ref(1)

        @refx.cross_barrier(jax.jit)
        def g(r2: refx.Ref[int]):
            r2.value += 1
            assert r1.ref is not r2.ref
            return r2

        r2 = g(r1)
        assert r1.ref is r2.ref
        assert r1.value == 2
        assert r2.value == 2

        r2 = g(r1)
        assert r1.ref is r2.ref
        assert r1.value == 3
        assert r2.value == 3

        # test passing a reference to a jitted function without refx.cross_barrier
        @jax.jit
        def f(r1):
            return None

        with pytest.raises(TypeError, match="Cannot interpret value of type"):
            f(r1)

        assert isinstance(r1.value, jax.Array)
        assert r1.value == 3


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
