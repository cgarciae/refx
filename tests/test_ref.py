import dataclasses

import jax
import pytest
from simple_pytree import Pytree

import refx


class TestPytreeRef:
    def test_ref(self):
        p1 = refx.PytreeRef(1)
        assert p1.value == 1

        def add_one(r):
            r.value += 1
            return r

        p2 = jax.tree_map(add_one, p1)

        assert p1.value == 2
        assert p2.value == 2
        assert p1 is not p2
        assert p1.ref is p2.ref

        p1.value = 3

        assert p1.value == 3
        assert p2.value == 3

    def test_ref_context(self):
        p1 = refx.PytreeRef(1)
        p2 = jax.tree_map(lambda x: x, p1)  # copy
        assert p1.value == 1
        assert p2.value == 1
        p1.value = 2  # OK
        assert p2.value == 2

        with refx.incremented_ref():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different context"
            ):
                p1.value = 3

            p1, p2 = refx.clone_references((p1, p2))
            assert p1.value == 2
            p2.value = 3  # OK
            assert p1.value == 3

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different context"
        ):
            p1.value = 4

        with pytest.raises(
            ValueError, match="Cannot clone ref from higher context level"
        ):
            p1, p2 = refx.clone_references((p1, p2))

    def test_ref_trace_level(self):
        p1: refx.PytreeRef[int] = refx.PytreeRef(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                p1.value = 2
            return 1

        f()

        @refx.cross_barrier(jax.jit)
        def g(p2: refx.PytreeRef[int], p3: refx.PytreeRef[int]):
            assert p2.ref is p3.ref

            p2.value = 2
            assert p1.ref is not p2.ref
            assert p1.id is p2.id
            return p2

        p2 = g(p1, p1)
        p2_ref = p2.ref

        assert p1.value == 2
        assert p2.value == 2

        p2.value = 3
        assert p1.value == 3
        assert p2.value == 3

        p3 = g(p1, p1)
        p3_ref = p3.ref

        assert p3_ref is p2_ref
        assert p3.value == 2

        @jax.jit
        def h():
            p4 = refx.clone_references(p3)
            with jax.ensure_compile_time_eval():
                assert p4.value == 2
            assert p4.ref is not p3.ref
            assert p4.id is not p3.id
            p4.value = 5
            assert p4.value == 5

        h()

    def test_cross_barrier(self):
        p1: refx.PytreeRef[int] = refx.PytreeRef(1)

        @refx.cross_barrier(jax.jit)
        def g(p2: refx.PytreeRef[int]):
            p2.value += 1
            assert p1.ref is not p2.ref
            return p2

        p2 = g(p1)
        assert p1.ref is p2.ref
        assert p1.value == 2
        assert p2.value == 2

        p2 = g(p1)
        assert p1.ref is p2.ref
        assert p1.value == 3
        assert p2.value == 3

        # test passing a reference to a jitted function without refx.cross_barrier
        @jax.jit
        def f(p1):
            return None

        with pytest.raises(TypeError, match="Cannot interpret value of type"):
            f(p1)

        assert isinstance(p1.value, jax.Array)
        assert p1.value == 3


class TestRefField:
    def test_ref_field_dataclass(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int = refx.field(type=refx.PytreeRef[int])

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
            a = refx.field(type=refx.PytreeRef[int])

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
