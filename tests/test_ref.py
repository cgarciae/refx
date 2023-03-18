from typing import Any, Optional

import jax
import pytest

from refx import Ref, deref, reref


class TestPytreeRef:
    def test_ref(self):
        r1 = Ref(1)
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

    def test_ref_trace_level(self):
        r1: Ref[int] = Ref(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @jax.jit
        def g(r2: Ref[int], r3: Ref[int]):
            r2, r3 = reref((r2, r3))
            assert r2 is r3

            r2.value = 2
            assert r1 is not r2
            assert r3.value == 2
            return deref(r2)

        r2 = reref(g(*deref((r1, r1))))

        assert r1.value == 1
        assert r2.value == 2

        r2.value = 3
        assert r1.value == 1
        assert r2.value == 3

        r3 = reref(g(*deref((r1, r1))))

        assert r3 is not r2
        assert r3.value == 2

    def test_deref_through_jit(self):
        r1 = Ref(1)
        r2 = Ref(2)

        pytree = pytree0 = {"a": [r1, r2], "b": r1}

        @jax.jit
        def f(pytree):
            pytree = reref(pytree)

            assert pytree["a"][0] is pytree["b"]
            assert pytree["a"][1] is not pytree["b"]

            return deref(pytree)

        pytree = f(deref(pytree))
        pytree = reref(pytree)

        assert pytree["a"][0] is pytree["b"]
        assert pytree["a"][1] is not pytree["b"]

        # compare with pytree0
        assert pytree["a"][0] is not pytree0["a"][0]
        assert pytree["a"][1] is not pytree0["a"][1]
        assert pytree["b"] is not pytree0["b"]

    def test_barrier_edge_case(self):
        r1: Optional[Ref[Any]] = None

        @jax.jit
        def f():
            nonlocal r1
            x = jax.numpy.empty(1)
            r1 = Ref(x)
            return x

        x = f()
        assert r1 is not None

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different trace level"
        ):
            r1.value = 2

        @jax.jit
        def g():
            nonlocal r1
            assert r1 is not None
            x = jax.numpy.empty(1)
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = x
            return x

        x = g()

    def test_cross_barrier(self):
        r1: Ref[int] = Ref(1)

        @jax.jit
        def g(r2: Ref[int]):
            r2 = reref(r2)
            r2.value += 1
            assert r1 is not r2
            return deref(r2)

        r2 = reref(g(deref(r1)))
        assert r1 is not r2
        assert r1.value == 1
        assert r2.value == 2

        r3 = reref(g(deref(r2)))
        assert r1 is not r2
        assert r2 is not r3
        assert r1.value == 1
        assert r2.value == 2
        assert r3.value == 3

        # test passing a reference to a jitted function without cross_barrier
        @jax.jit
        def f(r1):
            return None

        with pytest.raises(TypeError, match="Cannot interpret value of type"):
            f(r1)

        assert isinstance(r1.value, int)
        assert r1.value == 1

    def test_no_rejit(self):
        n = 0
        r1 = Ref(1)
        r2 = Ref(2)

        @jax.jit
        def g(r3: Ref[int], r4: Ref[int], r5: Ref[int]):
            r3, r4, r5 = reref((r3, r4, r5))
            nonlocal n
            n += 1
            assert r3 is r4
            assert r4 is not r5
            return deref(r3)

        r6 = reref(g(*deref((r1, r1, r2))))
        assert r6 is not r1
        assert r6.value == r1.value
        assert n == 1

        g(*deref((r1, r1, r2)))
        assert n == 1

        g(*deref((r2, r2, r1)))
        assert n == 1

        with pytest.raises(AssertionError):
            g(*deref((r1, r2, r1)))

        assert n == 2

    def test_deref_number_of_fields(self):
        r1 = Ref(1)
        r2 = Ref(2)
        v1 = 3
        pytree = {
            "a": [r1, r2, v1],
            "b": {"c": r1, "d": r2},
        }
        assert len(jax.tree_util.tree_leaves(pytree)) == 5

        pytree = deref(pytree)
        assert len(jax.tree_util.tree_leaves(pytree)) == 3

        pytree = reref(pytree)
        assert len(jax.tree_util.tree_leaves(pytree)) == 5
