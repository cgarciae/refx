from functools import partial
import pytest
import refx
import nnx


class TestJIT:
    def test_jit(self):
        r1: refx.Ref[int] = refx.Ref(1)

        @nnx.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                r1.value = 2
            return 1

        f()

        @nnx.jit
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

    def test_grad(self):
        p1 = nnx.Param(1.0)
        p2 = nnx.Param(2.0)

        pytree = {
            "a": [p1, p2],
            "b": p1,
            "c": 1,
            "d": 2.0,
        }

        @nnx.grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)

        assert grad[0].value == 2.0
        assert isinstance(grad[0], refx.Value)
        assert issubclass(grad[0].ref_type, nnx.Param)
        assert grad[1].value == 1.0
        assert isinstance(grad[1], refx.Value)
        assert issubclass(grad[1].ref_type, nnx.Param)
        assert isinstance(grad[2], refx.Index)
        assert issubclass(grad[2].ref_type, nnx.Param)
        assert grad[3] is nnx.NOTHING
        assert grad[4] is nnx.NOTHING

    def test_grad_with_multiple_ref_types(self):
        p1 = nnx.Param(1.0)
        p2 = nnx.BatchStat(2.0)

        pytree = {
            "a": [p1, p2],
            "b": p1,
            "c": 1,
            "d": 2.0,
        }

        @nnx.grad
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)

        assert grad[0].value == 2.0
        assert isinstance(grad[0], refx.Value)
        assert issubclass(grad[0].ref_type, nnx.Param)
        assert grad[1] is nnx.NOTHING
        assert isinstance(grad[2], refx.Index)
        assert issubclass(grad[2].ref_type, nnx.Param)
        assert grad[3] is nnx.NOTHING
        assert grad[4] is nnx.NOTHING

    def test_grad_with_type_predicate(self):
        p1 = nnx.Param(1.0)
        p2 = nnx.BatchStat(2.0)

        pytree = {
            "a": [p1, p2],
            "b": p1,
            "c": 1,
            "d": 2.0,
        }

        @partial(nnx.grad, type_predicate=nnx.BatchStat)
        def f(pytree):
            # sum all params
            return pytree["a"][0].value + pytree["a"][1].value + pytree["b"].value

        grad = f(pytree)

        assert grad[0] is nnx.NOTHING
        assert grad[1].value == 1.0
        assert isinstance(grad[1], refx.Value)
        assert issubclass(grad[1].ref_type, nnx.BatchStat)
        assert grad[2] is nnx.NOTHING
        assert grad[3] is nnx.NOTHING
        assert grad[4] is nnx.NOTHING
