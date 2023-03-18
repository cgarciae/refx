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

        pytree = pytree0 = {
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
        grad
