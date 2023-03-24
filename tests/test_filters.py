import typing as tp

import jax

import refx


class TestFilters:
    def test_dagify_jit(self):
        r1 = refx.Ref(10.0)
        r2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [r1, r2],
            "b": r1,
            "c": 7,
            "d": 5.0,
        }

        @refx.dagify(jax.jit)
        def f(pytree):
            pytree["a"][0].value *= -1
            return pytree

        pytree = f(pytree)

        assert pytree["a"][0].value == -10.0
        assert pytree["a"][1].value == 20.0
        assert pytree["b"].value == -10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0

    def test_dagify_jit_propagate_state(self):
        r1 = refx.Ref(10.0)
        r2 = refx.Ref(20.0)

        pytree: tp.Dict[str, tp.Any] = {
            "a": [r1, r2],
            "b": r1,
            "c": 7,
            "d": 5.0,
        }

        @refx.dagify(jax.jit, propagate_state=True)
        def f(pytree):
            pytree["a"][0].value *= -1

        f(pytree)

        assert pytree["a"][0].value == -10.0
        assert pytree["a"][1].value == 20.0
        assert pytree["b"].value == -10.0
        assert pytree["c"] == 7
        assert pytree["d"] == 5.0
