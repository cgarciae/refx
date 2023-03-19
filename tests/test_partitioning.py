import jax
import refx
import typing as tp


class Param(refx.Ref[tp.Any]):
    ...


class BatchStat(refx.Ref[tp.Any]):
    ...


class TestPartitioning:
    def test_partition_tree(self):
        p1 = Param(1)
        p2 = Param(2)
        s1 = BatchStat(3)

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        (params, rest), treedef = refx.partition_tree(pytree, Param)

        assert len(params) == 4
        assert len(rest) == 4

        # check params
        assert params[0] is p1
        assert params[1] is refx.NOTHING
        assert params[2] is p2
        assert params[3] is refx.NOTHING

        # check rest
        assert rest[0] is refx.NOTHING
        assert rest[1] is s1
        assert rest[2] is refx.NOTHING
        assert rest[3] == 100

        pytree = refx.merge_partitions((params, rest), treedef)

        assert pytree["a"][0] is p1
        assert pytree["a"][1] is s1
        assert pytree["b"] is p2
        assert pytree["c"] == 100

    def test_update_from(self):
        p1 = Param(1)
        p2 = Param(2)
        s1 = BatchStat(3)

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        derered = refx.deref(pytree)
        derered = jax.tree_map(lambda x: x * 2, derered)

        refx.update_from(pytree, derered)

        assert pytree["a"][0].value == 2
        assert pytree["a"][1].value == 6
        assert pytree["b"].value == 4
        assert pytree["c"] == 100

    def test_grad_example(self):
        p1 = Param(1.0)
        s1 = BatchStat(-10)

        pytree = {
            "a": [p1, s1],
            "b": p1,
            "c": 100,
        }

        params = refx.get_partition(refx.deref(pytree), Param)

        def loss(params):
            params = refx.reref(params)
            return sum(p.value for p in jax.tree_util.tree_leaves(params))

        grad = jax.grad(loss)(params)
        refx.update_partition(pytree, grad, Param)

        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == -10
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 100
