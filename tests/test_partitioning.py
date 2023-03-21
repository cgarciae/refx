import jax
import refx
import typing as tp


class TestPartitioning:
    def test_partition_tree(self):
        p1 = refx.Ref("params", 1)
        p2 = refx.Ref("params", 2)
        s1 = refx.Ref("batch_stats", 3)

        pytree = {
            "a": [p1, s1],
            "b": p2,
            "c": 100,
        }

        (params, rest), treedef = refx.partition_tree(pytree, "params")

        assert len(params) == 4
        assert len(rest) == 4

        # check params
        assert params[("a", "0")] is p1
        assert params[("a", "1")] is refx.NOTHING
        assert params[("b",)] is p2
        assert params[("c",)] is refx.NOTHING

        # check rest
        # assert rest[0] is refx.NOTHING
        assert rest[("a", "0")] is refx.NOTHING
        # assert rest[1] is s1
        assert rest[("a", "1")] is s1
        # assert rest[2] is refx.NOTHING
        assert rest[("b",)] is refx.NOTHING
        # assert rest[3] == 100
        assert rest[("c",)] == 100

        pytree = refx.merge_partitions((params, rest), treedef)

        assert pytree["a"][0] is p1
        assert pytree["a"][1] is s1
        assert pytree["b"] is p2
        assert pytree["c"] == 100

    def test_update_from(self):
        p1 = refx.Ref("params", 1)
        p2 = refx.Ref("params", 2)
        s1 = refx.Ref("batch_stats", 3)

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
        p1 = refx.Ref("params", 1.0)
        s1 = refx.Ref("batch_stats", -10)

        pytree = {
            "a": [p1, s1],
            "b": p1,
            "c": 100,
        }

        params = refx.get_partition(refx.deref(pytree), "params")

        def loss(params):
            params = refx.reref(params)
            return sum(p.value for p in jax.tree_util.tree_leaves(params))

        grad = jax.grad(loss)(params)
        refx.update_partition(pytree, grad, "params")

        assert pytree["a"][0].value == 2.0
        assert pytree["a"][1].value == -10
        assert pytree["b"].value == 2.0
        assert pytree["c"] == 100
