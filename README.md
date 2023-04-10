# Refx

_A **re**ference system **f**or ja**x**_

## What is Refx?

Refx defines a simple reference system that allows you to create DAGs on top of JAX PyTrees. It enables some key features:

* Shared state
* Tractable mutability
* Semantic partitioning

Refx is intended to be a low-level library that can be used as a building block within other JAX libraries.

<details><summary><b>Why Refx?</b></summary>

Functional systems like `flax` and `haiku` are powerful but add a lot of complexity that is often transfered to the user. On the other hand, pytree-based systems like `equinox` are simpler but lack the ability to share parameters and modules. 

Refx aims to create a system that can be used to build neural networks libraries that has the simplicity of pytree-based systems while also having the power of functional systems.

</details>

## Installation

```bash
pip install refx
```

## Getting Started

Refx's main data structure is the `Ref` class. It is a wrapper around a value that can be used as leaves in a pytree. It also has a `value` attribute that can be used to access and mutate the value.

```python
import jax
import refx

r1 = refx.Ref(1)
r2 = refx.Ref(2)

pytree = {
    'a': [r1, r1, r2],
    'b': r2
}

pytree['a'][0].value = 10

assert pytree['a'][1].value == 10
```
`Ref` is not a pytree node, therefore you cannot pass pytrees containing `Ref`s to JAX functions. To interact with JAX, `refx` provides the following functions:

* `deref`: converts a pytree of references to a pytree of values and indexes.
* `reref`: converts a pytree of values and indexes to a pytree of references.

`deref` must be called before crossing a JAX boundary and `reref` must be called after crossing a JAX boundary.

```python
pytree = refx.deref(pytree)

@jax.jit
def f(pytree):
    pytree = refx.reref(pytree)
    # perform some computation / update the references
    pytree['a'][2].value = 50
    return jax.deref(pytree)

pytree = f(pytree)
pytree = refx.reref(pytree)

assert pytree['b'].value == 50
```

As you see in the is example, we've effectively implemented shared state and tracktable mutability with pure pytrees. 

### Trace-level awareness
In JAX, unconstrained mutability can lead to tracer leakage. To prevent this, `refx` only allows mutating references from the same trace level they were created in.

```python
r = refx.Ref(1)

@jax.jit
def f():
    # ValueError: Cannot mutate ref from different trace level
    r.value = jnp.array(1.0)
    ...
```

### Partitioning
Each reference has a `collection: Hashable` attribute that can be used to partition references into different groups. `refx` provides the `tree_partition` and to partition a pytree based a predicate function.
```python
r1 = refx.Ref(1, collection="params")
r2 = refx.Ref(2, collection="batch_stats")

pytree = {
    'a': [r1, r1, r2],
    'b': r2
}

(params, rest), treedef = refx.tree_partition(
    pytree, lambda x: isinstance(x, refx.Ref) and x.collection == "params")

assert params == {
    ('a', '0'): r1,
    ('a', '1'): r1,
    ('a', '2'): refx.NOTHING,
    ('b',): refx.NOTHING
}
assert rest == {
    ('a', '0'): refx.NOTHING,
    ('a', '1'): refx.NOTHING,
    ('a', '2'): r2,
    ('b',): r2,
}
```
You can use more partitioning functions to partition a pytree into multiple groups, `tree_partition` will always return one more partition than the number of functions passed to it. The last partition (`rest`) will contain all the remaining elements of the pytree. `tree_partition` also returns a `treedef` that can be used to reconstruct the pytree by using the `merge_partitions` function:

```python
pytree = refx.merge_partitions(partitions, treedef)
```
If you only need a single partition, you can use the `get_partition` function:
```python
r1 = refx.Ref(1, collection="params")
r2 = refx.Ref(2, collection="batch_stats")

pytree = {
    'a': [r1, r1, r2],
    'b': r2
}

params = refx.get_partition(
    pytree, lambda x: isinstance(x, refx.Ref) and x.collection == "params")

assert params == {
    ('a', '0'): r1,
    ('a', '1'): r1,
    ('a', '2'): refx.NOTHING,
    ('b',): refx.NOTHING
}
```
### Updating references
`refx` provides the `update_refs` function to update references in a pytree. It updates a **target** pytree with references reference with the values from a **source** pytree. The source pytree can be either a pytree of references or a pytree of values. As an example, here is how you can use `update_refs` to perform gradient descent on a pytree of references:

```python
def by_collection(c):
    return lambda x: isinstance(x, refx.Ref) and x.collection == c

(params, rest), treedef = refx.tree_partition(
    refx.deref(pytree), by_collection("params"))

def loss(params):
    pytree = refx.merge_partitions((params, rest), treedef)
    ...

grads = jax.grad(loss)(params)

# gradient descent
params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)

refx.update_refs(
    refx.get_partition(pytree, by_collection("params")), params)
```