# Refx

_A **re**ference system **f**or ja**x**_

## What is Refx?

Refx is a library that allows you to define py**dags** (directed acyclic graphs) on top of JAX's pytrees via a simple reference system. It enable two key features:

* Shared parameters / modules
* Tractable mutability

## Why Refx?
Functional systems like `flax` and `haiku` are powerful but add a lot of complexity that is often transfered to the user. On the other hand, pytree-based systems like `equinox` are simpler but lack the ability to share parameters and modules. 

Refx aims to create a system that can be used to build neural networks libraries that has the simplicity of pytree-based systems while also having the power of functional systems.

## Installation

```bash
pip install refx
```

## Usage
