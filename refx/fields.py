import typing as tp
import dataclasses
from refx.ref import Ref

A = tp.TypeVar("A")


class RefField(dataclasses.Field, tp.Generic[A]):
    def __init__(self, *, default, ref_type: tp.Type[Ref[A]], **kwargs):
        super().__init__(default, **kwargs)
        self.ref_type = ref_type
        self.__first_get_request = True

    def __set_name__(self, owner, name):
        """__set_name__ initializer for properties as per [PEP-487](https://peps.python.org/pep-0487/)"""
        self.name = name
        self.__name = "__" + self.name

    def __get__(self, obj, objtype=None):
        if obj is None:
            if self.__first_get_request:
                self.__first_get_request = False
                return self
            else:
                raise AttributeError

        if not hasattr(obj, f"_ref_{self.name}"):
            raise AttributeError(f"Attribute {self.name} is not set")

        return getattr(obj, f"_ref_{self.name}").value

    def __set__(self, obj, value):
        if isinstance(value, RefField):
            return

        if hasattr(obj, f"_ref_{self.name}"):
            ref: Ref[A] = getattr(obj, f"_ref_{self.name}")
            ref.value = value
        elif isinstance(value, Ref):
            raise ValueError("Cannot change Ref")
        else:
            obj.__dict__[f"_ref_{self.name}"] = self.ref_type(value)


def field(
    default: tp.Union[A, dataclasses._MISSING_TYPE] = dataclasses.MISSING,
    *,
    type: tp.Type[Ref[A]],
    pytree_node: bool = True,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> A:
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("'pytree_node' found in metadata")

    metadata["pytree_node"] = pytree_node

    return RefField(
        default=default,
        ref_type=type,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
