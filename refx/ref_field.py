import typing as tp
import dataclasses
from refx.ref import Ref


A = tp.TypeVar("A")
K = tp.TypeVar("K", bound=tp.Hashable)


class RefField(dataclasses.Field, tp.Generic[A]):
    def __init__(
        self,
        *,
        collection: tp.Hashable = None,
        default: tp.Any = dataclasses.MISSING,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        metadata: tp.Optional[tp.Mapping[tp.Any, tp.Any]] = None,
    ):
        if metadata is None:
            metadata = {}
        super().__init__(default, default_factory, init, repr, hash, compare, metadata)
        self.collection = collection
        self._first_get_call = True
        self.class_field_name: tp.Optional[str] = None

    def __set_name__(self, cls, name):
        """__set_name__ initializer for properties as per [PEP-487](https://peps.python.org/pep-0487/)"""
        self.class_field_name = name

    @property
    def object_field_name(self):
        return f"{self.class_field_name}__ref"

    def __get__(self, obj, objtype=None):
        if obj is None:
            if self._first_get_call:
                self._first_get_call = False
                return self
            else:
                return

        if not hasattr(obj, self.object_field_name):
            raise AttributeError(f"Attribute '{self.class_field_name}' is not set")

        return getattr(obj, self.object_field_name).value

    def __set__(self, obj, value: A):
        if isinstance(value, Ref):
            raise ValueError("Cannot change Ref")
        elif hasattr(obj, self.object_field_name):
            ref: Ref[A] = getattr(obj, self.object_field_name)
            ref.value = value
        else:
            obj.__dict__[self.object_field_name] = Ref(
                value, collection=self.collection
            )
