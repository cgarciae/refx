from typing import Tuple


class StrPath(Tuple[str, ...]):
    ...


a = StrPath(("a", "b"))
b = ("a", "b")

tree = {a: 1}
print(tree[b])
