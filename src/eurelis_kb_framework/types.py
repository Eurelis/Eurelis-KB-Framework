from typing import TypeAlias, Sequence

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
PARAMS: TypeAlias = dict[str, "JSON"]
FACTORY: TypeAlias = str | PARAMS
CLASS: TypeAlias = str | PARAMS

EMBEDDING: TypeAlias = Sequence[float]
