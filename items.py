from typing import NamedTuple


class Word(NamedTuple):
    value: str  # Lowercase
    index: int  # 0-based, first item is most frequent
    freq: float  # relative frequency compared to the word 'the'

    def __str__(self):
        return self.value
