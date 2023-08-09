from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import NamedTuple, Iterable

from bs4 import Tag


class Word(NamedTuple):
    value: str  # Lowercase
    index: int  # 0-based, first item is most frequent
    freq: float  # relative frequency compared to the word 'the'

    def __str__(self):
        return self.value


@dataclass
class Solution:
    index: int
    value: str
    when: date

    @classmethod
    def from_triple(cls, ta: Tag, tb: Tag, tc: Tag):
        a = str(ta.contents[0])
        b = str(tb.contents[0])
        c = str(tc.contents[0])
        assert b[0] == '#'
        return cls(
            int(b[1:]),
            c.lower(),
            datetime.strptime(a, '%Y/%m/%d').date()
        )

    def __str__(self):
        return f"{self.value} (#{self.index} - {self.when})"


class Statistics(NamedTuple):
    n_words: int
    n_unsolved: int
    sum_depth: float
    n_words_weighted: float
    n_unsolved_weighted: float
    sum_depth_weighted: float
    max_depth: int

    @classmethod
    def combine(cls, stats: Iterable[Statistics]) -> Statistics:
        n = 0
        n_un = 0
        sum_depth = 0
        n_w = 0.0
        n_un_w = 0.0
        sum_depth_w = 0.0
        max_depth = 0
        for s in stats:
            n += s.n_words
            n_un += s.n_unsolved
            sum_depth += s.sum_depth
            n_w += s.n_words_weighted
            n_un_w += s.n_unsolved_weighted
            sum_depth_w += s.sum_depth_weighted
            max_depth = max(max_depth, s.max_depth)
        return Statistics(n, n_un, sum_depth, n_w, n_un_w, sum_depth_w, max_depth)

    @classmethod
    def headers(cls, ) -> [str]:
        return [
            "Words",
            "Unsolved %",
            "Unsolved % (weighted)",
            "Max Solves",
            "Average Solves",
            "Average Solves (weighted)",
        ]

    def describe(self) -> [str]:
        return [
            f"{self.n_words}",
            f"{100 * self.n_unsolved / self.n_words:.3f}%",
            f"{100 * self.n_unsolved_weighted / self.n_words_weighted:.3f}%",
            f"{self.max_depth}",
            f"{1 + self.sum_depth / self.n_words:.3f}",
            f"{1 + self.sum_depth_weighted / self.n_words_weighted:.3f}",
        ]
