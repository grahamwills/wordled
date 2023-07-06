from __future__ import annotations

import dataclasses
from typing import List

with open('rsrc/words.txt') as f:
    _DICT = set(f.replace('\n', '') for f in f.readlines())

WORDS = sorted(_DICT)


def score(choice: str, target: str) -> str:
    available = [c for c in target]
    outcome = ['-'] * 5
    # Handle greens
    for i in range(5):
        if choice[i] == target[i]:
            outcome[i] = 'G'
            available.remove(choice[i])
    # Handle yellows
    for i in range(5):
        if outcome[i] == '-' and choice[i] in available:
            outcome[i] = 'Y'
            available.remove(choice[i])
    return ''.join(outcome)


def compatible(word: str, guess: str, result: str):
    return score(guess, target=word) == result


@dataclasses.dataclass
class SearchNode:
    target: str
    guess: str
    result: str
    depth: int
    children: List[SearchNode]

    def __init__(self, target: str, guess: str, possibles: List[str], depth=1):
        self.target = target
        self.guess = guess
        self.result = score(guess, target)
        self.depth = depth

        if self.result == 'GGGGG' or not possibles:
            self.children = []
        else:
            self.children = self.search(possibles)


    def deepest(self):
        if self.result == 'GGGGG':
            return 1
        if not self.children:
            return 9999999
        return 1 + max(c.deepest() for c in self.children)

    def search(self, possibles: List[str]) -> List[SearchNode]:
        restricted = [w for w in possibles if compatible(w, self.guess, self.result)]

        return [SearchNode(self.target, w, restricted) for w in restricted]

    def __str__(self):
        return self.guess + '[' + self.target + '] -> ' + self.result
