from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Dict, Optional

from search import *

ALL = WORDS


@lru_cache(maxsize=500000)
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

    info = score.cache_info()
    if (1+info.currsize) % 10000 == 0:
        print('score cache size = ', info.currsize+1)

    return ''.join(outcome)



@dataclass
class Options:
    parent: Optional[Options]
    possibilities: List[str]
    guess: str
    options: Dict[str, List[str]]
    subsequent: Dict[str, Options]
    worst_case_guesses: int

    def __init__(self, possibilities:List[str], guess: str, parent:Options = None):
        self.parent = parent
        self.possibilities = possibilities
        self.guess = guess
        self.options = defaultdict(lambda: [])
        self.subsequent = {}
        for w in possibilities:
            self.options[score(guess, w)].append(w)
            self.worst_case_guesses = 99

    def display(self):
        pre = '            ' * self.depth()
        for k,v in sorted(self.options.items(), key=lambda item: (len(item[1]), item[0])):
            print(pre, self.worst_case_guesses, self.guess + ':' + k, v)

    def depth(self):
        return 1 + self.parent.depth() if self.parent else 0

    def solve(self, max_recursions=7):

        if max_recursions < 1:
            self.worst_case_guesses = 99
            return

        # Choose best guess for any option that has multiple possibilities
        self.worst_case_guesses = 1
        for result, possibles in self.options.items():
            # print(f'When {self.guess} yields {result}, possibilities are {possibles}')

            if len(possibles) == 1:
                # print(' .. which needs no further work')
                pass
            else:
                best:Options = None
                child_max_recursions = max_recursions-1
                for trial in possibles:
                    o = Options(possibles, trial, self)
                    o.solve(child_max_recursions)
                    if not best or o.worst_case_guesses < best.worst_case_guesses:
                        best = o
                        child_max_recursions = min(child_max_recursions, best.worst_case_guesses)

                    # One is the best we can do
                    if best.worst_case_guesses == 1:
                        break

                # print(f' .. best is {best} with {best.worst_case_guesses} guesses at worst -- from {possibles}')
                self.subsequent[result] = best
                self.worst_case_guesses = max(self.worst_case_guesses, 1 + best.worst_case_guesses)

        # print(f'Solution: Worst case of {self.worst_case_guesses} for {self.possibilities}')
        Options.OPS += 1
        if Options.OPS % 1000000 == 0:
            print(str(Options.OPS//1000000)+'M' , '-', self.trail())



    def __str__(self):
        return self.guess + ': ' + ', '.join(f'{a}:{b}' for a,b in self.options.items())

    def print_solution(self, pre=''):

        print(f"{pre}{self.guess} will get an answer within {self.worst_case_guesses} attempts")

        # First, all the easy ones
        for result, items in sorted(self.options.items()):
            if result not in self.subsequent:
                print(f"{pre}When {self.guess} yields {result} the answer is {items[0]}")

        # Now the tricky ones
        for result, items in sorted(self.options.items()):
            if result in self.subsequent:
                o = self.subsequent[result]
                print(f"{pre}When {self.guess} yields {result} try {o.guess}: ")
                o.print_solution(pre + '  ..  ')

    def trail(self):
        if self.parent:
            return self.parent.trail() + ' ' + self.guess
        else:
            return  self.guess


if __name__ == '__main__':
    Options.OPS = 0

    top = Options(ALL, 'stare')
    top.solve()
    top.print_solution()

