"""Play wordle"""
from __future__ import annotations

import random
from typing import NamedTuple, Tuple

from tqdm.contrib.concurrent import process_map

with open('resources/words.txt') as f:
    _DICT = set(f.replace('\n', '') for f in f.readlines())

WORDS = sorted(_DICT)

def _score(choice: str, target: str) -> str:
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


def _func(args):
    choice = args[0]
    score = sum(evaluate(start(target).choose(choice), g=args[1][0], y=args[1][1]) for target in WORDS)
    return (choice, score)


def _find_start_words(factors: Tuple[float, float]):
    file_name = f'resources/cache/best_{factors[0]:.3f}_{factors[1]:.3f}.txt'
    try:
        with open(file_name, 'r') as f:
            tops = f.readline().split()
            return tops
    except:
        pass

    trials = process_map(_func, zip(WORDS, [factors] * len(WORDS)),
                         total=len(WORDS), desc=f'Evaluating starting words for {factors}')
    trials.sort(key=lambda x: -x[1])
    tops = [t[0] for t in trials[:5]]

    with open(file_name, 'w') as f:
        f.write(' '.join(tops))
    return tops


class State(NamedTuple):
    target: str
    guesses: Tuple
    responses: Tuple

    def print(self):
        for i in range(len(self.guesses)):
            if i < len(self.responses):
                outcome = self.responses[i]
            else:
                outcome = '?????'
            print(self.guesses[i], '->', outcome)

    def last_response(self) -> str:
        return self.responses[-1] if self.responses else None

    def finished(self) -> bool:
        return self.last_response() == 'GGGGG'

    def valid(self) -> bool:
        return all(valid_word(w) for w in self.guesses)

    def best_choice(self, factors=(3.0, 1.0)):

        return suggestions(self.guesses, self.responses, factors)[0]

    def choose(self, choice: str) -> State:
        if not valid_word(choice):
            raise ValueError(f'Invalid word: {choice}')
        return State(self.target, self.guesses + (choice,), self.responses + (_score(choice, self.target),))

    def consistent(self, word: str) -> bool:
        return all(_score(guess, target=word) == response for guess, response in zip(self.guesses, self.responses))

    def guess_count(self):
        return len(self.guesses)


def random_start():
    return start(random.choice(WORDS))


def start(target: str) -> State:
    if not valid_word(target):
        raise ValueError(f'Invalid word: {target}')
    return State(target, tuple(), tuple())


def valid_word(s) -> bool:
    return s in _DICT


def evaluate(s: State, g: float, y: float):
    return sum(g if x == 'G' else (y if x == 'Y' else 0) for x in s.last_response())


def suggestions(guesses: Tuple[str], responses: Tuple[str], factors=(3, 1)):
    if not guesses:
        tops = _find_start_words(factors)
        return tops

    # Find words that are consistent with the reported pattern
    possibles = [w for w in WORDS if
                 all(_score(guess, target=w) == response for guess, response in zip(guesses, responses))]

    trials = []
    for choice in possibles:
        score = sum(evaluate(start(target).choose(choice), g=factors[0], y=factors[1]) for target in WORDS)
        trials.append((choice, score))

    trials.sort(key=lambda x: -x[1])
    return [x[0] for x in trials[:5]]
