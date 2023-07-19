from __future__ import annotations

import os
import pickle
from array import array
from dataclasses import dataclass
from datetime import datetime, date
from typing import NamedTuple, Iterable, List, Tuple, Callable

from bs4 import Tag

HARD: bool = True
MAX_CANDIDATES = 3


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


WORDS: List[Word]
N_WORDS: int
OUTCOMES: [str]
SCORES: bytes
SOLUTIONS: List[Solution]

# These are the words we will use
with open('resources/words.pickle', 'rb') as f:
    WORDS = pickle.load(f)
N_WORDS = len(WORDS)

WORD_MAP = dict(((z.value, i) for i, z in enumerate(WORDS)))

# Pickled outcomes
with open('resources/outcomes.pickle', 'rb') as f:
    OUTCOMES: [str] = pickle.load(f)

# Pickled scores
with open('resources/scores.pickle', 'rb') as f:
    SCORES: bytes = pickle.load(f)

# Pickled solutions
with open('resources/solutions.pickle', 'rb') as f:
    SOLUTIONS: bytes = pickle.load(f)

print(f'READ {N_WORDS} WORDS')
print(f'READ {len(OUTCOMES)} PRE_CALCULATED OUTCOMES')
print(f'READ {len(SCORES)} PRE_CALCULATED SCORES')
print(f'READ {len(SOLUTIONS)} KNOWN SOLUTIONS')


def words_as_str(indices: [int]) -> str:
    return '|'.join(WORDS[b].value for b in indices)


def frequency(index: int) -> float:
    return WORDS[index].freq


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


def score_depth(stats: Iterable[Statistics]) -> float:
    return max(s.max_depth * 100 + s.sum_depth / s.n_words for s in stats)


def score_weighted_unsolved_first(stats: Iterable[Statistics]) -> float:
    return sum(s.n_unsolved_weighted + 1e-6 * s.sum_depth_weighted for s in stats)


def score_unsolved_first(stats: Iterable[Statistics]) -> float:
    return sum(s.n_unsolved + 1e-6 * s.sum_depth for s in stats)


@dataclass
class Attempt:
    guess: int
    outcomes: List[(int, Node)]
    _score: float = -1

    def copy(self) -> Attempt:
        outcomes = [(i, n.copy()) for i, n in self.outcomes]
        return Attempt(self.guess, outcomes, -1)

    def score(self, scorer: Callable[[Iterable[Statistics]], float]) -> float:
        if self._score < 0:
            self._score = scorer(s.statistics for _, s in self.outcomes)
        return self._score

    def solve(self, word: int) -> List[str]:
        guess_result = SCORES[self.guess * N_WORDS + word]
        result_as_str = OUTCOMES[guess_result]
        outcome = f"{WORDS[self.guess]}:{result_as_str}"
        if result_as_str == 'GGGGG':
            return [outcome]
        for o, n in self.outcomes:
            if o == guess_result:
                return [outcome] + n.solve(word)
        raise RuntimeError('Could not find outcome: ' + result_as_str)

    def __repr__(self):
        return f"<Attempt: {WORDS[self.guess]} → {len(self.outcomes)}>"


class Node:
    __slots__ = ['possible', 'attempts', 'statistics']

    def __init__(self, possibles: array):
        self.possible = possibles
        self.attempts: List[Attempt] or None = None
        self.statistics: Statistics or None = None

    def copy(self) -> Node:
        other = Node(self.possible)
        if self.attempts:
            other.attempts = [a.copy() for a in self.attempts]
        other.statistics = None
        return other

    def __repr__(self):
        return f"〔{len(self.possible)} possibles, {len(self.attempts)} children〕"

    def max_guesses_needed(self) -> int:
        if self.only_one_possible():
            return 0
        if not self.attempts:
            raise RuntimeError('should have guesses')
        mx = 0
        for a in self.attempts:
            mx = max(mx, max(s.max_guesses_needed() for _, s in a.outcomes))
        return 1 + mx

    def display(self, depth: int = 0):
        if depth > 1:
            return
        leader = '  ' * depth
        print(leader, words_as_str(self.possible), ' [', self.max_guesses_needed(), ']', sep='')
        for a in self.attempts:
            print(leader, f"Guess@{depth}: {WORDS[a.guess]}", sep='')
            for outcome, node in OUTCOMES:
                print(leader, ' ', OUTCOMES[outcome], ': ', words_as_str(node.possible), sep='')
                if not node.only_one_possible():
                    node.display(depth + 1)

    def decides_outcome_immediately(self) -> bool:
        """ Don't need to follow any further; this guess solves it"""
        return all(s.node.only_one_possible() for _, s in self.attempts)

    def needs_following(self) -> bool:
        return not self.only_one_possible()

    def only_one_possible(self) -> bool:
        return len(self.possible) < 2

    def keep_best(self, scorer: Callable[[Iterable[Statistics]], float], depth: int = 0):
        """ Reduce the number of attempts to a single best option"""
        # First we follow the tree downward to set up all our children
        if self.attempts:
            best: Tuple[Attempt or None, float] = (None, 9e99)
            for a in self.attempts:
                for _, s in a.outcomes:
                    s.keep_best(scorer, depth + 1)
                score = a.score(scorer)
                if score < best[1]:
                    best = (a, score)
            if depth == 0:
                # Keep all the top attempts
                self.attempts = sorted(self.attempts, key=lambda a: a._score)
            else:
                # Lower down the tree, just keep the best
                self.attempts = [best[0]]
                self.statistics = Statistics.combine(node.statistics for _, node in best[0].outcomes)
        else:
            assert len(self.possible) == 1
            f = frequency(self.possible[0])
            n_unsolved = 0 if depth < 6 else 1
            self.statistics = Statistics(1, n_unsolved, depth, f, f * n_unsolved, f * depth, depth)

    def store(self, prefixes: dict, used: List[int]):
        keys = list(prefixes.keys())
        values = [str(prefixes[k]) for k in keys]

        need_header = not os.path.isfile('results.csv')

        with open('results.csv', 'at') as f:
            if need_header:
                f.write(
                    ', '.join(keys + ['Guess', 'Unsolved (NYT)', 'Average Depth (NYT)'] + Statistics.headers()) + '\n')
            for a in self.attempts:
                nyt = [len(a.solve(u)) for u in used]
                nyt_vals = [str(i) for i in (sum(1 for k in nyt if k > 6), sum(nyt) / len(nyt))]
                statistics = Statistics.combine(node.statistics for _, node in a.outcomes)
                f.write(', '.join(values + [str(WORDS[a.guess])] + nyt_vals + statistics.describe()) + '\n')

    def solve(self, word: int or str) -> List[str]:
        # Returns the solution chain for the given word
        if isinstance(word, str):
            word = WORD_MAP[word.lower()]
        if self.attempts:
            # Follow the top attempt
            return self.attempts[0].solve(word)
        else:
            assert self.possible[0] == word
            return [f"{WORDS[self.possible[0]]}:GGGGG"]


def evaluate_possibilities(guess: int, possible: list[int], limit: int) -> ([array], int) or None:
    """ limit is the largest set to split allowed"""
    outcomes: List[array or None] = [None] * 243
    longest = 0
    for p in possible:
        guess_result = SCORES[guess * N_WORDS + p]
        which = outcomes[guess_result]
        if which:
            which.append(p)
            longest = max(longest, len(which))
            if longest >= limit:
                return None, longest
        else:
            outcomes[guess_result] = array('i', [p])

    return outcomes, longest


def add_candidates_to_node(node: Node, best: [(int, array)], needs_splitting: [Node]):
    attempts = []
    for c, outcomes in best:
        segments = []
        for k, v in enumerate(outcomes):
            if v:
                child = Node(v)
                segments.append((k, child))
                if len(v) > 1:
                    needs_splitting.append(child)
        attempts.append(Attempt(c, segments))
    node.attempts = attempts


def find_best_candidates(node) -> [int, array]:
    # All possible words could be used
    candidates = node.possible if HARD else range(N_WORDS)

    # Sorted list, with the worst being the last
    best = []
    limit = 999999  # The worst we can tolerate
    for c in candidates:
        outcomes, longest = evaluate_possibilities(c, node.possible, limit)
        if not outcomes:
            continue
        if len(best) < MAX_CANDIDATES:  # 5 best candidates
            best.append((longest, c, outcomes))
        elif longest < best[-1][0]:
            # This is better than the worst of our best, so it replaces it
            limit = longest
            best[-1] = (longest, c, outcomes)
            best.sort(key=lambda x: x[:-1])
    return [b[1:] for b in best]


def step_downward(nodes: [Node], ) -> [Node]:
    n = len(nodes)
    time0 = datetime.now()
    print(f'Processing {n} nodes', flush=True)
    next_nodes = []
    for node in nodes:
        best = find_best_candidates(node)
        add_candidates_to_node(node, best, next_nodes)
    elapsed = (datetime.now() - time0).total_seconds()
    print(f'{n} nodes evaluated in {elapsed} seconds')
    return next_nodes


if __name__ == '__main__':

    if os.path.isfile('results.csv'):
        os.remove('results.csv')

    t0 = datetime.now()

    # Initialize the processing list
    EVERY = 5
    base = Node(array('i', (i for i in range(0, N_WORDS) if i % EVERY == 0)))
    to_process = [base]

    solutions = set(w.value for w in SOLUTIONS).intersection({WORDS[i].value for i in base.possible})
    nyt_words = [WORD_MAP[w] for w in solutions]
    print(f"{len(solutions)} of {len(SOLUTIONS)} are allowable for this fraction of words")

    # Loop until nothing left to process
    next_process = []
    while to_process:
        to_process = step_downward(to_process)

    t1 = datetime.now()

    for name, scorer in [
        ('Max Depth | Sum Depth', score_depth),
        ('Unsolved | Average', score_unsolved_first),
        ('Unsolved | Average (Weighted)', score_weighted_unsolved_first),
    ]:
        top = base.copy()
        t2 = datetime.now()
        top.keep_best(scorer)
        t3 = datetime.now()

        t = (t1 - t0).total_seconds() + (t3 - t2).total_seconds()

        # for solution in solutions:
        #     guesses = top.solve(solution)
        #     print(str(solution) + ' -> ' + ' | '.join(guesses))

        top.store({
            'N': len(top.possible),
            'Candidates': MAX_CANDIDATES,
            'Hard': HARD,
            'Method': name,
            'Time': int(t * 1000) / 1000,
        }, nyt_words)

        print(f"Time taken = {t}s ({(t3 - t2).total_seconds()} to prune)")
