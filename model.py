from __future__ import annotations

import pickle
from array import array
from datetime import datetime
from typing import NamedTuple, Iterable, List

from items import Word

HARD: bool = True
MAX_CANDIDATES = 5


class WordleContext:
    words: [Word]
    n_words: int
    outcomes: [str]
    scores: bytes

    def __init__(self):
        # These are the words we will use
        with open('resources/words.pickle', 'rb') as f:
            self.words = pickle.load(f)
        self.n_words = len(self.words)

        # Pickled outcomes
        with open('resources/outcomes.pickle', 'rb') as f:
            self.outcomes: [str] = pickle.load(f)

        # Pickled scores
        with open('resources/scores.pickle', 'rb') as f:
            self.scores: bytes = pickle.load(f)

        print(f'READ {self.n_words} WORDS')
        print(f'READ {len(self.outcomes)} PRE_CALCULATED OUTCOMES')
        print(f'READ {len(self.scores)} PRE_CALCULATED SCORES')

    def words_as_str(self, indices: [int]) -> str:
        return '|'.join(self.words[b].value for b in indices)

    def frequency(self, index: int) -> float:
        return self.words[index].freq


class Statistics(NamedTuple):
    n_words: int
    n_unsolved: int
    sum_depth: float
    n_words_weighted: float
    n_unsolved_weighted: float
    sum_depth_weighted: float
    max_depth: int

    @classmethod
    def sum(cls, stats: Iterable[Statistics]) -> Statistics:
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


class Attempt(NamedTuple):
    guess: int
    outcomes: [(int, Node)]


class Node:
    __slots__ = ['possible', 'attempts', 'statistics']

    def __init__(self, possibles: array):
        self.possible = possibles
        self.attempts = None
        self.statistics = None

    def max_guesses_needed(self) -> int:
        if self.only_one_possible():
            return 0
        if not self.attempts:
            raise RuntimeError('should have guesses')
        mx = 0
        for a in self.attempts:
            mx = max(mx, max(s.max_guesses_needed() for _, s in a.outcomes))
        return 1 + mx

    def display(self, ctx: WordleContext, depth: int = 0):
        if depth > 1:
            return
        leader = '  ' * depth
        print(leader, ctx.words_as_str(self.possible), ' [', self.max_guesses_needed(), ']', sep='')
        for a in self.attempts:
            print(leader, f"Guess@{depth}: {ctx.words[a.guess]}", sep='')
            for outcome, node in a.outcomes:
                print(leader, ' ', ctx.outcomes[outcome], ': ', ctx.words_as_str(node.possible), sep='')
                if not node.only_one_possible():
                    node.display(ctx, depth + 1)

    def decides_outcome_immediately(self) -> bool:
        """ Don't need to follow any further; this guess solves it"""
        return all(s.node.only_one_possible() for _, s in self.attempts)

    def needs_following(self) -> bool:
        return not self.only_one_possible()

    def only_one_possible(self) -> bool:
        return len(self.possible) < 2

    def keep_best(self, ctx: WordleContext, depth: int = 0):
        """ Reduce the number of attempts to a single best option"""
        if self.attempts:
            best = (None, 99)
            for guess in self.attempts:
                needed = 0
                for _, s in guess.outcomes:
                    s.keep_best(ctx, depth + 1)
                    needed = max(needed, s.max_guesses_needed())
                if needed < best[1]:
                    best = (guess, needed)
            self.attempts = [best[0]]
            self.statistics = Statistics.sum(node.statistics for _, node in best[0].outcomes)
        else:
            assert len(self.possible) == 1
            f = ctx.frequency(self.possible[0])
            n_unsolved = 0 if depth < 6 else 1
            self.statistics = Statistics(1, n_unsolved, depth, f, f * n_unsolved, f * depth, depth)

    def display_statistics(self):
        assert len(self.possible) == self.statistics.n_words
        print(f"# Words    =                  {self.statistics.n_words}")
        print(f"# Max Depth =                 {self.statistics.max_depth}")
        print(f"# Unsolved % =                {100 * self.statistics.n_unsolved / self.statistics.n_words:.3f}%")
        print(f"# Unsolved % (weighted) =     "
              f"{100 * self.statistics.n_unsolved_weighted / self.statistics.n_words_weighted:.3f}%")
        print(f"# Average Solves =            {1 + self.statistics.sum_depth / self.statistics.n_words:.3f}")
        print(f"# Average Solves (weighted) = "
              f"{1 + self.statistics.sum_depth_weighted / self.statistics.n_words_weighted:.3f}")


def evaluate_possibilities(guess: int, possible: list[int],
                           limit: int, scores: bytes, n_words: int) -> ([array], int) or None:
    """ limit is the largest set to split allowed"""
    outcomes:List[array or None] = [None] * 243
    longest = 0
    for p in possible:
        guess_result = scores[guess * n_words + p]
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


def find_best_candidates(node, scores: bytes, n_words: int) -> [int, array]:
    candidates = node.possible
    # Sorted list, with the worst being the last
    best = []
    limit = 999999  # The worst we can tolerate
    for c in candidates:
        outcomes, longest = evaluate_possibilities(c, node.possible, limit, scores, n_words)
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


def step_downward(nodes: [Node], ctx: WordleContext) -> [Node]:
    n = len(nodes)
    t0 = datetime.now()
    print(f'Processing {n} nodes', flush=True)
    next_nodes = []
    for node in nodes:
        best = find_best_candidates(node, ctx.scores, ctx.n_words)
        add_candidates_to_node(node, best, next_nodes)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f'{n} nodes evaluated in {elapsed} seconds')
    return next_nodes


if __name__ == '__main__':

    t0 = datetime.now()
    ctx = WordleContext()

    # Initialize the processing list
    EVERY = 5
    top = Node(array('i', (i for i in range(0, ctx.n_words) if i % EVERY == 0)))
    to_process = [top]

    # Loop until nothing left to process
    next_process = []
    while to_process:
        to_process = step_downward(to_process, ctx)

    t1 = datetime.now()

    top.keep_best(ctx)

    t2 = datetime.now()

    top.display_statistics()

    print(f"Time taken = {(t2 - t0).total_seconds()}s ({(t2 - t1).total_seconds()} to prune)")

    # top.display(ctx)
