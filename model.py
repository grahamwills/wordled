from __future__ import annotations

import os
import pickle
from array import array
from dataclasses import dataclass
from datetime import datetime, date
from typing import NamedTuple, Iterable, List, Tuple, Callable, Dict

from bs4 import Tag

HARD: bool = True
MAX_CANDIDATES = 10
PRUNE_LIMIT = 100000
EVERY = 5


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


def score_weighted_average_first(stats: Iterable[Statistics]) -> float:
    return sum(1e-6 * s.n_unsolved_weighted + s.sum_depth_weighted for s in stats)


def score_unsolved_first(stats: Iterable[Statistics]) -> float:
    return sum(s.n_unsolved + 1e-6 * s.sum_depth for s in stats)


@dataclass
class Attempt:
    guess: int
    max_split_len: int
    outcomes: List[(int, Node)]
    _score: float = -1

    def copy(self) -> Attempt:
        outcomes = [(i, n.copy()) for i, n in self.outcomes]
        return Attempt(self.guess, self.max_split_len, outcomes)

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


class Candidate(NamedTuple):
    max_split_len: int
    guess: int
    outcomes: Dict[int, array]


    def __hash__(self):
        return hash(tuple(self.outcomes.keys()))


class Node:
    __slots__ = ['possible', 'attempts', 'statistics', 'depth', 'parent']

    def __init__(self, possibles: array, parent: Node or None = None):
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.possible = possibles
        self.attempts: List[Attempt] or None = None
        self.statistics: Statistics or None = None

    def copy(self) -> Node:
        other = Node(self.possible, self.parent)
        if self.attempts:
            other.attempts = [a.copy() for a in self.attempts]
        other.statistics = None
        return other

    def __repr__(self):
        return f"〔{len(self.possible)} possibles, {len(self.attempts)} children〕"

    def decides_outcome_immediately(self) -> bool:
        """ Don't need to follow any further; this guess solves it"""
        return all(s.node.only_one_possible() for _, s in self.attempts)

    def needs_following(self) -> bool:
        return not self.only_one_possible()

    def only_one_possible(self) -> bool:
        return len(self.possible) < 2

    def keep_best(self, scorer: Callable[[Iterable[Statistics]], float]):
        """ Reduce the number of attempts to a single best option"""
        # First we follow the tree downward to set up all our children
        if self.attempts:
            best: Tuple[Attempt or None, float] = (None, 9e99)
            for a in self.attempts:
                for _, s in a.outcomes:
                    s.keep_best(scorer)
                score = a.score(scorer)
                if score < best[1]:
                    best = (a, score)
            if self.depth == 0:
                # Keep all the top attempts
                self.attempts = sorted(self.attempts, key=lambda a: a._score)
            else:
                # Lower down the tree, just keep the best
                self.attempts = [best[0]]
                self.statistics = Statistics.combine(node.statistics for _, node in best[0].outcomes)
        else:
            assert len(self.possible) == 1
            f = frequency(self.possible[0])
            n_unsolved = 0 if self.depth < 6 else 1
            self.statistics = Statistics(1, n_unsolved, self.depth, f, f * n_unsolved, f * self.depth, self.depth)

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


def evaluate_possibilities(guess: int, possible: list[int], limit: int) -> dict[int, array] or None:
    """ limit is the largest set to split allowed"""
    outcomes = {}
    for p in possible:
        guess_result = SCORES[guess * N_WORDS + p]
        outcome_possibles = outcomes.get(guess_result, None)
        if not outcome_possibles:
            outcomes[guess_result] = array('i', [p])
        elif len(outcome_possibles) < limit:
            outcome_possibles.append(p)
        else:
            return None
    return outcomes


def add_candidates_to_node(node: Node, candidates: [Candidate], needs_splitting: [Node]):
    attempts = []
    for c in candidates:
        segments = []
        for k, v in c.outcomes.items():
            child = Node(v, parent=node)
            segments.append((k, child))
            if len(v) > 1:
                needs_splitting.append(child)
        attempts.append(Attempt(c.guess, c.max_split_len, segments))
    node.attempts = attempts


def find_best_candidates(node) -> List[Candidate]:
    # All possible words could be used
    candidates = node.possible if HARD else range(N_WORDS)

    # Sorted list, with the worst being the last
    best: List[Candidate] = []
    limit = 999999  # The worst we can tolerate
    for c in candidates:
        outcomes = evaluate_possibilities(c, node.possible, limit)
        if not outcomes:
            continue

        candidate_max_split_len = max(len(a) for a in outcomes.values())
        candidate = Candidate(candidate_max_split_len, c, outcomes)
        if candidate_max_split_len == 1:
            # This is the best we could ever have
            return [candidate]
        if len(best) < MAX_CANDIDATES:  # 5 best candidates
            best.append(candidate)
            if len(best) == MAX_CANDIDATES:
                best.sort(key=lambda x: x[:-1])
        elif candidate_max_split_len < limit:
            # This is better than the worst of our best, so it replaces it
            best[-1] = candidate
            best.sort(key=lambda x: x.max_split_len + 1e-6 * x.guess)
            limit = best[-1].max_split_len
    return best


def step_downward(nodes: List[Node]) -> List[Node]:
    n = len(nodes)
    time0 = datetime.now()
    print(f'Processing {n} nodes', flush=True)
    next_nodes = []

    node_candidates: List[ Tuple[Node, List[Candidate]]] = []
    all_candidates = []
    for node in nodes:
        candidates = find_best_candidates(node)
        node_candidates.append((node, candidates))
        all_candidates += candidates

    if len(all_candidates) > PRUNE_LIMIT:
        all_candidates = sorted(all_candidates, key=lambda c: c.max_split_len + c.guess * 1e-6)
        valid = set(all_candidates[0:PRUNE_LIMIT])

        for node, best in node_candidates:
            allowed = [b for b in best if b in valid]
            if not allowed:
                allowed = [best[0]]
            add_candidates_to_node(node, allowed, next_nodes)

    else:
        for node, best in node_candidates:
            add_candidates_to_node(node, best, next_nodes)

    elapsed = (datetime.now() - time0).total_seconds()
    print(f'{n} nodes evaluated in {elapsed} seconds')
    return next_nodes


if __name__ == '__main__':

    if os.path.isfile('results.csv'):
        os.remove('results.csv')

    t0 = datetime.now()

    # Initialize the processing list
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

    TRIALS = [
        # ('Max Depth | Sum Depth', score_depth),
        # ('Unsolved | Average', score_unsolved_first),
        ('Unsolved | Average (Weighted)', score_weighted_unsolved_first),
        # ('Average | Unsolved (Weighted)', score_weighted_average_first),
    ]
    for name, scorer in TRIALS:
        print('Trying scoring method:', name)
        if name == TRIALS[-1][0]:
            # Last step through, we can kill things
            top = base
        else:
            # Need to make sure it's a copy so we don't pollute it
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
            'Prune Limit': PRUNE_LIMIT,
            'Candidates': MAX_CANDIDATES,
            'Hard': HARD,
            'Method': name,
            'Time': int(t * 1000) / 1000,
        }, nyt_words)

        print(f"Time taken = {t}s ({(t3 - t2).total_seconds()} to select best paths)")

        top.attempts = top.attempts[:1]
        with open('tree.pickle', 'wb') as f:
            pickle.dump(top, f)
