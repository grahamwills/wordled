from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

logging.basicConfig(level='DEBUG')
LOGGER = logging.getLogger('analyze')
LOGGER.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
LOGGER.addHandler(handler)

HARD: bool = True

# These are the words we will use
LOGGER.info('READING WORDS')
with open('../resources/words.pickle', 'rb') as f:
    WORDS = pickle.load(f)
N_WORDS = len(WORDS)
LOGGER.info(f'READ {N_WORDS} WORDS')

# Pickled outcomes
LOGGER.info('READING PRE_CALCULATED OUTCOMES')
with open('../resources/outcomes.pickle', 'rb') as f:
    OUTCOMES: [str] = pickle.load(f)
LOGGER.info(f'READ {len(OUTCOMES)} PRE_CALCULATED OUTCOMES')

# Pickled scores
LOGGER.info('READING PRE_CALCULATED SCORES')
with open('../resources/scores.pickle', 'rb') as f:
    SCORES: bytes = pickle.load(f)
LOGGER.info(f'READ {len(SCORES)} PRE_CALCULATED SCORES')


def pp(bb: [int]) -> str:
    return '|'.join(WORDS[b] for b in bb)


def evaluate_possibilities(guess: int, possible: list[int], limit: int) -> [[int]] or None:
    """ limit is the largest set to split allowed"""
    outcomes = [None] * 243
    for p in possible:
        guess_result = SCORES[guess * N_WORDS + p]
        which = outcomes[guess_result]
        if not which:
            outcomes[guess_result] = which = []

        if len(which) >= limit - 1:
            return None

        which.append(p)
    return outcomes


@dataclass
class Segment:
    outcome: int
    node: Node

    def display(self, depth: int = 0):
        leader = '  ' * (depth - 1) + ' '
        print(leader, OUTCOMES[self.outcome], ': ', pp(self.node.possible), sep='')
        if not self.node.only_one_possible():
            self.node.display()

    def only_one_possible(self):
        return self.node.only_one_possible()


@dataclass
class Node:
    possible: [int]
    guesses: [(int, [Segment])] = field(default_factory=list)
    depth: int = 0

    def max_guesses_needed(self) -> int:
        if self.only_one_possible():
            return 0
        if not self.guesses:
            raise RuntimeError('should have guesses')
        mx = 0
        for _, segments in self.guesses:
            mx = max(mx, max(s.node.max_guesses_needed() for s in segments))
        return 1 + mx

    def display(self):
        leader = '  ' * self.depth
        if not self.guesses:
            print(leader, 'FINAL RESULT: ', pp(self.possible), sep='')
            return

        print(leader, pp(self.possible), ' [', self.max_guesses_needed(), ']', sep='')
        for g, seg in self.guesses:
            print(leader, f"Guess@{self.depth}: {WORDS[g]}", sep='')
            for s in seg:
                s.display(self.depth + 1)

    def decides_outcome_immediately(self) -> bool:
        """ Don't need to follow any further; this guess solves it"""
        return all(s.only_one_possible() for _, s in self.guesses)

    def needs_following(self) -> bool:
        return not self.only_one_possible()

    def only_one_possible(self) -> bool:
        return len(self.possible) < 2

    def keep_best(self):
        """ The best is the min/max -- the guess with the minimum worst case length """
        if not self.guesses:
            # No work needed
            return
        best = (None, 99)
        for guess, segments in self.guesses:
            needed = 0
            for s in segments:
                s.node.keep_best()
                needed = max(needed, s.node.max_guesses_needed())
            if needed < best[1]:
                best = ((guess, segments), needed)
        self.guesses = [best[0]]


def build_node_children(node: Node, max_children: int = 5) -> [Node]:
    if HARD:
        candidates = node.possible
    else:
        candidates = range(0, N_WORDS)
    child_nodes = []

    # Sorted list, with the worst being the last
    best = []

    limit = 999999  # The worst we can tolerate
    for c in candidates:
        outcomes = evaluate_possibilities(c, node.possible, limit)
        if not outcomes:
            continue

        longest = max(len(o) for o in outcomes if o)
        if len(best) < max_children:
            best.append((longest, c, outcomes))
        elif longest < best[-1][0]:
            # This is better than the worst of our best, so it replaces
            limit = longest
            best[-1] = (longest, c, outcomes)
            best.sort(key=lambda x: x[:-1])

    # Use the best ones
    for _, c, outcomes in best:
        segments = [Segment(k, Node(v, depth=node.depth + 1)) for k, v in enumerate(outcomes) if v]
        node.guesses.append((c, segments))
        child_nodes += [s.node for s in segments]
    return child_nodes


def step_downward(nodes: [Node]) -> [Node]:
    n = len(nodes)
    print(f'Processing {n} nodes: ', end='', flush=True)

    next_nodes = []
    last_percent = 0
    for i, node in enumerate(nodes):
        percent = (i * 1000 // n) * 0.1
        if percent != last_percent:
            if last_percent % 5:
                print('.', end='', flush=True)
            else:
                print(f"\n({i} - {percent:0.1f}%)", end='', flush=True)
            last_percent = percent
        children = build_node_children(node)
        for c in children:
            if not c.only_one_possible():
                next_nodes.append(c)

    print('\n')
    return next_nodes


if __name__ == '__main__':
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.info(f"# valid words = {N_WORDS}")

    # Initialize the processing list

    top = Node([i for i in range(0, N_WORDS)])
    to_process = [top]

    next_process = []

    while to_process:
        to_process = step_downward(to_process)

    top.keep_best()

    top.display()
