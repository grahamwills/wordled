from __future__ import annotations

import logging
import math
import multiprocessing
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing.shared_memory import SharedMemory

HARD: bool = True
MAX_CANDIDATES = 5


class WordleContext:
    words: [str]
    n_words: int
    outcomes: [str]
    scores: bytes
    shared_scores: SharedMemory

    def __init__(self):
        # These are the words we will use
        with open('../resources/words.pickle', 'rb') as f:
            self.words = pickle.load(f)
        self.n_words = len(self.words)

        # Pickled outcomes
        with open('../resources/outcomes.pickle', 'rb') as f:
            self.outcomes: [str] = pickle.load(f)

        # Pickled scores
        with open('../resources/scores.pickle', 'rb') as f:
            self.scores: bytes = pickle.load(f)

        self.shared_scores = SharedMemory(create=True, size=len(self.scores))
        self.shared_scores.buf[:len(self.scores)] = self.scores


def pp(bb: [int], ctx: WordleContext) -> str:
    return '|'.join(ctx.words[b] for b in bb)


def evaluate_possibilities(guess: int, possible: list[int], limit: int, scores: bytes, n_words: int) -> [[int]] or None:
    """ limit is the largest set to split allowed"""
    outcomes = [None] * 243
    for p in possible:
        guess_result = scores[guess * n_words + p]
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

    def display(self, ctx: WordleContext, depth: int = 0):
        leader = '  ' * (depth - 1) + ' '
        print(leader, ctx.outcomes[self.outcome], ': ', pp(self.node.possible, ctx), sep='')
        if not self.node.only_one_possible():
            self.node.display(ctx)

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

    def display(self, ctx: WordleContext):
        leader = '  ' * self.depth
        if not self.guesses:
            print(leader, 'FINAL RESULT: ', pp(self.possible), sep='')
            return

        print(leader, pp(self.possible, ctx), ' [', self.max_guesses_needed(), ']', sep='')
        for g, seg in self.guesses:
            print(leader, f"Guess@{self.depth}: {ctx.words[g]}", sep='')
            for s in seg:
                s.display(ctx, self.depth + 1)

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


def add_candidates_to_node(node: Node, best, needs_splitting: [Node]):
    # Use the best ones
    for _, c, outcomes in best:
        segments = []
        for k, v in enumerate(outcomes):
            if v:
                child = Node(v, depth=node.depth + 1)
                segments.append(Segment(k, child))
                if len(v) > 1:
                    needs_splitting.append(child)


def find_best_candidates(node, scores: memoryview, n_words: int) -> []:
    candidates = node.possible
    # Sorted list, with the worst being the last
    best = []
    limit = 999999  # The worst we can tolerate
    for c in candidates:
        outcomes = evaluate_possibilities(c, node.possible, limit, scores, n_words)
        if not outcomes:
            continue

        longest = max(len(o) for o in outcomes if o)
        if len(best) < 5:
            # 5 best candidates
            best.append((longest, c, outcomes))
        elif longest < best[-1][0]:
            # This is better than the worst of our best, so it replaces
            limit = longest
            best[-1] = (longest, c, outcomes)
            best.sort(key=lambda x: x[:-1])
    return best


def find_best_candidates_for_block(nodes: [Node], scores_mem: SharedMemory, n_words: int) -> []:
    results = [find_best_candidates(n, scores_mem.buf, n_words) for n in nodes]
    return results


def step_downward(nodes: [Node], ctx: WordleContext) -> [Node]:
    n = len(nodes)

    t0 = datetime.now()
    print(f'Processing {n} nodes', flush=True)

    next_nodes = []

    cpu_count = multiprocessing.cpu_count()

    block_size = max(math.ceil(n / cpu_count), 100)
    n_blocks = math.ceil(n / block_size)
    assert n_blocks <= cpu_count

    blocks = [
        nodes[i * block_size: min((i + 1) * block_size, n)] for i in range(n_blocks)
    ]

    t0 = datetime.now()
    with multiprocessing.Pool(cpu_count) as pool:
        single = partial(find_best_candidates_for_block, scores_mem=ctx.shared_scores, n_words=ctx.n_words)
        best_for_node_block = pool.map(single, blocks)
    t1 = datetime.now()
    best_for_node = [x for block in best_for_node_block for x in block]
    t2 = datetime.now()

    for node, best in zip(nodes, best_for_node):
        add_candidates_to_node(node, best, next_nodes)
    t3 = datetime.now()
    print('Inner timings', (t1 - t0).total_seconds(), (t2 - t1).total_seconds(), (t3 - t2).total_seconds())

    # for i, node in enumerate(nodes):
    #     best = find_best_candidates(node, ctx.shared_scores, ctx.n_words)
    #     children = add_candidates_to_node(node, best)
    #     for c in children:
    #         if not c.only_one_possible():
    #             next_nodes.append(c)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f'{n} nodes evaluated in {elapsed} seconds')
    return next_nodes


if __name__ == '__main__':

    logging.basicConfig(level='DEBUG')
    LOGGER = logging.getLogger('analyze')
    LOGGER.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)

    LOGGER.setLevel(logging.DEBUG)
    ctx = WordleContext()
    LOGGER.info(f'READ {ctx.n_words} WORDS')
    LOGGER.info(f'READ {len(ctx.outcomes)} PRE_CALCULATED OUTCOMES')
    LOGGER.info(f'READ {len(ctx.scores)} PRE_CALCULATED SCORES')

    # Initialize the processing list

    top = Node([i for i in range(0, ctx.n_words)])
    to_process = [top]

    next_process = []

    while to_process:
        to_process = step_downward(to_process, ctx)

    top.keep_best()

    top.display(ctx)
