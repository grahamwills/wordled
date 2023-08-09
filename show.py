import pickle
from contextlib import redirect_stdout
from typing import Tuple, List

from analyze import Node, SOLUTIONS, OUTCOMES, WORDS, Attempt
from model import Statistics


def read_optimal_tree() -> Node:
    with open('optimal tree.pickle', 'rb') as file:
        top = pickle.load(file)
        top.statistics = Statistics.combine(node.statistics for _, node in top.attempts[0].outcomes)
        return top


def evaluate_on_nyt(node: Node) -> Tuple[List[int], List[List[str]]]:
    success = [0] * 7
    failures = []
    for solution in SOLUTIONS:
        results = node.solve(solution.value)
        if len(results) < 7:
            success[len(results)] += 1
        else:
            failures.append(results)
    return success, failures


def display_node_title(node, code_index: int = None) -> bool:
    if node.attempts is None:
        assert len(node.possible) == 1
        code = OUTCOMES[code_index]
        answer = WORDS[node.possible[0]].value
        print('#' * (node.depth + 1), f"**{code}**:", answer.upper())
        print('DONE', '\n')
        return False
    attempt: Attempt = node.attempts[0]
    word = WORDS[attempt.guess].value
    s: Statistics = node.statistics
    if code_index is None:
        print('#', word.upper())
    else:
        code = OUTCOMES[code_index]
        if node.depth == 0:
            print('#', f"**{code}**", word.upper())
        else:
            print('###', f"**`{code}`**", ' ⇒ ', word.upper())
    print(f'*Matches {s.n_words} words. Up to {s.max_depth} total guesses may be needed. '
          f'Average guesses is {s.sum_depth_weighted / s.n_words_weighted:.1f}*', '\n')
    return True


def display(node: Node):
    for code, child in sorted(node.attempts[0].outcomes):
        if display_node_title(child, code):

            # | Syntax | Description |
            # | ----------- | ----------- |
            # | Header | Title |
            # | Paragraph | Text |
            print("| Guesses | Outcome | Next Guess | Count | Max | Average |")
            print("| ------- | ------- | ---------- | ----- | --- | ------- |")
            for code1, child1 in sorted(child.attempts[0].outcomes):
                guesses = WORDS[node.attempts[0].guess].value.upper() + \
                          ' • ' + WORDS[child.attempts[0].guess].value.upper()
                code_s = OUTCOMES[code1]
                if child1.attempts is None:
                    answer = WORDS[child1.possible[0]].value
                    print(f"| {guesses} | **{code_s}**: | {answer.upper()} | DONE | | |")
                else:
                    attempt: Attempt = child1.attempts[0]
                    word = WORDS[attempt.guess].value
                    s: Statistics = child1.statistics
                    print(f"| {guesses} | **{code_s}**: | {word.upper()} | {s.n_words} | {s.max_depth} "
                          f"| {s.sum_depth_weighted / s.n_words_weighted:.1f} |")
            print()
        print('-' * 60)


if __name__ == '__main__':
    model = read_optimal_tree()

    with open('output.md', 'w') as f:
        with redirect_stdout(f):

            display_node_title(model)

            histogram, failures = evaluate_on_nyt(model)

            print('\n## Histogram of results on first 761 NYT words\n')

            m = max(histogram)
            for i, v in enumerate(histogram):
                if i:
                    print('   ', i, f'{v:3}', '*' * (30 * v // m))

            print('\n')

            print('\n## Failures on first 761 NYT words\n')

            for fail in failures:
                print('   ', ' • '.join(fail))

            print('\n# Guess Guide\n')

            display(model)
