import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup

from model import Solution, Word

G = 71
Y = 89
N = 45


@dataclass
class DictionaryEntry:
    value: str
    pos: str
    gloss: str

    @classmethod
    def from_line(cls, line: str):
        line = line.strip()
        if line[0] == '"':
            line = line[1:-1]
        p = line.index('(')
        q = line.index(')')
        return cls(
            line[:p].lower().strip(),
            line[p + 1:q].lower().strip(),
            line[q + 1:].strip()
        )

    def is_plural(self):
        return self.pos == 'n. pl.'

    def __str__(self):
        return f"{self.value} ({self.pos}) ({self.gloss})"

    def is_noun(self):
        return self.pos == 'n.'


def read_words_5() -> [Word]:
    result = set()
    with open('resources/base words.txt', 'rt') as file:
        for line in file.readlines():
            line = line.strip()
            if len(line) == 5:
                result.add(line)

    f_map = defaultdict(lambda: 10000.0)
    with open('resources/word frequencies.csv', 'rt') as file:
        file.readline()  # Skip header
        for line in file.readlines():
            line = line.split(',')
            f_map[line[0].strip()] = float(line[1].strip())

    base = f_map['the']

    missing = [w for w in result if w not in f_map]
    print('No freq:', missing)
    tuples = [(f_map[w], w) for w in result]
    tuples.sort(reverse=True)
    results = [Word(t[1], idx, t[0] / base) for idx, t in enumerate(tuples)]
    return results


def read_dict() -> List[DictionaryEntry]:
    results = []
    base = Path('resources/Dictionary')
    for path in base.glob('?.csv'):
        with open(path, 'rt', encoding='ISO 8859-1') as file:
            for line in file.readlines():
                if len(line) > 5:
                    word = DictionaryEntry.from_line(line)
                    results.append(word)
    return results


def read_solutions() -> List[Solution]:
    results = []
    solutions_url = 'https://www.stockq.org/life/wordle-answers.php#all'
    with urlopen(solutions_url) as response:
        soup = BeautifulSoup(response, 'html.parser')
        for tr in soup.find_all('tr'):
            items = tr.findAll('td')
            n_items = len(items)
            if n_items == 3 or n_items == 6:
                sol = Solution.from_triple(items[0], items[1], items[2])
                results.append(sol)
            if n_items == 6:
                sol = Solution.from_triple(items[3], items[4], items[5])
                results.append(sol)

    with open('resources/solutions.pickle', 'wb') as file:
        pickle.dump(results, file)
    return results


def evaluate(choice: str, target: str) -> bytes:
    outcome = bytearray('-----', 'ascii')
    available = [c for c in target]

    # Handle greens
    for idx in range(5):
        if choice[idx] == target[idx]:
            outcome[idx] = G
            available.remove(choice[idx])

    # Handle yellows
    for idx in range(5):
        if outcome[idx] == N and choice[idx] in available:
            outcome[idx] = Y
            available.remove(choice[idx])

    return bytes(outcome)


def build_score_possibilities() -> [bytes]:
    result = []
    for a1 in [N, Y, G]:
        for a2 in [N, Y, G]:
            for a3 in [N, Y, G]:
                for a4 in [N, Y, G]:
                    for a5 in [N, Y, G]:
                        result.append(bytes([a1, a2, a3, a4, a5]))
    return result


if __name__ == '__main__':
    solutions = read_solutions()
    words = read_words_5()
    dictionary = read_dict()

    print("#5 letter words: ", len(words))

    # Remove simple noun plurals (Wordle does not use them)
    noun_4 = {w.value + 's' for w in dictionary if len(w.value) == 4 and w.is_noun()}
    noun_3 = {w.value + 'es' for w in dictionary if len(w.value) == 3 and w.is_noun()}
    for w in list(words):
        if w.value in noun_4 or w in noun_3:
            words.remove(w)
    print("#non plural    : ", len(words))

    word_set = {w.value for w in words}

    for s in solutions:
        if s.value not in word_set:
            print("Missing solution:", s)

    n = len(words)
    with open('resources/words.pickle', 'wb') as f:
        pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)

    outcomes = build_score_possibilities()
    with open('resources/outcomes.pickle', 'wb') as f:
        strs = [o.decode('ascii') for o in outcomes]
        pickle.dump(strs, f, protocol=pickle.HIGHEST_PROTOCOL)

    rev_map = {}
    for i, s in enumerate(outcomes):
        rev_map[s] = i

    print("building scores")
    scores = bytearray(n * n)
    last = 0
    for a in range(n):
        percent = a * 100 // n
        if percent != last:
            if percent % 10:
                print('.', end='')
            else:
                print(f"({percent}%)", end='')
            last = percent
        for b in range(n):
            scores[a * n + b] = rev_map[evaluate(words[a].value, words[b].value)]
    with open('resources/scores.pickle', 'wb') as f:
        pickle.dump(bytes(scores), f, protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\nfinished')
