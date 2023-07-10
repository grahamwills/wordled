import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup, Tag

from items import Word

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


def read_words_5() -> [Word]:
    words = set()
    with open('resources/combined words.txt', 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 5:
                words.add(line)

    f_map = defaultdict(lambda: 10000.0)
    with open('resources/unigram_freq.csv', 'rt') as f:
        f.readline()  # Skip header
        for line in f.readlines():
            line = line.split(',')
            f_map[line[0].strip()] = float(line[1].strip())

    base = f_map['the']

    missing = [w for w in words if w not in f_map]
    print('No freq:', missing)
    tuples = [(f_map[w], w) for w in words]
    tuples.sort(reverse=True)
    results = [Word(t[1], i, t[0] / base) for i, t in enumerate(tuples)]
    return results


def read_dict(refresh: bool = False) -> List[DictionaryEntry]:
    if refresh:
        results = []
        base = Path('resources/Dictionary')
        for path in base.glob('?.csv'):
            with open(path, 'rt', encoding='ISO 8859-1') as f:
                for line in f.readlines():
                    if len(line) > 5:
                        word = DictionaryEntry.from_line(line)
                        results.append(word)

        with open('resources/dict.pickle', 'wb') as f:
            pickle.dump(results, f)


    else:
        with open('resources/dict.pickle', 'rb') as f:
            results = pickle.load(f)
    return results


def read_solutions(refresh: bool = False) -> List[Solution]:
    if refresh:
        results = []
        solutions_url = 'https://www.stockq.org/life/wordle-answers.php#all'
        with urlopen(solutions_url) as response:
            soup = BeautifulSoup(response, 'html.parser')
            for tr in soup.find_all('tr'):
                items = tr.findAll('td')
                n = len(items)
                if n == 3 or n == 6:
                    sol = Solution.from_triple(items[0], items[1], items[2])
                    results.append(sol)
                if n == 6:
                    sol = Solution.from_triple(items[3], items[4], items[5])
                    results.append(sol)

        with open('resources/solutions.pickle', 'wb') as f:
            pickle.dump(results, f)

        # Simple check we have all indices
        assert [s.index for s in results] == list(range(len(results)))
    else:
        with open('resources/solutions.pickle', 'rb') as f:
            results = pickle.load(f)
    return results


def evaluate(choice: str, target: str) -> bytes:
    outcome = bytearray('-----', 'ascii')
    available = [c for c in target]

    # Handle greens
    for i in range(5):
        if choice[i] == target[i]:
            outcome[i] = G
            available.remove(choice[i])

    # Handle yellows
    for i in range(5):
        if outcome[i] == N and choice[i] in available:
            outcome[i] = Y
            available.remove(choice[i])

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
    refresh = True
    solutions = read_solutions(refresh)
    words = read_words_5()
    dictionary = read_dict(refresh)

    print("#5 letter words: ", len(words))

    noun_4 = {w.value + 's' for w in dictionary if len(w.value) == 4 and w.is_noun()}
    noun_3 = {w.value + 'es' for w in dictionary if len(w.value) == 3 and w.is_noun()}

    for w in list(words):
        if w.value in noun_4 or w in noun_3:
            words.remove(w)

    print("#non plural    : ", len(words))

    word_set = {w.value for w in words}

    for s in solutions:
        if s.value not in word_set:
            print("Missing:", s)

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
