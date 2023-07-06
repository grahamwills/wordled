"""Pre-processes the total word list"""
import re
from typing import List


def read_all() -> List[str]:
    """Reads all the words"""
    with open('rsrc/corncob.txt') as f:
        return f.readlines()


def write_all(words: List[str]):
    with open('rsrc/words.txt', 'w') as f:
        f.writelines(w + '\n' for w in words)


def _trim(w: str) -> str:
    i = w.find('[')
    return w[:i]


def _normalize(old: str):
    new = old.lower()
    new = re.sub(r'[àáâãäå]', 'a', new)
    new = re.sub(r'[èéêë]', 'e', new)
    new = re.sub(r'[ìíîï]', 'i', new)
    new = re.sub(r'[òóôõö]', 'o', new)
    new = re.sub(r'[ùúûü]', 'u', new)
    return new


def _valid_5_letter(s: str) -> bool:
    return len(s) == 5 and s.isalpha()


if __name__ == '__main__':
    all_words = read_all()
    n = len(all_words)
    print(f"Read {n} base words")

    all_words = [_trim(w) for w in all_words]
    all_words = [_normalize(w) for w in all_words]
    all_words = sorted({w for w in all_words if _valid_5_letter(w)})
    print(all_words[:100])
    n = len(all_words)
    print(f"After processing we have {n} base words")
    write_all(all_words)
