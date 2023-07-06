"""main"""
import statistics

from tqdm.contrib.concurrent import process_map

import wordle

WORDS = wordle.WORDS[::10]


def autoplay(args) -> int:
    game = wordle.start(args[0])
    while not game.finished():
        game = game.choose(game.best_choice(factors=args[1]))
    return game.guess_count()


def average_game_length(factors):
    values = process_map(autoplay, zip(WORDS, [factors] * len(WORDS)),
                         total=len(WORDS), desc=f'Calculating average game length for {factors}')
    return statistics.mean(values)


if __name__ == '__main__':
    ratios = [3.0, 2.9, 2.95, 3.05, 3.1, 3.15, 3.2]

    for r in ratios:
        factors = (r, 1)
        print('Considering', factors)
        wordle._find_start_words(factors)
        mean = average_game_length(factors)
        print(f'Average for {factors} = {mean}')
