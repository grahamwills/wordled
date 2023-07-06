"""main"""
import wordle

if __name__ == '__main__':

    guesses = []
    responses = []
    finished = False
    while not finished:
        suggestions = wordle.suggestions(tuple(guesses), tuple(responses))
        print("Suggestions:", ', '.join(suggestions))
        x = input('You entered: ')
        y = input('Response was: ')
        guesses.append(x)
        responses.append(y)
        finished = (y == 'GGGGG')
