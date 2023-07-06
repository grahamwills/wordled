from wordle import start, State, evaluate


def test_play():
    a = start('panel').choose('hands')
    assert a.last_response() == '-GG--'
    b = a.choose('peels')
    assert b.last_response() == 'GY-Y-'


def test_play_2():
    a = start('point').choose('strap')
    assert a.last_response() == '-Y--Y'
    b = a.choose('apple')
    assert b.last_response() == '-Y---'
    c = a.choose('pivot')
    assert c.last_response() == 'GY-YG'
    d = a.choose('point')
    assert d.last_response() == 'GGGGG'
    assert d.finished()


def test_evaluate():
    a = State('', ('',), ('----',))
    assert evaluate(a, g=3, y=1) == 0.0

    a = State('', ('',), ('YG--Y',))
    assert evaluate(a, g=3, y=1) == 5.0

    a = State('', ('',), ('YG--Y',))
    assert evaluate(a, g=3, y=1.5) == 6.0


def test_best_choice_1():
    a = start('point').choose('strap').choose('apple').choose('pivot')
    assert a.best_choice() == 'point'


def test_best_choice_2():
    a = start('point').choose('strap').choose('apple')
    assert a.best_choice() == 'point'


def test_best_choice_3():
    a = start('point').choose('strap')
    assert a.best_choice() == 'piety'

def test_best_choice():
    a = start('point')
    assert a.best_choice([3,1]) == 'tares'


def test_consistent():
    a = start('point').choose('strap')
    assert not a.consistent('apple')
    assert a.consistent('pivot')
    assert a.consistent('point')


def test_strategy():
    factors = [3, 1]
    a = start('point').choose('tares')
    while not a.finished():
        a = a.choose(a.best_choice(factors))
    assert a == State(target='point', guesses=('tares', 'point'), responses=('Y----', 'GGGGG'))
