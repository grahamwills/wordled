from search import SearchNode, compatible, WORDS


def test_search_node():
    s = SearchNode('spell', 'heels', [])
    assert s.result == '--GGY'
    assert str(s) == 'heels[spell] -> --GGY'


def test_compatible():
    assert compatible('spell', 'heels', '--GGY')
    assert not compatible('spell', 'heels', 'G-GGY')
    assert compatible('xxabx', 'aaabb', '--GG-')


def test_filter_compatible():
    words = [w for w in WORDS if compatible(w, 'heels', '--GGY')]
    assert len(words) == 5
    assert " • ".join(words) == 'smell • smelt • spell • spelt • swell'


def test_search_a():
    top = SearchNode('spell', 'heels', WORDS)
    assert len(top.children) == 5
    assert top.deepest() == 2
