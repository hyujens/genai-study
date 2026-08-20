from concept_with_keyword_search import knowledge, naive_keyword_search


def test_keyword_search():
    ranks = naive_keyword_search("What is Underwhelming Spatula?", knowledge)
    assert ranks[0] == knowledge[0]
