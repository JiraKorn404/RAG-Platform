import math

from rag_lab.core.embed import document_vectors, query_vector
from rag_lab.serve.search import yes_probability


def _weight(vector, term: str) -> float:
    indices, values = vector
    return dict(zip(indices, values))[query_vector(term)[0][0]]


def test_query_vector_has_one_weight_per_distinct_lowercased_term():
    indices, values = query_vector("Rare42 rare42 beta")
    assert len(indices) == 2 and values == [1.0, 1.0]
    assert sorted(indices) == sorted(query_vector("beta, RARE42!")[0])


def test_document_weight_grows_with_count_but_saturates_and_falls_with_length():
    short, longer = document_vectors(["cat cat cat dog", "cat dog " + "filler " * 40])
    assert _weight(short, "cat") > _weight(short, "dog")
    assert _weight(short, "cat") < 3 * _weight(short, "dog")  # three times the count is less than three times the weight
    assert _weight(short, "dog") > _weight(longer, "dog")  # the same count in a longer text weighs less


def test_yes_probability_sums_spellings_and_ignores_other_tokens():
    top = [
        {"token": "yes", "logprob": math.log(0.5)},
        {"token": " Yes", "logprob": math.log(0.1)},
        {"token": "No", "logprob": math.log(0.2)},
        {"token": "no", "logprob": math.log(0.1)},
        {"token": "The", "logprob": math.log(0.05)},
    ]
    assert math.isclose(yes_probability(top), 0.6 / 0.9)
    assert yes_probability([{"token": "The", "logprob": -1.0}]) == 0.0
