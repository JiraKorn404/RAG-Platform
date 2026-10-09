from rag_lab.core.events import Hit
from rag_lab.serve.chat_documents import above, after_grade, after_rewrite, is_new


def _hit(rank: int, score: float) -> Hit:
    return Hit(rank, score, 1 - score, "text", "a.pdf", 1, "text", [], "doc", f"chunk-{rank}")


def test_chunks_below_the_floor_are_left_out_and_the_rest_keep_their_numbers():
    hits = [_hit(1, 0.9), _hit(2, 0.4), _hit(3, 0.05)]
    assert [h.rank for h in above(hits, 0.4)] == [1, 2]  # at the floor is kept
    assert above(hits, 0.0) == hits
    assert above(hits, 0.95) == []


def test_enough_chunks_answer_whatever_is_left():
    assert after_grade(True, rewrites=0, max_rewrites=1) == "generate"
    assert after_grade(True, rewrites=1, max_rewrites=1) == "generate"


def test_weak_chunks_rewrite_until_the_rewrites_are_used_up_then_abstain():
    assert after_grade(False, rewrites=0, max_rewrites=2) == "rewrite"
    assert after_grade(False, rewrites=1, max_rewrites=2) == "rewrite"
    assert after_grade(False, rewrites=2, max_rewrites=2) == "abstain"
    assert after_grade(False, rewrites=0, max_rewrites=0) == "abstain"


def test_only_a_new_query_is_searched_and_a_repeat_has_used_its_attempt():
    assert after_rewrite(True, rewrites=1, max_rewrites=1) == "retrieve"
    assert after_rewrite(False, rewrites=1, max_rewrites=2) == "rewrite"
    assert after_rewrite(False, rewrites=2, max_rewrites=2) == "abstain"


def test_case_and_spacing_do_not_make_a_new_query():
    tried = ["What does PMON do?"]
    assert not is_new("  what does  PMON do? ", tried)
    assert not is_new("", tried)
    assert is_new("PMON background process", tried)


def _searches(max_rewrites: int, rewrite_is_new: bool) -> int:
    """How often a turn whose chunks are never enough searches, by the two decisions alone."""
    searches, rewrites, step = 1, 0, after_grade(False, 0, max_rewrites)
    while step != "abstain":
        if step == "rewrite":
            rewrites += 1
            step = after_rewrite(rewrite_is_new, rewrites, max_rewrites)
        else:  # retrieve, then grade
            searches += 1
            step = after_grade(False, rewrites, max_rewrites)
    return searches


def test_a_turn_never_searches_more_than_max_rewrites_plus_one_times():
    for max_rewrites in range(4):
        assert _searches(max_rewrites, rewrite_is_new=True) == max_rewrites + 1
        assert _searches(max_rewrites, rewrite_is_new=False) == 1
