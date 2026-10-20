import pytest


@pytest.mark.skip(reason="Week 1, Person B: implement removal.py first")
def test_removing_no_tokens_reproduces_outputs():
    """Plan Section 9: removing no tokens must reproduce the original outputs exactly (CPU, float32)."""
