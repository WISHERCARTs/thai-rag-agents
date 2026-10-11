import pytest

from tests.fakes import CHUNKS, FakeKnowledgeBase


@pytest.fixture
def kb() -> FakeKnowledgeBase:
    return FakeKnowledgeBase(CHUNKS)
