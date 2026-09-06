import pytest
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import TopicAdherenceScore

@pytest.mark.asyncio
async def test_topic_adherence(llm_wrapper, getData):
    topic_adherence = TopicAdherenceScore(llm=llm_wrapper)
    score = await topic_adherence.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8

@pytest.fixture
def getData():
    conversation = [
        HumanMessage(content="how many articles are there in the selenium web driver python course?"),
        AIMessage(content="There are 23 articles in the Selenium WebDriver Python course."),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course.")
    ]
    reference=[
    """
    The AI should:
    1. Give results related to the selenium webdriver python course.
    2. There are 23 articles and 9 downloadable resources,
    """
    ]
    sample = MultiTurnSample(
        user_input=conversation,
        reference_topics=reference
    )
    return sample
