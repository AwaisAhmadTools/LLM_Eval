import pytest
from langchain_community.utils.ernie_functions import convert_pydantic_to_ernie_tool
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import Faithfulness, TopicAdherenceScore

from utils import load_test_data, get_llm_response


#@pytest.mark.parametrize("getData", load_test_data("test4_data.json"), indirect=True)
@pytest.mark.asyncio
async def test_topic_adherence(llm_wrapper, getData):
    topic_adherence = TopicAdherenceScore(llm=llm_wrapper)
    score = await topic_adherence.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8

@pytest.fixture
def getData():
    #test_data = request.param
    #responseDict = get_llm_response(test_data)
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
