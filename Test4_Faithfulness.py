import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils import load_test_data, get_llm_response


@pytest.mark.parametrize("getData", load_test_data("test4_data.json"), indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, getData):
    faithful = Faithfulness(llm=llm_wrapper)
    score = await faithful.single_turn_ascore(getData)
    print(score)
    assert score > 0.8

@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")]
    )
    return sample
