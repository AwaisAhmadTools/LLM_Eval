import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils import load_test_data, get_llm_response, assert_score

DATA = load_test_data("dataset.json")


@pytest.mark.parametrize("get_data", DATA, indirect=True, ids=[d["id"] for d in DATA])
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, get_data):
    faithful = Faithfulness(llm=llm_wrapper)
    score = await faithful.single_turn_ascore(get_data)
    print(f"{get_data.user_input} → faithfulness: {score:.3f}")
    assert_score(score, 0.8, "faithfulness")


@pytest.fixture
def get_data(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[
            doc["page_content"] for doc in responseDict.get("retrieved_docs")
        ],
    )
    return sample
