import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from utils import load_test_data, get_llm_response, assert_score

DATA = load_test_data("dataset.json")


@pytest.mark.parametrize("get_data", DATA, indirect=True, ids=[d["id"] for d in DATA])
@pytest.mark.asyncio
async def test_context_precision(llm_wrapper, get_data):
    # create object of class for that specific metric

    # power of llm + method metric -> score
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)

    # Get the score
    score = await context_precision.single_turn_ascore(get_data)
    print(f"{get_data.user_input} → precision: {score:.3f}")
    assert_score(score, 0.8, "context precision")


@pytest.fixture
def get_data(request):

    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[
            response_dict["retrieved_docs"][0]["page_content"],
            response_dict["retrieved_docs"][1]["page_content"],
            response_dict["retrieved_docs"][2]["page_content"],
        ],
    )

    return sample
