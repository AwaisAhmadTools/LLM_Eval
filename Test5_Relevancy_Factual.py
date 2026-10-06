import math
import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness
from utils import load_test_data, get_llm_response, assert_score

DATA = load_test_data("dataset.json")


@pytest.mark.parametrize("get_data", DATA, indirect=True, ids=[d["id"] for d in DATA])
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, get_data):

    metrics = [
        ResponseRelevancy(llm=llm_wrapper),
        FactualCorrectness(llm=llm_wrapper, mode="recall"),
    ]

    # use EvaluationDataSet for multiple metrics
    eval_dataset = EvaluationDataset([get_data])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    """
    results = evaluate(dataset=eval_dataset)
    If metrics are not provided it defaults to answer_relevancy, context_precision, faithfulness, context_recall
    """

    relevancy = results["answer_relevancy"][0]
    factual = results["factual_correctness(mode=recall)"][0]  # key encodes the mode

    print(results)
    print(
        f"{get_data.user_input} → relevancy: {relevancy:.3f} | factual: {factual:.3f}"
    )
    assert_score(relevancy, 0.8, "relevancy")
    assert_score(factual, 0.8, "factual correctness")  # known limitation — see README


# assert all (float(r['answer_relevancy']) > 0.8 for r in results)


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    # grabbing all the data to use on different metrics
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[
            doc["page_content"] for doc in response_dict.get("retrieved_docs")
        ],
        reference=test_data["reference"],
    )

    return sample
