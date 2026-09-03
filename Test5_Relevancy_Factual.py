import json
import os

import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness

from utils import load_test_data, get_llm_response

with open('testdata/env_config.json') as f:
    env_config = json.load(f)

os.environ["RAGAS_APP_TOKEN"] = env_config["RAGAS_APP_TOKEN"]
@pytest.mark.parametrize("get_data", load_test_data("test5_data.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, get_data):

    metrics = [ResponseRelevancy(llm=llm_wrapper), FactualCorrectness(llm=llm_wrapper)]

    # use EvaluationDataSet for multiple metrics
    eval_dataset = EvaluationDataset([get_data])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    '''
    results = evaluate(dataset=eval_dataset)
    If metrics are not provided it defaults to answer_relevancy, context_precision, faithfulness, context_recall
    '''
    print(results)
    print (results['answer_relevancy'])
    assert results['answer_relevancy'][0] > 0.8
    # results.upload()
   # assert all (float(r['answer_relevancy']) > 0.8 for r in results)



@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    #grabbing all the data to use on different metrics
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in response_dict.get("retrieved_docs")],
        reference=test_data["reference"]
    )

    return sample
