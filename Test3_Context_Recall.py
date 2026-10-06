import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

from utils import get_llm_response, load_test_data, assert_score

DATA = load_test_data("dataset.json")


@pytest.mark.asyncio
@pytest.mark.parametrize("get_data", DATA, indirect=True, ids=[d["id"] for d in DATA])
async def test_context_recall(llm_wrapper, get_data):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(get_data)
    print(f"{get_data.user_input} → recall: {score:.3f}")
    assert_score(score, 0.7, "context recall")


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[
            response_dict["retrieved_docs"][0]["page_content"],
            response_dict["retrieved_docs"][1]["page_content"],
            response_dict["retrieved_docs"][2]["page_content"],
        ],
        reference=test_data["reference"],
    )
    return sample
