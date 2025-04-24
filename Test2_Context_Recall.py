import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall


@pytest.mark.asyncio
async def test_context_recall():
    question = "How many articles are there in the Selenium webdriver python course?"
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm=lang_chain_llm)

    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": [
        ]
    }).json()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[response_dict["retrieved_docs"][0]["page_content"],
                            response_dict["retrieved_docs"][1]["page_content"],
                            response_dict["retrieved_docs"][2]["page_content"]
                            ],
        reference="23"
    )

    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7