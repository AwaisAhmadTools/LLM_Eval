import json
import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

with open('testdata/env_config.json') as f:
    env_config = json.load(f)
os.environ["OPENAI_API_KEY"] = env_config["OPENAI_API_KEY"]

@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return LangchainLLMWrapper(llm)