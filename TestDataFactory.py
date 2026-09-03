import json
import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
import nltk

with open('testdata/env_config.json') as f:
    env_config = json.load(f)

os.environ["RAGAS_APP_TOKEN"] = env_config["RAGAS_APP_TOKEN"]

def test_data_creation():
    nltk.data.path.append(r"C:\Users\aahma\Documents\AI\LLM+Evaluation_Resources\nltk_data")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    # convert data to vector format
    embed = OpenAIEmbeddings()
    loader = DirectoryLoader(
        path=r"C:\Users\aahma\Documents\AI\LLM+Evaluation_Resources\LLM Evaluation_Resources\fs11",
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    #ragas wrapper
    generate_embeddings = LangchainEmbeddingsWrapper(embed)
    generator = TestsetGenerator(llm=langchain_llm,embedding_model=generate_embeddings)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=20)
    print(dataset.to_list())
    # dataset.upload()
