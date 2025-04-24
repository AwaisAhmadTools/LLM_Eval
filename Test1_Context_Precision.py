import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


# user input -> query
# response -> response by LLM
# reference -> ground truth, similar to expected result
# retrieved_context -> top k retrieved docs
@pytest.mark.asyncio
async def test_context_precision():
    # create object of class for that specific metric

    # power of llm + method metric -> score
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"

    # Feed data
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": [
        ]
    }).json()

    print(response_dict)

    sample = SingleTurnSample(
        user_input=question,
        response=response_dict["answer"],
        retrieved_contexts=[response_dict["retrieved_docs"][0]["page_content"],
                            response_dict["retrieved_docs"][1]["page_content"],
                            response_dict["retrieved_docs"][2]["page_content"]
                            ]
    )
    # Get the score
    score = await context_precision.single_turn_ascore(sample)
    print(score)
    assert score > 0.8

    # sample = SingleTurnSample(
    #     user_input="How many articles are the in the Selenium webdriver python course.",
    #     response="There are 23 articles in the course.",
    #     retrieved_contexts=[
    #         "Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\"Last "
    #         "but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design "
    #         "Stage\nThis course includes:\n17.5 hours on-demand video\nAssignments\n23 articles\n9 downloadable "
    #         "resources\nAccess on mobile and TV\nCertificate of completion\nRequirements",
    #         "What you'll learn\n*****By the end of this course,You will be Mastered on Selenium Webdriver with strong "
    #         "Core JAVA basics\n****You will gain the ability to design PAGEOBJECT, DATADRIVEN&HYBRID Automation "
    #         "FRAMEWORKS from scratch\n*** InDepth understanding of real time Selenium CHALLENGES with 100 + "
    #         "examples\n*Complete knowledge on TestNG, MAVEN,ANT, JENKINS,LOG4J, CUCUMBER, HTML REPORTS,EXCEL API, "
    #         "GRID PARALLEL TESTING",
    #         "What you'll learn\nAt the end of this course, You will get complete knowledge on Python Automation using "
    #         "Selenium WebDriver\nYou will be able to implement Python Test Automation Frameworks from Scratch with "
    #         "all latest Technlogies\nComplete Understanding of Python Basics with many practise Examples to gain a "
    #         "solid exposure\nYou will be learning Python Unit Test Frameworks like PyTest which will helpful for Unit "
    #         "and Integration Testing",
    #         "Join in our Selenium Training community where 3 Million Students Learning Together which you will not "
    #         "see in any other Selenium courses on Udemy\nDescription\n**Learn Everything You Need to Know About "
    #         "Python Selenium Automation including Framework Even If You've Never Programmed Before in Python**"
    #     ]
    # )
