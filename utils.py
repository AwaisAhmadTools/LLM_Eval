import json
from pathlib import Path

import requests


def get_llm_response(test_data):
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": test_data["question"],
        "chat_history": [
        ]
    }).json()
    return response_dict

def load_test_data(filename):
    test_data_path = Path(__file__).parent.absolute()/"testdata"/filename
    with open (test_data_path) as f:
        return json.load(f)