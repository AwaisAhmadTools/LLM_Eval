import json, os
from pathlib import Path
import requests
import math

CONFIG = json.loads(
    (Path(__file__).parent / "testdata" / "env_config.json").read_text()
)
TARGET = os.getenv("RAG_TARGET", CONFIG["active_target"])
API_URL = CONFIG["targets"][TARGET]["api_url"]
DATA_DIR = Path(__file__).parent / "testdata" / TARGET


def get_llm_response(test_data):
    return requests.post(
        API_URL, json={"question": test_data["question"], "chat_history": []}
    ).json()


def load_test_data(filename):
    test_data_path = DATA_DIR / filename
    if not test_data_path.exists():
        raise FileNotFoundError(f"target '{TARGET}' has no {filename}")
    with open(test_data_path, encoding="utf-8") as f:
        return json.load(f)


def assert_score(score: float, threshold: float, metric: str) -> None:
    """Two distinct failure modes, kept apart.

    NaN  -> infrastructure failure (judge API / embeddings), NOT a quality result
    low  -> quality failure: the system scored below the bar
    """
    assert math.isfinite(
        score
    ), f"{metric}: judge returned NaN — LLM/API failure, not an answer failure"
    assert score > threshold, f"{metric}: {score:.3f} ≤ threshold {threshold}"
