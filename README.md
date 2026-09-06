


## How a test is built — the five steps

Every test in this suite follows the same skeleton:

1. Build the judge — the LLM that scores. `ChatOpenAI` at `temperature=0`
   (deterministic judging — the score reflects the answer, not judge variance),
   wrapped for RAGAS via `LangchainLLMWrapper`. Supplied to tests as the
   `llm_wrapper` fixture (`conftest.py`).
2. Define the metric — instantiate the RAGAS metric to evaluate, e.g.
   `LLMContextPrecisionWithoutReference(llm=llm_wrapper)`.
3. Ask the system under test — one call to the RAG endpoint
   (`get_llm_response`). The entire pipeline — embedding, retrieval, prompt
   assembly, generation — happens inside this single call, which returns the
   `answer` and the `retrieved_docs` (top-K chunks).
4. Build the sample — reshape the response into RAGAS's data structure:
   `SingleTurnSample(user_input, response, retrieved_contexts, reference?)`.
5. Score + assert — run the metric and gate on a threshold:
   `await metric.single_turn_ascore(sample)` → `assert score > 0.8`.

```python
1. Build the judge (conftest.py fixture)
llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))

2. Define the metric
metric = LLMContextPrecisionWithoutReference(llm=llm_wrapper)

3. Ask the system under test — one call, the whole pipeline happens inside
response = get_llm_response(test_data)

4. Build the sample
sample = SingleTurnSample(
    user_input=test_data["question"],
    response=response["answer"],
    retrieved_contexts=[doc["page_content"] for doc in response["retrieved_docs"]],
)

5. Score + assert
score = await metric.single_turn_ascore(sample)
assert score > 0.8
```
## The metrics

The suite measures quality at three layers of the RAG pipeline: what was
retrieved, what was generated, and — for conversations — whether the
system stayed on task.

### Retrieval quality — did we pull the right sources?

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Context Precision | Test1 | Of the top-K chunks retrieved, how many were actually relevant to the question? | ❌ |
| Context Recall | Test3 | Of all the chunks that should have been retrieved, how many were? | ✅ |

Precision punishes noise — a high-precision retriever returns only what matters.
Recall punishes omission — a high-recall retriever misses nothing. Real systems
trade the two; this suite watches both.

### Generation quality — is the answer trustworthy?

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Faithfulness | Test4 | Is every claim in the answer traceable to the retrieved context? Catches hallucination — invented facts not in the sources. | ❌ |
| Answer Relevancy | Test5 | Does the answer actually address the question? A relevant-but-wrong answer passes this and fails the next — that's the point. | ❌ |
| Factual Correctness | Test5 | Does the answer match the ground truth? The final arbiter of right and wrong. | ✅ |

These three form the metric triangle — each catches a failure mode the
others miss: grounded-but-false (bad sources) passes faithfulness, fails
factual; fluent-but-invented passes relevancy, fails faithfulness.

### Conversation & custom criteria

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Topic Adherence | Test6 | Across a multi-turn conversation, does the assistant stay within the allowed topic space? The seed of agent evaluation. | ✅ (reference_topics) |
| Rubric Score | Test7 | The judge scores the answer against a scale you write in plain English — how you evaluate domain-specific quality (e.g. regulatory or ethical criteria) that generic metrics can't express. | ✅ |