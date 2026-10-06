# LLM_Eval — an evaluation suite for RAG systems

RAG (Retrieval-Augmented Generation) systems don't fail like normal software.
They don't throw exceptions — they return confident, fluent, plausible answers
that may be wrong. Traditional asserts can't catch a hallucination.

This suite treats a RAG pipeline as a system under test and measures its
quality across three layers: what it retrieves, what it generates, and
how it behaves in conversation — 7 metrics, LLM-judged, threshold-gated.

## Layout

| File | Metric(s) | Layer |
|---|---|---|
| `Test1_Context_Precision.py` | Context Precision | retrieval |
| `Test3_Context_Recall.py` | Context Recall | retrieval |
| `Test4_Faithfulness.py` | Faithfulness (hallucination check) | generation |
| `Test5_Relevancy_Factual.py` | Answer Relevancy + Factual Correctness | generation |
| `Test6_Topic_Adherence.py` | Topic Adherence (multi-turn) | conversation |
| `Test7_Rubric_Score.py` | Rubric scoring (custom criteria) | custom |
| `conftest.py` | judge fixture (gpt-4o, temperature=0) | harness |
| `utils.py` | the system-under-test seam + `assert_score` | harness |
| `data_factory.py` | synthetic test-set generation from source docs | tooling |
| `testdata/<target>/dataset.json` | questions + references — one dataset per target | data |

## The seam — and how to point the suite at another system

The system under test is a **target**: a name that resolves to an API URL *and* a data folder.
`utils.get_llm_response()` makes one call to the RAG endpoint — embedding, retrieval, prompt
assembly and generation all happen inside it. The suite doesn't know or care how the system is built.

| Piece | Where it lives |
|---|---|
| `api_url` — the endpoint to test | `testdata/env_config.json` → `targets.<name>.api_url` |
| `dataset.json` — questions + references | `testdata/<name>/dataset.json` |

`utils.py` resolves both from a single variable:

```python
TARGET   = os.getenv("RAG_TARGET", CONFIG["active_target"])
API_URL  = CONFIG["targets"][TARGET]["api_url"]
DATA_DIR = Path(__file__).parent / "testdata" / TARGET
```

**The invariant: the config key must equal the folder name.** To add a target, add a config entry and
a folder of the same name — the tests never change.

**The filename is the interface.** Every target must provide the same filenames, because that is what
the tests ask for: `dataset.json` serves Test1, Test3, Test4 and Test5 (the single-turn metrics).
Test6 and Test7 carry their own inline data (a scripted transcript; rubric definitions), which is why
they have no data file.

## Running

```
pip install -r requirements.txt
cp testdata/env_config_sample.json testdata/env_config.json   # add your OpenAI key + targets
pytest                                  # runs `active_target` from env_config.json
```

Override the target for a single run:

| Shell | Command |
|---|---|
| PowerShell | `$env:RAG_TARGET="course"; pytest` |
| bash | `RAG_TARGET=course pytest` |

`-s -v` is set in `pytest.ini`, so individual scores and case ids print by default.

## How a test is built — the five steps

Every test in this suite follows the same skeleton:

1. Build the judge — the LLM that scores. `ChatOpenAI` at `temperature=0`, wrapped for RAGAS via
   `LangchainLLMWrapper`. Temperature 0 **reduces** judge variance but does not eliminate it: the
   same inputs have produced different scores across runs (see Known limitations), so a single run
   is an indication, not a verdict. Supplied to tests as the `llm_wrapper` fixture (`conftest.py`).
2. Define the metric — instantiate the RAGAS metric to evaluate, e.g.
   `LLMContextPrecisionWithoutReference(llm=llm_wrapper)`.
3. Ask the system under test — one call to the RAG endpoint
   (`get_llm_response`). The entire pipeline — embedding, retrieval, prompt
   assembly, generation — happens inside this single call, which returns the
   `answer` and the `retrieved_docs` (top-K chunks).
4. Build the sample — reshape the response into RAGAS's data structure:
   `SingleTurnSample(user_input, response, retrieved_contexts, reference?)`.
5. Score + assert — run the metric, then gate through the shared helper:
   `await metric.single_turn_ascore(sample)` → `assert_score(score, 0.8, "context precision")`.
   `assert_score` keeps two failure modes apart: **NaN** means the judge or API failed
   (infrastructure); a **low score** means the system scored below the bar (quality).

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
assert_score(score, 0.8, "context precision")   # NaN → infra failure; low → quality failure
```
## The metrics

The suite measures quality at three layers of the RAG pipeline: what was
retrieved, what was generated, and — for conversations — whether the
system stayed on task.

### Retrieval quality — did we pull the right sources?

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Context Precision | Test1 | Of the top-K chunks retrieved, how many were actually relevant to the question? | ❌ |
| Context Recall | Test3 | How much of the reference can be attributed to the retrieved context? Missing sources lower recall. | ✅ |

Precision punishes noise — a high-precision retriever returns only what matters.
Recall punishes omission — a high-recall retriever misses nothing. Real systems
trade the two; this suite watches both.

### Generation quality — is the answer trustworthy?

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Faithfulness | Test4 | Is every claim in the answer traceable to the retrieved context? Catches hallucination — invented facts not in the sources. | ❌ |
| Answer Relevancy | Test5 | Does the answer actually address the question? A relevant-but-wrong answer passes this and fails the next — that's the point. | ❌ |
| Factual Correctness | Test5 | Does the answer match the ground truth? `mode` matters: `f1` also penalises extra *correct* content, while `recall` measures coverage only — see Known limitations. | ✅ |

These three form the metric triangle — each catches a failure mode the
others miss: grounded-but-false (bad sources) passes faithfulness, fails
factual; fluent-but-invented passes relevancy, fails faithfulness.

### Conversation & custom criteria

| Metric | Test | What it measures | Needs reference? |
|---|---|---|---|
| Topic Adherence | Test6 | Across a multi-turn conversation, does the assistant stay within the allowed topic space? The seed of agent evaluation. | ✅ (reference_topics) |
| Rubric Score | Test7 | The judge scores the answer against a scale you write in plain English — how you evaluate domain-specific quality (e.g. regulatory or ethical criteria) that generic metrics can't express. | ✅ |

## Known limitations — and why the board is red

`test_relevancy_factual` currently fails 4 of 6 cases on the local target. **This is a limitation of
the instrument (the metric and its ground truth), not a defect found in the system under test.**
Every failing answer was read by hand and verified as grounded, complete and on-topic.

**Evidence**
- **Run-to-run variance:** identical inputs produced different scores between runs — q6 faithfulness
  0.846 → 1.000; q4 relevancy 0.9963 → 1.000. A metric that moves between runs cannot support a
  fixed threshold.
- **Hand audit, 6 of 6:** every low factual-correctness score traced to a reference/measurement cause,
  in three modes — *scope* (the reference demands claims the question never asked), *source variance*
  (the corpus states the same fact two ways; the answer cites one, the reference the other),
  *thinness* (a reference less complete than a good answer, so `f1`'s precision half penalises extra
  *correct* content).
- **Cross-metric disagreement:** faithfulness = 1.00 (fully grounded in retrieved context) alongside
  factual correctness = 0.40. When the anti-invention metric passes and the reference-based metric
  fails, the reference is the suspect.

**Latest local-target results:** precision 1.00×6 · recall 1.00×6 · faithfulness 1.00×6 ·
relevancy 0.908–1.000 · factual correctness (`mode=recall`) 0.45–0.87 (4 of 6 below the 0.8 gate).

**Deliberately not done:** the threshold is *not* lowered to make the board green. The metric is shown
to be unstable, so its gate stands as a known, visible defect. The fix is to measure variance (N≥5
runs) and set thresholds empirically — not to move the bar.

**Degradation test (2026-10-05):** feeding Test3 only the first retrieved chunk (`docs[:1]`) — a
deliberate retrieval regression — left context recall at **1.000 for 5 of 6 questions** and dropped
it only on q6 (0.500). The metric can fail (it is not dead), but it is close to blind: five
references are fully satisfiable from a single chunk, so a real retrieval regression would not be
caught. **Next step:** write references whose claims span more than one source, as q6's does, then
run the degradation test permanently as proof the suite *can* fail.
