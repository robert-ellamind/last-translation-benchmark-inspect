"""Last Translation Benchmark (LTBv1-eval) as an Inspect AI task.

Each item is a source text plus one or more natural-language verification
rules. A judge model answers pass/fail per rule, and a translation counts as
correct only if every rule passes. The headline metric is the leaderboard
score: the mean over items of all(rules passed).

Mirrors the official leaderboard scorer, scripts/41-score_leaderboard.py in
https://github.com/zouharvi/last-translation-benchmark: same verifier prompt
(server/utils.py::get_prompt_verify), same pass/fail parse, empty translations
fail every rule, and the same default judge, google/gemini-3.8-flash.

The only piece not taken from upstream is the instruction given to the model
under test, since leaderboard submissions bring their own translations.
"""

import json
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

HF_REPO = "zouhar/last-translation-benchmark"
HF_FILE = "data/v1.json"
# LTBv1 as released; sha256 of data/v1.json at this revision is
# 99fa3d547530b64b45fa3181811fba11281a013c53b9959091f770a5f6594b9f
HF_REVISION = "a483825ddbe2d7756f5bdfb1e4f611bee9026c4c"
EVAL_TAG = "LTBv1-eval"

DEFAULT_JUDGE = "google/gemini-3.8-flash"

TRANSLATE_PROMPT = (
    "Translate the following text from {source_lang} to {target_lang}.\n"
    "Output only the translation, with no explanation or extra text."
)

# verbatim from server/utils.py::get_prompt_verify (text-only branch)
VERIFY_PROMPT = (
    "Your goal is to verify whether a translation fulfills a criterion.\n\n"
    "Criterion: {rule}\n\nInput: {source_text}\n\nTranslation to verify: {translation}\n\n"
    "Output only pass or fail and nothing else."
)


def parse_verify(res_text: str) -> bool:
    """Same pass/fail parse as scripts/41-score_leaderboard.py."""
    tokens = res_text.strip().lower().strip(" \t\n\r.,!?\"'*").split()
    text_clean = tokens[-1] if tokens else ""
    if "pass" in text_clean:
        return True
    if "fail" in text_clean:
        return False
    if "pass" in res_text.lower():
        return True
    if "fail" in res_text.lower():
        return False
    return False


def record_to_sample(r: dict[str, Any]) -> Sample:
    prompt = TRANSLATE_PROMPT.format(source_lang=r["source_lang"], target_lang=r["target_lang"])
    if r["source_instructions"]:
        prompt += f"\n\nAdditional instructions: {r['source_instructions']}"
    return Sample(
        id=r["id"],
        input=f"{prompt}\n\n{r['source_text']}",
        metadata={
            "source_text": r["source_text"],
            "lang_pair": f"{r['source_lang']}->{r['target_lang']}",
            "verification_rules": r["verification_rules"],
            "linguistics": r["linguistics"],
        },
    )


def load_dataset() -> MemoryDataset:
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", revision=HF_REVISION)
    with open(path, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if EVAL_TAG in r["tags"]]
    # the eval subset is text-only, so the media branch of the verifier prompt is never needed
    assert not any(r["source_media"] for r in rows)
    return MemoryDataset([record_to_sample(r) for r in rows], name=EVAL_TAG)


@metric
def pass_rate_by_lang_pair() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        by_pair: dict[str, list[float]] = {}
        for s in scores:
            by_pair.setdefault(s.sample_metadata["lang_pair"], []).append(s.score.as_float())
        return {pair: sum(v) / len(v) for pair, v in sorted(by_pair.items())}

    return compute


# accuracy over 0/1 item scores is the leaderboard score
@scorer(metrics=[accuracy(), stderr(), pass_rate_by_lang_pair()])
def ltb_verifier(judge_model: str = DEFAULT_JUDGE) -> Scorer:
    """One judge call per verification rule; correct iff all rules pass."""

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(role="grader", default=judge_model)
        translation = state.output.completion
        rules = state.metadata["verification_rules"]

        # empty translations fail every rule without a judge call
        if translation == "":
            verdicts = ["(empty translation)"] * len(rules)
            results = [False] * len(rules)
        else:
            verdicts = []
            for rule in rules:
                prompt = VERIFY_PROMPT.format(
                    rule=rule, source_text=state.metadata["source_text"], translation=translation
                )
                verdicts.append((await judge.generate(prompt)).completion)
            results = [parse_verify(v) for v in verdicts]

        return Score(
            value=1 if all(results) else 0,
            answer=translation,
            explanation="\n".join(f"{'PASS' if ok else 'FAIL'}: {rule}" for ok, rule in zip(results, rules)),
            metadata={"rule_results": results, "judge_outputs": verdicts},
        )

    return score


@task
def last_translation_benchmark(judge_model: str = DEFAULT_JUDGE) -> Task:
    """LTBv1-eval (911 text-only items).

    Args:
        judge_model: Verifier model. Overridden by a `grader` model role if one is given.
    """
    return Task(
        dataset=load_dataset(),
        solver=generate(),
        scorer=ltb_verifier(judge_model),
        version="1.0.0",
    )
