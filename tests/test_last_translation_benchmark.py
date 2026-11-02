from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

from last_translation_benchmark import last_translation_benchmark
from last_translation_benchmark.last_translation_benchmark import (
    parse_verify,
    pass_rate_by_lang_pair,
    record_to_sample,
)


def test_record_to_sample():
    sample = record_to_sample(
        {
            "id": 337,
            "source_text": "Our new nurse used to play in the men's professional football team.",
            "source_lang": "English",
            "target_lang": "German",
            "source_instructions": "Formal register.",
            "verification_rules": ["Nurse must be translated to its male version (Krankenpfleger)."],
            "linguistics": ["Morphological"],
        }
    )
    assert sample.id == 337
    assert sample.input.startswith("Translate the following text from English to German.")
    assert "Additional instructions: Formal register." in sample.input
    assert sample.input.endswith("men's professional football team.")
    assert sample.metadata["lang_pair"] == "English->German"


def test_pass_rate_by_lang_pair():
    metadata = [{"lang_pair": "en->de"}, {"lang_pair": "en->de"}, {"lang_pair": "it->en"}]
    metrics = [pass_rate_by_lang_pair()]
    # each item scored over two epochs; the mean reducer averages them first
    values = run_metrics(metrics, [[1, 1], [0, 1], [0, 0]], metadata=metadata)
    assert values == {"en->de": 0.75, "it->en": 0.0}
    assert_agreeing_epochs_change_nothing(metrics, [1, 0, 1], metadata=metadata)


def test_parse_verify():
    assert parse_verify("Pass.")
    assert not parse_verify("fail")
    assert parse_verify("The translation is a pass.")
    # the last token decides, as upstream
    assert not parse_verify("Not a pass, it is a fail")
    assert not parse_verify("")


def test_dataset():
    samples = last_translation_benchmark().dataset
    assert len(samples) == 911
    assert all(s.metadata["verification_rules"] for s in samples)


def test_scoring_end_to_end():
    # an item is correct only if the judge passes every rule
    target = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "translation")] * 2)
    judge = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", v) for v in ["pass", "pass", "pass", "fail"]],
    )
    # samples 40 and 63 have two rules each
    [log] = inspect_eval(
        last_translation_benchmark(),
        model=target,
        model_roles={"grader": judge},
        sample_id=[40, 63],
        display="none",
    )
    scores = [s.scores["ltb_verifier"].value for s in log.samples]
    assert sorted(scores) == [0, 1]
