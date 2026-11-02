# Last Translation Benchmark for Inspect AI

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of the [Last Translation Benchmark](https://last-translation-benchmark.vilda.net/) (LTB, [arXiv:2609.04173](https://arxiv.org/abs/2609.04173)). It runs the `LTBv1-eval` subset: 911 text-only, hard-to-translate inputs across many language pairs, each with one or more natural-language verification rules.

A judge model checks the translation against each rule (pass/fail), and an item counts as correct only if every rule passes. The headline metric `accuracy`, the share of items that pass all their rules, is the score shown on the [LTB leaderboard](https://last-translation-benchmark.vilda.net/leaderboard-results). The task also reports its `stderr` and the pass rate per language pair.

## Usage

```bash
git clone https://github.com/robert-ellamind/last-translation-benchmark-inspect
```

```bash
cd last-translation-benchmark-inspect && uv sync
```

```bash
uv run --with openai inspect eval src/last_translation_benchmark/last_translation_benchmark.py --model openrouter/google/gemma-4-31b-it
```

`--with openai` adds the SDK that Inspect's OpenRouter provider needs; use the SDK for your provider instead (see the [Inspect providers docs](https://inspect.aisi.org.uk/providers.html)).

The judge defaults to `google/gemini-3.8-flash`, the verifier the leaderboard uses, so it needs a `GEMINI_API_KEY`. Override it with `-T judge_model=<model>` or `--model-role grader=<model>`. Scores are only comparable to the leaderboard when the same judge is used.

## Fidelity to the official scorer

This mirrors `scripts/41-score_leaderboard.py` in [zouharvi/last-translation-benchmark](https://github.com/zouharvi/last-translation-benchmark):

- the verifier prompt is copied from `server/utils.py::get_prompt_verify`;
- the pass/fail parse is the same (the last word of the judge's answer decides);
- an empty translation fails every rule;
- the default judge is the same.

The only part not taken from upstream is the instruction given to the model under test (`TRANSLATE_PROMPT`). The leaderboard scores submitted translations and defines no prompt, so this one is kept minimal and neutral.

## Results

This implementation reproduces the official leaderboard scores well. These are full runs on all 911 items (commit `58c335b`, judge `google/gemini-3.8-flash`, November 2026), and both leaderboard scores fall inside our 95% confidence intervals:

| Model | This implementation: accuracy [95% CI] | [Leaderboard](https://last-translation-benchmark.vilda.net/leaderboard-results) (Blind, LTBv1-eval) |
|---|---|---|
| Gemma 4 31B (`openrouter/google/gemma-4-31b-it`) | 0.071 [0.055, 0.088] | 0.071 |
| DeepSeek V4.1 Flash (`openrouter/deepseek/deepseek-v4.1-flash`) | 0.127 [0.106, 0.149] | 0.138 |

The intervals are percentile bootstrap intervals over items: the 911 per-item scores (0 or 1) of each run were resampled with replacement 10,000 times (seed 0), and the 2.5th and 97.5th percentiles of the resampled accuracies are reported. They capture how much the score depends on which items happen to be in the set, from a single run. They do not capture run-to-run noise from the model's or judge's sampling, which would need repeated runs (`--epochs`).

Each run cost about $2 at OpenRouter and Gemini API list prices, mostly for the judge.

## Data

The task downloads `data/v1.json` from the official Hugging Face dataset [`zouhar/last-translation-benchmark`](https://huggingface.co/datasets/zouhar/last-translation-benchmark) (about 90 MB, CC BY 4.0), pinned to revision `a483825ddbe2d7756f5bdfb1e4f611bee9026c4c`, and keeps the items tagged `LTBv1-eval`.

## Tests

```bash
uv run pytest
```

## Citation

```bibtex
@misc{zouhar2026translationbenchmark,
  title={Last Translation Benchmark},
  author={LTB},
  year={2026},
  eprint={2609.04173},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2609.04173},
}
```

The code here is MIT-licensed. The LTB data is CC BY 4.0 and is not redistributed in this repository.
