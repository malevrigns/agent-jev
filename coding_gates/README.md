# Coding gates

A coding agent has just run the tests. Did they pass? What should it do next? How many failed? The README's own examples ask AgentJev exactly this.

Every label comes from how the project was built, never from reading the output back.

## What is in it

| | |
|---|---|
| Runs | 2,100 real test runs: pytest, unittest, jest, mocha (450 each); vitest, node --test (150 each) |
| Outcomes | **pass** (sometimes with skipped tests) · **fail** (1–6 assertion failures) · **error** (the code under test raises) · **broken** (syntax or import error, nothing runs) · **no tests** found |
| Check | Each run's exit code must match its planted outcome. All 2,100 do. |
| Summaries | 1,800 one-line results in the shapes agents and CI print: `23 tests passed, 1 failed.`, `Tests run: 14, Failures: 1, Errors: 0, Skipped: 0`, `测试结果：共 24 个，通过 23 个，失败 1 个` |
| Questions | gate (boolean) · next action (choice of 5) · failure count (score, 4 levels; asked only when tests ran) |
| Splits | train / dev / test 70 / 15 / 15 within each runner × outcome. **vitest and node --test are held out entirely** (`ood`): never trained on, never used for selection. |

The projects are small modules (15 functions, in Python and JavaScript twins) with generated tests. The output is what the real runners print, with local paths replaced by `C:/dev/...`; `tests/test_coding_gates.py` checks that no machine path survives.

## Protocol

- `results/protocol.json` holds the selection rule. It was written before training step 1.
- A step is eligible if its Typed Decisions dev soft CE is at most step 0's plus 0.01. Among those, the lowest gate dev soft CE wins, with each dev gate also asked in the four wordings of `evaluate.PARAPHRASES`. Those wordings never appear in training (`tests/test_coding_gates.py` checks).
- Temperatures are fitted on the Typed Decisions calibration split only, as for v1.
- Gate test, `ood`, five more wordings (`evaluate.FINAL_PARAPHRASES`, one of them inverted), `ecosystem_probes.py`, and the Typed Decisions test split are read once, at the end, for both models.

## Results

**v1** is the release file (revision `7d433994`, BF16). The **candidate** continues it on this training split. Both run through `jev_service` with temperatures fitted on Typed Decisions calibration only. Every held-out row was read once, for both models.

| | v1 | candidate |
| --- | ---: | ---: |
| Gate, test: accuracy over 280 runs and 271 one-line summaries | 77.7% | **100.0%** |
| Gate, test: failing cases called green | 123 / 348 | **0 / 348** |
| … of which one-line summaries | 115 / 164 | **0 / 164** |
| Gate, test: passing cases blocked | 0 / 203 | 0 / 203 |
| All three questions, test | 74.8% | **99.6%** |
| Gate, held-out runners (vitest, node --test): accuracy over 300 runs | 89.3% | **95.0%** |
| Gate, held-out runners: failing runs called green | 31 / 198 | **7 / 198** |
| Gate, held-out runners: passing runs blocked | 1 / 102 | 8 / 102 |
| All three questions, held-out runners | 75.1% | **90.3%** |
| Gate in the five held-out wordings, test: mean accuracy | 78.4% | **95.6%** |
| Gate in the five held-out wordings, held-out runners: mean accuracy | 85.1% | **90.7%** |
| Gate in the four dev wordings, test: mean accuracy | 78.6% | **99.9%** |
| Other test tools and agent prose (`ecosystem_probes.py`): correct | 21 / 23 | 22 / 23 |
| Typed Decisions test: top-1 | 79.50% | **80.85%** |
| Typed Decisions test: soft CE ↓ | 0.8494 | **0.8461** |

A paired bootstrap over cases (`results/compare.json`) puts every accuracy gain above zero. Typed Decisions moves by **+1.35 points, 95% CI +0.20 to +2.50**. The 79.50% for v1 is the BF16 release through the serving engine; the main README's 79.25% is the FP32 checkpoint scored offline.

What the rows say:

- **v1 reads one-line summaries badly.** On test it calls 115 of 164 failing summaries green, against 8 of 184 failing full runs.
- **On runners it never trained on**, the candidate lets 7 failing runs through (v1: 31) and blocks 8 passing ones (v1: 1). All 15 are `node --test`; vitest is perfect for both models. The 8 blocks are node's `dot` reporter with no exit code in the state, where a passing run prints only dots. Four of the 7 passes are files with no tests, which node itself reports as one passing test (`# tests 1`, `# pass 1` in TAP, a single dot in `dot`). The other three are TAP runs that report failures.
- **Wording.** In the held-out wordings the candidate stays above 95% on test, except the terse one, "Tests passing?" with criteria `yes` / `no`: 83.1% on test and 78.7% on held-out runners, with 93 and 58 failing cases called green.
- **Other tools.** The candidate gets Go, cargo, Maven, Gradle, RSpec, PHPUnit, dotnet, CTest, and agent prose right, except a passing `go test ./...`. It reads the `? pkg [no test files]` line as "no tests ran" and blocks with P(true) 0.00. v1 misses that case too (0.47), and also passes a failing `cargo test` (0.53).

### The first attempt

The first run (`results/attempt1_*`) trained on one fixed wording. On dev it answered that wording perfectly and learned it more than the task. Asked "Are all tests passing?", it called the README's own example, `23 tests passed, 1 failed.`, green with probability 1.00 (v1: 0.84). It was dropped on dev, before any held-out set was read.

The candidate rewords training gates (`wordings.py`): English and Chinese, with and without criteria, and inverted questions such as "Does this test run show a problem?". It selects on dev asked in five wordings. The five held-out wordings (`evaluate.FINAL_PARAPHRASES`) and `ecosystem_probes.py` were fixed after that, before the candidate was scored.

### Limits

- The projects are small: 15 functions in Python and JavaScript twins, with generated tests. Real repositories print longer and noisier logs: warnings, coverage, build steps, interleaved workers.
- The one-line summaries come from templates, and test shares them with training. `23 tests passed, 1 failed.` matches one of them, so getting the README example right is in-distribution.
- One recording: Windows 11 and fixed tool versions.
- The 8 README probes and 23 ecosystem probes are hand-written and small. Read them as spot checks, not rates.

## Reproduce

```bash
python fetch_v1_weights.py                  # the v1 release in models/agentjev-v1
python coding_gates/build_dataset.py        # coding_gates/data, checked against manifest.json
python typed_decisions/prepare_data.py      # typed_decisions/prepared, for --typed and for training

python coding_gates/evaluate.py --checkpoint models/agentjev-v1/model.safetensors \
    --model-path models/agentjev-v1 --temperatures models/agentjev-v1/temperatures.json \
    --splits test ood --paraphrase --final-paraphrase --ecosystems --typed test \
    --out v1.json --predictions v1_pred.jsonl
python coding_gates/compare.py coding_gates/results/final_v1_pred.jsonl.gz coding_gates/results/final_candidate_pred.jsonl.gz
```

The candidate's run took about 30 minutes on an RTX 5060, evaluations included, and peaked at 4,164 MiB of GPU memory:

```bash
python coding_gates/train.py --steps 900 --vary-wording --dev-paraphrase --out coding_gates/runs/gates_v2
python coding_gates/calibrate.py --checkpoint coding_gates/runs/gates_v2/selected.safetensors \
    --model-path models/agentjev-v1 --out coding_gates/runs/gates_v2/temperatures.json
```

`results/protocol.json` and `results/selection.json` record that run: every argument, the rule, and the dev curve.

`make_runs.py` regenerates the runs themselves. It needs pytest and a folder with jest, mocha, and vitest installed; its docstring has the versions used here. Other runner versions print slightly different text, so `runs.jsonl.gz` is the recorded set.
