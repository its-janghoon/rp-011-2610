# Replication package

Anonymous package for the submission *A 2,049-Parameter Head Predicts When a Small
LLM Will Answer Correctly: Answer-or-Delegate Routing from the Model's Own Hidden
State, and Three Reversals of the Adoption Verdict*.

Every number in the paper comes from a file below. Nothing was computed in the
manuscript directory; the measurements are the artifact repository's, and this
package is a curated, scrubbed copy of the parts the paper cites.

## What maps to what

| Paper | Artifact |
| --- | --- |
| Table 1, own test split (0.7100, 0.6704) | `results/09_ctrl_auroc_ci.base.parquet`, `.sc06.parquet`; `code/scoring/evaluate.py` |
| Table 1, head size (2,049 par. / 41 KB, 1,025 / 21 KB) | `heads/26_llamacpp_head.json`, `.sc06.json`, and the `.npz` beside each |
| Table 1, public benchmarks (0.8138, 0.7136 on 482 items) | `results/41_bench_head.json`; `code/scoring/eval_bench_head.py` |
| Table 2, end to end (0.7195 -> 0.8064, +8.69) | `results/25_e2e_cascade_gguf.json`, `25_e2e_cascade_sc06.json` |
| Table 3, quantisation (0.7516, 0.7397, 0.9746, 0.8359) | `results/36_q4_control.json`, `09_ctrl_auroc_ci.q4.parquet`; `code/scoring/q4_head_control.py` |
| Section 6, latency (486/1,330 ms, 51.4/19.9 tok/s, RSS) | `results/32_ttft.json`; `code/scoring/bench_ttft.py` |
| Section 6, blob sizes (805, 2,065, 2,165, 1,223 MB) | `BLOBS.json` -- both SI megabytes and mebibytes, because the paper uses both units; the blobs themselves are too large to ship and the bundle manifests carry their SHA-256 |
| Table 4, server accuracy (0.8659, 0.683, 0.783) | `results/34_server_as.json`, `35_rejudge.json`; `code/cascade/rejudge_with_server.py` |
| Section 6, 0.7887 / 0.7302, 52.8%, 47.2% | `results/35_rejudge.json`; `code/cascade/report_verdict06.py` |
| Table 5, trivial stratum (0.9430, 0.8727, 0.9492, 0.9633) | `results/40_rejudge_trivial.json`, `37_trivial_router_probe*.json` |
| Section 6, critical fraction 6.5% | `code/cascade/trivial_traffic_sensitivity.py` |
| Section 7, tool calling (0.5116, threshold 0.9335) | `results/42_tool_head.json`, `44_threshold_suite.json`; `code/scoring/eval_tool_head.py` |
| Section 3, corpus (5,000 / 2,648 / 21,184) | `corpus/`, `configs/` |
| Section 6, server accuracy per item | `dataset/server_grades.jsonl` |
| Appendix A, registered thresholds | `SPEC.md` |
| Appendix B, the three bundles | `bundles/` |

## Reproducing the headline numbers without a GPU

The graded corpus, the frozen heads and the per-item router scores are all here, so
Table 1 and Table 2 recompute from files:

    python3 code/scoring/check_paper_numbers.py

That script is the artifact repository's own gate: it reads each number out of the
data file and fails if the write-up disagrees with it. It needs `$REPO` to point at
an unpacked copy of this package's `code/`, `results/` and `corpus/` laid out as
`scripts/`, `results/` and `data/` -- the paths inside the scripts were rewritten to
environment variables, see below.

Re-running generation or extraction needs the weight blobs, which are not in this
package. `BLOBS.json` records each blob's name and byte size, and each bundle's
`MANIFEST.json` records its SHA-256, so a reader who rebuilds the quantisations from
the public base model can confirm they have the same bytes.

## What is deliberately absent

The server model's response texts. The Ethics statement gives the reason: we did not
seek a determination on whether redistributing another provider's outputs conflicts
with that provider's terms. `dataset/server_grades.jsonl` carries the per-item
grading label and a SHA-256 of each response instead, and recomputing server
accuracy from those labels returns the value reported in Section 6.

The ungraded 2,352 corpus items are present as prompts but carry no grade, which is
Section 3's statement rather than an omission here.

## Paths

Absolute paths were replaced with environment variables (`$REPO`, `$HOME`,
`$HF_HOME`, `$LLAMA_CPP`, `$DATASET`, `$VENV`, `$WORK`) so nothing points at the
machine the measurements were taken on. Set them before running. The inference
runtime for the quantised bundles is the public llama.cpp project, fetched at first
run rather than redistributed; the bundles' `delegate/config.json` records the build
line with the upstream URL removed for anonymity.

## License

To be released under Apache-2.0 on publication, which is the licence the base
model's permits and the one the delegation head is contributed under.
