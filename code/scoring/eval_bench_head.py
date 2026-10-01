"""동결 헤드를 **외부 공개 분포**에서 평가한다 (SPEC 13-AL).

지금까지 모든 분별력 수치는 자체 생성 합성 코퍼스에서 나왔고, 그 코퍼스에는 13-C 에서 발견한
구조적 결함(셀 동질성)이 있다. 헤드가 **다른 분포**에서도 신호를 담는지가 이식·공개의 관문이다.

GSM8K 250 + MMLU 232(채점이 불안정한 18건 제외)에 대해
  * student 정확도(8시드) -> 라벨
  * 배포 경로의 동결 헤드 점수 -> 예측
로 AUROC 를 낸다. **헤드는 이 분포로 다시 적합하지 않았다.**

통제: 원래 3단 통제는 (카테고리, 난이도, 언어) 26셀이었다. 여기서는 난이도·언어가 상수이므로
**카테고리 2셀** 통제다. 같은 종류의 통제(카테고리 정체성 제거)이지만 셀 수가 다르므로
원 수치와 직접 비교하지 않고 나란히 놓는다.

    .venv/bin/python scripts/eval_bench_head.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RESULTS, read_rows, write_json  # noqa: E402
from signal_probe import auroc, auroc_within  # noqa: E402

FAIL_THR = 0.75            # llamacpp_head 와 같은 라벨 정의
ARMS = {
    "gguf": {"tag": "bench", "label": "1.7B", "esc": 0.20,
             "in_dist_test_ctrl": 0.7100},
    "sc06": {"tag": "bench.sc06", "label": "0.6B", "esc": 0.50,
             "in_dist_test_ctrl": 0.6704},
}
BOOT = 400
SEED = 0


def boot_ci(scores, labels, cells, n=BOOT):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(scores))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        a, npair, _ = auroc_within([scores[i] for i in s], [labels[i] for i in s],
                                   [cells[i] for i in s])
        if npair > 0 and a == a:
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main() -> int:
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.bench.parquet")}
    # 선택지 본문을 못 뽑아 채점이 불안정한 MMLU 항목은 제외한다(정답이 낱글자로 남아 있다).
    drop = {i for i, k in keys.items()
            if i.startswith("MMLU") and len(str(k["answer"]).strip()) == 1}
    print(f"채점 불안정으로 제외한 MMLU 항목 {len(drop)}건")

    out = {"excluded_unstable_mmlu": sorted(drop), "fail_thr": FAIL_THR}
    for arm, cfg in ARMS.items():
        acc = {r["id"]: float(r["accuracy"])
               for r in read_rows(DATA / f"06_accuracy.{cfg['tag']}.parquet")}
        cat = {r["id"]: r["category"]
               for r in read_rows(DATA / "01_prompts.bench.parquet")}
        routes = [json.loads(x) for x in
                  (DATA / f"38_route_layer.{cfg['tag']}.jsonl").read_text(
                      encoding="utf-8").splitlines() if x.strip()]
        rows = [r for r in routes if r["id"] in acc and r["id"] not in drop]
        sc = [float(r["score"]) for r in rows]
        ok = [bool(acc[r["id"]] >= FAIL_THR) for r in rows]      # True = student 성공
        cl = [cat[r["id"]] for r in rows]

        a_raw = auroc(sc, ok)
        a_ct, npair, ncell = auroc_within(sc, ok, cl)
        lo, hi = boot_ci(sc, ok, cl)
        n_del = sum(1 for r in rows if r["route"] == "delegate")
        arm_out = {
            "n": len(rows), "n_success": int(sum(ok)),
            "auroc_uncontrolled": float(a_raw),
            "auroc_controlled_category": float(a_ct),
            "ci95": [lo, hi], "n_pair": int(npair), "n_cell": int(ncell),
            "esc_rate_on_bench": n_del / len(rows),
            "in_dist_test_3way_ctrl": cfg["in_dist_test_ctrl"],
        }
        # 카테고리별로도 본다 — 한쪽에서만 신호가 있으면 그것도 결과다
        for c in sorted(set(cl)):
            m = [i for i, x in enumerate(cl) if x == c]
            if len({ok[i] for i in m}) < 2:
                arm_out[f"auroc_{c}"] = None
                continue
            arm_out[f"auroc_{c}"] = float(auroc([sc[i] for i in m], [ok[i] for i in m]))
        out[arm] = arm_out

        print(f"\n[{cfg['label']}]  n={len(rows)} (성공 {sum(ok)})")
        print(f"  통제 없는 AUROC       {a_raw:.4f}")
        print(f"  **카테고리 통제 AUROC** {a_ct:.4f}  CI95 [{lo:.4f}, {hi:.4f}] "
              f"(셀 {ncell}, 짝 {npair})")
        print(f"  참고: 자체 코퍼스 test 3단 통제 {cfg['in_dist_test_ctrl']:.4f}")
        print(f"  이 분포에서의 위임률   {100*n_del/len(rows):.1f}% "
              f"(운용점 esc {cfg['esc']:.0%})")
        for c in sorted(set(cl)):
            v = arm_out.get(f"auroc_{c}")
            print(f"    {c} 단독 AUROC       {'—' if v is None else f'{v:.4f}'}")

    write_json(out, RESULTS / "41_bench_head.json")
    print(f"\n-> {RESULTS / '41_bench_head.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
