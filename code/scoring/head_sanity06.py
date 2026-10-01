"""위임 헤드 위생 검사 — 셀 내부 라벨 섞기 (SPEC 13-U 의 G8 필수 검사).

동결된 헤드의 cal 점수는 그대로 두고 **라벨만 셀(카테고리/난이도/언어) 안에서 섞어** 통제
AUROC 를 다시 잰다. 신호가 진짜라면 섞은 뒤에는 0.5 주변이어야 한다. 0.5 를 포함하지 않으면
통제 구현이나 특징에 누출이 있다는 뜻이므로 판정을 무효로 봐야 한다.

  .venv/bin/python scripts/head_sanity06.py [--tag sc06] [--n 400]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, read_json, require  # noqa: E402

FAIL_THR = 0.75


def main() -> int:
    import numpy as np

    from signal_probe import auroc_within

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sc06", help="빈 문자열이면 1.7B 기본 헤드")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    sfx = f".{a.tag}" if a.tag else ""

    feat = DATA / f"26_llamacpp_features{sfx}.npz"
    hw = DATA / f"26_llamacpp_head{sfx}.npz"
    hj = DATA / f"26_llamacpp_head{sfx}.json"
    for p in (feat, hw, hj):
        require(p.exists(), f"{p} 가 없다")

    z = np.load(feat, allow_pickle=False)
    X, split, acc, cell3 = z["X"], z["split"], z["acc"], z["cell3"]
    h = np.load(hw, allow_pickle=False)
    head = read_json(hj)

    ca = split == "cal"
    Xca = (X[ca] - h["mean"]) / h["scale"]
    # 헤드의 점수 규약: 1 - P(틀림). 높을수록 로컬 유지.
    s = 1.0 - 1.0 / (1.0 + np.exp(-(Xca @ h["coef"] + float(h["intercept"][0]))))
    y = ~(acc[ca] < FAIL_THR)                     # True = 맞힘
    c3 = list(cell3[ca])

    real, n_pair, n_cell = auroc_within(s, y, c3)
    print(f"태그 {a.tag or '(1.7B 기본)'} / cal {ca.sum()}건 / 셀 {n_cell} / 짝 {n_pair}")
    print(f"실제 통제 AUROC  {real:.4f}   (헤드 json 기록 {head['cal_auroc_within_cell_lang']:.4f})")
    require(abs(real - head["cal_auroc_within_cell_lang"]) < 5e-3,
            f"동결 헤드 재계산이 기록과 다르다: {real:.4f} vs "
            f"{head['cal_auroc_within_cell_lang']:.4f}")

    # 셀 내부에서만 라벨을 섞는다 — 셀 구성은 보존하고 항목 수준 신호만 파괴한다
    idx_by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(c3):
        idx_by_cell.setdefault(c, []).append(i)

    rng = np.random.default_rng(20260917)
    vals = []
    for _ in range(a.n):
        ys = y.copy()
        for ix in idx_by_cell.values():
            if len(ix) > 1:
                ys[ix] = y[rng.permutation(ix)]
        v = auroc_within(s, ys, c3)[0]
        if v == v:
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    print(f"라벨 섞기 {len(vals)}회: 평균 {vals.mean():.4f} / 95% CI [{lo:.4f}, {hi:.4f}]")

    ok = lo <= 0.5 <= hi
    print(f"\n위생 검사: {'통과 — 0.5 를 포함한다 (누출 없음)' if ok else '실패 — 0.5 를 포함하지 않는다'}")

    from _common import write_json
    write_json({"tag": a.tag or "", "n_cal": int(ca.sum()), "n_cell": int(n_cell),
                "n_pair": int(n_pair), "real_ctrl_auroc": float(real),
                "shuffle_n": int(len(vals)), "shuffle_mean": float(vals.mean()),
                "shuffle_ci": [float(lo), float(hi)], "passed": bool(ok)},
               DATA / f"26_head_sanity{sfx}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
