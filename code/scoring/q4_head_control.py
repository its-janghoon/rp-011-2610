"""Q4 재적합의 값을 대조군으로 확인한다 (SPEC 13-AH 의 Q4·Q5 예측 검증).

사전 등록한 두 가지를 잰다.
  Q5 대조군 — **재적합하지 않은** Q8_0 헤드를 Q4 임베딩에 그대로 얹으면 분별력이 어떻게 되나.
              이것이 "양자화를 바꾸면 헤드를 다시 적합해야 한다" 는 주장의 근거다.
  Q4 예측   — 재적합한 Q4 헤드의 계수가 Q8_0 헤드와 상관은 높지만 동일하지는 않은가.

cal 만 쓴다. test 는 이 스크립트에서 열지 않는다.

    .venv/bin/python scripts/q4_head_control.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RESULTS, write_json  # noqa: E402
from llamacpp_head import FAIL_THR  # noqa: E402
from signal_probe import auroc_within  # noqa: E402


def score(head: dict, X: np.ndarray) -> np.ndarray:
    """헤드를 특징에 적용해 점수를 낸다. 높을수록 로컬 유지 (1 - P(실패))."""
    z = (X - head["mean"]) / head["scale"]
    p = 1.0 / (1.0 + np.exp(-(z @ head["coef"] + float(np.ravel(head["intercept"])[0]))))
    return 1.0 - p


def load(p: Path) -> dict:
    z = np.load(p)
    return {k: z[k] for k in ("coef", "intercept", "mean", "scale")}


def main() -> int:
    zq4 = np.load(DATA / "26_llamacpp_features.q4.npz")
    X, split, acc, cell3 = zq4["X"], zq4["split"], zq4["acc"], zq4["cell3"]
    ca = split == "cal"
    y = acc < FAIL_THR                     # True = student 가 틀렸다
    Xca, yca, c3 = X[ca], ~y[ca], list(cell3[ca])
    print(f"cal {ca.sum()}건 (틀림 {int(y[ca].sum())})")

    h_q4 = load(DATA / "26_llamacpp_head.q4.npz")
    h_q8 = load(DATA / "26_llamacpp_head.npz")

    out = {}
    print("\n같은 Q4 임베딩에 두 헤드를 얹는다 (cal 3단 통제 AUROC)")
    for name, h in (("Q4 재적합 헤드", h_q4), ("Q8_0 동결 헤드 (재적합 없음)", h_q8)):
        a, npair, ncell = auroc_within(score(h, Xca), yca, c3)
        key = "q4_refit" if "재적합 헤드" in name else "q8_frozen_on_q4"
        out[key] = {"cal_ctrl_auroc": float(a), "n_pair": int(npair), "n_cell": int(ncell)}
        print(f"  {name:<28} {a:.4f}  (셀 {ncell}, 짝 {npair})")

    d = out["q4_refit"]["cal_ctrl_auroc"] - out["q8_frozen_on_q4"]["cal_ctrl_auroc"]
    out["refit_gain"] = float(d)
    print(f"  재적합의 이득: {d:+.4f}")

    print("\n계수 비교 (Q4 재적합 대 Q8_0)")
    r = float(np.corrcoef(h_q4["coef"].astype(np.float64),
                          h_q8["coef"].astype(np.float64))[0, 1])
    cos = float(h_q4["coef"] @ h_q8["coef"]
                / (np.linalg.norm(h_q4["coef"]) * np.linalg.norm(h_q8["coef"])))
    ident = bool(np.array_equal(h_q4["coef"], h_q8["coef"]))
    out["coef"] = {"pearson_r": r, "cosine": cos, "identical": ident,
                   "l2_q4": float(np.linalg.norm(h_q4["coef"])),
                   "l2_q8": float(np.linalg.norm(h_q8["coef"]))}
    print(f"  피어슨 r {r:.4f} / 코사인 {cos:.4f} / 완전히 동일? {ident}")
    print(f"  L2 노름  Q4 {out['coef']['l2_q4']:.4f}  Q8_0 {out['coef']['l2_q8']:.4f}")

    # 임베딩 자체가 얼마나 달라졌는가 — 원인 쪽 증거
    zq8 = np.load(DATA / "26_llamacpp_features.npz")
    if list(zq8["ids"]) == list(zq4["ids"]):
        A, B = zq8["X"].astype(np.float64), X.astype(np.float64)
        cs = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        out["embedding_cosine"] = {"mean": float(cs.mean()), "min": float(cs.min()),
                                   "p05": float(np.percentile(cs, 5))}
        print(f"\n임베딩 코사인 유사도 (Q8_0 대 Q4, 같은 항목 {len(cs)}건)")
        print(f"  평균 {cs.mean():.4f} / 5분위 {np.percentile(cs,5):.4f} / 최소 {cs.min():.4f}")
    else:
        print("\n(항목 순서가 달라 임베딩 직접 비교는 건너뜀)")

    write_json(out, RESULTS / "36_q4_control.json")
    print(f"\n-> {RESULTS / '36_q4_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
