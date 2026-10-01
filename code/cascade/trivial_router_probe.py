"""라우터가 사소한 질의를 로컬로 유지하는가 — 분포 밖 점검.

헤드는 **10토큰 이상** 프롬프트로만 적합됐다(코퍼스 최소 10, 5분위 17). "안녕"·"1+1" 같은
1~5토큰 질의는 헤드에게 **분포 밖**이다. 그런 질의를 라우터가 로컬로 유지하는지는 가정이 아니라
측정으로 확인해야 한다 — 만약 낮은 점수를 줘서 위임한다면 가장 나쁜 결과다(사소한 질의에
서버 비용을 쓰고 지연도 늘어난다).

서버가 이미 떠 있어야 한다(`cascade start`).

    .venv/bin/python scripts/trivial_router_probe.py [--esc 0.20]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, write_json  # noqa: E402

ART = Path("$REPO/artifact/cascade-1.7b-gguf")
BASE = "http://127.0.0.1:8099"
# 사소한 질의. 실제 트래픽의 머리 부분을 대표한다 — 인사·아주 쉬운 산수·단문 사실.
TRIVIAL = [
    ("인사", "안녕"), ("인사", "안녕하세요"), ("인사", "hi"), ("인사", "hello"),
    ("인사", "고마워"), ("인사", "thanks"),
    ("산수", "1+1"), ("산수", "2+2는?"), ("산수", "10 x 10"), ("산수", "what is 3+3"),
    ("단문사실", "대한민국 수도"), ("단문사실", "capital of France"),
    ("단문사실", "물의 화학식"), ("단문사실", "how many days in a week"),
    ("잡담", "ㅇㅇ"), ("잡담", "ok"), ("잡담", "뭐해?"), ("잡담", "what's up"),
]
# 대조군: 코퍼스 분포 안의 짧은 항목과 명백히 어려운 항목
CONTROL = [
    ("대조-어려움", "1979년 이란 혁명이 OPEC 의 유가 결정 구조에 미친 장기적 영향을 설명해봐"),
    ("대조-어려움", "Derive the variance of the OLS estimator under heteroskedasticity"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc", type=float, default=0.20)
    ap.add_argument("--artifact", default=str(ART),
                    help="다른 산출물로 재려면 경로를 준다 (0.6B arm 등)")
    ap.add_argument("--out", default="37_trivial_router_probe.json")
    a = ap.parse_args()

    art = Path(a.artifact)
    cfg = json.loads((art / "delegate" / "config.json").read_text())
    thr = float(cfg["thresholds"][f"{a.esc:.2f}"])
    spec = importlib.util.spec_from_file_location("_infer", art / "infer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not m.alive(BASE):
        print(f"서버가 없다: {BASE} — `cascade start` 를 먼저 해라", file=sys.stderr)
        return 1
    head = m.load_head()

    print(f"산출물 {art.name} / 위임률 목표 {a.esc:.0%} -> 문턱 {thr:.4f}")
    print(f"{'분류':<12}{'질의':<52}{'점수':>8}  결정")
    rows, n_del = [], 0
    for kind, q in TRIVIAL + CONTROL:
        r = m.route(BASE, head, q, thr)
        deleg = r["route"] == "delegate"
        if kind != "대조-어려움" and deleg:
            n_del += 1
        rows.append({"kind": kind, "prompt": q, "score": r["score"],
                     "route": r["route"]})
        print(f"{kind:<12}{q[:50]:<52}{r['score']:>8.4f}  "
              f"{'위임 ⚠' if deleg else '로컬'}")

    triv = [x for x in rows if x["kind"] != "대조-어려움"]
    print(f"\n사소한 질의 {len(triv)}건 중 위임 {n_del}건 ({100*n_del/len(triv):.0f}%)")
    scores = [x["score"] for x in triv]
    print(f"  점수 최소 {min(scores):.4f} / 중앙 {sorted(scores)[len(scores)//2]:.4f} / "
          f"최대 {max(scores):.4f}")
    ctrl = [x for x in rows if x["kind"] == "대조-어려움"]
    print(f"  대조(어려움) 점수: {[round(x['score'],4) for x in ctrl]} "
          f"-> {[x['route'] for x in ctrl]}")
    write_json({"target_esc": a.esc, "threshold": thr, "rows": rows,
                "trivial_delegated": n_del, "trivial_n": len(triv)},
               RESULTS / a.out)
    print(f"\n-> {RESULTS / a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
