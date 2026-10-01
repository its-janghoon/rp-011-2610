"""사소 질의 층의 라우팅을 실측한다 (SPEC 13-AK).

산출물의 배포 경로(llama.cpp Q8_0 서버 + 동결 헤드)를 그대로 써서 각 항목의 라우팅 결정을
낸다. 헤드는 **동결**이고 이 층으로 다시 적합하지 않는다 — 사소 층은 평가 전용이다.

서버가 해당 arm 으로 떠 있어야 한다.

    ~/bin/cascade start                                    # 1.7B
    .venv/bin/python scripts/route_layer.py --tag triv --arm gguf --esc 0.20

    CASCADE_ARTIFACT=.../gguf06-v1 ~/bin/cascade start      # 0.6B
    .venv/bin/python scripts/route_layer.py --tag triv.sc06 --arm sc06 --esc 0.40
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
from _common import DATA, REPO, read_rows, write_jsonl  # noqa: E402

ARTS = {"gguf": "cascade-1.7b-gguf", "sc06": "cascade-0.6b-gguf"}
BASE = "http://127.0.0.1:8099"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="출력 접미사 (예: triv, triv.sc06)")
    ap.add_argument("--arm", required=True, choices=sorted(ARTS))
    ap.add_argument("--esc", type=float, required=True, help="운용점 위임률")
    ap.add_argument("--prompts", default=str(DATA / "01_prompts.triv.parquet"))
    a = ap.parse_args()

    art = REPO / "artifact" / ARTS[a.arm]
    cfg = json.loads((art / "delegate" / "config.json").read_text())
    thr = float(cfg["thresholds"][f"{a.esc:.2f}"])
    spec = importlib.util.spec_from_file_location("_infer", art / "infer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not m.alive(BASE):
        print(f"서버가 없다: {BASE} — 해당 arm 으로 `cascade start` 를 먼저 해라",
              file=sys.stderr)
        return 1
    head = m.load_head()

    rows = sorted(read_rows(Path(a.prompts)), key=lambda r: r["id"])
    print(f"arm {a.arm} / 운용점 {a.esc:.0%} -> 문턱 {thr:.4f} / 항목 {len(rows)}건")
    out, n_del = [], 0
    for n, r in enumerate(rows, 1):
        d = m.route(BASE, head, r["prompt"], thr)
        deleg = d["route"] == "delegate"
        n_del += deleg
        out.append({"id": r["id"], "prompt": r["prompt"], "route": d["route"],
                    "score": round(d["score"], 6), "threshold": round(thr, 6)})
        if n % 40 == 0:
            print(f"  {n}/{len(rows)}  위임 누계 {n_del}", flush=True)

    p = DATA / f"38_route_layer.{a.tag}.jsonl"
    write_jsonl(out, p)
    print(f"\n위임 {n_del}/{len(rows)} ({100*n_del/len(rows):.1f}%) -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
