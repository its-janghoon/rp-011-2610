"""도구 호출 층에서 라우터가 쓸모 있는가 (SPEC 13-AM).

물어야 할 것은 정확도가 아니라 **라우터가 실패를 알아보는가** 다. 도구 호출은 easy/medium 은
거의 성공하고 hard 에서 급락한다(실측). 라우터가 그 hard 를 골라 위임하면 캐스케이드가 이득이고,
못 고르면 이 층에서는 쓸모가 없다.

    .venv/bin/python scripts/eval_tool_head.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RESULTS, read_rows, write_json  # noqa: E402
from signal_probe import auroc, auroc_within  # noqa: E402

FAIL_THR = 0.75
ARMS = {"gguf": ("1.7B", "tool", 0.20), "sc06": ("0.6B", "tool.sc06", 0.50)}


def main() -> int:
    diff = {r["id"]: r["difficulty"] for r in read_rows(DATA / "01_prompts.tool.parquet")}
    out = {}
    for arm, (label, tag, esc) in ARMS.items():
        acc = {r["id"]: float(r["accuracy"])
               for r in read_rows(DATA / f"06_accuracy.{tag}.parquet")}
        routes = [json.loads(x) for x in
                  (DATA / f"38_route_layer.{tag}.jsonl").read_text(
                      encoding="utf-8").splitlines() if x.strip()]
        rows = [r for r in routes if r["id"] in acc]
        sc = [float(r["score"]) for r in rows]
        ok = [bool(acc[r["id"]] >= FAIL_THR) for r in rows]
        dl = [diff[r["id"]] for r in rows]
        n_del = sum(1 for r in rows if r["route"] == "delegate")

        a_raw = auroc(sc, ok) if len(set(ok)) > 1 else float("nan")
        a_ct, npair, ncell = auroc_within(sc, ok, dl)

        # 라우터가 hard 를 골라내는가 — 난이도별 위임률
        by = {}
        for d in ("easy", "medium", "hard"):
            m = [r for r in rows if diff[r["id"]] == d]
            if not m:
                continue
            by[d] = {"n": len(m),
                     "esc": sum(1 for r in m if r["route"] == "delegate") / len(m),
                     "acc": st.mean(acc[r["id"]] for r in m),
                     "score_mean": st.mean(float(r["score"]) for r in m)}

        out[arm] = {"n": len(rows), "esc_rate": n_del / len(rows),
                    "auroc_uncontrolled": a_raw,
                    "auroc_controlled_difficulty": a_ct,
                    "n_pair": int(npair), "n_cell": int(ncell), "by_difficulty": by,
                    "local_only_acc": st.mean(acc[r["id"]] for r in rows)}

        # **보정 진단.** 순서가 보존돼도 문턱이 맞지 않으면 아무것도 위임되지 않는다.
        # 등록된 격자를 훑어 각 문턱에서 난이도별 위임률이 어떻게 되는지 본다 — 도구 호출
        # 프롬프트가 이 헤드에게 일률적으로 '쉬워 보이는' 것인지, 아니면 순서 자체가 없는지
        # 구분하는 진단이다.
        import json as _json
        art = ("cascade-1.7b-gguf" if arm == "gguf"
               else "cascade-0.6b-gguf")
        cfg = _json.loads((Path(__file__).resolve().parent.parent / "artifact" / art
                           / "delegate" / "config.json").read_text())
        sweep = []
        for k in sorted(cfg["thresholds"], key=float):
            t = float(cfg["thresholds"][k])
            row = {"target_esc": float(k), "threshold": t}
            for d in ("easy", "medium", "hard"):
                m = [float(r["score"]) for r in rows if diff[r["id"]] == d]
                row[d] = sum(1 for s in m if s < t) / len(m) if m else None
            sweep.append(row)
        out[arm]["threshold_sweep"] = sweep
        print(f"  문턱 훑기 (난이도별 위임률) — 순서가 살아 있으면 hard 가 먼저 넘어가야 한다")
        print(f"  {'esc':>6}{'문턱':>9}{'easy':>8}{'medium':>8}{'hard':>8}")
        for r in sweep:
            if r["easy"] is None:
                continue
            print(f"  {r['target_esc']:>6.2f}{r['threshold']:>9.4f}"
                  f"{100*r['easy']:>7.0f}%{100*r['medium']:>7.0f}%{100*r['hard']:>7.0f}%")

        print(f"\n[{label}]  n={len(rows)} / 운용점 esc {esc:.0%} / 실제 위임 "
              f"{100*n_del/len(rows):.1f}%")
        print(f"  로컬 전용 정확도 {out[arm]['local_only_acc']:.4f}")
        print(f"  통제 없는 AUROC  {a_raw:.4f}" if a_raw == a_raw else "  AUROC 계산 불가")
        print(f"  난이도 통제 AUROC {a_ct:.4f} (셀 {ncell}, 짝 {npair})"
              if a_ct == a_ct else "  난이도 통제 AUROC 계산 불가")
        print(f"  {'난이도':<8}{'n':>4}{'정확도':>9}{'평균점수':>10}{'위임률':>9}")
        for d, v in by.items():
            print(f"  {d:<8}{v['n']:>4}{v['acc']:>9.4f}{v['score_mean']:>10.4f}"
                  f"{100*v['esc']:>8.0f}%")

    write_json(out, RESULTS / "42_tool_head.json")
    print(f"\n-> {RESULTS / '42_tool_head.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
