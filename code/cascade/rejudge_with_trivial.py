"""사소 질의 층을 섞어 두 arm 을 재판정한다 (SPEC 13-AK).

13-AJ 에서 **산술로** 추정했던 것을 이제 **실측으로** 바꾼다. 그때는 "사소한 질의는 양쪽 다
로컬에서 정답" 이라고 가정했는데, 실측은 다르다:
  * 1.7B 사소 층 정확도 0.9344 / 라우터가 위임 0.6%
  * 0.6B 사소 층 정확도 0.8539 / 라우터가 위임 13.8%   <- 가정보다 나쁘다

사소 층은 **평가 전용**이다. 헤드를 이 층으로 다시 적합하지 않았다(동결 유지).

혼합 규칙: 실제 트래픽의 f 비율이 사소하다면
    정확도(arm, f) = (1-f) x 비사소_캐스케이드(arm) + f x 사소_캐스케이드(arm)
    위임률(arm, f) = (1-f) x 비사소_위임률(arm) + f x 사소_위임률(arm)
비사소 쪽 곡선은 `rejudge_with_server.py` 의 것을 그대로 재사용한다(항목별 서버 실측).

    .venv/bin/python scripts/rejudge_with_trivial.py
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
from grade import grade_contains, grade_exact  # noqa: E402
from rejudge_with_server import curve, interp, server_outcomes  # noqa: E402

LOCAL_FLOOR = 0.50          # 13-U 사전 등록 기준 B
ARMS = {
    "gguf": {"tag": "triv", "escmap": "09_escalation_map.parquet", "cond": "G-gguf",
             "acc": "06_accuracy.parquet", "target": 0.20, "label": "1.7B"},
    "sc06": {"tag": "triv.sc06", "escmap": "09_escalation_map.sc06.parquet",
             "cond": "G-sc06", "acc": "06_accuracy.sc06.parquet", "target": 0.40,
             "label": "0.6B"},
}


def trivial_stats(arm: str) -> dict:
    """사소 층의 캐스케이드 정확도와 위임률. 위임 항목은 서버 답을 채점한다."""
    cfg = ARMS[arm]
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.triv.parquet")}
    prompts = {r["id"]: r["prompt"] for r in read_rows(DATA / "01_prompts.triv.parquet")}
    acc = {r["id"]: float(r["accuracy"])
           for r in read_rows(DATA / f"06_accuracy.{cfg['tag']}.parquet")}
    routes = [json.loads(x) for x in
              (DATA / f"38_route_layer.{cfg['tag']}.jsonl").read_text(
                  encoding="utf-8").splitlines() if x.strip()]
    srv_path = DATA / f"39_triv_server.{cfg['tag']}.jsonl"
    srv = {}
    if srv_path.exists():
        srv = {json.loads(x)["id"]: json.loads(x)["response"] for x in
               srv_path.read_text(encoding="utf-8").splitlines() if x.strip()}

    def grade(i: str, text: str) -> bool:
        k = keys[i]
        if k["grading"] == "exact":
            return bool(grade_exact(text, k, prompts[i])[0])
        return bool(grade_contains(text, k)[0])

    vals, n_del, missing = [], 0, []
    for r in routes:
        i = r["id"]
        if r["route"] == "delegate":
            n_del += 1
            if i in srv:
                vals.append(1.0 if grade(i, srv[i]) else 0.0)
            else:
                missing.append(i)
        else:
            vals.append(acc[i])
    return {"cascade_acc": st.mean(vals) if vals else float("nan"),
            "esc_rate": n_del / len(routes), "n": len(routes),
            "local_only_acc": st.mean(acc[r["id"]] for r in routes),
            "n_delegate": n_del, "missing_server": missing}


def main() -> int:
    srv_nt, model = server_outcomes()
    out = {"server_model": model}
    triv, nont = {}, {}
    for arm, cfg in ARMS.items():
        t = trivial_stats(arm)
        if t["missing_server"]:
            print(f"!! {cfg['label']}: 위임된 사소 항목의 서버 응답 "
                  f"{len(t['missing_server'])}건이 없다 — "
                  f"scripts/fill_triv_server.py 를 먼저 돌려라\n   {t['missing_server'][:5]}",
                  file=sys.stderr)
            return 1
        acc = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / cfg["acc"])}
        c = curve(cfg["escmap"], cfg["cond"], acc, srv_nt)
        op = next(r for r in c if abs(float(r["target_esc"]) - cfg["target"]) < 1e-3)
        triv[arm], nont[arm] = t, {"curve": c, "op": op}
        print(f"[{cfg['label']}]")
        print(f"  비사소 운용점 : 캐스케이드 {op['cascade_acc']:.4f} / "
              f"위임률 {100*float(op['esc_rate']):.1f}% (n={op['n']})")
        print(f"  사소 층      : 캐스케이드 {t['cascade_acc']:.4f} / "
              f"위임률 {100*t['esc_rate']:.1f}% / 로컬전용 {t['local_only_acc']:.4f} "
              f"(n={t['n']}, 위임 {t['n_delegate']})")

    print()
    print("혼합 재판정 — 실제 트래픽의 f 비율이 사소할 때")
    print(f"  {'f':>6}{'1.7B 정확도':>12}{'0.6B 필요위임':>14}{'0.6B 전체위임':>14}"
          f"{'0.6B 로컬처리':>14}   기준 B")
    rows = []
    for f in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        # 1.7B 목표 정확도 (혼합)
        a17 = (1 - f) * nont["gguf"]["op"]["cascade_acc"] + f * triv["gguf"]["cascade_acc"]
        # 0.6B 가 그 정확도에 닿으려면 비사소 쪽에서 필요한 위임률
        # (1-f)*C06(e) + f*T06 = a17  ->  C06(e) = (a17 - f*T06) / (1-f)
        if f >= 1.0:
            continue
        need_c = (a17 - f * triv["sc06"]["cascade_acc"]) / (1 - f)
        e_nt = interp(nont["sc06"]["curve"], need_c)
        if e_nt is None:
            rows.append({"f": f, "acc17": a17, "need_esc_nontrivial": None})
            print(f"  {100*f:>5.1f}%{a17:>12.4f}{'도달불가':>14}{'—':>14}{'—':>14}   —")
            continue
        esc_all = (1 - f) * e_nt + f * triv["sc06"]["esc_rate"]
        loc = 1 - esc_all
        ok = loc >= LOCAL_FLOOR
        rows.append({"f": f, "acc17": a17, "need_esc_nontrivial": e_nt,
                     "esc_all_06": esc_all, "local_rate_06": loc, "criterion_b": ok})
        print(f"  {100*f:>5.1f}%{a17:>12.4f}{100*e_nt:>13.1f}%{100*esc_all:>13.1f}%"
              f"{100*loc:>13.1f}%   {'통과' if ok else '실패'}")

    # 판정이 뒤집히는 f 를 이분법으로 찾는다
    lo, hi = 0.0, 0.95
    for _ in range(60):
        mid = (lo + hi) / 2
        a17 = (1 - mid) * nont["gguf"]["op"]["cascade_acc"] + mid * triv["gguf"]["cascade_acc"]
        need_c = (a17 - mid * triv["sc06"]["cascade_acc"]) / (1 - mid)
        e = interp(nont["sc06"]["curve"], need_c)
        if e is None:
            lo = mid
            continue
        loc = 1 - ((1 - mid) * e + mid * triv["sc06"]["esc_rate"])
        if loc >= LOCAL_FLOOR:
            hi = mid
        else:
            lo = mid
    print(f"\n  >>> 기준 B 가 통과로 바뀌는 임계 비율: f* = {100*hi:.1f}%")
    print(f"      (13-AJ 의 산술 추정은 6.6% 였다 — 실측 사소층 정확도·위임률을 넣은 값이 이것이다)")

    out.update({"trivial": {k: {kk: vv for kk, vv in v.items() if kk != "missing_server"}
                            for k, v in triv.items()},
                "nontrivial_op": {k: v["op"] for k, v in nont.items()},
                "mix": rows, "f_star": hi, "local_floor": LOCAL_FLOOR})
    write_json(out, RESULTS / "40_rejudge_trivial.json")
    print(f"\n-> {RESULTS / '40_rejudge_trivial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
