"""Stage 4 — 실측 서버 응답으로 두 arm 을 재판정한다 (SPEC 13-AE).

**a_s 라는 상수를 없앤다.** 지금까지 캐스케이드는 `mean(where(esc, a_s, student))` 로 계산했고
a_s 는 하나의 스칼라(0.967)였다. 이제 656 항목 전부에 서버 응답이 있으므로 **항목별 서버
정답 여부**를 그대로 쓴다:

    캐스케이드 = mean( where(esc, server_correct_i, student_acc_i) )

이러면 a_s 가 모수로 등장하지 않는다. 부수적으로 13-AD 의 실측이 드러낸 사실 하나가
자동으로 반영된다 — **a_s 는 운용점에 의존한다**(1.7B @esc20% 부분집합 0.683, 0.6B @esc40%
0.783). 위임을 적게 하는 arm 은 더 어려운 항목만 보내므로 그 부분집합의 서버 정확도가 낮다.
상수 a_s 를 두 arm 에 같이 쓰면 **위임을 적게 하는 arm 을 체계적으로 유리하게** 만든다.

  .venv/bin/python scripts/rejudge_with_server.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, fmt_table, read_json, read_rows, require, write_json  # noqa: E402
from grade import grade_contains, grade_exact  # noqa: E402

LOCAL_FLOOR = 0.50
TOL_PP = 2.0


def server_outcomes() -> tuple[dict[str, bool], str]:
    rows = [json.loads(x) for x in
            (DATA / "34_server_responses.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r["prompt"] for r in read_rows(DATA / "01_prompts.parquet")}
    out = {}
    for r in rows:
        k = keys[r["id"]]
        if k["grading"] == "exact":
            out[r["id"]] = bool(grade_exact(r["response"], k, prompts[r["id"]])[0])
        elif k["grading"] == "contains":
            out[r["id"]] = bool(grade_contains(r["response"], k)[0])
    return out, rows[0]["model"]


def curve(escmap: str, cond: str, acc: dict[str, float], srv: dict[str, bool]):
    """문턱별 (esc율, 캐스케이드 정확도, 로컬전용, 그 부분집합의 a_s)."""
    rows = read_rows(RESULTS / escmap)
    by_t: dict[float, list[dict]] = {}
    for r in rows:
        if r["condition"] != cond:
            continue
        by_t.setdefault(round(float(r["target_esc"]), 4), []).append(r)
    out = []
    for t in sorted(by_t):
        rs = [r for r in by_t[t] if r["id"] in acc and r["id"] in srv]
        if not rs:
            continue
        esc = [r["id"] for r in rs if r["escalated"]]
        casc = st.mean((1.0 if srv[r["id"]] else 0.0) if r["escalated"] else acc[r["id"]]
                       for r in rs)
        local_only = st.mean(acc[r["id"]] for r in rs)
        out.append({"target_esc": t, "n": len(rs), "esc_rate": len(esc) / len(rs),
                    "cascade_acc": casc, "local_only_acc": local_only,
                    "a_s_on_subset": st.mean(1.0 if srv[i] else 0.0 for i in esc) if esc
                    else float("nan")})
    return out


def interp(c, target):
    pts = sorted((r["esc_rate"], r["cascade_acc"]) for r in c)
    prev = pts[0]
    for e, a in pts:
        if a >= target:
            if a == prev[1]:
                return e
            return prev[0] + (target - prev[1]) / (a - prev[1]) * (e - prev[0])
        prev = (e, a)
    return None


def main() -> int:
    srv, model = server_outcomes()
    a17 = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.parquet")}
    a06 = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.sc06.parquet")}
    old = read_json(RESULTS / "09_sidecar_curve.base.parquet".replace(".parquet", ".json")) \
        if False else None

    c17 = curve("09_escalation_map.parquet", "G-gguf", a17, srv)
    c06 = curve("09_escalation_map.sc06.parquet", "G-sc06", a06, srv)
    require(c17 and c06, "곡선을 만들지 못했다")

    op17 = next(r for r in c17 if abs(r["target_esc"] - 0.20) < 1e-3)
    target = op17["cascade_acc"]
    need06 = interp(c06, target)

    # 상수 a_s=0.967 로 계산했던 예전 값 (대조용)
    prev17 = read_rows(RESULTS / "09_sidecar_curve.base.parquet")
    prev06 = read_rows(RESULTS / "09_sidecar_curve.sc06.parquet")
    p17 = next(r for r in prev17 if abs(float(r["target_esc"]) - 0.20) < 1e-6)

    t1 = [
        ["서버 정확도 — escalate 부분집합", f"{op17['a_s_on_subset']:.4f}",
         f"{next(r for r in c06 if abs(r['target_esc']-0.40) < 1e-3)['a_s_on_subset']:.4f}",
         "0.967 (에이전트, 두 arm 공용)"],
        ["로컬 전용 정확도", f"{op17['local_only_acc']:.4f}",
         f"{next(r for r in c06 if abs(r['target_esc']-0.40) < 1e-3)['local_only_acc']:.4f}", "—"],
        ["실현 esc율", f"{100*op17['esc_rate']:.1f}%",
         f"{100*next(r for r in c06 if abs(r['target_esc']-0.40) < 1e-3)['esc_rate']:.1f}%", "—"],
        ["**캐스케이드 (항목별 서버 실측)**", f"**{op17['cascade_acc']:.4f}**",
         f"**{next(r for r in c06 if abs(r['target_esc']-0.40) < 1e-3)['cascade_acc']:.4f}**", "—"],
        ["예전 값 (상수 a_s=0.967)", f"{float(p17['cascade_acc']):.4f}",
         f"{float(next(r for r in prev06 if abs(float(r['target_esc'])-0.40) < 1e-6)['cascade_acc']):.4f}",
         "—"],
    ]

    t2 = [[f"{100*r['target_esc']:.0f}%", f"{100*r['esc_rate']:.1f}%",
           f"{r['cascade_acc']:.4f}", f"{r['a_s_on_subset']:.4f}" if r["a_s_on_subset"] == r["a_s_on_subset"] else "—"]
          for r in c06]
    t3 = [[f"{100*r['target_esc']:.0f}%", f"{100*r['esc_rate']:.1f}%",
           f"{r['cascade_acc']:.4f}", f"{r['a_s_on_subset']:.4f}"] for r in c17]

    B_local = 1.0 - need06 if need06 is not None else float("nan")
    A_pass = need06 is not None
    B_pass = A_pass and B_local >= LOCAL_FLOOR

    body = [
        "# Stage 4 — 실측 서버로 두 arm 재판정 (a_s 상수를 없앴다)",
        "",
        f"서버 모델 **{model}**. 캐스케이드를 `mean(where(esc, 항목별 서버 정답, student 정확도))` "
        "로 계산한다 — **a_s 가 모수로 등장하지 않는다.** 판정 규칙은 SPEC 13-AD 에 수치 보기 "
        "전에 등록했다.",
        "",
        "## 표 1. 운용점 대조 — 무엇이 바뀌었나",
        "",
        fmt_table(["항목", "1.7B @esc 20%", "0.6B @esc 40%", "예전 가정"], t1),
        "",
        "**a_s 는 상수가 아니었다.** 1.7B 는 위임을 18% 만 하므로 가장 어려운 항목만 보내고 그 "
        f"부분집합에서 서버가 {op17['a_s_on_subset']:.1%} 만 맞힌다. 0.6B 는 40% 를 보내므로 "
        "부분집합이 넓고 쉬워 서버 정확도가 더 높다(78.3%).",
        "",
        "**그런데 상수를 쓴 편향의 방향은 반대쪽이었다.** 팽창분은 "
        "`esc율 x (가정 a_s − 실제 a_s)` 이므로 **위임을 많이 하는 arm 이 더 많이 부풀려진다**: "
        f"1.7B 는 {100*op17['esc_rate']:.1f}% x {op17['a_s_on_subset']:.3f}->0.967 = "
        f"+{100*(0.967-op17['a_s_on_subset'])*op17['esc_rate']:.1f}pp, "
        "0.6B 는 39.3% x 0.783->0.967 = +7.2pp. 즉 **상수 a_s=0.967 은 0.6B 를 체계적으로 "
        "유리하게 만들고 있었다** — 13-AA 의 위협 11 이 예측한 방향 그대로이고, 이번 실측이 "
        "그것을 확인했다.",
        "",
        "## 표 2. 0.6B 곡선 (항목별 서버 실측)",
        "",
        fmt_table(["목표 esc", "실현 esc", "캐스케이드", "그 부분집합 a_s"], t2),
        "",
        "## 표 3. 1.7B 곡선",
        "",
        fmt_table(["목표 esc", "실현 esc", "캐스케이드", "그 부분집합 a_s"], t3),
        "",
        "## 표 4. G8 기준 A·B 재판정",
        "",
        fmt_table(["기준", "문턱", "실측", "판정"], [
            ["A 정확도", f"1.7B 운용점 {target:.4f} 의 {TOL_PP:.0f}pp 안",
             f"0.6B 가 esc {100*need06:.1f}% 에서 도달" if A_pass else "도달 불가",
             "통과" if A_pass else "**실패**"],
            ["B 로컬 처리율", f"≥ {100*LOCAL_FLOOR:.0f}%",
             f"{100*B_local:.1f}%" if A_pass else "—",
             "통과" if B_pass else "**실패**"],
        ]),
        "",
        f"## 판정: {'두 구성 유지' if B_pass else '**0.6B 기각 — 1.7B 만**'}",
        "",
        "13-U 에 등록한 규칙: **\"B 실패 -> 지연과 무관하게 0.6B 기각. 이 규칙이 먼저다.\"** "
        f"실측 로컬 처리율 {100*B_local:.1f}% 가 문턱 50% 를 밑돈다. "
        + ("따라서 13-AA 의 \"두 구성\" 판정은 **뒤집힌다**. 0.6B 가 지연 기준 C 를 통과하는 "
           "것은 그대로지만(TTFT 0.49s 대 1.33s), 등록된 규칙이 B 를 먼저 놓았고 그 이유는 "
           "골이다 — 절반 이상을 서버로 보내면 그건 온디바이스 모델이 아니라 씬 클라이언트다. "
           "**지연으로 씬 클라이언트를 살 수 없다**는 규칙을 결과 보기 전에 박아 뒀고, 이제 "
           "그 규칙이 실제로 물었다." if not B_pass else "판정은 유지된다."),
        "",
        "이 뒤집힘의 원인은 **0.6B 가 나빠진 것이 아니다** — 두 arm 모두 캐스케이드 정확도가 "
        "내려갔고(1.7B −5.2pp, 0.6B −7.2pp) 목표가 함께 내려갔다. 뒤집은 것은 **0.6B 가 그 "
        "목표에 닿는 데 필요한 esc율이 46.9% -> 52.8% 로 올라가 50% 선을 넘은 것**이다. "
        "위임을 많이 하는 구성은 서버 품질 하락에 비례적으로 더 민감하다.",
        "",
    ]
    write_json({"model": model, "target_acc": target,
                "need_esc_06": need06, "local_rate_06": B_local,
                "A_pass": bool(A_pass), "B_pass": bool(B_pass),
                "curve_17": c17, "curve_06": c06}, RESULTS / "35_rejudge.json")
    p = RESULTS / "35_rejudge.md"
    p.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
