"""실제 서버 모델의 a_s 를 채점해 산출한다 (SPEC 13-AD / Stage 3).

student 와 **같은 채점 규칙**을 쓴다 — `grade.py` 의 `grade_exact` / `grade_contains` 를
그대로 import 한다. 채점기가 다르면 a_s 가 아니라 채점기를 재게 된다.

산출:
  - 전체 a_s (656 항목)
  - **각 arm 의 escalate 부분집합별 a_s** — 이게 캐스케이드 계산에 실제로 들어가는 값이다.
    escalate 된 항목은 어려운 항목 쪽으로 치우쳐 있으므로 전체 a_s 보다 낮게 나온다.
  - 채점 규칙별 분해 — 실제 모델은 답을 길게 쓴다. `contains` 계열엔 유리하고
    `label_unique` 처럼 라벨이 하나만 나와야 하는 규칙엔 불리할 수 있다. 그 차이가 a_s 에
    어떻게 들어오는지 보이게 쪼갠다.
  - 손익분기 대조: 같은 부분집합에서 student 정확도와 비교한다.

  .venv/bin/python scripts/grade_server_as.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, fmt_table, read_rows, require, write_json  # noqa: E402
from grade import grade_contains, grade_exact  # noqa: E402

RES = DATA / "34_server_responses.jsonl"
OUT_J = RESULTS / "34_server_as.json"
OUT_MD = RESULTS / "34_server_as.md"

# 각 arm 의 운용점: (이름, escalation map 파일, condition, target_esc, 설명)
ARMS = [
    ("1.7B @esc 20%", "09_escalation_map.parquet", "G-gguf", 0.20, "13-Q 채택 운용점"),
    ("0.6B @esc 40%", "09_escalation_map.sc06.parquet", "G-sc06", 0.40, "13-Z e2e 운용점"),
    ("0.6B @esc 50%", "09_escalation_map.sc06.parquet", "G-sc06", 0.50, "동률점 46.9% 위쪽"),
]


def main() -> int:
    require(RES.exists(), f"{RES} 가 없다 — askllm --batch 를 먼저")
    rows = [json.loads(x) for x in RES.read_text(encoding="utf-8").splitlines() if x.strip()]
    resp = {r["id"]: r["response"] for r in rows}
    models = sorted({r.get("model", "?") for r in rows})
    require(len(models) == 1, f"모델이 섞였다: {models}")

    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r["prompt"] for r in read_rows(DATA / "01_prompts.parquet")}
    a17 = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.parquet")}
    a06 = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.sc06.parquet")}
    meta = {r["id"]: r for r in read_rows(DATA / "06_accuracy.parquet")}

    def grade(i: str, text: str):
        k = keys[i]
        if k["grading"] == "exact":
            ok, rule = grade_exact(text, k, prompts[i])[:2]
        elif k["grading"] == "contains":
            ok, rule = grade_contains(text, k)[:2]
        else:
            return None, None
        return bool(ok), rule

    graded: dict[str, bool] = {}
    rule_of: dict[str, str] = {}
    for i, t in resp.items():
        ok, rule = grade(i, t)
        if ok is not None:
            graded[i] = ok
            rule_of[i] = str(rule)

    ids = sorted(graded)
    a_s_all = st.mean(graded[i] for i in ids)
    print(f"모델 {models[0]} / 채점 {len(ids)}건 / **전체 a_s {a_s_all:.4f}**")
    require(len(ids) == len(resp),
            f"채점 불가 항목이 응답에 섞였다: 응답 {len(resp)} / 채점 {len(ids)}")

    # ---- 채점 규칙별 ----
    by_rule: dict[str, list[str]] = {}
    for i in ids:
        by_rule.setdefault(rule_of[i], []).append(i)
    t_rule = [[r, len(v), f"{st.mean(graded[i] for i in v):.4f}"]
              for r, v in sorted(by_rule.items(), key=lambda kv: -len(kv[1]))]

    # ---- 언어·난이도별 ----
    def group(field):
        g: dict[str, list[str]] = {}
        for i in ids:
            g.setdefault(str(meta[i][field]), []).append(i)
        return [[k, len(v), f"{st.mean(graded[i] for i in v):.4f}"] for k, v in sorted(g.items())]

    # ---- arm 별 escalate 부분집합 ----
    t_arm, arm_json = [], {}
    for name, fn, cond, tgt, note in ARMS:
        p = RESULTS / fn
        if not p.exists():
            continue
        # target_esc 는 parquet 에 float32 로 저장돼 0.20 이 0.20000000298 이 된다.
        # 허용오차를 1e-9 로 두면 0.5 처럼 정확히 표현되는 값만 통과해 **조용히 행이 빠진다**
        # (실제로 처음 실행에서 1.7B @20% 와 0.6B @40% 가 사라졌다). 1e-6 을 쓴다.
        esc = [r["id"] for r in read_rows(p)
               if r["condition"] == cond and abs(float(r["target_esc"]) - tgt) < 1e-6
               and r["escalated"] and r["id"] in graded]
        if not esc:
            continue
        a_s = st.mean(graded[i] for i in esc)
        stu = a06 if "0.6B" in name else a17
        stu_acc = st.mean(stu[i] for i in esc if i in stu)
        t_arm.append([name, len(esc), f"{a_s:.4f}", f"{stu_acc:.4f}",
                      f"{a_s - stu_acc:+.4f}", "값 함" if a_s > stu_acc else "**손해**", note])
        arm_json[name] = {"n_escalated": len(esc), "a_s": a_s,
                          "student_acc_on_subset": stu_acc, "margin": a_s - stu_acc,
                          "target_esc": tgt, "condition": cond}
        print(f"  {name}: n={len(esc)} a_s={a_s:.4f} vs student {stu_acc:.4f} "
              f"({a_s-stu_acc:+.4f})")
    require(len(arm_json) == len([1 for _, fn, _, _, _ in ARMS if (RESULTS / fn).exists()]),
            f"arm 이 빠졌다 — 기대 {len(ARMS)} / 실제 {len(arm_json)}: {list(arm_json)}")

    body = [
        "# Stage 3 — 실제 서버 모델의 a_s 실측",
        "",
        f"모델 **{models[0]}** / temp 0 / max_tokens 512 / student 와 같은 프롬프트·같은 채점 규칙. "
        f"선택 근거와 판정 규칙은 SPEC 13-AD 에 **수치 보기 전에** 등록했다.",
        "",
        "## 표 1. a_s — 전체와 arm 별 escalate 부분집합",
        "",
        fmt_table(["대상", "n", "a_s (서버)", "student 정확도", "차이", "판정", "비고"],
                  [["전체 (라벨 있는 평가 항목)", len(ids), f"{a_s_all:.4f}", "—", "—", "—",
                    "모든 문턱의 상위 집합"]] + t_arm),
        "",
        "**escalate 부분집합의 a_s 가 전체보다 낮은 것이 정상이다** — 라우터가 어려운 항목을 "
        "골라 보내므로 그 부분집합은 전체보다 어렵다. 캐스케이드 계산에 들어가는 값은 "
        "**부분집합 a_s** 다.",
        "",
        "## 표 2. 채점 규칙별 — 실제 모델은 답을 길게 쓴다",
        "",
        fmt_table(["규칙", "n", "a_s"], t_rule),
        "",
        "에이전트가 서버 역할을 할 때는 라벨을 하나만 쓰고 숫자를 마지막에 두도록 **의도적으로 "
        "조심했다**(13-J·13-R). 실제 모델은 그러지 않으므로, 형식을 요구하는 규칙"
        "(`label_unique`, `numeric_last`)에서 손실이 생길 수 있다. 그 손실은 서버의 "
        "**지식 부족이 아니라 형식 불일치**이며, 실제 제품이라면 서버 프롬프트에 형식 지시를 "
        "넣어 회수할 수 있는 부분이다. 여기서는 student 와 같은 프롬프트를 쓴다는 원칙을 지켰다.",
        "",
        "## 표 3. 언어별 / 난이도별",
        "",
        fmt_table(["언어", "n", "a_s"], group("lang")),
        "",
        fmt_table(["난이도", "n", "a_s"], group("difficulty")),
        "",
    ]
    OUT_MD.write_text("\n".join(body), encoding="utf-8")
    write_json({"model": models[0], "n_graded": len(ids), "a_s_all": a_s_all,
                "arms": arm_json,
                "by_rule": {r: {"n": len(v), "a_s": st.mean(graded[i] for i in v)}
                            for r, v in by_rule.items()},
                "agent_as_previous": 0.967,
                "note": "student 와 같은 프롬프트·같은 채점 규칙. SPEC 13-AD"}, OUT_J)
    print("\n".join(body[-14:]))
    print(f"\n-> {OUT_MD}\n-> {OUT_J}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
