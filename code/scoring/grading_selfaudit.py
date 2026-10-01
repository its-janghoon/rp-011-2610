"""채점 규칙 자기 감사: **정답 키 자신을 응답으로 넣으면 통과하는가.**

통과하지 못하는 항목은 어떤 모델도 맞힐 수 없으므로 student/server 오답률을 구조적으로
부풀린다. 2026-09-16 에 server 응답을 직접 서비스하다가 E-en-0162 에서 "Answer: Flood" 가
오답 처리되는 것을 발견해 추가했다.

각 exact/contains 항목에 대해 최소 응답을 만들어 채점한다:
  exact    -> "정답: <answer>"  (마커 + 정답)
  contains -> must_include 를 모두 담은 문장
통과하지 못하면 그 항목은 결함이다.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, fmt_table, read_rows, write_parquet  # noqa: E402


def main() -> int:
    import pyarrow as pa

    from grade import grade_contains, grade_exact, label_set

    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}

    bad, rows = [], []
    by_rule = Counter()
    by_rule_bad = Counter()
    for i, k in keys.items():
        if k["grading"] not in ("exact", "contains"):
            continue
        pr = prompts[i]["prompt"]
        if k["grading"] == "exact":
            # 정답이 문장부호 하나뿐인 항목(백틱, 마침표)은 최소 응답이 퇴화한다 —
            # 실제 모델은 "백틱입니다" 처럼 답하고 변형으로 통과한다. 감사가 거짓 경보를
            # 내지 않도록 그런 경우엔 첫 변형을 쓴다.
            ans = str(k["answer"])
            if not re.search(r"[0-9A-Za-z\uac00-\ud7a3]", ans) and list(k["variants"]):
                ans = str(k["variants"][0])      # 문장부호만인 정답(백틱, 마침표)일 때만
            resp = f"정답: {ans}"
            ok, rule = grade_exact(resp, k, pr)
        else:
            resp = "정답: " + " ".join(str(x) for x in k["must_include"])
            ok, rule = grade_contains(resp, k)
        by_rule[rule] += 1
        if not ok:
            by_rule_bad[rule] += 1
            labs = label_set(pr) or []
            bad.append((i, k["grading"], rule, str(k["answer"])[:40],
                        len(labs), str(labs)[:70]))
        rows.append({"id": i, "grading": k["grading"], "rule": rule, "self_pass": bool(ok)})

    print(f"감사 대상 {len(rows)}건 / 자기 통과 실패 **{len(bad)}건** "
          f"({100*len(bad)/len(rows):.2f}%)")
    print()
    print(fmt_table(["규칙", "항목", "실패", "실패율"],
                    [[r, by_rule[r], by_rule_bad[r], f"{100*by_rule_bad[r]/by_rule[r]:.1f}%"]
                     for r in sorted(by_rule)]))
    if bad:
        print("\n=== 실패 항목 (최대 15건) ===")
        print(fmt_table(["id", "grading", "rule", "answer", "라벨수", "추출된 라벨"],
                        [list(b) for b in bad[:15]]))
    out = RESULTS / "16_grading_selfaudit.parquet"
    write_parquet(rows, out, pa.schema([("id", pa.string()), ("grading", pa.string()),
                                       ("rule", pa.string()), ("self_pass", pa.bool_())]))
    print(f"\n{len(rows)}행 -> {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
