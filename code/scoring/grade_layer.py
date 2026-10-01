"""새 코퍼스 층을 채점한다 (SPEC 13-AK).

`grade.py` 도 동결 파일이므로 고치지 않고 **채점 함수를 import 해서 그대로 호출**한다.
바뀌는 것은 어느 키 파일·어느 생성 파일을 읽고 어디에 쓰는가뿐이다.

judge/clarify 는 이 경로에서 지원하지 않는다 — 새 층은 전부 exact/contains 로 설계했다.

    .venv/bin/python scripts/grade_layer.py --keys data/02_answer_keys.triv.parquet \\
        --prompts data/01_prompts.triv.parquet --tag triv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import read_rows, require, write_parquet  # noqa: E402
import grade as G  # noqa: E402   동결 모듈 — 읽기만 한다
import run_student as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True, help="생성 파일과 출력의 접미사")
    a = ap.parse_args()

    keys = {r["id"]: r for r in read_rows(Path(a.keys))}
    prompts = {r["id"]: r["prompt"] for r in read_rows(Path(a.prompts))}
    gen_path = R.paths(a.tag)["gen"]
    require(gen_path.exists(), f"{gen_path} 가 없다 — gen_student_layer.py 를 먼저 돌려라")
    gens = read_rows(gen_path)

    graded, per_item, broken_ct = [], defaultdict(list), defaultdict(int)
    for g in gens:
        rid = g["id"]
        k = keys.get(rid)
        require(k is not None, f"{rid} 의 정답 키가 없다")
        rule = k["grading"]
        require(rule in ("exact", "contains"),
                f"{rid}: 이 경로는 exact/contains 만 지원한다 (받은 값 {rule})")
        if g["broken"]:
            ok, how = False, "broken"
            broken_ct[rid] += 1
        elif rule == "exact":
            ok, how = G.grade_exact(g["response"], k, prompts[rid])
        else:
            ok, how = G.grade_contains(g["response"], k)
        graded.append({"id": rid, "model": g["model"], "seed": int(g["seed"]),
                       "grading": rule, "correct": bool(ok),
                       "broken": bool(g["broken"]), "rule_used": how})
        if not g["broken"]:
            per_item[rid].append(bool(ok))

    acc = []
    for rid, oks in sorted(per_item.items()):
        k = keys[rid]
        acc.append({"id": rid, "category": k["category"], "lang": k["lang"],
                    "difficulty": k["difficulty"], "grading": k["grading"],
                    "n_graded": int(len(oks)), "n_broken": int(broken_ct.get(rid, 0)),
                    "accuracy": float(sum(oks) / len(oks))})

    gp = G.paths(a.tag)
    write_parquet(graded, gp["graded"], G.GRADED_SCHEMA)
    write_parquet(acc, gp["acc"], G.ACC_SCHEMA)

    import statistics as st
    m = st.mean(x["accuracy"] for x in acc)
    perfect = sum(1 for x in acc if x["accuracy"] >= 1.0)
    zero = sum(1 for x in acc if x["accuracy"] <= 0.0)
    print(f"채점 {len(graded)}건 / 항목 {len(acc)}개")
    print(f"  평균 정확도 {m:.4f} / 만점 {perfect}건 ({100*perfect/len(acc):.1f}%) / "
          f"0점 {zero}건 ({100*zero/len(acc):.1f}%)")
    for lang in sorted({x["lang"] for x in acc}):
        s = [x["accuracy"] for x in acc if x["lang"] == lang]
        print(f"  {lang}: 평균 {st.mean(s):.4f} (n={len(s)})")
    print(f"  -> {gp['graded']}")
    print(f"  -> {gp['acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
