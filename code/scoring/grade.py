#!/usr/bin/env python3
"""6단계: 채점 — exact / contains 경로 (SPEC §5, §8).

judge / clarify 는 teacher(에이전트) 호출이 필요해 이 스크립트가 다루지 않는다.
여기서 채점되는 것은 프로그램적으로 판정 가능한 exact + contains 뿐이고,
그 부분집합만으로 G4(accuracy 분포)를 판정할 수 있다 — 에이전트 비용 0.

    python scripts/grade.py --run [--tag T]
    python scripts/grade.py --report [--tag T]

## 채점 규칙

정규화: NFKC -> 소문자 -> 양끝 공백·따옴표·마침표 제거 -> 연속 공백 1개 -> 수치의 천단위 구분자 제거.
한국어 조사는 제거하지 않는다(정답 키의 variants 가 조사 변형을 담당한다).

exact:
  - 답이 **수치**면 응답의 마지막 수치와 비교한다. 결론은 뒤에 오고, 스쳐 지나간 숫자를
    정답으로 인정하지 않기 위해서다. 최종답 표지(`정답:` `답:` `answer:` 마지막 볼드)가 있으면
    그 구간의 수치를 우선한다.
  - 답이 **JSON 형태**면 공백을 전부 제거하고 포함 여부를 본다(키 순서·들여쓰기 차이 흡수).
  - 그 밖에는 정규화된 응답에서 answer/variants 를 **단어 경계 일치**로 찾는다.
    절단된 응답도 답이 나왔으면 정답이다(SPEC §5).
  - 과대 인정 방지: 프롬프트에서 라벨 집합을 뽑을 수 있으면(`A/B/C 중`, `one of A, B, or C`)
    그 집합 중 **정확히 하나만** 등장해야 정답으로 본다. 여러 라벨을 나열한 응답은 오답.

contains: must_include 전부가 정규화된 응답에 있어야 정답.

accuracy: 생성별 0/1 을 프롬프트별로 **비-broken 생성에 대해 평균**한다. 이진화하지 않는다.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, fmt_table, load_scale, read_rows, require, write_parquet  # noqa: E402

import pyarrow as pa  # noqa: E402

GRADED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("model", pa.string()),
        ("seed", pa.int32()),
        ("grading", pa.string()),
        ("correct", pa.bool_()),
        ("broken", pa.bool_()),
        ("rule_used", pa.string()),
    ]
)

ACC_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("category", pa.string()),
        ("lang", pa.string()),
        ("difficulty", pa.string()),
        ("grading", pa.string()),
        ("n_graded", pa.int32()),
        ("n_broken", pa.int32()),
        ("accuracy", pa.float32()),
    ]
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSAND = re.compile(r"(?<=\d),(?=\d{3}\b)")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_MARKER = re.compile(
    r"(?:정답|답|최종\s*답|answer|result|total)\s*(?:은|는|이|가)?\s*[:=]\s*(.+)",
    re.IGNORECASE,
)
_LABELS_KO = re.compile(r"([^\s,()]+(?:\s*/\s*[^\s,()]+){1,5})\s*(?:중|가운데)")
_LABELS_EN = re.compile(
    r"(?:one of|classify (?:this|it) as|sort (?:this|it) into(?: one of)?)\s*[:\s]\s*(.+?)(?:[.?\n]|$)",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _THOUSAND.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip("\"'`.。 ").strip()


def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def is_numeric(s: str) -> bool:
    t = normalize(s)
    return bool(t) and bool(re.fullmatch(r"-?\d+(?:\.\d+)?", t))


def looks_json(s: str) -> bool:
    t = (s or "").strip()
    return t.startswith(("{", "[")) and t.endswith(("}", "]"))


def numbers(s: str) -> list[str]:
    return _NUM.findall(_THOUSAND.sub("", s or ""))


def marker_span(resp: str) -> str | None:
    """최종답 표지 뒤 구간, 없으면 마지막 볼드."""
    hits = list(_MARKER.finditer(resp or ""))
    if hits:
        return hits[-1].group(1)
    bolds = _BOLD.findall(resp or "")
    if bolds:
        return bolds[-1]
    return None


def wb_contains(hay: str, needle: str) -> bool:
    """단어 경계 일치. 한글은 \\b 가 동작하지 않아 경계 문자를 직접 본다."""
    n = normalize(needle)
    if not n:
        return False
    for m in re.finditer(re.escape(n), hay):
        before = hay[m.start() - 1] if m.start() > 0 else " "
        after = hay[m.end()] if m.end() < len(hay) else " "
        ok_b = not (before.isalnum() and n[0].isalnum() and before.isascii() and n[0].isascii())
        ok_a = not (after.isalnum() and n[-1].isalnum() and after.isascii() and n[-1].isascii())
        if ok_b and ok_a:
            return True
    return False


def label_set(prompt: str) -> list[str] | None:
    m = _LABELS_KO.search(prompt or "")
    if m:
        labs = [x.strip() for x in m.group(1).split("/") if x.strip()]
        if len(labs) >= 2:
            return labs
    m = _LABELS_EN.search(prompt or "")
    if m:
        # 2026-09-16 수정: 라벨 목록이 콜론/인용부호에서 끊기지 않아 **마지막 라벨이 분류 대상
        # 인용문에 붙어버렸다**. 그러면 그 라벨이 독립적으로 존재하지 않아 정답을 써도
        # label_unique 가 실패한다(자기 감사에서 label_unique 193건 중 32건이 이 결함).
        # 예: "one of Heat, Cold, Storm, or Flood: \"river levels rising fast, ...\""
        #     -> ['Heat','Cold','Storm','Flood: "river levels rising fast','low streets ...']
        raw = re.split(r'[:"\u201c\u201d]', m.group(1), maxsplit=1)[0]
        raw = re.sub(r"\bor\b", ",", raw, flags=re.IGNORECASE)
        labs = [x.strip(" \"'*.") for x in raw.split(",")]
        # 관사를 떼어낸다: "a Scale or a Chord" -> ['Scale','Chord'].
        # 떼지 않으면 정답 'Scale' 이 라벨 'a Scale' 과 일치하지 않아 label_unique 가 실패한다.
        labs = [re.sub(r"^(?:a|an|the)\s+", "", x, flags=re.IGNORECASE).strip() for x in labs]
        labs = [x for x in labs if x and len(x) < 40]
        if len(labs) >= 2:
            return labs
    return None


def grade_exact(resp: str, key: dict, prompt: str) -> tuple[bool, str]:
    answer = key["answer"]
    cands = [answer] + list(key["variants"])
    nresp = normalize(resp)

    if looks_json(answer):
        a = strip_ws(normalize(answer))
        return (a in strip_ws(nresp), "json_ws_strip")

    if is_numeric(answer):
        want = normalize(answer)
        span = marker_span(resp)
        if span:
            sn = numbers(span)
            if sn:
                return (sn[-1] == want, "numeric_marker")
        rn = numbers(resp)
        return (bool(rn) and rn[-1] == want, "numeric_last")

    labs = label_set(prompt)
    if labs:
        present = [l for l in labs if wb_contains(nresp, l)]
        # 라벨이 서로 부분문자열이면(한국어 '정답' ⊂ '부분정답', 영어 'New' ⊂ 'Like new')
        # 짧은 쪽이 함께 탐지되어 개수가 2 가 된다. **최대 일치만 남긴다.**
        present = [l for l in present
                   if not any(l != o and normalize(l) in normalize(o) for o in present)]
        hit = any(wb_contains(nresp, c) for c in cands)
        # 라벨 집합 중 정확히 하나만 등장해야 인정한다
        return (hit and len(present) == 1, "label_unique")

    return (any(wb_contains(nresp, c) for c in cands), "word_boundary")


def grade_contains(resp: str, key: dict) -> tuple[bool, str]:
    nresp = normalize(resp)
    need = list(key["must_include"])
    require(need, "contains 인데 must_include 가 비었다")
    ok = all(normalize(x) in nresp for x in need)
    return (ok, "contains_all")


def paths(tag: str) -> dict[str, Path]:
    sfx = f".{tag}" if tag else ""
    return {
        "gen": DATA / f"04_student_gen{sfx}.parquet",
        "graded": DATA / f"06_graded_student{sfx}.parquet",
        "acc": DATA / f"06_accuracy{sfx}.parquet",
        "null": DATA / f"06_null_bucket{sfx}.parquet",
        "jreq": DATA / f"06_judge_sample_requests{sfx}.jsonl",
        "jres": DATA / f"06_judge_sample_responses{sfx}.jsonl",
        "jacc": DATA / f"06_judge_sample_accuracy{sfx}.parquet",
    }


# --------------------------------------------------------------------------- #
# judge 표본 브리지 — G4 를 완결하기 위한 최소 측정
#
# exact/contains 는 생성당 0/1 만 낼 수 있어 항목 accuracy 가 0 또는 1 로 몰린다.
# judge 는 rubric 통과 비율이라 생성당 {0, 1/3, 1/2, 2/3, 1} 을 내므로 분포가 다르다.
# 코퍼스의 38.8% 를 빼고 G4 를 판정할 수 없으니, 층화 표본만 채점해 mid-band 비중을 추정한다.
# --------------------------------------------------------------------------- #
def cmd_emit_judge(tag: str, n_items: int, n_seeds: int) -> int:
    import numpy as np

    from _common import derive_seed as dseed
    from _common import write_jsonl

    p = paths(tag)
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}
    gens = read_rows(p["gen"])

    jids = sorted(rid for rid, k in keys.items() if k["grading"] == "judge")
    cells: dict[tuple, list[str]] = {}
    for rid in jids:
        pr = prompts[rid]
        cells.setdefault((pr["category"], pr["lang"], pr["difficulty"]), []).append(rid)
    ckeys = sorted(cells)
    from _common import allocate as alloc
    quota = alloc(n_items, {str(k): float(len(cells[k])) for k in ckeys},
                  [str(k) for k in ckeys])
    picked: list[str] = []
    for k in ckeys:
        items = sorted(cells[k], key=lambda i: (dseed("judge_sample", i), i))
        picked.extend(items[: quota[str(k)]])

    seeds = sorted({g["seed"] for g in gens})[:n_seeds]
    by_key = {(g["id"], g["seed"]): g for g in gens}
    reqs = []
    for rid in sorted(picked):
        for sd in seeds:
            g = by_key[(rid, sd)]
            reqs.append({
                "id": rid, "seed": int(sd),
                "category": prompts[rid]["category"], "lang": prompts[rid]["lang"],
                "difficulty": prompts[rid]["difficulty"],
                "prompt": prompts[rid]["prompt"],
                "rubric": list(keys[rid]["rubric"]),
                "response": g["response"],
                "truncated": g["finish_reason"] == "length",
            })
    write_jsonl(reqs, p["jreq"])
    print(f"judge 표본 요청 {len(reqs)}건 (항목 {len(picked)} x 시드 {len(seeds)}) -> {p['jreq']}")
    print(f"rubric 기준 수 분포: ", end="")
    rc: dict[int, int] = {}
    for rid in picked:
        n = len(keys[rid]["rubric"])
        rc[n] = rc.get(n, 0) + 1
    print("  ".join(f"{k}개: {v}항목" for k, v in sorted(rc.items())))
    return EXIT_NEED_AGENT_LOCAL


EXIT_NEED_AGENT_LOCAL = 2


def cmd_ingest_judge(tag: str) -> int:
    import numpy as np

    from _common import read_jsonl

    p = paths(tag)
    reqs = {(r["id"], r["seed"]): r for r in read_jsonl(p["jreq"])}
    res = read_jsonl(p["jres"])
    require(res, "judge 응답이 비었다")

    seen = set()
    per_gen: dict[str, list[float]] = {}
    for i, r in enumerate(res, 1):
        where = f"{p['jres']}:{i}"
        for f in ("id", "seed", "rubric_pass"):
            require(f in r, f"{where} '{f}' 누락")
        k = (r["id"], int(r["seed"]))
        require(k in reqs, f"{where} 요청에 없는 (id, seed): {k}")
        require(k not in seen, f"{where} 중복: {k}")
        seen.add(k)
        want = len(reqs[k]["rubric"])
        pv = r["rubric_pass"]
        require(isinstance(pv, list) and len(pv) == want,
                f"{where} rubric_pass 길이 {len(pv) if isinstance(pv,list) else '?'} != {want}")
        require(all(isinstance(x, bool) for x in pv), f"{where} rubric_pass 는 bool 배열")
        per_gen.setdefault(r["id"], []).append(sum(pv) / want)

    missing = set(reqs) - seen
    require(not missing, f"judge 판정 누락 {len(missing)}건 예: {sorted(missing)[:3]}")

    prompts = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}
    rows = []
    for rid, vals in per_gen.items():
        pr = prompts[rid]
        rows.append({"id": rid, "category": pr["category"], "lang": pr["lang"],
                     "difficulty": pr["difficulty"], "grading": "judge",
                     "n_graded": len(vals), "n_broken": 0,
                     "accuracy": float(np.mean(vals))})
    write_parquet(sorted(rows, key=lambda r: r["id"]), p["jacc"], ACC_SCHEMA)
    print(f"judge 표본 accuracy {len(rows)}항목 -> {p['jacc']}")
    return cmd_g4(tag)


def cmd_g4(tag: str) -> int:
    """측정된 exact/contains + judge 표본으로 코퍼스 전체 G4 를 추정한다."""
    import numpy as np

    p = paths(tag)
    keys = read_rows(DATA / "02_answer_keys.parquet")
    n_by_grading: dict[str, int] = {}
    for k in keys:
        n_by_grading[k["grading"]] = n_by_grading.get(k["grading"], 0) + 1

    meas = np.array([r["accuracy"] for r in read_rows(p["acc"])], dtype="float64")
    jud = np.array([r["accuracy"] for r in read_rows(p["jacc"])], dtype="float64")

    def band(a):
        return float(((a >= 0.3) & (a <= 0.7)).mean()) if len(a) else float("nan")

    def ext(a):
        return float((((a <= 1e-9) | (a >= 1 - 1e-9))).mean()) if len(a) else float("nan")

    n_ec = n_by_grading.get("exact", 0) + n_by_grading.get("contains", 0)
    n_j = n_by_grading.get("judge", 0)
    n_cl = n_by_grading.get("clarify", 0)
    n_all = n_ec + n_j + n_cl

    b_ec, b_j = band(meas), band(jud)
    # clarify 는 생성당 0/1 이라 exact/contains 와 같은 분포로 **보수적으로** 가정한다.
    b_cl = b_ec
    est = (b_ec * n_ec + b_j * n_j + b_cl * n_cl) / n_all
    se_j = float(np.sqrt(b_j * (1 - b_j) / len(jud))) if len(jud) else float("nan")

    print("\n=== G4 코퍼스 추정 (측정 + judge 표본) ===")
    print(fmt_table(
        ["부분", "항목수", "가중", "0.3~0.7", "0 또는 1", "출처"],
        [["exact+contains", n_ec, f"{100.0*n_ec/n_all:.1f}%", f"{100*b_ec:.1f}%",
          f"{100*ext(meas):.1f}%", "전량 측정"],
         ["judge", n_j, f"{100.0*n_j/n_all:.1f}%", f"{100*b_j:.1f}%",
          f"{100*ext(jud):.1f}%", f"표본 {len(jud)}항목"],
         ["clarify", n_cl, f"{100.0*n_cl/n_all:.1f}%", f"{100*b_cl:.1f}%", "-",
          "보수적 가정(=이진)"]]))
    lo = est - 1.96 * se_j * n_j / n_all
    hi = est + 1.96 * se_j * n_j / n_all
    print(f"\n코퍼스 0.3~0.7 추정: {100*est:.1f}%  (95% CI {100*lo:.1f}~{100*hi:.1f}%)")
    verdict = "통과" if lo >= 0.25 else ("통과(점추정) — CI 하한 미달" if est >= 0.25 else "미달")
    print(f">>> G4: {verdict}   기준 25%")
    return 0


def cmd_run(tag: str) -> int:
    from tqdm import tqdm

    scale = load_scale()
    max_broken = int(scale["broken"]["max_broken_per_prompt"])
    p = paths(tag)
    gens = read_rows(p["gen"])
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}

    target = {"exact", "contains"}
    graded: list[dict] = []
    for g in tqdm(gens, desc="채점", unit="건"):
        k = keys[g["id"]]
        if k["grading"] not in target:
            continue
        if g["broken"]:
            graded.append({"id": g["id"], "model": g["model"], "seed": g["seed"],
                           "grading": k["grading"], "correct": False,
                           "broken": True, "rule_used": "broken_excluded"})
            continue
        if k["grading"] == "exact":
            ok, rule = grade_exact(g["response"], k, prompts[g["id"]]["prompt"])
        else:
            ok, rule = grade_contains(g["response"], k)
        graded.append({"id": g["id"], "model": g["model"], "seed": g["seed"],
                       "grading": k["grading"], "correct": bool(ok),
                       "broken": False, "rule_used": rule})

    require(graded, "채점 대상이 없다")
    write_parquet(graded, p["graded"], GRADED_SCHEMA)

    # 프롬프트별 accuracy: 비-broken 생성에 대한 평균. 이진화하지 않는다.
    by_id: dict[str, list[dict]] = {}
    for r in graded:
        by_id.setdefault(r["id"], []).append(r)

    acc_rows: list[dict] = []
    null_rows: list[dict] = []
    for rid, rs in by_id.items():
        nb = sum(1 for r in rs if r["broken"])
        good = [r for r in rs if not r["broken"]]
        if nb > max_broken or not good:
            null_rows.append({"id": rid, "n_broken": nb, "n_total": len(rs)})
            continue
        pr = prompts[rid]
        acc_rows.append(
            {
                "id": rid, "category": pr["category"], "lang": pr["lang"],
                "difficulty": pr["difficulty"], "grading": rs[0]["grading"],
                "n_graded": len(good), "n_broken": nb,
                "accuracy": float(sum(r["correct"] for r in good) / len(good)),
            }
        )
    write_parquet(acc_rows, p["acc"], ACC_SCHEMA)
    if null_rows:
        write_parquet(null_rows, p["null"],
                      pa.schema([("id", pa.string()), ("n_broken", pa.int32()),
                                 ("n_total", pa.int32())]))

    print(f"\n채점 {len(graded)}건 -> {p['graded']}")
    print(f"프롬프트 accuracy {len(acc_rows)}건 -> {p['acc']}")
    print(f"null bucket {len(null_rows)}건")
    return cmd_report(tag)


def cmd_server() -> int:
    """server 응답을 **student 와 완전히 같은 규칙**으로 채점한다.

    별도 구현을 두면 student/server 정확도가 비교 불가가 되므로 grade_exact / grade_contains
    를 그대로 재사용한다. 대상은 data/05_server_gen.parquet (run_server.py --ingest 산출물).
    """
    from tqdm import tqdm

    src = DATA / "05_server_gen.parquet"
    require(src.exists(), f"{src} 없음 — run_server.py --ingest 먼저")
    gens = read_rows(src)
    keys = {r["id"]: r for r in read_rows(DATA / "02_answer_keys.parquet")}
    prompts = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}

    graded: list[dict] = []
    skipped = 0
    for g in tqdm(gens, desc="server 채점", unit="건"):
        k = keys[g["id"]]
        if k["grading"] not in ("exact", "contains"):
            skipped += 1
            continue
        if k["grading"] == "exact":
            ok, rule = grade_exact(g["response"], k, prompts[g["id"]]["prompt"])
        else:
            ok, rule = grade_contains(g["response"], k)
        graded.append({"id": g["id"], "model": "server", "seed": int(g["gen_index"]),
                       "grading": k["grading"], "correct": bool(ok),
                       "broken": False, "rule_used": rule})
    require(graded, "채점 대상이 없다 (exact/contains 항목이 없음)")
    out = DATA / "05_graded_server.parquet"
    write_parquet(graded, out, GRADED_SCHEMA)

    by_id: dict[str, list[dict]] = {}
    for r in graded:
        by_id.setdefault(r["id"], []).append(r)
    acc_rows = []
    for rid, rs in by_id.items():
        pr = prompts[rid]
        acc_rows.append({"id": rid, "category": pr["category"], "lang": pr["lang"],
                         "difficulty": pr["difficulty"], "grading": rs[0]["grading"],
                         "n_graded": len(rs), "n_broken": 0,
                         "accuracy": float(sum(r["correct"] for r in rs) / len(rs))})
    acc_out = DATA / "05_server_accuracy.parquet"
    write_parquet(sorted(acc_rows, key=lambda r: r["id"]), acc_out, ACC_SCHEMA)
    n_ok = sum(1 for r in graded if r["correct"])
    print(f"\nserver 채점 {len(graded)}건 (정답 {n_ok}, 정확도 {100*n_ok/len(graded):.1f}%) -> {out}")
    print(f"항목 accuracy {len(acc_rows)}건 -> {acc_out}")
    if skipped:
        print(f"judge/clarify 항목 {skipped}건은 프로그램 채점 대상이 아니라 제외")
    return 0


def cmd_report(tag: str) -> int:
    import numpy as np

    p = paths(tag)
    acc = read_rows(p["acc"])
    graded = read_rows(p["graded"])
    a = np.array([r["accuracy"] for r in acc], dtype="float64")
    n = len(a)

    print(f"\n=== accuracy 분포 (프롬프트 {n}건, k=8 평균) ===")
    bins = [(0.0, 0.0), (0.001, 0.124), (0.125, 0.249), (0.25, 0.374), (0.375, 0.499),
            (0.5, 0.624), (0.625, 0.749), (0.75, 0.874), (0.875, 0.999), (1.0, 1.0)]
    rows = []
    for lo, hi in bins:
        c = int(((a >= lo - 1e-9) & (a <= hi + 1e-9)).sum())
        lab = f"{lo:.3f}" if lo == hi else f"{lo:.3f}~{hi:.3f}"
        rows.append([lab, c, f"{100.0*c/n:.1f}%", "#" * int(round(60.0 * c / n))])
    print(fmt_table(["구간", "건수", "비율", ""], rows))

    band = int(((a >= 0.3) & (a <= 0.7)).sum())
    at0 = int((a <= 1e-9).sum())
    at1 = int((a >= 1 - 1e-9).sum())
    print(fmt_table(
        ["G4 판정 항목", "값", "기준"],
        [["0.3~0.7 구간", f"{band}건 ({100.0*band/n:.1f}%)", "25% 이상 필요"],
         ["accuracy = 0", f"{at0}건 ({100.0*at0/n:.1f}%)", "-"],
         ["accuracy = 1", f"{at1}건 ({100.0*at1/n:.1f}%)", "-"],
         ["0 또는 1 합", f"{at0+at1}건 ({100.0*(at0+at1)/n:.1f}%)", "여기에만 몰리면 실험 불가"],
         ["평균 / 중앙", f"{a.mean():.3f} / {np.median(a):.3f}", "-"]]))
    verdict = "통과" if band >= 0.25 * n else "미달 — 실험 불가"
    print(f"\n>>> G4: {verdict} (0.3~0.7 = {100.0*band/n:.1f}%)")

    print("\n=== 카테고리 × 난이도 평균 accuracy ===")
    cats = sorted({r["category"] for r in acc})
    diffs = ["easy", "medium", "hard"]
    crows = []
    for c in cats:
        sub = [r for r in acc if r["category"] == c]
        cells = []
        for d in diffs:
            v = [r["accuracy"] for r in sub if r["difficulty"] == d]
            cells.append(f"{np.mean(v):.2f}" if v else "-")
        bandc = len([r for r in sub if 0.3 <= r["accuracy"] <= 0.7])
        crows.append([c, len(sub), f"{np.mean([r['accuracy'] for r in sub]):.2f}",
                      *cells, f"{100.0*bandc/len(sub):.0f}%"])
    print(fmt_table(["코드", "건수", "전체", *diffs, "0.3~0.7"], crows))

    print("\n=== 언어별 ===")
    lrows = []
    for lg in ("ko", "en"):
        v = [r["accuracy"] for r in acc if r["lang"] == lg]
        b = len([x for x in v if 0.3 <= x <= 0.7])
        lrows.append([lg, len(v), f"{np.mean(v):.3f}", f"{100.0*b/len(v):.1f}%"])
    print(fmt_table(["lang", "건수", "평균", "0.3~0.7"], lrows))

    print("\n=== 적용된 채점 규칙 ===")
    ru: dict[str, int] = {}
    for r in graded:
        ru[r["rule_used"]] = ru.get(r["rule_used"], 0) + 1
    print(fmt_table(["규칙", "생성수", "비율"],
                    [[k, v, f"{100.0*v/len(graded):.1f}%"]
                     for k, v in sorted(ru.items(), key=lambda kv: -kv[1])]))

    print("\n=== grading 별 ===")
    grows = []
    for g in ("exact", "contains"):
        v = [r["accuracy"] for r in acc if r["grading"] == g]
        if not v:
            continue
        b = len([x for x in v if 0.3 <= x <= 0.7])
        grows.append([g, len(v), f"{np.mean(v):.3f}", f"{100.0*b/len(v):.1f}%"])
    print(fmt_table(["grading", "건수", "평균", "0.3~0.7"], grows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="6단계 채점 (exact/contains + judge 표본)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true", help="exact/contains 전량 채점")
    g.add_argument("--report", action="store_true")
    g.add_argument("--emit-judge", action="store_true", help="judge 표본 요청 작성 (exit 2)")
    g.add_argument("--ingest-judge", action="store_true", help="judge 판정 검증 후 G4 추정")
    g.add_argument("--g4", action="store_true", help="G4 추정만 다시 계산")
    g.add_argument("--server", action="store_true",
                   help="server 응답(05_server_gen.parquet)을 같은 규칙으로 채점")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--sample", type=int, default=60, help="judge 표본 항목 수")
    ap.add_argument("--seeds", type=int, default=3, help="judge 표본 시드 수")
    a = ap.parse_args()

    if a.run:
        return cmd_run(a.tag)
    if a.report:
        return cmd_report(a.tag)
    if a.emit_judge:
        return cmd_emit_judge(a.tag, a.sample, a.seeds)
    if a.ingest_judge:
        return cmd_ingest_judge(a.tag)
    if a.server:
        return cmd_server()
    return cmd_g4(a.tag)


if __name__ == "__main__":
    sys.exit(main())
