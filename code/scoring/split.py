#!/usr/bin/env python3
"""3단계: 층화 분할 (SPEC §6).

층 = 카테고리 × 언어 × 난이도 = 54 셀. train 2500 / cal 1250 / test 1250.
**무작위 분할 금지.** 두 단계 모두 결정적이다.

1) 셀별 배분: 50/25/25 최대잉여, 동률은 train > cal > test 순으로 깬다
2) 셀 안에서 어떤 항목이 어느 split 인지: `(domain, id)` 순으로 정렬한 뒤
   남은 결손 비율이 가장 큰 split 에 차례로 배정한다(체계적 추출).
   id 순으로 앞에서 자르면 도메인이 한 split 에 몰리는데, 이 방식은 도메인이
   세 split 에 번갈아 퍼진다.
3) 전역 합 보정: 셀별 반올림이 쌓여 합이 2500/1250/1250 에서 벗어나면,
   잉여 split 을 가진 큰 셀부터 한 건씩 결손 split 으로 옮긴다(결정적).

    python scripts/split.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA,
    allocate,
    derive_seed,
    fmt_table,
    load_categories,
    load_scale,
    read_rows,
    require,
    write_parquet,
)

import pyarrow as pa  # noqa: E402

TRAIN, CAL, TEST = "train", "cal", "test"
SPLITS = [TRAIN, CAL, TEST]
DIFFS = ["easy", "medium", "hard"]

SPLIT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("category", pa.string()),
        ("lang", pa.string()),
        ("difficulty", pa.string()),
        ("domain", pa.string()),
        ("split", pa.string()),
        ("stratum_id", pa.string()),
    ]
)
TEST_ID_SCHEMA = pa.schema([("id", pa.string())])


def assign_within_cell(cell_id: str, items: list[dict], targets: dict[str, int]) -> dict[str, str]:
    """셀 안에서 항목별 split 배정. 결정적이지만 도메인 편향이 없어야 한다.

    `(domain, id)` 정렬은 쓰지 않는다. 그러면 알파벳 앞쪽 도메인이 모든 셀에서 항상 먼저
    오고, 배정 초반 슬롯은 언제나 train 이라 같은 도메인이 매번 train 으로 몰린다.
    (실측: 그 방식에서 test 의 고유 도메인이 70개 중 54개로 줄었다.)
    대신 셀마다 도메인 순서를 MASTER_SEED 파생값으로 회전시킨다 — 재현 가능하면서
    특정 도메인이 항상 앞자리를 차지하지 않는다.
    """
    order = sorted(
        items,
        key=lambda r: (derive_seed("split_domain_order", f"{cell_id}|{r['domain']}"),
                       r["domain"], r["id"]),
    )
    need = dict(targets)
    out: dict[str, str] = {}
    for r in order:
        total_left = sum(need.values())
        require(total_left > 0, "배정할 자리가 없다")
        # 결손이 가장 큰 split. 동률은 SPLITS 순서(train>cal>test)로 깬다
        sp = max(SPLITS, key=lambda s: (need[s] / total_left, -SPLITS.index(s)))
        out[r["id"]] = sp
        need[sp] -= 1
    require(all(v == 0 for v in need.values()), f"셀 배정 잔여: {need}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="3단계 층화 분할")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    scale = load_scale()
    cat_order = [c["code"] for c in load_categories()["categories"]]
    rows = read_rows(DATA / "01_prompts.parquet")
    if a.limit > 0:
        rows = sorted(rows, key=lambda r: r["id"])[: a.limit]
    n = len(rows)

    want = {s: int(scale["splits"][s]) for s in SPLITS}
    if a.limit > 0:  # 소량 실행 시 비율만 유지
        want = allocate(n, {s: float(want[s]) for s in SPLITS}, SPLITS)
    require(sum(want.values()) == n, f"split 합 {sum(want.values())} != 항목 {n}")

    # ---- 1) 셀 구성과 셀별 배분 ------------------------------------------- #
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["category"], r["lang"], r["difficulty"]), []).append(r)
    require(len(cells) <= 54, f"셀 수 이상: {len(cells)}")

    cell_targets: dict[tuple, dict[str, int]] = {}
    for cell, items in cells.items():
        cell_targets[cell] = allocate(
            len(items), {TRAIN: 50.0, CAL: 25.0, TEST: 25.0}, SPLITS
        )

    got = {s: sum(t[s] for t in cell_targets.values()) for s in SPLITS}
    drift = {s: got[s] - want[s] for s in SPLITS}
    print("=== 셀별 배분 후 전역 합 ===")
    print(fmt_table(["split", "배분", "목표", "차이"],
                    [[s, got[s], want[s], f"{drift[s]:+d}"] for s in SPLITS]))

    # ---- 2) 전역 합 보정 --------------------------------------------------- #
    moves: list[list] = []
    guard = 0
    while any(d != 0 for d in drift.values()):
        guard += 1
        require(guard < 10000, "보정 루프가 끝나지 않는다")
        src = max(SPLITS, key=lambda s: (drift[s], -SPLITS.index(s)))
        dst = min(SPLITS, key=lambda s: (drift[s], SPLITS.index(s)))
        require(drift[src] > 0 and drift[dst] < 0, f"보정 불가 상태: {drift}")
        # 잉여 split 을 가진 셀 중에서 고른다. 큰 셀부터 고르면 보정이 A/C easy 셀에만
        # 몰려 난이도별 비율이 한쪽으로 틀어지므로, 이동 횟수에 따라 난이도를 순환시킨다.
        prefer = DIFFS[len(moves) % len(DIFFS)]
        cand = sorted(
            (c for c in cell_targets if cell_targets[c][src] > 0),
            key=lambda c: (0 if c[2] == prefer else 1,
                           -cell_targets[c][src] / len(cells[c]),
                           -len(cells[c]),
                           cat_order.index(c[0]), c[1], DIFFS.index(c[2])),
        )
        require(cand, f"{src} 를 줄일 셀이 없다")
        c = cand[0]
        cell_targets[c][src] -= 1
        cell_targets[c][dst] += 1
        drift[src] -= 1
        drift[dst] += 1
        moves.append(["-".join(c), f"{src}->{dst}"])

    if moves:
        print(f"\n=== 전역 합 보정 {len(moves)}건 ===")
        print(fmt_table(["셀", "이동"], moves))

    # ---- 3) 셀 안에서 항목별 배정 ------------------------------------------ #
    assign: dict[str, str] = {}
    for cell, items in cells.items():
        assign.update(assign_within_cell("-".join(cell), items, cell_targets[cell]))
    require(len(assign) == n, f"배정 {len(assign)} != 항목 {n}")

    out = []
    for r in sorted(rows, key=lambda r: r["id"]):
        out.append(
            {
                "id": r["id"],
                "category": r["category"],
                "lang": r["lang"],
                "difficulty": r["difficulty"],
                "domain": r["domain"],
                "split": assign[r["id"]],
                "stratum_id": f"{r['category']}-{r['lang']}-{r['difficulty']}",
            }
        )
    write_parquet(out, DATA / "03_splits.parquet", SPLIT_SCHEMA)

    # test id 목록은 별도 파일. 이 파일 경로는 configs/test_path.yaml 에만 있고
    # evaluate.py 외 어떤 스크립트도 읽지 않는다 (check_test_isolation.py 가 강제).
    held = [{"id": r["id"]} for r in out if r["split"] == TEST]
    write_parquet(held, DATA / "03_test_ids.parquet", TEST_ID_SCHEMA)

    # ---- 검증과 보고 ------------------------------------------------------- #
    final = {s: len([r for r in out if r["split"] == s]) for s in SPLITS}
    require(final == want, f"최종 합 불일치: {final} != {want}")
    require(len({r["id"] for r in out}) == n, "id 중복")

    print("\n=== 최종 split ===")
    print(fmt_table(["split", "건수", "비율"],
                    [[s, final[s], f"{100.0*final[s]/n:.1f}%"] for s in SPLITS]))

    print("\n=== 카테고리 × split ===")
    crows = []
    for code in cat_order:
        sub = [r for r in out if r["category"] == code]
        if not sub:
            continue
        cs = [len([r for r in sub if r["split"] == s]) for s in SPLITS]
        crows.append([code, len(sub), *cs,
                      "  ".join(f"{100.0*c/len(sub):.0f}%" for c in cs)])
    print(fmt_table(["코드", "건수", *SPLITS, "비율"], crows))

    print("\n=== 난이도·언어 × split (층화가 유지되는지) ===")
    drows = []
    for key, sel in [(d, lambda r, d=d: r["difficulty"] == d) for d in DIFFS] + \
                    [(lg, lambda r, lg=lg: r["lang"] == lg) for lg in ("ko", "en")]:
        sub = [r for r in out if sel(r)]
        cs = [len([r for r in sub if r["split"] == s]) for s in SPLITS]
        drows.append([key, len(sub), *cs,
                      "  ".join(f"{100.0*c/len(sub):.1f}%" for c in cs)])
    print(fmt_table(["층", "건수", *SPLITS, "비율"], drows))

    print("\n=== 도메인 균형 (split 별 고유 도메인 수 / 최다 도메인 비중) ===")
    rrows = []
    for s in SPLITS:
        sub = [r for r in out if r["split"] == s]
        cnt: dict[str, int] = {}
        for r in sub:
            cnt[r["domain"]] = cnt.get(r["domain"], 0) + 1
        top = max(cnt.items(), key=lambda kv: kv[1])
        rrows.append([s, len(cnt), f"{top[0]} {100.0*top[1]/len(sub):.1f}%"])
    print(fmt_table(["split", "고유도메인", "최다도메인"], rrows))

    print(f"\n분할 -> {DATA / '03_splits.parquet'}")
    print(f"test id {len(held)}건 -> {DATA / '03_test_ids.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
