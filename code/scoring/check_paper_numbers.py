"""PAPER.md 의 0.6B arm 수치를 데이터 파일에 대고 검증한다 (SPEC 13-AB).

`PAPER.md` 는 산문 문서라 스크립트가 생성하지 않는다 — 그래서 수치를 손으로 적는다. 손으로
적은 수치는 재실행 때 낡아도 아무도 모른다(13-M 에서 `path_choice.py` 가 실제로 그랬다).
그래서 **문서에 적힌 값이 파일과 같은지 기계로 검사**한다. 불일치가 있으면 exit 1.

  .venv/bin/python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, REPO, RESULTS, read_json, read_rows  # noqa: E402

PAPER = REPO / "PAPER.md"


def main() -> int:
    txt = PAPER.read_text(encoding="utf-8")

    h06 = read_json(DATA / "26_llamacpp_head.sc06.json")
    h17 = read_json(DATA / "26_llamacpp_head.json")
    base = read_json(DATA / "18_delegate_head.json")
    ci06 = read_rows(RESULTS / "09_ctrl_auroc_ci.sc06.parquet")[0]
    ci17 = read_rows(RESULTS / "09_ctrl_auroc_ci.base.parquet")[0]
    frozen = {r["condition"] + "/" + r["variant"]: r
              for r in read_rows(RESULTS / "09_ctrl_auroc_ci.parquet")}
    s06 = read_json(DATA / "26_head_sanity.sc06.json")
    s17 = read_json(DATA / "26_head_sanity.json")
    tt = read_json(RESULTS / "32_ttft.json")["arms"]
    c06 = read_rows(RESULTS / "09_sidecar_curve.sc06.parquet")
    c17 = read_rows(RESULTS / "09_sidecar_curve.base.parquet")
    o17 = next(r for r in c17 if abs(float(r["target_esc"]) - 0.20) < 1e-9)
    o06 = next(r for r in c06 if abs(float(r["target_esc"]) - 0.20) < 1e-9)
    e06 = read_json(RESULTS / "25_e2e_cascade_sc06.json")
    e17 = read_json(RESULTS / "25_e2e_cascade_gguf.json")
    g06 = next((REPO / "artifact" / "cascade-0.6b-gguf" / "model").glob("*.gguf"))
    g17 = next((REPO / "artifact" / "cascade-1.7b-gguf" / "model").glob("*.gguf"))
    z06 = None
    try:
        import numpy as np
        z = np.load(DATA / "26_llamacpp_features.sc06.npz", allow_pickle=False)
        z06 = float((z["acc"][z["split"] == "train"] < 0.75).mean())
        z17f = np.load(DATA / "26_llamacpp_features.npz", allow_pickle=False)
        z17 = float((z17f["acc"][z17f["split"] == "train"] < 0.75).mean())
    except Exception:
        z17 = None

    # (문서에 나타나야 하는 문자열, 그 값의 출처 설명)
    checks: list[tuple[str, str]] = [
        (f"{h06['cal_auroc_within_cell_lang']:.4f}", "0.6B cal 3단통제"),
        (f"{h17['cal_auroc_within_cell_lang']:.4f}", "1.7B cal 3단통제"),
        (f"{float(base['baseline_e_a_en']):.4f}", "E/A-en cal 기준선"),
        (f"{float(ci06['ctrl_auroc']):.4f}", "0.6B test 3단통제"),
        (f"{float(ci17['ctrl_auroc']):.4f}", "1.7B test 3단통제"),
        (f"{h06['n_params']:,}", "0.6B 헤드 파라미터"),
        (f"{h17['n_params']:,}", "1.7B 헤드 파라미터"),
        (f"{ci06['n_pair']:,}", "0.6B test 통제 짝"),
        (f"{ci17['n_pair']:,}", "1.7B test 통제 짝"),
        (f"{s06['n_pair']:,}", "0.6B cal 통제 짝"),
        (f"{s17['n_pair']:,}", "1.7B cal 통제 짝"),
        (f"{tt['qwen3-0.6b']['ttft_ms']['mean']/1000:.2f}", "0.6B TTFT s"),
        (f"{tt['qwen3-1.7b']['ttft_ms']['mean']/1000:.2f}", "1.7B TTFT s"),
        (f"{tt['qwen3-0.6b']['tokens_per_sec']:.1f}", "0.6B tok/s"),
        (f"{tt['qwen3-1.7b']['tokens_per_sec']:.1f}", "1.7B tok/s"),
        (f"{g06.stat().st_size/1e6:,.0f}", "0.6B blob MB"),
        (f"{g17.stat().st_size/1e6:,.0f}", "1.7B blob MB"),
        (f"{tt['qwen3-0.6b']['rss_mb']:,.0f}", "0.6B RSS MB"),
        (f"{tt['qwen3-1.7b']['rss_mb']:,.0f}", "1.7B RSS MB"),
        (f"{100*float(o17['esc_rate']):.1f}%", "1.7B 실현 esc"),
        (f"{float(o06['routing_efficiency']):.3f}", "0.6B 라우팅 효율"),
        (f"{float(o17['routing_efficiency']):.3f}", "1.7B 라우팅 효율"),
        (f"{float(e06['paired']['gain_pp']):+.1f}pp", "0.6B e2e 짝지은 이득"),
        (f"{float(e17['paired']['gain_pp']):+.1f}pp", "1.7B e2e 짝지은 이득"),
    ]
    if z06 is not None and z17 is not None:
        checks += [(f"{z06:.3f}", "0.6B train 라벨 비율"),
                   (f"{z17:.3f}", "1.7B train 라벨 비율")]

    # 실제 서버로 재측정한 값 (SPEC 13-AE) — 뒤집힌 판정의 근거 수치도 문서와 일치해야 한다
    sa = RESULTS / "34_server_as.json"
    rj = RESULTS / "35_rejudge.json"
    if sa.exists() and rj.exists():
        s = read_json(sa)
        r = read_json(rj)

        def op(curve, tgt):
            return next(x for x in curve if abs(float(x["target_esc"]) - tgt) < 1e-3)

        o17, o06 = op(r["curve_17"], 0.20), op(r["curve_06"], 0.40)
        checks += [
            (s["model"], "서버 모델 id"),
            (f"{s['a_s_all']:.3f}", "전체 a_s"),
            (f"{float(o17['a_s_on_subset']):.3f}", "1.7B 부분집합 a_s"),
            (f"{float(o06['a_s_on_subset']):.3f}", "0.6B 부분집합 a_s"),
            (f"{float(o17['cascade_acc']):.4f}", "1.7B 캐스케이드 (실측 서버)"),
            (f"{float(o06['cascade_acc']):.4f}", "0.6B 캐스케이드 (실측 서버)"),
            (f"{100*float(r['need_esc_06']):.1f}%", "0.6B 필요 esc율 (실측)"),
            (f"{100*float(r['local_rate_06']):.1f}%", "0.6B 로컬 처리율 (실측)"),
        ]

    bad = []
    for needle, what in checks:
        if needle not in txt:
            bad.append((needle, what))
        else:
            print(f"  OK  {what:24s} {needle}")

    # 위협 목록 번호가 1..N 로 연속인지 (항목을 추가하다 번호가 겹치기 쉽다).
    # 해결된 항목은 `11. ~~**...**~~` 처럼 취소선으로 표기하므로 `**` 만 기대하면 놓친다.
    sec = txt.split("## 4. 타당성 위협과 한계")[1].split("\n---")[0]
    nums = [int(m) for m in re.findall(r"^(\d+)\. [~*]", sec, flags=re.M)]
    seq_ok = nums == list(range(1, len(nums) + 1))
    print(f"\n위협 목록 번호 {nums[0]}..{nums[-1]} ({len(nums)}개) — "
          f"{'연속' if seq_ok else '불연속!'}")

    if bad:
        print(f"\n!! PAPER.md 에서 찾지 못한 값 {len(bad)}건:", file=sys.stderr)
        for needle, what in bad:
            print(f"   {what}: '{needle}'", file=sys.stderr)
        return 1
    if not seq_ok:
        return 1
    print(f"\n판정: 통과 — 검사한 {len(checks)}개 값이 모두 파일과 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
