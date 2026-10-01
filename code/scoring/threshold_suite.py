"""문턱 스위트 — 층별·arm별 **최저 경계값**을 한 화면에 낸다 (SPEC 13-AN).

13-AM 에서 도구 호출은 운용점(esc 0.20)에서 아무것도 위임되지 않지만 esc 0.80 에서는 hard 가
100% 위임되는 것을 봤다. 즉 **문턱은 질의 유형마다 다른 곳에 있어야 한다.** 이 스크립트는 네 층
(자체 코퍼스 / 사소 / 공개 벤치 / 도구 호출) x 두 arm x 등록된 문턱 전부를 훑어 각 조합의
**최저 경계값**을 찾는다.

경계값의 정의 (셋 다 낸다 — 목적이 다르다)
  * `esc_breakeven`  : 캐스케이드 정확도가 로컬 전용을 처음으로 넘는 최저 esc
  * `esc_knee`       : 도달 가능한 최대 정확도의 99% 에 처음 닿는 최저 esc (수익 감소 지점)
  * `esc_best`       : 정확도가 최대인 esc (여러 개면 최저)
모두 **등록된 문턱 격자 안에서만** 고른다 — 새 문턱을 만들지 않는다.

캐스케이드 정확도는 항목별로 계산한다: 위임이면 서버 정답 여부, 로컬이면 student 8시드 정확도.
서버 답은 arm 과 무관하므로 층별로 한 번만 받아 뒀다(`43_server_all.<층>.jsonl`).

    .venv/bin/python scripts/threshold_suite.py [--md]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, REPO, RESULTS, read_rows, write_json  # noqa: E402
from grade import grade_contains, grade_exact  # noqa: E402

ARTS = {"gguf": "cascade-1.7b-gguf", "sc06": "cascade-0.6b-gguf"}
ARM_LABEL = {"gguf": "1.7B", "sc06": "0.6B"}

# 층: (표시 이름, 프롬프트, 키, arm별 accuracy 접미사, arm별 라우팅 태그)
LAYERS = {
    "core": ("자체 코퍼스", "01_prompts.parquet", "02_answer_keys.parquet",
             {"gguf": "", "sc06": ".sc06"},
             {"gguf": "core", "sc06": "core.sc06"}),
    "triv": ("사소 질의", "01_prompts.triv.parquet", "02_answer_keys.triv.parquet",
             {"gguf": ".triv", "sc06": ".triv.sc06"},
             {"gguf": "triv", "sc06": "triv.sc06"}),
    "bench": ("공개 벤치", "01_prompts.bench.parquet", "02_answer_keys.bench.parquet",
              {"gguf": ".bench", "sc06": ".bench.sc06"},
              {"gguf": "bench", "sc06": "bench.sc06"}),
    "tool": ("도구 호출", "01_prompts.tool.parquet", "02_answer_keys.tool.parquet",
             {"gguf": ".tool", "sc06": ".tool.sc06"},
             {"gguf": "tool", "sc06": "tool.sc06"}),
}


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def server_ok(layer: str, keys: dict, prompts: dict) -> dict[str, bool]:
    """항목별 서버 정답 여부. student 와 같은 채점 규칙을 쓴다."""
    out = {}
    for r in jl(DATA / f"43_server_all.{layer}.jsonl"):
        i, txt = r["id"], r["response"]
        k = keys.get(i)
        if k is None:
            continue
        if k["grading"] == "exact":
            out[i] = bool(grade_exact(txt, k, prompts[i])[0])
        elif k["grading"] == "contains":
            out[i] = bool(grade_contains(txt, k)[0])
    return out


def scores_for(layer: str, arm: str, cfg) -> dict[str, float] | None:
    """항목별 라우터 점수. core 는 escalation map 에서, 나머지는 route_layer 기록에서."""
    tags = cfg[4]
    if tags is not None:
        rows = jl(DATA / f"38_route_layer.{tags[arm]}.jsonl")
        return {r["id"]: float(r["score"]) for r in rows} if rows else None
    # core: escalation map 의 target_esc=1.0 행에 전 항목이 있다 (13-AD 와 같은 경로)
    name = ("09_escalation_map.parquet" if arm == "gguf"
            else "09_escalation_map.sc06.parquet")
    p = RESULTS / name
    if not p.exists():
        return None
    cand = [r for r in read_rows(p) if abs(float(r["target_esc"]) - 1.0) < 1e-6]
    if not cand or "score" not in cand[0]:
        return None
    return {r["id"]: float(r["score"]) for r in cand}


def sweep(scores, acc, srv, grid) -> list[dict]:
    ids = [i for i in scores if i in acc]
    local_only = st.mean(acc[i] for i in ids)
    out = []
    for esc, thr in grid:
        vals, n_del, n_gr = [], 0, 0
        for i in ids:
            if scores[i] < thr:
                n_del += 1
                if i in srv:
                    vals.append(1.0 if srv[i] else 0.0)
                    n_gr += 1
            else:
                vals.append(acc[i])
                n_gr += 1
        out.append({"target_esc": esc, "threshold": thr,
                    "esc_rate": n_del / len(ids),
                    "cascade_acc": st.mean(vals) if vals else float("nan"),
                    "n_graded": n_gr, "n": len(ids)})
    return local_only, out


def boundaries(local_only: float, curve: list[dict]) -> dict:
    """경계값 네 가지. 목적이 다르므로 하나로 뭉치지 않는다.

    처음엔 '무릎' 을 최대 이득의 99% 지점으로 뒀는데, 정확도가 위임에 따라 계속 오르므로
    그 무릎이 esc 0.80~1.00 (거의 전부 서버로) 에 붙었다 — 온디바이스 모델의 목적을 부정하는
    지표였다. 그래서 **위임 단위당 이득**과 **기준 B 한계**를 쓴다.
    """
    ok = [r for r in curve if r["cascade_acc"] == r["cascade_acc"]]
    if not ok:
        return {}
    best = max(r["cascade_acc"] for r in ok)
    bst = next(r for r in ok if r["cascade_acc"] == best)
    # ① 손익분기: 캐스케이드가 로컬 전용을 처음 넘는 최저 esc  = 문자 그대로의 최저 경계
    be = next((r for r in ok if r["cascade_acc"] > local_only), None)
    # ② 위임 효율 최고: (이득)/(실제 위임률) 이 최대인 지점 — 한 번의 서버 호출이 가장 값진 곳
    eff = [(r, (r["cascade_acc"] - local_only) / r["esc_rate"])
           for r in ok if r["esc_rate"] > 0.02 and r["cascade_acc"] > local_only]
    mb = max(eff, key=lambda t: t[1])[0] if eff else None
    mb_eff = max(eff, key=lambda t: t[1])[1] if eff else None
    # ③ 기준 B 한계: 실제 위임률이 50% 이하인 가장 높은 esc (로컬 처리율 >= 50% 유지)
    under = [r for r in ok if r["esc_rate"] <= 0.50]
    ub = max(under, key=lambda r: r["target_esc"]) if under else None
    return {
        "local_only_acc": local_only, "best_acc": best,
        "esc_breakeven": be["target_esc"] if be else None,
        "acc_breakeven": be["cascade_acc"] if be else None,
        "esc_rate_breakeven": be["esc_rate"] if be else None,
        "esc_marginal_best": mb["target_esc"] if mb else None,
        "acc_marginal_best": mb["cascade_acc"] if mb else None,
        "esc_rate_marginal_best": mb["esc_rate"] if mb else None,
        "gain_per_delegation": mb_eff,
        "esc_under50": ub["target_esc"] if ub else None,
        "acc_under50": ub["cascade_acc"] if ub else None,
        "esc_rate_under50": ub["esc_rate"] if ub else None,
        "esc_best": bst["target_esc"], "esc_rate_best": bst["esc_rate"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="results/44_threshold_suite.md 도 쓴다")
    a = ap.parse_args()

    out, missing = {}, []
    lines = ["# 문턱 스위트 — 층별·arm별 최저 경계값", "",
             "캐스케이드 정확도는 항목별 실측이다(위임이면 서버 정답 여부, 로컬이면 student "
             "8시드 정확도). 문턱은 **등록된 격자** 안에서만 고른다.", ""]

    for lk, cfg in LAYERS.items():
        name, pf, kf, accs, _ = cfg
        prompts = {r["id"]: r["prompt"] for r in read_rows(DATA / pf)}
        keys = {r["id"]: r for r in read_rows(DATA / kf)}
        # 채점 가능한 항목만 (judge/clarify 는 제외, 불안정 MMLU 도 제외)
        keys = {i: k for i, k in keys.items() if k["grading"] in ("exact", "contains")}
        if lk == "bench":
            keys = {i: k for i, k in keys.items()
                    if not (i.startswith("MMLU") and len(str(k["answer"]).strip()) == 1)}
        srv = server_ok(lk, keys, prompts)
        if not srv:
            missing.append(f"{lk}: 서버 응답 없음 (fill_server_all.py 를 먼저 돌려라)")
            continue

        for arm in ("gguf", "sc06"):
            accp = DATA / f"06_accuracy{accs[arm]}.parquet"
            if not accp.exists():
                missing.append(f"{lk}/{arm}: {accp.name} 없음")
                continue
            acc = {r["id"]: float(r["accuracy"]) for r in read_rows(accp)
                   if r["id"] in keys}
            sc = scores_for(lk, arm, cfg)
            if not sc:
                missing.append(f"{lk}/{arm}: 라우터 점수 없음")
                continue
            cfgj = json.loads((REPO / "artifact" / ARTS[arm] / "delegate"
                               / "config.json").read_text())
            grid = sorted((float(k), float(v)) for k, v in cfgj["thresholds"].items())
            lo, curve = sweep(sc, acc, srv, grid)
            b = boundaries(lo, curve)
            out[f"{lk}/{arm}"] = {"layer": name, "arm": ARM_LABEL[arm],
                                  "curve": curve, **b}

            print(f"\n[{name} / {ARM_LABEL[arm]}]  n={curve[0]['n']} "
                  f"로컬전용 {lo:.4f}  최대 {b['best_acc']:.4f}")
            print(f"  {'esc':>6}{'문턱':>9}{'실제위임':>9}{'캐스케이드':>11}   경계")
            for r in curve:
                mk = []
                if r["target_esc"] == b.get("esc_breakeven"):
                    mk.append("손익분기")
                if r["target_esc"] == b.get("esc_marginal_best"):
                    mk.append("**효율최고**")
                if r["target_esc"] == b.get("esc_under50"):
                    mk.append("기준B한계")
                if r["target_esc"] == b.get("esc_best"):
                    mk.append("정확도최대")
                print(f"  {r['target_esc']:>6.2f}{r['threshold']:>9.4f}"
                      f"{100*r['esc_rate']:>8.0f}%{r['cascade_acc']:>11.4f}   "
                      f"{' '.join(mk)}")

    print("\n" + "=" * 92)
    print("최저 경계값 요약  —  손익분기 = 위임이 처음 값을 내는 곳 / 효율최고 = 서버 호출 "
          "한 번이 가장 값진 곳 / 기준B = 로컬 50% 유지")
    hdr = (f"  {'층':<12}{'arm':<7}{'로컬전용':>9}{'손익분기 esc':>13}{'효율최고 esc':>13}"
           f"{'(위임)':>8}{'(정확도)':>10}{'기준B esc':>11}{'(정확도)':>10}")
    print(hdr)
    lines += ["## 최저 경계값 요약", "",
              "- **손익분기**: 캐스케이드가 로컬 전용을 처음 넘는 최저 esc — 문자 그대로의 "
              "최저 경계", "- **효율최고**: (이득 / 실제 위임률) 이 최대인 esc — 서버 호출 "
              "한 번이 가장 값진 곳", "- **기준B 한계**: 실제 위임률 ≤ 50% 를 지키는 가장 "
              "높은 esc (로컬 처리율 ≥ 50%)", "",
              "| 층 | arm | 로컬 전용 | 손익분기 esc | 효율최고 esc | 그때 위임 | "
              "그때 정확도 | 기준B esc | 그때 정확도 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in out.items():
        f = lambda x, s="{:.2f}": "—" if x is None else s.format(x)  # noqa: E731
        print(f"  {v['layer']:<12}{v['arm']:<7}{v['local_only_acc']:>9.4f}"
              f"{f(v.get('esc_breakeven')):>13}{f(v.get('esc_marginal_best')):>13}"
              f"{f(v.get('esc_rate_marginal_best'), '{:.0%}'):>8}"
              f"{f(v.get('acc_marginal_best'), '{:.4f}'):>10}"
              f"{f(v.get('esc_under50')):>11}"
              f"{f(v.get('acc_under50'), '{:.4f}'):>10}")
        lines.append(
            f"| {v['layer']} | {v['arm']} | {v['local_only_acc']:.4f} | "
            f"{f(v.get('esc_breakeven'))} | **{f(v.get('esc_marginal_best'))}** | "
            f"{f(v.get('esc_rate_marginal_best'), '{:.0%}')} | "
            f"{f(v.get('acc_marginal_best'), '{:.4f}')} | "
            f"{f(v.get('esc_under50'))} | {f(v.get('acc_under50'), '{:.4f}')} |")

    if missing:
        print("\n빠진 조합:")
        for m in missing:
            print(f"  - {m}")
    out["_missing"] = missing
    write_json(out, RESULTS / "44_threshold_suite.json")
    if a.md:
        (RESULTS / "44_threshold_suite.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
        print(f"-> {RESULTS / '44_threshold_suite.md'}")
    print(f"-> {RESULTS / '44_threshold_suite.json'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
