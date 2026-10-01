"""코퍼스에 "안녕"·"1+1" 같은 사소한 질의가 있는가, 없다면 판정이 어떻게 흔들리는가.

사용자 지적: 실제 트래픽에는 사소한 질의가 많고 그건 0.6B 도 당연히 처리한다. 우리 코퍼스가
그 질량을 담고 있지 않다면, 로컬 처리율 47.2% 는 **실제보다 비관적인 하한**이다.

두 가지를 잰다.
  1. 코퍼스의 난이도 실태 — 프롬프트 길이, student 정확도 분포, 만점 항목 비율
  2. 민감도 — 실제 트래픽의 f 비율이 사소하다면(양쪽 다 로컬에서 정답) 0.6B 의 로컬 처리율이
     사전 등록 문턱 50% 를 언제 넘는가

    .venv/bin/python scripts/trivial_traffic_sensitivity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RESULTS, read_rows  # noqa: E402

LOCAL_FLOOR = 0.50          # 13-U 에 사전 등록한 기준 B


def main() -> int:
    pr = {r["id"]: r for r in read_rows(DATA / "01_prompts.parquet")}
    acc17 = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.parquet")}
    acc06 = {r["id"]: float(r["accuracy"])
             for r in read_rows(DATA / "06_accuracy.sc06.parquet")}

    print("=" * 68)
    print("1) 이 코퍼스에 사소한 질의가 있는가")
    ntok = np.array([int(pr[i]["n_tokens"]) for i in acc17 if i in pr])
    print(f"  프롬프트 토큰 수: 최소 {ntok.min()} / 5분위 {np.percentile(ntok,5):.0f} / "
          f"중앙 {np.median(ntok):.0f} / 최대 {ntok.max()}")
    print(f"    -> '안녕'(1~2토큰)·'1+1'(3~5토큰) 급은 {int((ntok <= 8).sum())}건 "
          f"({100*(ntok<=8).mean():.1f}%)")

    for name, acc in (("1.7B", acc17), ("0.6B", acc06)):
        a = np.array(list(acc.values()))
        print(f"  {name} student 8시드 평균 정확도: 만점(1.0) {int((a>=1.0).sum())}건 "
              f"({100*(a>=1.0).mean():.1f}%) / 0점 {int((a<=0.0).sum())}건 "
              f"({100*(a<=0.0).mean():.1f}%) / 평균 {a.mean():.4f}")

    # 가장 쉬운 셀조차 만점이 아니면 "사소한 질의가 없다" 는 증거다
    easy = [i for i in acc17 if pr.get(i, {}).get("difficulty") == "easy"]
    e17 = np.array([acc17[i] for i in easy])
    e06 = np.array([acc06[i] for i in easy if i in acc06])
    print(f"  easy 항목 {len(easy)}건: 1.7B 평균 {e17.mean():.4f} (만점 "
          f"{100*(e17>=1.0).mean():.1f}%) / 0.6B 평균 {e06.mean():.4f} (만점 "
          f"{100*(e06>=1.0).mean():.1f}%)")
    print("  * 사소한 질의라면 양쪽 다 만점에 몰려야 한다. 그렇지 않다면 이 코퍼스의 'easy' 는")
    print("    '안녕' 급이 아니라 '여러 사실을 뽑아내는 짧은 과제' 다.")

    print("=" * 68)
    print("2) 실제 트래픽에 사소한 질의가 f 비율 있다면 (양쪽 다 로컬에서 정답)")
    rj = json.loads((RESULTS / "35_rejudge.json").read_text())
    need = float(rj["need_esc_06"])           # 비사소 부분에서 0.6B 가 동률에 필요한 위임률
    print(f"  비사소 부분에서 0.6B 가 1.7B 정확도에 닿는 위임률: {100*need:.1f}%")
    print(f"  사소한 질의는 양쪽 다 로컬·정답이므로 그 부분의 위임률은 0 이다.")
    print(f"  따라서 전체 위임률 = (1-f) x {100*need:.1f}%,  로컬 처리율 = 1 - 그 값")
    print()
    print(f"  {'f (사소 비율)':>14}{'전체 위임률':>12}{'로컬 처리율':>12}   기준 B(>=50%)")
    for f in (0.0, 0.03, 0.053, 0.10, 0.20, 0.30, 0.50):
        esc = (1 - f) * need
        loc = 1 - esc
        print(f"  {100*f:>13.1f}%{100*esc:>11.1f}%{100*loc:>11.1f}%   "
              f"{'통과' if loc >= LOCAL_FLOOR else '실패'}")
    f_star = 1 - (1 - LOCAL_FLOOR) / need
    print()
    print(f"  >>> 판정이 뒤집히는 임계 비율: f* = {100*f_star:.1f}%")
    print(f"      실제 트래픽의 {100*f_star:.1f}% 만 사소하면 0.6B 가 기준 B 를 통과한다.")

    # 사소한 질의를 라우터가 전부 로컬로 두지는 않는다 — 실측값을 넣어 다시 계산한다.
    probe = RESULTS / "37_trivial_router_probe_06.json"
    if probe.exists():
        p = json.loads(probe.read_text())
        d_triv = p["trivial_delegated"] / p["trivial_n"]
        print()
        print("=" * 68)
        print(f"3) 실측 보정 — 0.6B 라우터는 사소한 질의의 {100*d_triv:.0f}% 를 위임한다")
        print(f"   (results/{probe.name}: {p['trivial_delegated']}/{p['trivial_n']}건)")
        print(f"   전체 위임률 = (1-f) x {100*need:.1f}% + f x {100*d_triv:.0f}%")
        fs2 = (need - (1 - LOCAL_FLOOR)) / (need - d_triv)
        print(f"   >>> 보정된 임계 비율: f* = {100*fs2:.1f}%")
        print(f"   {'f':>6}{'전체 위임률':>12}{'로컬 처리율':>12}   기준 B")
        for f in (0.0, 0.05, fs2, 0.10, 0.20, 0.30):
            esc = (1 - f) * need + f * d_triv
            print(f"   {100*f:>5.1f}%{100*esc:>11.1f}%{100*(1-esc):>11.1f}%   "
                  f"{'통과' if (1-esc) >= LOCAL_FLOOR else '실패'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
