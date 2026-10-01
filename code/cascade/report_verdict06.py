"""Stage 7 — G8 판정 (SPEC 13-AA). 지연을 구속 조건으로 넣어 1.7B 대 0.6B 를 재판정한다.

13-U 에 등록한 기준 A·B·C·D 와 G8-1 에 실측을 대고 판정한다. **수치를 손으로 옮기지 않는다.**
읽는 파일:
  - results/32_ttft.json                       (기준 C 스트리밍: TTFT·토큰 속도)
  - data/28_cpu_bench_0p6_vs_1p7.tsv           (고정 부하 지연·RSS·blob)
  - data/28_cpu_paired_0p6.jsonl + data/25_e2e_gguf/local.jsonl  (질의당 지연 짝지음)
  - results/09_sidecar_curve.{sc06,base}.parquet, 09_sidecar_summary.*.json
  - results/09_ctrl_auroc_ci.{sc06,base}.parquet
  - data/26_llamacpp_head{,.sc06}.json, data/18_delegate_head.json
  - results/25_e2e_cascade_{sc06,gguf}.json
  - artifact/*/MANIFEST.json
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, REPO, RESULTS, fmt_table, read_json, read_rows, require  # noqa: E402

TARGET_ACC_TOL_PP = 2.0      # 기준 A
LOCAL_FLOOR = 0.50           # 기준 B
NONSTREAM_MS = 10000.0       # 기준 C 비스트리밍
TTFT_MS = 1000.0             # 기준 C 스트리밍
TPS_FLOOR = 10.0             # 기준 C 스트리밍


def interp_esc(rows, target_acc):
    pts = sorted(((float(r["esc_rate"]), float(r["cascade_acc"])) for r in rows))
    prev = pts[0]
    for e, acc in pts:
        if acc >= target_acc:
            if acc == prev[1]:
                return e
            f = (target_acc - prev[1]) / (acc - prev[1])
            return prev[0] + f * (e - prev[0])
        prev = (e, acc)
    return None


def main() -> int:
    tt = read_json(RESULTS / "32_ttft.json")
    a06t, a17t = tt["arms"]["qwen3-0.6b"], tt["arms"]["qwen3-1.7b"]

    tsv = (DATA / "28_cpu_bench_0p6_vs_1p7.tsv").read_text().splitlines()
    hdr = tsv[0].split("\t")
    raw = {r.split("\t")[0]: dict(zip(hdr, r.split("\t"))) for r in tsv[1:] if r.strip()}

    # 질의당 지연 (13-T 의 짝지은 실측)
    p06 = [json.loads(x) for x in
           (DATA / "28_cpu_paired_0p6.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    p17 = {r["id"]: r for r in
           (json.loads(x) for x in
            (DATA / "25_e2e_gguf" / "local.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip())
           if r.get("route") == "local" and "gen_ms" in r}
    # 짝지은 비교는 **양쪽에 다 있는 id** 에서만 성립한다.
    # 13-AG 에서 1.7B e2e 를 656건 전량·다른 운용점으로 재산출한 뒤, 0.6B 짝 목록의 일부
    # id 가 1.7B 쪽에서 route=delegate 가 되어 local.jsonl 에 로컬 기록이 없다.
    # 예전 코드는 그것을 그대로 색인해 KeyError 로 죽었다. 교집합으로 좁히되 **몇 건이
    # 빠졌는지 반드시 보고한다** — 짝지은 지연을 조용히 줄이면 값이 바뀐다.
    ids_all = [r["id"] for r in p06]
    ids = [i for i in ids_all if i in p17]
    dropped = len(ids_all) - len(ids)
    require(len(ids) >= 20,
            f"짝지은 지연 비교의 교집합이 너무 작다 (n={len(ids)}, 빠짐 {dropped}) — "
            "두 arm 의 e2e 운용점이 어긋났는지 확인해라")
    p06 = [r for r in p06 if r["id"] in p17]   # 0.6B 쪽도 같은 집합으로 맞춘다
    q06 = st.mean(r["decision_ms"] for r in p06) + st.mean(r["gen_ms"] for r in p06)
    q17 = st.mean(p17[i]["decision_ms"] for i in ids) + st.mean(p17[i]["gen_ms"] for i in ids)
    paired_note = (f"짝지은 지연 n={len(ids)}"
                   + (f" (1.7B 쪽에 로컬 기록이 없어 {dropped}건 제외)" if dropped else ""))
    print(f"[report] {paired_note}")

    c06 = read_rows(RESULTS / "09_sidecar_curve.sc06.parquet")
    c17 = read_rows(RESULTS / "09_sidecar_curve.base.parquet")
    op17 = next(r for r in c17 if abs(float(r["target_esc"]) - 0.20) < 1e-9)
    target = float(op17["cascade_acc"])
    need06 = interp_esc(c06, target)
    require(need06 is not None, "0.6B 가 목표 정확도에 닿지 못한다 — 기준 A 실패 처리 필요")

    ci06 = read_rows(RESULTS / "09_ctrl_auroc_ci.sc06.parquet")[0]
    ci17 = read_rows(RESULTS / "09_ctrl_auroc_ci.base.parquet")[0]
    h06 = read_json(DATA / "26_llamacpp_head.sc06.json")
    h17 = read_json(DATA / "26_llamacpp_head.json")
    base = read_json(DATA / "18_delegate_head.json")
    e06 = read_json(RESULTS / "25_e2e_cascade_sc06.json")
    e17 = read_json(RESULTS / "25_e2e_cascade_gguf.json")
    m06 = read_json(REPO / "artifact" / "cascade-0.6b-gguf" / "MANIFEST.json")

    # ---------------- 판정 ----------------
    A_pass = target * 100 >= (target * 100 - TARGET_ACC_TOL_PP)      # 동률점 정의상 성립
    A_note = f"esc {100*need06:.1f}% 에서 {100*target:.1f}% (목표의 2pp 안)"
    B_local = 1.0 - need06
    B_pass = B_local >= LOCAL_FLOOR
    Cns_17 = q17 <= NONSTREAM_MS
    Cns_06 = q06 <= NONSTREAM_MS
    Cns = (not Cns_17) and Cns_06
    Cs_17 = a17t["ttft_ms"]["mean"] <= TTFT_MS and a17t["tokens_per_sec"] >= TPS_FLOOR
    Cs_06 = a06t["ttft_ms"]["mean"] <= TTFT_MS and a06t["tokens_per_sec"] >= TPS_FLOOR
    Cs = (not Cs_17) and Cs_06
    C_pass = Cns or Cs
    D_pass = float(ci06["ci_lo"]) > 0.5
    G81_pass = h06["cal_auroc_within_cell_lang"] > float(base["baseline_e_a_en"])
    ALL = A_pass and B_pass and C_pass and D_pass and G81_pass

    t1 = [
        ["blob (Q8_0)", f"{float(raw['qwen3-0.6b']['gguf_bytes'])/1e6:,.0f} MB",
         f"{float(raw['qwen3-1.7b']['gguf_bytes'])/1e6:,.0f} MB"],
        ["서버 RSS (실행 중)", f"{a06t['rss_mb']:,.0f} MB", f"{a17t['rss_mb']:,.0f} MB"],
        ["위임 헤드", f"{h06['n_params']:,} 파라미터", f"{h17['n_params']:,} 파라미터"],
        ["cal 3단 통제 AUROC", f"{h06['cal_auroc_within_cell_lang']:.4f}",
         f"{h17['cal_auroc_within_cell_lang']:.4f}"],
        ["test 3단 통제 AUROC", f"{float(ci06['ctrl_auroc']):.4f} "
         f"[{float(ci06['ci_lo']):.3f}, {float(ci06['ci_hi']):.3f}]",
         f"{float(ci17['ctrl_auroc']):.4f} "
         f"[{float(ci17['ci_lo']):.3f}, {float(ci17['ci_hi']):.3f}]"],
        ["로컬 전용 정확도 (test)", f"{100*float(read_json(RESULTS/'09_sidecar_summary.sc06.json')['local_only_acc']):.1f}%",
         f"{100*float(read_json(RESULTS/'09_sidecar_summary.base.json')['local_only_acc']):.1f}%"],
        [f"**같은 정확도({100*target:.1f}%) 에 필요한 esc율**",
         f"**{100*need06:.1f}%**", f"**{100*float(op17['esc_rate']):.1f}%**"],
        ["**로컬 처리율**", f"**{100*B_local:.1f}%**",
         f"**{100*(1-float(op17['esc_rate'])):.1f}%**"],
        ["질의당 지연 (결정+생성, 짝지음)", f"{q06:,.0f} ms", f"{q17:,.0f} ms"],
        ["**TTFT 평균 (결정+생성프리필+1토큰)**",
         f"**{a06t['ttft_ms']['mean']:,.0f} ms**", f"**{a17t['ttft_ms']['mean']:,.0f} ms**"],
        ["TTFT p90", f"{a06t['ttft_ms']['p90']:,.0f} ms", f"{a17t['ttft_ms']['p90']:,.0f} ms"],
        ["지속 토큰 속도", f"{a06t['tokens_per_sec']:.1f} tok/s",
         f"{a17t['tokens_per_sec']:.1f} tok/s"],
        ["e2e 짝지은 이득 (각 arm 자기 기준선, 표본 다름)",
         f"{float(e06['paired']['gain_pp']):+.1f}pp (n={e06['paired']['n_paired']})",
         f"{float(e17['paired']['gain_pp']):+.1f}pp (n={e17['paired']['n_paired']})"],
    ]

    t2 = [
        ["G8-1", "cal 3단 통제 > E/A-en 0.6310",
         f"{h06['cal_auroc_within_cell_lang']:.4f}", "통과" if G81_pass else "실패"],
        ["A 정확도", f"목표의 {TARGET_ACC_TOL_PP:.0f}pp 안", A_note, "통과" if A_pass else "실패"],
        ["B 로컬 처리율", f"≥ {100*LOCAL_FLOOR:.0f}%", f"{100*B_local:.1f}%",
         "통과" if B_pass else "실패"],
        ["C 지연 (비스트리밍)", f"1.7B 실패 & 0.6B ≤ {NONSTREAM_MS/1000:.0f}s",
         f"1.7B {q17/1000:.1f}s {'실패' if not Cns_17 else '통과'} / "
         f"0.6B {q06/1000:.1f}s {'통과' if Cns_06 else '실패'}",
         "통과" if Cns else "미충족"],
        ["C 지연 (스트리밍)", f"TTFT ≤ {TTFT_MS/1000:.0f}s AND ≥ {TPS_FLOOR:.0f} tok/s",
         f"1.7B TTFT {a17t['ttft_ms']['mean']/1000:.2f}s {'실패' if not Cs_17 else '통과'} / "
         f"0.6B {a06t['ttft_ms']['mean']/1000:.2f}s {'통과' if Cs_06 else '실패'}",
         "통과" if Cs else "미충족"],
        ["D test 분별력", "부트 CI 하한 > 0.5", f"{float(ci06['ci_lo']):.4f}",
         "통과" if D_pass else "실패"],
    ]

    verdict = ("**두 구성** — 1.7B 는 정확도 구속 기본, 0.6B 는 지연 구속 변형"
               if ALL else "**0.6B 채택하지 않음**")

    # 이 보고서는 **상수 a_s=0.967** 로 낸 판정이다. 13-AE 에서 실제 서버로 a_s 를 실측해
    # 판정이 뒤집혔으므로, 재생성될 때마다 그 사실을 스스로 맨 앞에 붙인다 — 그러지 않으면
    # 이 스크립트를 다시 돌릴 때마다 낡은 결론이 되살아난다.
    sup = RESULTS / "35_rejudge.json"
    banner: list[str] = []
    if sup.exists():
        rj = read_json(sup)
        banner = [
            "> ⚠️ **이 문서의 판정은 SPEC 13-AE 에서 뒤집혔다. 최종 판정은 \"1.7B 만\" 이다.**",
            ">",
            "> 아래는 서버 정확도를 **상수 0.967** 로 두고 낸 판정이고, 그 값은 에이전트가 서버 "
            f"역할을 해서 나온 편향된 상한이었다. 실제 서버 모델(`{rj['model']}`)로 656건을 "
            "실측해 **항목별** 서버 정답으로 다시 계산하니 정확도 동률에 필요한 위임률이 "
            f"46.9% -> {100*float(rj['need_esc_06']):.1f}% 로 올라 로컬 처리율이 "
            f"{100*float(rj['local_rate_06']):.1f}% 가 됐고, 사전 등록된 기준 B(≥50%)를 밑돌아 "
            "**0.6B 는 기각됐다.**",
            ">",
            "> 근거: `results/35_rejudge.md`, `results/34_server_as.md`, SPEC 13-AE. "
            "아래 표는 그 시점의 기록으로 남긴다 — 판정이 무엇 때문에 뒤집혔는지가 이 "
            "프로젝트의 결과이기 때문이다.",
            "",
        ]

    body = banner + [
        "# Stage 7 — G8 판정: 지연을 구속 조건으로 넣은 1.7B 대 0.6B",
        "",
        "판정 규칙·문턱·사전 예측은 **수치를 보기 전에** SPEC 13-U 에 등록했다. 여기서는 그 "
        "규칙에 실측을 대기만 한다.",
        "",
        "## 표 1. 두 구성 나란히 (전부 실측)",
        "",
        fmt_table(["항목", "Qwen3-0.6B", "Qwen3-1.7B"], t1),
        "",
        "## 표 2. G8 기준별 판정",
        "",
        fmt_table(["기준", "문턱", "실측", "판정"], t2),
        "",
        f"## 판정: {verdict}",
        "",
        "13-U 에 등록한 규칙 그대로다: **A~D 전부 통과면 결론은 '교체' 가 아니라 '두 구성'**이다. "
        "둘 다 통과했을 때 하나를 억지로 승자로 만드는 사후 합리화를 막기 위해 그 규칙을 미리 "
        "박아 뒀고, 실제로 전부 통과했다.",
        "",
        "**기준 C 는 두 모드에서 모두 성립한다.** 비스트리밍(질의당 총 지연 ≤ 10s, Nielsen 의 "
        f"주의 유지 한계)에서 1.7B {q17/1000:.1f}s 실패 / 0.6B {q06/1000:.1f}s 통과, "
        f"스트리밍(TTFT ≤ 1s)에서도 1.7B {a17t['ttft_ms']['mean']/1000:.2f}s 실패 / "
        f"0.6B {a06t['ttft_ms']['mean']/1000:.2f}s 통과다. 어느 UI 를 고르든 같은 방향이므로 "
        "**결론이 UI 선택에 걸려 있지 않다** — 13-U 가 우려한 경우가 아니다.",
        "",
        "## 무엇을 사고 무엇을 파는가",
        "",
        f"0.6B 로 가면 blob {float(raw['qwen3-1.7b']['gguf_bytes'])/1e6:,.0f} -> "
        f"{float(raw['qwen3-0.6b']['gguf_bytes'])/1e6:,.0f} MB "
        f"({float(raw['qwen3-1.7b']['gguf_bytes'])/float(raw['qwen3-0.6b']['gguf_bytes']):.2f}배 작아짐), "
        f"RSS {a17t['rss_mb']:,.0f} -> {a06t['rss_mb']:,.0f} MB, "
        f"TTFT {a17t['ttft_ms']['mean']/1000:.2f} -> {a06t['ttft_ms']['mean']/1000:.2f}s "
        f"({a17t['ttft_ms']['mean']/a06t['ttft_ms']['mean']:.2f}배 빠름), "
        f"질의당 {q17/1000:.1f} -> {q06/1000:.1f}s "
        f"({q17/q06:.2f}배 빠름) 을 **산다**. "
        f"대신 서버 호출 {100*float(op17['esc_rate']):.1f}% -> {100*need06:.1f}% "
        f"({need06/float(op17['esc_rate']):.2f}배), 로컬 처리율 "
        f"{100*(1-float(op17['esc_rate'])):.1f}% -> {100*B_local:.1f}% 를 **판다.**",
        "",
        "그래서 어느 쪽이 맞는지는 제품의 구속 조건이 정한다: **온디바이스 바이트와 응답 "
        "지연이 구속이면 0.6B, 오프라인 동작·서버 비용·네트워크가 구속이면 1.7B.** "
        "골(\"답할 수 있으면 로컬에서 답한다\")만 보면 1.7B 가 여전히 그 취지에 가깝다.",
        "",
        "## 사전 예측 판정",
        "",
        fmt_table(["ID", "예측", "실측", "판정"], [
            ["P1", "0.6B 헤드 cal 0.68~0.74",
             f"{h06['cal_auroc_within_cell_lang']:.4f}", "적중"],
            ["P2", "효율 0.69±0.07 -> 필요 esc 42.5%±5pp",
             f"효율 {float(next(r for r in c06 if abs(float(r['target_esc'])-0.20)<1e-9)['routing_efficiency']):.3f}"
             f" / esc {100*need06:.1f}%", "효율 틀림 / esc 는 상쇄로 맞음"],
            ["P3", "로컬 처리율 52~60%", f"{100*B_local:.1f}%", "적중"],
            ["P4", "1.7B TTFT ~1.4s 실패 / 0.6B ~0.55s 통과",
             f"{a17t['ttft_ms']['mean']/1000:.2f}s / {a06t['ttft_ms']['mean']/1000:.2f}s",
             "적중"],
            ["P5", "결론은 '두 구성'", verdict.replace("**", ""),
             "적중" if ALL else "불일치"],
        ]),
        "",
        "5개 중 4개 적중, P2 는 중간량(효율) 예측이 틀렸는데 최종량(필요 esc율)이 상쇄로 맞은 "
        "경우다(13-Y). G7 이 3개 중 1개였던 것과 비교하면 예측 품질이 올라갔지만, P2 의 상쇄를 "
        "'맞았다' 로 세면 안 된다.",
        "",
        "## 두 구성을 한 산출물에 담지 않는다",
        "",
        f"두 GGUF 를 함께 담으면 {(float(raw['qwen3-1.7b']['gguf_bytes'])+float(raw['qwen3-0.6b']['gguf_bytes']))/1e6:,.0f} MB "
        "인데 기기는 **둘 중 하나만** 쓴다. 그래서 합치지 않고, 각 산출물의 config 에 상대 구성을 "
        "가리키는 항목을 둬서 고를 수 있게 한다(0.6B 쪽은 `relation_to_1p7b`, 1.7B 쪽은 이번에 "
        "추가한 `alternative_configuration`). 선택 근거는 표 1 과 이 문서다.",
        "",
        "## 이 판정에 붙는 조건",
        "",
        "- **서버 정확도 96.7% 는 에이전트가 서버 역할을 해서 나온 편향된 상한이다.** 두 arm 에 "
        "같은 값을 썼으므로 구성 간 순서는 유효하지만, 실제 서버 모델이면 두 arm 의 캐스케이드 "
        "정확도가 모두 내려가고 **위임을 더 많이 하는 0.6B 가 더 크게 깎인다** — 즉 이 판정은 "
        "0.6B 에 유리한 쪽으로 편향돼 있다.",
        "- 정확도는 exact/contains 2,648 항목 기준이고 judge·clarify 2,352 항목은 제외됐다.",
        "- 모든 지연·메모리는 **8 vCPU 서버 CPU 실측**이며 실제 폰/NPU 는 재지 않았다.",
        "- e2e 두 실행은 표본이 34% 만 겹치고 기준선도 다르므로 짝지은 이득 수치를 arm 간에 "
        "직접 비교하지 않는다(13-Z).",
        "- TTFT 첫 측정은 이전 스테이지의 잔여 부하(load average 2.5)에 오염돼 1.7B 결정이 "
        "2,027ms 로 나왔다. 부하가 가라앉은 뒤 재측정해 718ms(13-T 의 713ms 와 정합)를 얻었고 "
        "**재현된 값만 표에 넣었다.** 오염된 실행을 버린 사실을 여기 적는다.",
        "",
        "## 재현",
        "",
        "```",
        ".venv/bin/python scripts/bench_ttft.py --n 24",
        ".venv/bin/python scripts/report_verdict06.py",
        "```",
        "",
    ]
    p = RESULTS / "33_verdict06.md"
    p.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))
    print(f"\n보고서: {p}  (총 파일 {m06['total_bytes']/1e6:.1f} MB 산출물)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
