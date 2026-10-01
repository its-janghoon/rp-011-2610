"""9단계: test 1회 로드 후 결과표 (SPEC §12). 에이전트 비용 0.

**이 스크립트만 test 를 로드한다.** `configs/test_path.yaml` 의 경로를 읽는 유일한 곳이고
`scripts/check_test_isolation.py` 가 그것을 기계적으로 보장한다. 정책은
`data/08_policy.json` 에 이미 동결되어 있으므로 여기서 학습·보정은 일어나지 않는다.

## 무엇을 비교하는가

주 비교축은 **동일 escalate 건수에서의 오탈출률**이다. server 를 실행하지 않았으므로 정확도의
절대 수준은 a_s(server 평균 정확도)를 스윕해 부표로 낸다. escalate 건수를 맞추면 a_s 항이
상수가 되어 조건 간 순서는 a_s 에 불변이다(근거는 conditions.py 서두).

### 단위를 맞춘다 — 8단계 cal AUROC 를 그대로 비교하면 안 된다

조건 A 는 프롬프트만 보므로 시드에 무관한 **항목 수준** 예측을 하고, B/C/D 는 생성마다 다른
**생성 수준** 예측을 한다. 항목 수준 라벨(8시드 평균의 이진화)은 생성 수준 라벨보다 잡음이 적어
더 쉬운 과제다. 그래서 cal 에서 본 A 0.757 vs D 0.680 은 공정한 비교가 아니다.
여기서는 **두 단위 모두**에서 낸다:
  - 항목 수준: A 의 점수와, B/C/D 점수의 8시드 평균을 같은 라벨에 대해 비교
  - 생성 수준: A 의 점수를 8시드에 브로드캐스트해 같은 라벨에 대해 비교
주 결과표(오탈출률)는 8-복제 시뮬레이션이므로 단위 문제가 없다 —
정책을 시드 s 의 특징으로 돌리고 시드 s 의 결과로 채점한다(SPEC §10).

### A 의 이득이 코퍼스 인공물인지 검증한다

A 는 train 항목의 **실측 accuracy** 로 만든 라벨의 kNN 이다. 우리 코퍼스는 카테고리 x 난이도
셀 안에서 프롬프트가 구조적으로 유사하므로(중복제거 임계 0.95), A 의 이웃이 사실상 같은 셀만
가져오고 있다면 A 는 "난이도 예측" 이 아니라 **셀 평균 조회**를 하고 있는 것이다.
그래서 사소한 사전 예측기 두 개를 같은 축에서 함께 낸다:
  - `cell-lookup` : (카테고리, 난이도) 셀의 train 평균 accuracy. 임베딩을 전혀 쓰지 않는다
  - `A-shuffled`  : train 라벨을 무작위 치환한 A. 임베딩 구조만 남고 라벨 정보는 없앤다
A 가 `cell-lookup` 을 유의하게 넘지 못하면 A 의 성능은 코퍼스 설계의 부산물이다.
A-en 과 A-multi 가 사실상 같은 값이라는 8단계 관찰이 이미 그 방향을 시사한다.

    python scripts/evaluate.py --run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA,
    REPO,
    derive_seed,
    fmt_table,
    load_scale,
    load_yaml,
    read_json,
    read_rows,
    require,
    write_json,
    write_parquet,
)
from features import SCORERS  # noqa: E402

import pyarrow as pa  # noqa: E402

RESULTS = REPO / "results"
# 특징만 바뀌고 라벨은 공유하는 태그(같은 기반 모델의 다른 양자화). SPEC 13-AH.
FEATURE_ONLY_TAGS = {"q4"}
OUT = RESULTS / "09_results.parquet"
CURVE = RESULTS / "09_risk_coverage.parquet"
MD = RESULTS / "09_results.md"

RES_SCHEMA = pa.schema([
    ("condition", pa.string()), ("variant", pa.string()), ("unit", pa.string()),
    ("stratum", pa.string()), ("target_esc", pa.float32()), ("seed", pa.int32()),
    ("esc_rate", pa.float32()), ("local_rate", pa.float32()),
    ("escape_rate", pa.float32()), ("local_acc", pa.float32()),
    ("auroc", pa.float32()), ("ece", pa.float32()),
    ("local_ms", pa.float32()), ("n_esc", pa.int32()), ("n_eval", pa.int32()),
])

CURVE_SCHEMA = pa.schema([
    ("condition", pa.string()), ("variant", pa.string()), ("stratum", pa.string()),
    ("coverage", pa.float32()), ("escape_rate", pa.float32()), ("seed", pa.int32()),
])


def ece_equal_mass(scores, labels, n_bins: int) -> float:
    """SPEC §12: 15개 동일질량 구간, Σ (n_b/N)|conf_b - acc_b|. 라벨은 연속값 허용."""
    import numpy as np

    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    ok = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[ok], y[ok]
    if len(s) == 0:
        return float("nan")
    # 신뢰도로 쓰려면 [0,1] 이어야 한다. min-max 정규화한다(SPEC: B 는 cal min-max).
    lo, hi = float(s.min()), float(s.max())
    c = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
    order = np.argsort(c, kind="mergesort")
    c, y = c[order], y[order]
    tot = 0.0
    for b in np.array_split(np.arange(len(c)), n_bins):
        if len(b) == 0:
            continue
        tot += (len(b) / len(c)) * abs(float(c[b].mean()) - float(y[b].mean()))
    return tot


def cmd_run() -> int:
    import numpy as np

    from signal_probe import auroc, auroc_within

    RESULTS.mkdir(parents=True, exist_ok=True)
    pol = read_json(DATA / "08_policy.json")
    scale = load_scale()
    seeds = [int(x) for x in pol["seeds"]]
    n_bins = int(scale["ece"]["n_bins"])
    rtts = [float(x) for x in scale["server_rtt_ms"]]

    # ---------------- test 로드 (이 스크립트에서 단 1회) ----------------
    tcfg = load_yaml(REPO / "configs" / "test_path.yaml")
    test_ids_all = [r["id"] for r in read_rows(REPO / tcfg["test_ids_path"])]
    accs = {r["id"]: float(r["accuracy"]) for r in read_rows(DATA / "06_accuracy.parquet")}
    meta = {r["id"]: r for r in read_rows(DATA / "06_accuracy.parquet")}
    te = sorted(set(test_ids_all) & set(accs))
    print(f"test 로드 1회: 전체 {len(test_ids_all)}건 중 라벨 있는 {len(te)}건 평가")
    print("(judge 15,528 + clarify 3,288 생성의 채점은 에이전트 작업으로 미완 — 그 부분은 제외)")

    graded: dict[tuple[str, int], bool] = {}
    for r in read_rows(DATA / "06_graded_student.parquet"):
        if not r["broken"]:
            graded[(r["id"], int(r["seed"]))] = bool(r["correct"])
    feats: dict[tuple[str, int], dict] = {}
    ntok: dict[tuple[str, int], int] = {}
    for r in read_rows(DATA / "07_features_uqlm.parquet"):
        k = (r["id"], int(r["seed"]))
        feats[k] = {s: float(r[s]) for s in SCORERS}
        ntok[k] = int(r["n_tok"])

    z = np.load(DATA / "07_embeddings.npz", allow_pickle=True)
    pos = {str(i): n for n, i in enumerate(z["ids"])}
    tr = pol["train_ids"]
    tl = np.array([1.0 if b else 0.0 for b in pol["train_route_local"]])
    tp = np.array([pos[i] for i in tr])
    qp = np.array([pos[i] for i in te])
    k_nn = int(pol["knn_k"])
    strat = {i: ("det" if (accs[i] <= 1e-9 or accs[i] >= 1 - 1e-9) else "mix") for i in accs}

    def knn(D, lab):
        sim = D[qp] @ D[tp].T
        kk = min(k_nn, sim.shape[1])
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        return lab[idx].mean(axis=1)

    # ---------------- 조건별 항목 점수 (시드 무관) ----------------
    item_scores: dict[tuple[str, str], np.ndarray] = {}
    for name in [k for k in z.files if k != "ids"]:
        item_scores[("A", name)] = knn(z[name], tl)
    # E: 학습된 프롬프트 라우터. 동결된 LR 을 그대로 적용한다 (같은 임베딩·같은 라벨, 매핑만 학습).
    require("E" in pol, "정책에 조건 E 가 없다 — conditions.py --fit 을 다시 돌려라")
    for name in [k for k in z.files if k != "ids"]:
        e = pol["E"][name]
        Ze = (z[name][qp] - np.array(e["mean"])) / np.array(e["scale"])
        item_scores[("E", name)] = 1.0 / (1.0 + np.exp(-(Ze @ np.array(e["coef"])
                                                        + float(e["intercept"]))))
    rng = np.random.default_rng(derive_seed("a_shuffled"))
    item_scores[("A-shuffled", "A-multi")] = knn(z["A-multi"], rng.permutation(tl))
    # cell-lookup: 임베딩 없이 (카테고리, 난이도) train 평균
    cell: dict[tuple[str, str], list[float]] = {}
    for i in tr:
        m = meta[i]
        cell.setdefault((m["category"], m["difficulty"]), []).append(accs[i])
    gmean = float(np.mean([accs[i] for i in tr]))
    item_scores[("cell-lookup", "none")] = np.array(
        [float(np.mean(cell.get((meta[i]["category"], meta[i]["difficulty"]), [gmean])))
         for i in te])

    # G: 모델 **내부** 위임 헤드 (SPEC 13-L/13-M). student 자신이 프롬프트를 읽은 은닉상태에
    # cal 에서 동결한 2049 파라미터 선형 헤드를 적용한다. 여기서 적합하는 것은 없다 —
    # 계수·표준화·문턱 전부 cal 산출물을 읽어 쓴다. 은닉상태는 delegate_extract_test.py 가
    # 미리 뽑아 뒀다(라벨 미열람, 근거는 check_test_isolation.py 와 SPEC 13-M).
    HD_J, HD_W = DATA / "18_delegate_head.json", DATA / "18_delegate_head.npz"
    HD_F = DATA / "19_delegate_features_test.npz"
    if HD_J.exists() and HD_W.exists() and HD_F.exists():
        hj = read_json(HD_J)
        hw = np.load(HD_W)
        hf = np.load(HD_F, allow_pickle=False)
        pos = {str(v): n for n, v in enumerate(hf["ids"])}
        miss = [i for i in te if i not in pos]
        require(not miss, f"test 은닉상태 누락 {len(miss)}건 — delegate_extract_test.py 재실행")
        idx = np.array([pos[i] for i in te])
        Xh = (hf[hj["variant"]][idx] - hw["mean"]) / hw["scale"]
        p_fail = 1.0 / (1.0 + np.exp(-(Xh @ hw["coef"] + float(hw["intercept"][0]))))
        item_scores[("G", "internal_head")] = 1.0 - p_fail      # 높을수록 로컬 유지
        n_tok_test = {str(v): int(t) for v, t in zip(hf["ids"], hf["n_tokens"])}
        # G-mid: 실제 운용 구성(중간 층). 층 선택은 SPEC 13-M 의 비용 규칙이 했고, 여기서는
        # 그 구성의 test 수치를 보고한다 — 배포하는 것과 측정하는 것이 달라선 안 된다.
        HM_J, HM_W = DATA / "18_delegate_head_mid.json", DATA / "18_delegate_head_mid.npz"
        if HM_J.exists() and HM_W.exists():
            hjm = read_json(HM_J)
            hwm = np.load(HM_W)
            Xm = (hf[hjm["variant"]][idx] - hwm["mean"]) / hwm["scale"]
            pf_m = 1.0 / (1.0 + np.exp(-(Xm @ hwm["coef"] + float(hwm["intercept"][0]))))
            item_scores[("G-mid", "internal_head_mid")] = 1.0 - pf_m
        else:
            hjm = None
        # G-i8: **실제 배포 구성** — int8 은닉상태 + int8 재교정 문턱. 계수는 cal 동결분 그대로.
        HI_J = DATA / "18_delegate_head_mid_int8.json"
        HI_F = DATA / "19_delegate_features_test_int8.npz"
        if HI_J.exists() and HI_F.exists():
            hji = read_json(HI_J)
            hwi = np.load(DATA / "18_delegate_head_mid_int8.npz")
            fi = np.load(HI_F, allow_pickle=False)
            pos_i = {str(v): n for n, v in enumerate(fi["ids"])}
            idx_i = np.array([pos_i[i] for i in te])
            Xi = (fi[hji["variant"]][idx_i] - hwi["mean"]) / hwi["scale"]
            pf_i = 1.0 / (1.0 + np.exp(-(Xi @ hwi["coef"] + float(hwi["intercept"][0]))))
            item_scores[("G-i8", "deployed_int8")] = 1.0 - pf_i
        else:
            hji = None
        # G-gguf: **llama.cpp Q8_0 경로** — 진짜 int8 가중치로 돈다. 특징은 서버의 최종
        # 임베딩이고 헤드는 그 특징으로 cal 에서 다시 적합했다(SPEC 13-Q). 여기서 적합하는
        # 것은 없다 — 계수·표준화·문턱 전부 cal 산출물을 읽어 쓴다.
        HG_J = DATA / "26_llamacpp_head.json"
        HG_F = DATA / "19_delegate_features_test_llamacpp.npz"
        if HG_J.exists() and HG_F.exists():
            hjg = read_json(HG_J)
            hwg = np.load(DATA / "26_llamacpp_head.npz")
            fg = np.load(HG_F, allow_pickle=False)
            pos_g = {str(v): n for n, v in enumerate(fg["ids"])}
            if all(i in pos_g for i in te):
                idx_g = np.array([pos_g[i] for i in te])
                Xg = (fg["llamacpp"][idx_g] - hwg["mean"]) / hwg["scale"]
                pf_g = 1.0 / (1.0 + np.exp(-(Xg @ hwg["coef"]
                                            + float(hwg["intercept"][0]))))
                item_scores[("G-gguf", "llamacpp_q8")] = 1.0 - pf_g
            else:
                hjg = None
        else:
            hjg = None
    else:
        n_tok_test = {}
        hjm = None
        hji = None
        hjg = None

    # ---------------- 조건별 생성 점수 (시드 의존) ----------------
    b_name = pol["B"]["scorer"]
    mu = np.array(pol["D"]["mean"]); sg = np.array(pol["D"]["scale"])
    co = np.array(pol["D"]["coef"]); ic = float(pol["D"]["intercept"])

    def d_prob(f):
        x = (np.array([f[s] for s in SCORERS]) - mu) / sg
        return float(1.0 / (1.0 + np.exp(-(x @ co + ic))))

    gen_scores: dict[tuple[str, str], dict[int, np.ndarray]] = {("B", b_name): {}, ("D", "lr"): {}}
    require("F" in pol, "정책에 조건 F 가 없다 — conditions.py --fit 을 다시 돌려라")
    gen_scores[("F", "n_tokens_out")] = {}
    for sd in seeds:
        gen_scores[("B", b_name)][sd] = np.array(
            [feats[(i, sd)][b_name] if (i, sd) in feats else np.nan for i in te])
        # F: 짧은 응답 = 로컬 유지. 결정 비용 0 (생성하면 토큰 수는 공짜)
        gen_scores[("F", "n_tokens_out")][sd] = np.array(
            [-float(ntok[(i, sd)]) if (i, sd) in ntok else np.nan for i in te])
        gen_scores[("D", "lr")][sd] = np.array(
            [d_prob(feats[(i, sd)]) if (i, sd) in feats
             and all(feats[(i, sd)][s] == feats[(i, sd)][s] for s in SCORERS) else np.nan
             for i in te])

    # ---------------- 평가 ----------------
    def outcome(i, sd):
        c = graded.get((i, sd))
        return (1.0 if c else 0.0) if c is not None else float(accs[i] >= 0.5)

    def student_ms(i, sd):
        return pol["latency"]["prefill_ms"] + pol["latency"]["per_token_ms"] * ntok.get((i, sd), 0)

    def decision_ms(cond, variant):
        """escalate 여부를 정하기까지의 로컬 비용."""
        if cond in ("A", "A-shuffled"):
            key = variant if variant in pol["embed_ms_batch1"] else "A-multi"
            return pol["embed_ms_batch1"][key] + pol["knn_ms"]
        if cond == "E":
            # 임베딩 + 384차원 내적 1회 (측정값). kNN 보다 오히려 싸다.
            return pol["embed_ms_batch1"][variant] + float(pol["lr_router_ms"])
        if cond == "cell-lookup":
            return 0.001
        return 0.0        # B/C/D 의 결정 비용은 student 생성 자체다 (아래에서 계상)

    def total_local_ms(cond, variant, ids_, sd, keep_mask):
        """SPEC §11 의 비대칭을 정확히 계상한다.

        **사전 진단(A, cell-lookup)**: 결정 비용을 전 항목에 지불하고, student 생성은
        **로컬로 유지한 항목에만** 지불한다. escalate 항목은 student 를 돌리지 않는 것이
        사전 방식의 구조적 이득이다.
        **사후(B/C/D)**: student 생성을 전 항목에 지불한다. escalate 해도 환불이 없다.

        (2026-09-16 수정: 이전 구현은 A 에게 결정 비용만 청구하고 로컬 유지 항목의 student
        생성비를 누락해 비대칭을 181배로 과대 보고했다. 실제 비대칭은
        `esc_rate * student_gen_ms - decision_ms` 이며 이것이 Bouchard 의 structural cost 항이다.)
        """
        pre_hoc = cond in ("A", "A-shuffled", "E", "cell-lookup")
        tot = 0.0
        for n, i in enumerate(ids_):
            tot += decision_ms(cond, variant)
            if pre_hoc:
                if keep_mask[n]:
                    tot += student_ms(i, sd)
            else:
                tot += student_ms(i, sd)
        return tot

    rows: list[dict] = []
    curves: list[dict] = []
    all_conds = list(item_scores) + list(gen_scores)

    for cond, variant in all_conds:
        is_item = (cond, variant) in item_scores
        for sd in seeds:
            sc = item_scores[(cond, variant)] if is_item else gen_scores[(cond, variant)][sd]
            y = np.array([outcome(i, sd) for i in te])
            for st in ("all", "det", "mix"):
                sel = [n for n, i in enumerate(te) if st == "all" or strat[i] == st]
                if not sel:
                    continue
                s_, y_ = sc[sel], y[sel]
                ids_ = [te[n] for n in sel]
                good = ~np.isnan(s_)
                au = auroc(s_[good], y_[good].astype(bool)) if good.any() else float("nan")
                ee = ece_equal_mass(s_, y_, n_bins)
                # risk-coverage 곡선: 임계값 전 범위를 쓸어 곡선 자체를 낸다 (동작점 선택 없음)
                order = np.argsort(np.where(np.isnan(s_), -np.inf, s_), kind="mergesort")
                ys = y_[order]
                n = len(ys)
                for cut in range(0, n + 1):
                    # 하위 cut 개를 escalate -> coverage = (n-cut)/n
                    esc_wrong_kept = float((1.0 - ys[cut:]).sum())
                    curves.append({"condition": cond, "variant": variant, "stratum": st,
                                   "coverage": (n - cut) / n,
                                   "escape_rate": esc_wrong_kept / n, "seed": sd})
                # 동결 임계값에서의 동작점
                thr_src = (pol["A"][variant]["thresholds"] if cond == "A"
                           else pol["E"][variant]["thresholds"] if cond == "E"
                           else pol["A"]["A-multi"]["thresholds"] if cond == "A-shuffled"
                           else pol["B"]["thresholds"] if cond == "B"
                           else pol["F"]["thresholds"] if cond == "F"
                           else pol["D"]["thresholds"] if cond == "D"
                           # G 계열 문턱은 cal 에서 동결됐다 (18_delegate_head*.json)
                           else hj["thresholds"] if cond == "G"
                           else hjm["thresholds"] if cond == "G-mid"
                           else hji["thresholds"] if cond == "G-i8"
                           else hjg["thresholds"] if cond == "G-gguf" else None)
                targets = pol["targets"] if thr_src else [0.2]
                for t in targets:
                    if thr_src:
                        thr = float(thr_src[f"{t:.2f}"])
                    else:
                        thr = float(np.quantile(s_[good], t)) if good.any() else 0.0
                    esc = np.where(np.isnan(s_), True, s_ < thr)
                    keep = ~esc
                    rows.append({
                        "condition": cond, "variant": variant,
                        "unit": "item" if is_item else "gen", "stratum": st,
                        "target_esc": t, "seed": sd,
                        "esc_rate": float(esc.mean()), "local_rate": float(keep.mean()),
                        "escape_rate": float((1.0 - y_[keep]).sum() / n) if keep.any() else 0.0,
                        "local_acc": float(y_[keep].mean()) if keep.any() else float("nan"),
                        "auroc": au, "ece": ee,
                        "local_ms": float(total_local_ms(cond, variant, ids_, sd, keep)),
                        "n_esc": int(esc.sum()), "n_eval": n,
                    })

    # ---- 셀 내부 층화 분석: 오염 경로를 닫고 잔여 신호를 본다 ----
    # 사전 지정된 교란 통제이고 동작점 탐색이 아니다 — 동결된 정책·점수를 다시 집계할 뿐이다.
    # 2단 통제: (카테고리, 난이도) 와 (카테고리, 난이도, 언어).
    # 후자를 넣는 이유: 셀 안에 ko/en 이 반반 섞여 있고, A-en 은 한국어를 좁은 원뿔에 뭉치므로
    # (무관쌍 코사인 0.474, SPEC §9) 같은 언어 이웃을 끌어온다. 언어가 정답률과 상관되면
    # A-en 의 마진은 난이도 예측이 아니라 언어 탐지다. 통제해서 사라지는지 본다.
    keys2 = [f"{meta[i]['category']}/{meta[i]['difficulty']}" for i in te]
    keys3 = [f"{meta[i]['category']}/{meta[i]['difficulty']}/{meta[i]['lang']}" for i in te]
    wrows = []
    wnum: dict[tuple[str, str], tuple[float, float, float]] = {}
    for cond, variant in all_conds:
        is_item = (cond, variant) in item_scores
        v2, v3, p2, p3, u2, u3 = [], [], 0, 0, 0, 0
        for sd in seeds:
            sc = item_scores[(cond, variant)] if is_item else gen_scores[(cond, variant)][sd]
            yb = np.array([outcome(i, sd) for i in te], dtype=bool)
            a, p, u = auroc_within(sc, yb, keys2)
            b, q, w = auroc_within(sc, yb, keys3)
            if a == a:
                v2.append(a); p2, u2 = p, u
            if b == b:
                v3.append(b); p3, u3 = q, w
        pooled = [x["auroc"] for x in rows
                  if x["condition"] == cond and x["variant"] == variant
                  and x["stratum"] == "all" and abs(x["target_esc"] - 0.2) < 1e-6]
        pm = np.mean(pooled) if pooled else float("nan")
        wnum[(cond, variant)] = (float(pm),
                                 float(np.mean(v2)) if v2 else float("nan"),
                                 float(np.mean(v3)) if v3 else float("nan"))
        wrows.append([cond, variant, f"{pm:.3f}" if pm == pm else "—",
                      f"{np.mean(v2):.3f}±{np.std(v2):.3f}" if v2 else "—",
                      f"{np.mean(v3):.3f}±{np.std(v3):.3f}" if v3 else "—",
                      f"{np.mean(v3) - pm:+.3f}" if v3 and pm == pm else "—",
                      f"{p2}/{p3}", f"{u2}/{u3}"])
    print("\n=== 표 G. 층화 일치도 — 교란 경로를 닫은 뒤 남는 신호 ===")
    print("같은 셀 안의 (정답, 오답) 짝만 비교한다. cell-lookup 은 구성상 **정확히 0.500** (위생검사)")
    print("+언어 열은 (카테고리, 난이도, 언어) 3단 통제 — A-en 의 마진이 언어 탐지인지 가린다")
    blk_g = fmt_table(["조건", "변형", "전체", "셀내부", "+언어", "3단-전체 차이",
                       "짝수 2단/3단", "유효셀 2단/3단"], wrows)
    print(blk_g)

    # ---- 표 H. 라우터에 용량을 주면 이득이 얼마나 남는가 (kNN -> 학습 LR) ----
    # A 와 E 는 같은 임베딩·같은 라우팅 라벨을 쓰고 매핑만 다르다. 따라서 두 조건의 차이는
    # 순수하게 "매핑을 학습으로 바꾼 효과" 이고, 그 효과를 통제 전후로 쪼개면 학습이 실제
    # 난이도를 더 배운 것인지 셀 정체성을 더 착취한 것인지가 갈린다.
    hrows = []
    for name in sorted({v for c, v in all_conds if c == "E"}):
        if ("A", name) not in wnum:
            continue
        a0, _, a3 = wnum[("A", name)]
        e0, _, e3 = wnum[("E", name)]
        surv = (e3 - a3) / (e0 - a0) * 100.0 if abs(e0 - a0) > 1e-9 else float("nan")
        hrows.append([name, f"{a0:.3f}", f"{e0:.3f}", f"{e0-a0:+.3f}",
                      f"{a3:.3f}", f"{e3:.3f}", f"{e3-a3:+.3f}",
                      f"{surv:.0f}%" if surv == surv else "—"])
    blk_h = fmt_table(["인코더", "A 통제없음", "E 통제없음", "이득",
                       "A 3단통제", "E 3단통제", "이득", "생존율"], hrows)
    print("\n=== 표 H. kNN -> 학습 라우터: 이득 중 통제 하에서 살아남는 비율 ===")
    print("같은 임베딩·같은 라벨, 매핑만 다르다. 생존율이 낮으면 학습은 난이도가 아니라")
    print("셀 정체성을 더 잘 착취한 것이다 (13-C 의 오염을 라우터 용량이 증폭)")
    print(blk_h)

    # ---- 표 I. 통계적 검정력 — 이 표본으로 조건을 가를 수 있는가 ----
    # 조건 간 격차(0.003~0.010)가 시드 산포(±0.013~0.033)보다 작다. 그러면 "누가 이겼다" 를
    # 말할 수 없고, 남은 질문은 **몇 개의 라벨이 있으면 말할 수 있는가** 다. 그것이 judge 채점
    # 19,000건(에이전트 작업)을 살 값어치가 있는지를 정한다.
    #
    # 항목 단위 클러스터 부트스트랩: test 항목을 복원추출하고, 보고된 추정량(8시드 평균의
    # 3단 통제 AUROC)을 매 재표본에서 다시 계산한다. 시드가 아니라 **항목**이 불확실성의
    # 단위다 — 같은 항목의 8시드는 독립 표본이 아니다.
    B = int(scale.get("bootstrap", {}).get("n_power", 200))
    brng = np.random.default_rng(derive_seed("power_boot"))
    te_arr = np.array(te)
    pos_of = {i: n for n, i in enumerate(te)}
    boot: dict[tuple[str, str], list[float]] = {c: [] for c in all_conds}
    for _b in range(B):
        pick = brng.integers(0, len(te), len(te))
        ids_b = te_arr[pick]
        k3_b = [f"{meta[i]['category']}/{meta[i]['difficulty']}/{meta[i]['lang']}" for i in ids_b]
        for cond, variant in all_conds:
            is_item = (cond, variant) in item_scores
            vals = []
            for sd in seeds:
                sc = (item_scores[(cond, variant)] if is_item
                      else gen_scores[(cond, variant)][sd])[pick]
                yb = np.array([outcome(i, sd) for i in ids_b], dtype=bool)
                v = auroc_within(sc, yb, k3_b)[0]
                if v == v:
                    vals.append(v)
                if is_item:
                    break                      # 항목 수준 조건은 시드 불변
            if vals:
                boot[(cond, variant)].append(float(np.mean(vals)))

    irows = []
    ci_rows = []
    for cond, variant in all_conds:
        v = np.array(boot[(cond, variant)])
        if not len(v):
            continue
        lo, hi = np.quantile(v, [0.025, 0.975])
        irows.append([cond, variant, f"{wnum[(cond, variant)][2]:.3f}",
                      f"{v.std():.4f}", f"[{lo:.3f}, {hi:.3f}]", f"{hi - lo:.3f}"])
        ci_rows.append({"condition": cond, "variant": variant,
                        "ctrl_auroc": float(wnum[(cond, variant)][2]),
                        "boot_se": float(v.std()), "ci_lo": float(lo), "ci_hi": float(hi)})
    # 기계가 읽을 수 있게 덤프한다 — 하류 스크립트가 md 표를 손으로 옮겨 적으면 재실행 때
    # 낡은 값을 보고한다.
    CI_MAP = RESULTS / "09_ctrl_auroc_ci.parquet"
    write_parquet(ci_rows, CI_MAP, pa.schema([
        ("condition", pa.string()), ("variant", pa.string()),
        ("ctrl_auroc", pa.float64()), ("boot_se", pa.float64()),
        ("ci_lo", pa.float64()), ("ci_hi", pa.float64())]))
    print(f"3단 통제 CI {len(ci_rows)}행 -> {CI_MAP}")
    blk_i = fmt_table(["조건", "변형", "3단통제", "부트 SE", "95% CI", "CI 폭"], irows)
    print(f"\n=== 표 I. 3단 통제 AUROC 의 항목 단위 부트스트랩 ({B}회 재표본) ===")
    print("불확실성의 단위는 시드가 아니라 항목이다 — 같은 항목의 8시드는 독립 표본이 아니다")
    print(blk_i)

    # 주요 대비의 짝지은 차이 + 필요 표본 크기
    PAIRS = [(("D", "lr"), ("A", "A-en")), (("D", "lr"), ("E", "A-en")),
             (("D", "lr"), ("B", "mean_token_negentropy")),
             (("E", "A-en"), ("A", "A-en")), (("D", "lr"), ("A", "A-multi")),
             (("B", "mean_token_negentropy"), ("F", "n_tokens_out"))]
    jrows = []
    n_items = len(te)
    for a_, b_ in PAIRS:
        if a_ not in boot or b_ not in boot or not boot[a_] or not boot[b_]:
            continue
        n = min(len(boot[a_]), len(boot[b_]))
        d = np.array(boot[a_][:n]) - np.array(boot[b_][:n])   # 같은 재표본끼리 = 짝지은 차이
        obs = wnum[a_][2] - wnum[b_][2]
        lo, hi = np.quantile(d, [0.025, 0.975])
        sig = "예" if lo * hi > 0 else "아니오"
        # CI 폭 ~ 1/sqrt(n) 이므로, 관측 차이를 유의하게 만들려면
        # n_need = n_now * (1.96*SE / |obs|)^2
        need = (1.96 * d.std() / abs(obs)) ** 2 * n_items if abs(obs) > 1e-9 else float("inf")
        jrows.append([f"{a_[0]}/{a_[1]}", f"{b_[0]}/{b_[1]}", f"{obs:+.4f}",
                      f"[{lo:+.3f}, {hi:+.3f}]", sig,
                      f"{need:,.0f}" if need < 1e7 else ">10M",
                      f"{need / n_items:,.0f}x" if need < 1e7 else "—"])
    blk_j = fmt_table(["조건 A", "조건 B", "관측 차이", "차이 95% CI", "유의",
                       "필요 라벨 항목", "현재 대비"], jrows)
    print("\n=== 표 J. 주요 대비 — 이 표본으로 가를 수 있는가, 못 가르면 몇 개가 필요한가 ===")
    print(f"현재 test 라벨 항목 {n_items}건. 코퍼스 전체(1,250)를 채점해도 배율 1.9x 가 상한이다")
    print(blk_j)

    write_parquet(rows, OUT, RES_SCHEMA)

    # ---- escalate 집합 덤프: 5단계 server 요청 대상을 여기서만 고른다 ----
    # run_server.py 는 이 파일의 id 목록만 읽는다. test 경로를 읽는 스크립트를 늘리지 않기 위해
    # **선택도 evaluate.py 안에서** 한다.
    ESC_POLICIES = [("E", "A-en"), ("E", "A-multi"), ("D", "lr"),
                    ("B", b_name), ("A", "A-en"), ("G", "internal_head"),
                    ("G-mid", "internal_head_mid"), ("G-i8", "deployed_int8"),
                    ("G-gguf", "llamacpp_q8")]
    esc_rows = []
    for cond, variant in ESC_POLICIES:
        if (cond, variant) not in item_scores and (cond, variant) not in gen_scores:
            continue
        is_item = (cond, variant) in item_scores
        thr_src = (pol["A"][variant]["thresholds"] if cond == "A"
                   else pol["E"][variant]["thresholds"] if cond == "E"
                   else pol["B"]["thresholds"] if cond == "B"
                   else pol["D"]["thresholds"] if cond == "D"
                   else hj["thresholds"] if cond == "G"
                   else hjm["thresholds"] if cond == "G-mid"
                   else hji["thresholds"] if cond == "G-i8"
                   else hjg["thresholds"] if cond == "G-gguf" else None)
        for t in (0.10, 0.20, 0.30):
            thr = float(thr_src[f"{t:.2f}"])
            # 생성 수준 정책은 시드 다수결로 항목 결정을 만든다 (배포는 항목 단위 결정이다)
            for n, i in enumerate(te):
                if is_item:
                    e = bool(item_scores[(cond, variant)][n] < thr)
                else:
                    votes = [bool(gen_scores[(cond, variant)][sd][n] < thr) for sd in seeds]
                    e = sum(votes) * 2 > len(votes)
                esc_rows.append({"id": i, "condition": cond, "variant": variant,
                                 "target_esc": float(t), "escalated": e})
    ESC_MAP = RESULTS / "09_escalation_map.parquet"
    write_parquet(esc_rows, ESC_MAP, pa.schema([
        ("id", pa.string()), ("condition", pa.string()), ("variant", pa.string()),
        ("target_esc", pa.float32()), ("escalated", pa.bool_())]))

    # 항목별 점수 덤프: 실제 위임 라벨(server 실측)로 조건부 재측정을 하려면 결정 여부가
    # 아니라 **점수**가 필요하다. 생성 수준 조건은 시드 평균을 항목 점수로 쓴다.
    sc_rows = []
    for cond, variant in all_conds:
        is_item = (cond, variant) in item_scores
        for n, i in enumerate(te):
            if is_item:
                v = float(item_scores[(cond, variant)][n])
            else:
                vals = [gen_scores[(cond, variant)][sd][n] for sd in seeds]
                vals = [x for x in vals if x == x]
                v = float(np.mean(vals)) if vals else float("nan")
            sc_rows.append({"id": i, "condition": cond, "variant": variant, "score": v})
    SC_MAP = RESULTS / "09_item_scores.parquet"
    write_parquet(sc_rows, SC_MAP, pa.schema([
        ("id", pa.string()), ("condition", pa.string()), ("variant", pa.string()),
        ("score", pa.float32())]))
    print(f"항목 점수 {len(sc_rows)}행 -> {SC_MAP}")

    # server 요청 id: 선도 사전 정책의 esc20% escalate 집합에서 셀 층화로 N 개
    N_REQ = int(scale.get("server", {}).get("n_validation_requests", 60))
    pick_pol = ("E", "A-en")
    cand = [r["id"] for r in esc_rows
            if (r["condition"], r["variant"]) == pick_pol
            and abs(r["target_esc"] - 0.20) < 1e-6 and r["escalated"]]
    bycell: dict[str, list[str]] = {}
    for i in cand:
        bycell.setdefault(f"{meta[i]['category']}/{meta[i]['difficulty']}", []).append(i)
    srng = np.random.default_rng(derive_seed("server_req_pick"))
    chosen: list[str] = []
    ck = sorted(bycell)
    while len(chosen) < min(N_REQ, len(cand)):
        grew = False
        for c in ck:
            pool = [x for x in bycell[c] if x not in set(chosen)]
            if pool and len(chosen) < N_REQ:
                chosen.append(pool[int(srng.integers(len(pool)))]); grew = True
        if not grew:
            break
    REQ_IDS = RESULTS / "09_server_request_ids.txt"
    REQ_IDS.write_text("\n".join(sorted(chosen)) + "\n", encoding="utf-8")
    print(f"\nescalate 지도 {len(esc_rows)}행 -> {ESC_MAP}")
    print(f"server 요청 대상 {len(chosen)}개 ({pick_pol[0]}/{pick_pol[1]} @ esc20% escalate "
          f"{len(cand)}개 중 셀 층화) -> {REQ_IDS}")
    write_parquet(curves, CURVE, CURVE_SCHEMA)
    print(f"{len(rows)}행 -> {OUT}\n{len(curves)}행 -> {CURVE}")
    return report(rows, curves, rtts, pol, blk_g, blk_h, blk_i, blk_j)


def report(rows, curves, rtts, pol, blk_g="", blk_h="", blk_i="", blk_j="") -> int:
    import numpy as np

    lines: list[str] = ["# 9단계 결과 — 온디바이스 캐스케이드 조건 비교", ""]

    def sel(cond, variant, st="all", t=None, unit=None):
        out = [r for r in rows if r["condition"] == cond and r["variant"] == variant
               and r["stratum"] == st and (t is None or abs(r["target_esc"] - t) < 1e-6)
               and (unit is None or r["unit"] == unit)]
        return out

    def ms(v):
        return (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"),) * 2

    keys = sorted({(r["condition"], r["variant"]) for r in rows})

    # ---- 표 A. 단위를 맞춘 AUROC ----
    t = [["조건", "변형", "단위", "AUROC(전체)", "결정적", "혼합"]]
    for c, v in keys:
        u = sel(c, v, "all", 0.2)[0]["unit"] if sel(c, v, "all", 0.2) else "?"
        r = [c, v, u]
        for st in ("all", "det", "mix"):
            m, s = ms([x["auroc"] for x in sel(c, v, st, 0.2)])
            r.append("—" if m != m else f"{m:.3f}±{s:.3f}")
        t.append(r)
    blk = fmt_table(t[0], t[1:])
    print("\n=== 표 A. 분별력 AUROC (단위 병기, 8-복제) ===")
    print("A 계열은 항목 수준(시드 무관), B/D 는 생성 수준. 단위가 다르면 직접 비교하지 않는다")
    print(blk)
    lines += ["## 표 A. 분별력 AUROC", "", "```", blk, "```", ""]

    # ---- 표 B. A 가 인공물인지 ----
    print("\n=== 표 B. 사전 라우터의 이득이 코퍼스 인공물인가 ===")
    bb = []
    for c, v in keys:
        if c not in ("A", "E", "A-shuffled", "cell-lookup"):
            continue
        m, s = ms([x["auroc"] for x in sel(c, v, "all", 0.2)])
        bb.append([f"{c}/{v}", f"{m:.3f}±{s:.3f}"])
    blk = fmt_table(["예측기", "항목 수준 AUROC"], bb)
    print("사전 라우터가 cell-lookup(오라클 라벨 조회)을 못 넘으면 셀 평균 조회에 불과하고,")
    print("A-shuffled 는 0.5 여야 한다. 학습된 라우터 E 도 같은 상한에 걸어 본다")
    print(blk)
    lines += ["## 표 B. 사전 라우터의 인공물 검증", "", "```", blk, "```", ""]

    # ---- 표 C. 주 결과: 동일 escalate 율에서 오탈출률 ----
    tg = [0.0, 0.10, 0.20, 0.30, 0.50]
    head = ["조건", "변형"] + [f"esc {int(100*x)}%" for x in tg]
    cb = []
    for c, v in keys:
        r = [c, v]
        for x in tg:
            m, s = ms([q["escape_rate"] for q in sel(c, v, "all", x)])
            e, _ = ms([q["esc_rate"] for q in sel(c, v, "all", x)])
            r.append("—" if m != m else f"{100*m:.1f}±{100*s:.1f} ({100*e:.0f}%)")
        cb.append(r)
    blk = fmt_table(head, cb)
    print("\n=== 표 C. 주 결과 — 오탈출률 % (괄호 = 실제 달성 escalate 율) ===")
    print("임계값은 cal 에서 동결했으므로 달성 escalate 율은 목표와 다를 수 있다. 낮은 오탈출률이 우수")
    print(blk)
    lines += ["## 표 C. 주 결과 — 오탈출률", "", "```", blk, "```", ""]

    # ---- 표 D. risk-coverage AUC (임계값 무관) ----
    print("\n=== 표 D. risk-coverage AUC (임계값 선택 없는 정면 비교) ===")
    db = []
    for c, v in keys:
        for st in ("all", "det", "mix"):
            pts = [(q["coverage"], q["escape_rate"]) for q in curves
                   if q["condition"] == c and q["variant"] == v and q["stratum"] == st]
            if not pts:
                continue
            agg: dict[float, list[float]] = {}
            for cov, er in pts:
                agg.setdefault(round(cov, 4), []).append(er)
            xs = sorted(agg)
            ys = [float(np.mean(agg[x])) for x in xs]
            # numpy 는 vLLM 제약으로 1.26.4 고정이므로 np.trapz 를 쓴다
            # (np.trapezoid 는 numpy 2.0 에서 도입된 새 이름이라 여기서는 없다).
            auc = float(np.trapz(ys, xs)) if len(xs) > 1 else float("nan")
            if st == "all":
                db.append([c, v, f"{auc:.4f}"])
            else:
                db[-1].append(f"{auc:.4f}")
    blk = fmt_table(["조건", "변형", "AUC(전체)", "결정적", "혼합"], db)
    print("곡선 아래 면적 = 커버리지 전 구간 평균 오탈출률. **낮을수록 우수**")
    print(blk)
    lines += ["## 표 D. risk-coverage AUC (낮을수록 우수)", "", "```", blk, "```", ""]

    # ---- 표 E. 비용 비대칭 ----
    print("\n=== 표 E. 로컬 계산비 비대칭 (SPEC §11 의 측정 대상) ===")
    eb = []
    for c, v in keys:
        s_ = sel(c, v, "all", 0.2)
        if not s_:
            continue
        m, _ = ms([q["local_ms"] for q in s_])
        n = s_[0]["n_eval"]
        ne, _ = ms([q["n_esc"] for q in s_])
        row = [c, v, f"{m/n:.2f}", f"{m/1000:.1f}"]
        for rtt in rtts:
            row.append(f"{(m + ne*rtt)/1000:.1f}")
        eb.append(row)
    blk = fmt_table(["조건", "변형", "로컬 ms/항목", "로컬 합계 s"]
                    + [f"+RTT {int(r)}ms 합계 s" for r in rtts], eb)
    print("server 생성 시간은 미실행이라 포함하지 않는다. RTT 만 더한 하한이다")
    print(blk)
    lines += ["## 표 E. 비용 비대칭", "", "```", blk, "```", ""]

    # ---- 표 F. 정확도 (a_s 스윕) ----
    print("\n=== 표 F. 정확도 % (server 정확도 a_s 스윕, esc 20% 목표) ===")
    fb = []
    for c, v in keys:
        s_ = sel(c, v, "all", 0.2)
        if not s_:
            continue
        m, _ = ms([q["escape_rate"] for q in s_])
        e, _ = ms([q["esc_rate"] for q in s_])
        fb.append([c, v, f"{100*e:.0f}%"] + [f"{100*(1-m-e*(1-a)):.1f}" for a in (0.7, 0.85, 1.0)])
    blk = fmt_table(["조건", "변형", "esc율", "a_s=0.70", "a_s=0.85", "a_s=1.00"], fb)
    print("정확도 = 1 - 오탈출률 - esc율*(1-a_s).  ECE 는 표 A 와 같은 축이라 별도 표 생략")
    print(blk)
    lines += ["## 표 F. 정확도 (a_s 스윕)", "", "```", blk, "```", ""]

    lines += ["## 표 G. 셀 내부 층화 일치도 (교란 통제)", "",
              "같은 (카테고리, 난이도) 셀 안의 (정답, 오답) 짝만 비교. cell-lookup 은 구성상 0.500.",
              "", "```", blk_g, "```", ""]
    lines += ["## 표 H. kNN -> 학습 라우터: 이득 중 통제 하 생존율", "",
              "A 와 E 는 같은 임베딩·같은 라우팅 라벨을 쓰고 **매핑만** 다르다.",
              "생존율이 낮으면 학습은 난이도가 아니라 셀 정체성을 더 잘 착취한 것이다.",
              "", "```", blk_h, "```", ""]
    lines += ["## 표 I. 3단 통제 AUROC 의 항목 단위 부트스트랩", "",
              "불확실성의 단위는 시드가 아니라 **항목**이다 — 같은 항목의 8시드는 독립 표본이 아니다.",
              "", "```", blk_i, "```", ""]
    lines += ["## 표 J. 주요 대비 — 가를 수 있는가, 못 가르면 몇 개가 필요한가", "",
              "`필요 라벨 항목` = 관측된 차이를 95% CI 로 유의하게 만드는 데 필요한 test 라벨 항목 수.",
              "코퍼스 전체(test 1,250)를 채점해도 현재 대비 **1.9배**가 상한이다.",
              "", "```", blk_j, "```", ""]
    lines += ["## 정책 출처 (읽는 법)", "",
              f"- 조건 B 스코어러 `{pol['B']['scorer']}` 는 "
              f"**{pol['B'].get('selection_criterion', 'cal AUROC')}** 로 골랐다 (SPEC 13-E).",
              f"  통제 없는 cal AUROC 1위는 `{pol['B'].get('scorer_uncontrolled_top', '-')}` 였다.",
              "- 조건 E 의 정규화 강도 C 도 **같은 통제 기준**으로 골랐다: "
              + ", ".join(f"{k} C={pol['E'][k]['C']:g}"
                          f"(통제 전 최적 {pol['E'][k].get('C_uncontrolled_best', float('nan')):g})"
                          for k in sorted(pol["E"])) + ".",
              "- 따라서 **표 A·C·D 의 통제 없는 수치로 B·E 를 평가하면 안 된다.** 그 수치는 코퍼스의",
              "  셀 구조에 무임승차하는 예측기를 보상한다(13-C). 판정은 표 G 의 `+언어` 열과 표 H 다.",
              "- A 계열과 cell-lookup 은 선택 없이 정의로 고정된 조건이라 스코어러·C 교체와 무관하다.",
              f"- **조건 F(길이 기준선) 주의**: 점수가 정수이고 절단 상한 "
              f"{pol['F'].get('cap_tokens', 0)}토큰에 동점 덩어리가 "
              f"{100*pol['F'].get('tie_mass_at_cap', 0):.1f}% 있다. 그래서 표 C 의 낮은 "
              "escalate 율은 **실현 불가**해 esc 0% 로 퇴화하고(F 과소평가), 표 D 의 "
              "risk-coverage 는 동점을 임의로 끊어 **낙관적 상한**이다(F 과대평가).",
              "  F 의 공정한 판정은 표 G 의 `+언어` 열이다.", ""]
    lines += ["## 미완 항목", "",
              "- server 생성 11,250건(train 2,500 + test 1,250 x k=3)은 에이전트 작업으로 미실행.",
              "  따라서 정확도의 **절대 수준**은 a_s 스윕으로만 제시된다. 조건 간 순서는 영향받지 않는다.",
              "- judge 15,528 + clarify 3,288 생성의 채점도 에이전트 작업으로 미완.",
              f"  이 표들은 exact/contains 채점 항목만 사용한다 (test {rows[0]['n_eval']}건).", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nmarkdown -> {MD}")
    return 0


def cmd_sidecar(tag: str) -> int:
    """0.6B(sidecar) arm 의 test 평가 — **좁은 모드**. SPEC 13-Y.

    왜 `cmd_run` 에 조건을 더하지 않는가: `cmd_run` 은 `06_accuracy.parquet`(1.7B 라벨) 하나로
    A~G 전 조건을 평가한다. 0.6B 헤드가 예측하는 것은 "**이** student 가 틀릴까" 이므로
    라벨이 `06_accuracy.sc06.parquet` 이어야 하고, 그 라벨로 B/C/D 를 평가하면 0.6B 의
    logprob 특징이 없으므로 라벨 출처가 섞인다. 그래서 이 모드는 **sidecar 헤드 하나만**
    계산하고 태그 붙은 파일로 쓴다. 이 파일은 이미 ALLOWED 예외이므로 게이트를 넓히지 않는다.

    여기서 적합하는 것은 없다 — 계수·표준화·문턱 전부 cal 산출물(13-X)을 읽어 쓴다.
    """
    import numpy as np

    from signal_probe import auroc, auroc_within

    RESULTS.mkdir(parents=True, exist_ok=True)
    # 입력 접미사와 출력 접미사를 분리한다. tag='base' 는 1.7B(접미사 없음) 파일을 읽되
    # 출력은 `.base` 로 쓴다 — cmd_run 이 만든 09_escalation_map.parquet 등을 덮으면 안 된다.
    in_sfx = "" if tag == "base" else f".{tag}"
    sfx = f".{tag}"
    # **라벨 접미사는 따로 둔다.** `q4` 처럼 같은 기반 모델을 다른 양자화로 임베딩만 바꾼
    # 태그는 라벨("동결 student 가 맞혔는가")이 그대로여서 접미사 없는 라벨을 읽어야 한다.
    # 이걸 합쳐 두면 존재하지 않는 06_accuracy.q4.parquet 를 찾다 죽는다.
    lab_sfx = "" if tag in FEATURE_ONLY_TAGS else in_sfx
    pol = read_json(DATA / "08_policy.json")

    HJ = DATA / f"26_llamacpp_head{in_sfx}.json"
    HW = DATA / f"26_llamacpp_head{in_sfx}.npz"
    HF = DATA / f"19_delegate_features_test_llamacpp{in_sfx}.npz"
    ACC = DATA / f"06_accuracy{lab_sfx}.parquet"
    GRD = DATA / f"06_graded_student{lab_sfx}.parquet"
    for p in (HJ, HW, HF, ACC, GRD):
        require(p.exists(), f"{p} 가 없다")

    tcfg = load_yaml(REPO / "configs" / "test_path.yaml")
    test_ids_all = [r["id"] for r in read_rows(REPO / tcfg["test_ids_path"])]
    accs = {r["id"]: float(r["accuracy"]) for r in read_rows(ACC)}
    meta = {r["id"]: r for r in read_rows(ACC)}
    te = sorted(set(test_ids_all) & set(accs))
    require(te, "평가할 test 항목이 없다")

    graded: dict[tuple[str, int], bool] = {}
    for r in read_rows(GRD):
        if not r["broken"]:
            graded[(r["id"], int(r["seed"]))] = bool(r["correct"])
    seeds = sorted({sd for _, sd in graded})

    hj, hw = read_json(HJ), np.load(HW)
    hf = np.load(HF, allow_pickle=False)
    pos = {str(v): n for n, v in enumerate(hf["ids"])}
    miss = [i for i in te if i not in pos]
    require(not miss, f"test 특징 누락 {len(miss)}건")
    idx = np.array([pos[i] for i in te])
    X = (hf["llamacpp"][idx] - hw["mean"]) / hw["scale"]
    p_fail = 1.0 / (1.0 + np.exp(-(X @ hw["coef"] + float(hw["intercept"][0]))))
    score = 1.0 - p_fail                       # 높을수록 로컬 유지 (cal 규약과 동일)

    def outcome(i, sd):
        c = graded.get((i, sd))
        return (1.0 if c else 0.0) if c is not None else float(accs[i] >= 0.5)

    cell3 = [f"{meta[i]['category']}/{meta[i]['difficulty']}/{meta[i]['lang']}" for i in te]
    # `cmd_run` 과 **같은 정의**를 쓴다: 점수는 시드 불변이지만 결과(outcome)는 시드마다
    # 다르므로, 시드별 AUROC 를 재고 **8시드 평균**을 점 추정으로 삼는다. 이 정의로 1.7B 를
    # 돌리면 09_ctrl_auroc_ci.parquet 의 G-gguf 0.7100 이 재현돼야 한다(교차 확인).
    ys = {sd: np.array([outcome(i, sd) for i in te], dtype=bool) for sd in seeds}
    plain = float(np.mean([auroc(score, ys[sd]) for sd in seeds]))
    per_seed, n_pair, n_cell = [], 0, 0
    for sd in seeds:
        a, p, u = auroc_within(score, ys[sd], cell3)
        if a == a:
            per_seed.append(a)
            n_pair, n_cell = p, u
    ctrl = float(np.mean(per_seed))

    B = 400
    brng = np.random.default_rng(derive_seed("sidecar_boot"))
    te_arr = np.array(te)
    vals = []
    for _b in range(B):
        pick = brng.integers(0, len(te), len(te))
        ids_b = te_arr[pick]
        k3 = [f"{meta[i]['category']}/{meta[i]['difficulty']}/{meta[i]['lang']}" for i in ids_b]
        sb = score[pick]
        vs = [auroc_within(sb, ys[sd][pick], k3)[0] for sd in seeds]
        vs = [x for x in vs if x == x]
        if vs:
            vals.append(float(np.mean(vs)))
    v = np.array(vals)
    lo, hi = np.quantile(v, [0.025, 0.975])

    # 로컬 전용 정확도·결과 배열도 시드 평균으로 둔다 (정확도 축의 단위를 통일한다)
    y_mean = np.mean([ys[sd].astype(float) for sd in seeds], axis=0)

    print(f"[sidecar {tag}] test 라벨 있는 {len(te)}건 / 시드 {len(seeds)}개")
    print(f"  통제 없는 AUROC {plain:.4f} / **3단 통제 {ctrl:.4f}** "
          f"(셀 {n_cell}, 짝 {n_pair})")
    print(f"  부트 95% CI [{lo:.4f}, {hi:.4f}] (SE {v.std():.4f}, {len(v)}회)")

    # ---- escalation map: cal 동결 문턱을 그대로 적용한다 ----
    a_s = float(pol.get("server_acc_measured", 0.967))
    local_only = float(np.mean(y_mean))
    oracle_ceiling = None
    rows, esc_rows = [], []
    for t, thr in sorted(hj["thresholds"].items(), key=lambda kv: float(kv[0])):
        esc = score < float(thr)
        keep = ~esc
        n_esc = int(esc.sum())
        esc_rate = n_esc / len(te)
        loc_acc = float(np.mean(y_mean[keep])) if keep.any() else float("nan")
        cascade = float(np.mean(np.where(esc, a_s, y_mean)))
        # 오라클 상한: 같은 esc 예산을 '가장 못 맞히는 항목부터' 썼다면
        order = np.argsort(y_mean, kind="stable")                # 정확도 낮은 것 먼저
        orc = np.zeros(len(te), dtype=bool)
        orc[order[:n_esc]] = True
        oracle = float(np.mean(np.where(orc, a_s, y_mean)))
        if abs(float(t) - 0.20) < 1e-9:
            oracle_ceiling = oracle
        eff = (cascade - local_only) / (oracle - local_only) if oracle > local_only else float("nan")
        rows.append({"target_esc": float(t), "threshold": float(thr),
                     "esc_rate": esc_rate, "n_esc": n_esc,
                     "local_acc_kept": loc_acc, "local_only_acc": local_only,
                     "cascade_acc": cascade, "oracle_acc": oracle,
                     "routing_efficiency": eff, "server_acc": a_s})
        for n, i in enumerate(te):
            esc_rows.append({"id": i, "condition": f"G-{tag}", "variant": "llamacpp_q8",
                             "target_esc": float(t), "escalated": bool(esc[n])})

    OUT_S = RESULTS / f"09_sidecar_curve{sfx}.parquet"
    write_parquet(rows, OUT_S, pa.schema([
        ("target_esc", pa.float64()), ("threshold", pa.float64()),
        ("esc_rate", pa.float64()), ("n_esc", pa.int64()),
        ("local_acc_kept", pa.float64()), ("local_only_acc", pa.float64()),
        ("cascade_acc", pa.float64()), ("oracle_acc", pa.float64()),
        ("routing_efficiency", pa.float64()), ("server_acc", pa.float64())]))
    ESC_S = RESULTS / f"09_escalation_map{sfx}.parquet"
    write_parquet(esc_rows, ESC_S, pa.schema([
        ("id", pa.string()), ("condition", pa.string()), ("variant", pa.string()),
        ("target_esc", pa.float32()), ("escalated", pa.bool_())]))
    CI_S = RESULTS / f"09_ctrl_auroc_ci{sfx}.parquet"
    write_parquet([{"condition": f"G-{tag}", "variant": "llamacpp_q8",
                    "ctrl_auroc": float(ctrl), "plain_auroc": float(plain),
                    "boot_se": float(v.std()), "ci_lo": float(lo), "ci_hi": float(hi),
                    "n_cell": int(n_cell), "n_pair": int(n_pair), "n_eval": len(te)}], CI_S,
                  pa.schema([("condition", pa.string()), ("variant", pa.string()),
                             ("ctrl_auroc", pa.float64()), ("plain_auroc", pa.float64()),
                             ("boot_se", pa.float64()), ("ci_lo", pa.float64()),
                             ("ci_hi", pa.float64()), ("n_cell", pa.int64()),
                             ("n_pair", pa.int64()), ("n_eval", pa.int64())]))
    write_json({"tag": tag, "n_eval": len(te), "local_only_acc": local_only,
                "server_acc_used": a_s, "oracle_at_esc20": oracle_ceiling,
                "ctrl_auroc": float(ctrl), "ctrl_ci": [float(lo), float(hi)],
                "plain_auroc": float(plain), "seeds": seeds,
                "label_source": ACC.name, "head": HJ.name,
                "note": "cal 동결 문턱을 그대로 적용했다. test 위에서 적합·선택한 것은 없다"},
               RESULTS / f"09_sidecar_summary{sfx}.json")
    print(f"  로컬 전용 정확도 {local_only:.4f} / 서버 정확도 {a_s:.3f} 고정")
    print(f"  -> {OUT_S}\n  -> {ESC_S}\n  -> {CI_S}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="9단계 평가 (test 1회 로드)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sidecar-tag", default="",
                    help="0.6B arm 의 좁은 평가 모드 (예: sc06). 태그 붙은 라벨·헤드·특징만 쓴다")
    a = ap.parse_args()
    if a.sidecar_tag:
        return cmd_sidecar(a.sidecar_tag)
    if not a.run:
        ap.error("--run 또는 --sidecar-tag")
    return cmd_run()


if __name__ == "__main__":
    sys.exit(main())
