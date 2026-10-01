"""Stage 8: **진짜 int8 경로.** llama.cpp Q8_0 에서 위임 특징을 뽑는다.

왜 새로 뽑아야 하는가. 파이썬 산출물은 int8 을 적재 시 bf16 으로 역양자화해 RAM 이 5,622MB
였다(13-P). llama.cpp 는 Q8_0 가중치를 그대로 두고 연산하므로 RSS 2,400MB 다 — int8 절감이
실제로 나온다. 대신 llama.cpp 는 **중간 층 은닉상태를 노출하지 않는다.** 노출하는 것은
최종 임베딩(`output_norm(h_28)`)이고, 그것은 헤드가 적합된 특징(`hidden_states[14]`, norm 이전)과
다른 공간이다. 그래서 **그 특징으로 헤드를 다시 적합한다.**

이 교체는 오히려 유리하다: 최종 층은 cal 통제 0.7530 / test 0.710 으로 중간 층(0.7420 / 0.685)
보다 좋았고, 중간 층을 고른 이유는 파이썬 경로의 프리필 비용뿐이었다. llama.cpp 는 28층
프리필이 싸므로 그 제약이 사라진다.

절차는 기존과 **동일**하다: train 에서 적합, C 는 cal 3단 통제 AUROC 로 선택, 문턱은 cal
분위수로 동결, test 는 열지 않는다.

  --extract --split train,cal   특징 추출 (llama-server 필요)
  --fit                          헤드 적합 + 문턱 동결

**0.6B arm (SPEC 13-U/13-W)**: `--tag sc06 --model sidecar` 를 주면 특징·헤드 경로가
`.sc06` 으로 분리되고 라벨을 `06_accuracy.sc06.parquet` 에서 읽는다. 기본값은 손대지 않았으므로
1.7B 경로는 플래그 없이 그대로 재현된다. 8081 에 어떤 모델이 떠 있는지는 이 파일이 알 수
없으므로, 호출하는 쪽이 서버가 맞는 GGUF 인지 확인할 책임을 진다(`run_extract06.sh` 가 한다).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, RESULTS, fmt_table, read_json, read_rows, require, write_json  # noqa: E402

FEAT = DATA / "26_llamacpp_features.npz"
HEAD_J = DATA / "26_llamacpp_head.json"
HEAD_W = DATA / "26_llamacpp_head.npz"
ACC = DATA / "06_accuracy.parquet"
MODEL_KEY = "primary"
TAG = ""
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
FAIL_THR = 0.75
URL = "http://127.0.0.1:8081/embedding"
# 특징만 바뀌고 라벨은 그대로인 태그. 같은 기반 모델을 다른 양자화로 임베딩하는 경우다.
FEATURE_ONLY = {"q4"}


def apply_tag(tag: str, model: str) -> None:
    """0.6B arm 용 경로·라벨·모델 교체. 태그가 없으면 아무것도 바뀌지 않는다.

    **특징만 바뀌는 태그**(FEATURE_ONLY)는 라벨을 갈아치우지 않는다. 예: `q4` 는 같은
    Qwen3-1.7B 를 다른 양자화로 임베딩할 뿐이므로, 라벨("동결 student 가 맞혔는가")은
    양자화와 무관하고 `06_accuracy.parquet` 를 그대로 써야 한다. 라벨까지 갈면 존재하지 않는
    `06_accuracy.q4.parquet` 를 찾다 죽는다.
    """
    global FEAT, HEAD_J, HEAD_W, ACC, MODEL_KEY, TAG
    TAG = tag
    MODEL_KEY = model
    if not tag:
        return
    sfx = f".{tag}"
    FEAT = DATA / f"26_llamacpp_features{sfx}.npz"
    HEAD_J = DATA / f"26_llamacpp_head{sfx}.json"
    HEAD_W = DATA / f"26_llamacpp_head{sfx}.npz"
    if tag not in FEATURE_ONLY:
        ACC = DATA / f"06_accuracy{sfx}.parquet"


def embed(text: str, retries: int = 3):
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            # 응답 모양은 버전에 따라 [{"embedding": [[...]]}] 또는 {"embedding":[...]} 다
            if isinstance(d, list):
                d = d[0]
            e = d["embedding"]
            while isinstance(e[0], list):
                e = e[-1]           # pooling=last 여도 중첩이 올 수 있다 — 마지막 행을 쓴다
            return e
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError):
            if k == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def render(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def cmd_extract(splits: list[str]) -> int:
    import numpy as np
    import yaml
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent
                          / "configs" / "student.yaml").read_text())
    model_id = cfg[MODEL_KEY]["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"템플릿 토크나이저 {model_id} (MODEL_KEY={MODEL_KEY})")

    split = {r["id"]: r["split"] for r in read_rows(DATA / "03_splits.parquet")}
    prompts = {r["id"]: r["prompt"] for r in read_rows(DATA / "01_prompts.parquet")}
    require(ACC.exists(), f"{ACC} 가 없다")
    acc = {r["id"]: float(r["accuracy"]) for r in read_rows(ACC)}
    meta = {r["id"]: r for r in read_rows(ACC)}

    ids = sorted([i for i, s in split.items() if s in splits and i in acc])
    print(f"대상 {len(ids)}건 (splits={splits})")

    X, keep, t0 = [], [], time.perf_counter()
    for n, i in enumerate(ids):
        X.append(embed(render(tok, prompts[i])))
        keep.append(i)
        if (n + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {n+1}/{len(ids)}  {el:.0f}s  ({1000*el/(n+1):.0f}ms/건)", flush=True)
    X = np.asarray(X, dtype=np.float32)
    print(f"특징 {X.shape}")

    old = {}
    if FEAT.exists():
        z = np.load(FEAT, allow_pickle=False)
        old = {str(v): z["X"][k] for k, v in enumerate(z["ids"])}
    for k, i in enumerate(keep):
        old[i] = X[k]
    ids_all = sorted(old)
    np.savez_compressed(
        FEAT, X=np.stack([old[i] for i in ids_all]).astype(np.float32),
        ids=np.array(ids_all),
        split=np.array([split[i] for i in ids_all]),
        acc=np.array([acc[i] for i in ids_all], dtype=np.float32),
        cell3=np.array([f"{meta[i]['category']}/{meta[i]['difficulty']}/{meta[i]['lang']}"
                        for i in ids_all]))
    print(f"-> {FEAT} ({FEAT.stat().st_size/1e6:.1f} MB, 누적 {len(ids_all)}건)")
    return 0


def cmd_fit() -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from signal_probe import auroc, auroc_within

    require(FEAT.exists(), f"{FEAT} 가 없다 — 먼저 --extract")
    z = np.load(FEAT, allow_pickle=False)
    X, split, acc, cell3 = z["X"], z["split"], z["acc"], z["cell3"]
    tr, ca = split == "train", split == "cal"
    require(tr.sum() > 0 and ca.sum() > 0, "train/cal 특징이 모두 필요하다")
    y = acc < FAIL_THR
    print(f"train {tr.sum()} (틀림 {y[tr].sum()}) / cal {ca.sum()} (틀림 {y[ca].sum()})")

    sc = StandardScaler().fit(X[tr])
    Xtr, Xca = sc.transform(X[tr]), sc.transform(X[ca])
    rows, best = [], None
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        lr.fit(Xtr, y[tr])
        s_tr = -lr.predict_proba(Xtr)[:, 1]
        s_ca = -lr.predict_proba(Xca)[:, 1]
        a_tr, a_ca = auroc(s_tr, ~y[tr]), auroc(s_ca, ~y[ca])
        a_ct, n_pair, n_cell = auroc_within(s_ca, ~y[ca], list(cell3[ca]))
        rows.append([C, f"{a_tr:.3f}", f"{a_ca:.3f}", f"{a_ct:.3f}", f"{a_tr-a_ca:+.3f}"])
        if best is None or a_ct > best["cal_ctrl"]:
            best = {"C": C, "train": a_tr, "cal": a_ca, "cal_ctrl": a_ct,
                    "n_pair": n_pair, "n_cell": n_cell,
                    "coef": lr.coef_[0].copy(), "intercept": float(lr.intercept_[0])}
    print(fmt_table(["C", "train", "cal", "**cal 통제**", "과적합폭"], rows))
    print(f"\n선택 C={best['C']} / cal 통제 **{best['cal_ctrl']:.4f}** "
          f"(셀 {best['n_cell']}, 짝 {best['n_pair']})")

    rng = np.random.default_rng(20260917)
    yca, c3 = ~y[ca], list(cell3[ca])
    s_ca = -(1.0 / (1.0 + np.exp(-(Xca @ best["coef"] + best["intercept"]))))
    bs = [auroc_within(s_ca[ix], yca[ix], [c3[j] for j in ix])[0]
          for ix in (rng.integers(0, len(s_ca), len(s_ca)) for _ in range(400))]
    bs = np.array([v for v in bs if v == v])
    lo, hi = np.quantile(bs, [0.025, 0.975])

    p_keep = 1.0 - (1.0 / (1.0 + np.exp(-(Xca @ best["coef"] + best["intercept"]))))
    pol_targets = read_json(DATA / "08_policy.json")["targets"]
    head = {"feature": "llama.cpp Q8_0 embedding, pooling=last, "
                       "--embd-normalize 플래그를 주지 않음(서버 기본값) "
                       "(= output_norm(h_final) 마지막 프롬프트 토큰, 실측 L2 노름 1.0)",
            "base_model_key": MODEL_KEY, "tag": TAG or "(1.7B 기본)",
            "label_source": str(ACC.name),
            "C": best["C"], "label": f"accuracy < {FAIL_THR}",
            "score_convention": "1 - P(accuracy < 0.75); 높을수록 로컬 유지",
            "cal_auroc": best["cal"], "cal_auroc_within_cell_lang": best["cal_ctrl"],
            "cal_ctrl_ci": [float(lo), float(hi)],
            "train_auroc": best["train"], "overfit_gap": best["train"] - best["cal"],
            "thresholds": {f"{float(t):.2f}": float(np.quantile(p_keep, float(t)))
                           for t in pol_targets},
            "thresholds_frozen_on": "cal 점수 분위수 (test 미열람)",
            "numerics": "llama.cpp Q8_0 (블록 32 대칭 int8) — 배포 형태 그 자체",
            "n_params": int(best["coef"].size + 1),
            "selected_on": "cal AUROC, (카테고리/난이도/언어) 통제 — 13-E·13-K 와 동일 기준"}
    write_json(head, HEAD_J)
    np.savez(HEAD_W, coef=best["coef"], intercept=np.array([best["intercept"]]),
             mean=sc.mean_, scale=sc.scale_)
    print(f"부트 95% CI [{lo:.3f}, {hi:.3f}] / 헤드 {head['n_params']} 파라미터 -> {HEAD_W}")
    print(f"문턱(esc20%) {head['thresholds']['0.20']:.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--split", default="train,cal")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tag", default="", help="0.6B arm 은 sc06 — 특징·헤드·라벨 경로를 분리한다")
    ap.add_argument("--model", default="primary", choices=("primary", "sidecar"),
                    help="채팅 템플릿을 어느 모델의 토크나이저로 만들 것인가")
    a = ap.parse_args()
    apply_tag(a.tag, a.model)
    if a.extract:
        return cmd_extract([s.strip() for s in a.split.split(",") if s.strip()])
    if a.fit:
        return cmd_fit()
    ap.error("--extract 또는 --fit")
    return 2


if __name__ == "__main__":
    sys.exit(main())
