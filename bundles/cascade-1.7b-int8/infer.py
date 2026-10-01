#!/usr/bin/env python3
"""온디바이스 위임 래퍼. 질의 하나를 받아 **로컬로 답하거나 위임 신호를 낸다.**

    python infer.py "질문"                 # 라우팅 후 필요하면 로컬 생성
    python infer.py --decide-only "질문"    # 라우팅 결정만 (프리필까지만 돈다)
    python infer.py --jsonl in.jsonl       # {"id","prompt"} 줄들을 일괄 라우팅

설계상 중요한 점 두 가지.

1. **결정은 블록 1..N 만 돌려서 낸다** (N = delegate/config.json 의 decision_layer).
   그 출력이 헤드가 적합된 특징(`hidden_states[N]`, 최종 norm 이전)과 같은 텐서다.
   위임될 질의는 나머지 층과 256토큰 생성을 아예 치르지 않는다.
2. **로컬로 유지되는 질의에는 추가 비용이 없다.** 프리필은 생성에 어차피 필요하므로
   결정에 쓴 계산이 버려지지 않는다.

가중치는 rtn int8 W8 group128 로 저장돼 있고 적재 시 역양자화한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(dtype: str = "bfloat16", decision_only: bool = False, device: str = "cpu"):
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((HERE / "delegate" / "config.json").read_text())
    dt = getattr(torch, dtype)
    conf = AutoConfig.from_pretrained(HERE / "model")
    tok = AutoTokenizer.from_pretrained(HERE / "model")

    # 결정만 필요하면 판정 층까지만 만든다 — 폰에서 라우터로만 쓸 때의 형태다.
    if decision_only:
        conf.num_hidden_layers = int(cfg["decision_layer"])

    # 적재 봉우리를 모델 크기 근처로 누른다. 처음 구현은 (a) `from_config` 가 float32 로
    # 전체를 할당한 뒤 캐스팅하고 (b) 역양자화한 텐서를 딕셔너리로 통째로 들고 있어서
    # **2GB 산출물에 RSS 8.4GB(디스크의 4.01배)** 가 들었다 — 폰에서 불가능한 값이다.
    # 이제 파라미터를 처음부터 bf16 으로 만들고, 텐서를 하나씩 읽어 제자리에 복사한다.
    #
    # meta 장치를 쓰지 않는 이유: rotary 의 `inv_freq` 는 비영속 버퍼라 체크포인트에 없고
    # 초기화되지 않은 쓰레기가 남는다(특징 최대 절대차 22.5 로 드러났다). 정상 생성 경로는
    # `inv_freq` 를 명시적으로 float32 로 계산하므로 기본 dtype 을 bf16 으로 바꿔도 안전하다.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        model = AutoModelForCausalLM.from_config(conf)
    finally:
        torch.set_default_dtype(prev)

    keep = int(cfg["decision_layer"]) if decision_only else 10**9
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    seen: set[str] = set()
    with safe_open(str(HERE / "model" / "weights_int8.safetensors"),
                   framework="pt", device="cpu") as f:
        keys = list(f.keys())
        bases = sorted({k[: -len(".qweight")] for k in keys if k.endswith(".qweight")})
        plain = [k for k in keys if not k.endswith((".qweight", ".scale", ".in_features"))]

        def wanted(name: str) -> bool:
            if not name.startswith("model.layers."):
                return True
            return int(name.split(".")[2]) < keep

        for base in bases:
            if not wanted(base):
                continue
            tgt = params.get(base)
            if tgt is None:
                raise RuntimeError(f"산출물에 없는 파라미터: {base}")
            q = f.get_tensor(base + ".qweight").float()
            sc = f.get_tensor(base + ".scale").float()
            in_f = int(f.get_tensor(base + ".in_features")[0])
            g = q.shape[1] // sc.shape[1]
            w = (q.reshape(q.shape[0], -1, g) * sc.unsqueeze(-1)).reshape(q.shape[0], -1)
            tgt.data.copy_(w[:, :in_f].to(dt))
            seen.add(base)
            del q, sc, w                       # 다음 텐서 전에 즉시 놓아준다
        for k in plain:
            if not wanted(k):
                continue
            t = f.get_tensor(k)
            if k in params:
                params[k].data.copy_(t.to(dt))
            elif k in buffers:
                buffers[k].data.copy_(t.to(buffers[k].dtype))
            else:
                raise RuntimeError(f"예상치 못한 텐서: {k}")
            seen.add(k)
            del t

    # tie_word_embeddings 인 모델은 lm_head 가 저장되지 않으므로 그것만은 정상 결측이다.
    missing = [n for n in params
               if n not in seen and wanted(n) and "lm_head" not in n]
    if missing:
        raise RuntimeError(f"결측 텐서 {len(missing)}개: {missing[:3]}")
    model.tie_weights()
    model.eval()
    if device != "cpu":
        model.to(device)          # dtype 은 건드리지 않는다 — 버퍼를 float32 로 남긴다
    if decision_only:
        # 헤드는 **최종 norm 이전**의 블록 출력으로 적합됐다. 결정 경로만 적재했을 때는
        # 마지막 블록 뒤에 norm 이 붙으므로 그것을 걷어내야 같은 텐서가 나온다.
        model.model.norm = torch.nn.Identity()

    hd = np.load(HERE / "delegate" / "head.npz")
    return tok, model, cfg, hd


def decision_feature(model, enc, n_layer_decide: int):
    """`hidden_states[n]`(블록 n 출력, 최종 norm 이전)의 마지막 프롬프트 토큰을 돌려준다.

    결정 경로만 적재한 모델(층이 정확히 n개)은 최종 norm 을 Identity 로 바꿔 두었으므로
    `last_hidden_state` 가 곧 블록 n 출력이다. 로터리·마스크는 HF 자체 순전파가 처리한다.
    전체 모델일 때는 `output_hidden_states` 로 같은 텐서를 집는다.
    """
    import torch

    # 호출자가 CPU 텐서를 넘기든 GPU 텐서를 넘기든 여기서 맞춘다 — 검증과 벤치마크가
    # 각자 enc 를 만들기 때문이다.
    ids = enc["input_ids"].to(next(model.parameters()).device)
    with torch.inference_mode():
        if len(model.model.layers) == n_layer_decide:
            h = model.model(input_ids=ids).last_hidden_state
            return h[0, -1].float().cpu()
        # 전체 모델일 때는 기반 트랜스포머만 돌린다 — lm_head 는 결정에 필요 없고
        # 어휘 151k 투영을 전 위치에 치르면 프리필 비용이 과대측정된다.
        hs = model.model(input_ids=ids, output_hidden_states=True).hidden_states
        return hs[n_layer_decide][0, -1].float().cpu()


def route(tok, model, cfg, hd, prompt: str, threshold: float | None = None):
    import numpy as np

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=cfg["chat_template"]["add_generation_prompt"],
        enable_thinking=cfg["chat_template"]["enable_thinking"])
    enc = tok(text, return_tensors="pt").to(model.device)
    f = decision_feature(model, enc, int(cfg["decision_layer"])).numpy()
    z = (f - hd["mean"]) / hd["scale"]
    p_fail = 1.0 / (1.0 + np.exp(-(z @ hd["coef"] + float(hd["intercept"][0]))))
    score = float(1.0 - p_fail)                     # 높을수록 로컬 유지
    thr = float(cfg["default_threshold"] if threshold is None else threshold)
    return {"route": "delegate" if score < thr else "local",
            "score": score, "threshold": thr,
            "n_prompt_tokens": int(enc["input_ids"].shape[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--jsonl")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    tok, model, cfg, hd = load(a.dtype,
                               decision_only=a.decide_only or bool(a.jsonl),
                               device=a.device)

    if a.jsonl:
        for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out = route(tok, model, cfg, hd, r["prompt"], a.threshold)
            out["id"] = r.get("id")
            print(json.dumps(out, ensure_ascii=False), flush=True)
        return 0

    if not a.prompt:
        ap.error("prompt 또는 --jsonl 이 필요하다")
    out = route(tok, model, cfg, hd, a.prompt, a.threshold)
    if out["route"] == "local" and not a.decide_only:
        import torch
        text = tok.apply_chat_template(
            [{"role": "user", "content": a.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt")
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                               do_sample=False)
        out["answer"] = tok.decode(g[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
