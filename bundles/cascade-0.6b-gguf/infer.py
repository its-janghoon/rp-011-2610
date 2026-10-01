#!/usr/bin/env python3
"""온디바이스 위임 래퍼 — llama.cpp Q8_0 경로. **numpy + 표준 라이브러리만** 쓴다.

torch 도 transformers 도 필요 없다. 채팅 템플릿은 직접 조립하고(그 조립이 헤드를 적합할 때
쓴 `apply_chat_template` 결과와 바이트 단위로 같은 것을 `scripts/check_template.py` 로
확인했다), 가중치는 llama.cpp 가 Q8_0 로 들고 있다.

    # 서버를 직접 띄우게 하려면
    python infer.py --llama-bin /path/to/llama-server "질문"
    # 이미 띄워 뒀으면
    python infer.py "질문"
    python infer.py --decide-only "질문"       # 라우팅만
    python infer.py --jsonl in.jsonl           # {"id","prompt"} 줄들을 일괄 라우팅

동작
  1. 프롬프트를 템플릿으로 감싸 `/embedding` 에 보낸다 -> 최종 임베딩(2048차원).
  2. 동결 헤드(2,049 파라미터)를 적용해 점수를 낸다. **높을수록 로컬 유지.**
  3. 점수 < 문턱이면 `delegate`, 아니면 `local`.
  4. `local` 이면 같은 서버의 `/completion` 으로 답한다 — **두 번째 모델이 없다.**
  5. `delegate` 면 `delegate_contract` 모양의 요청을 돌려준다. 서버 왕복이 실패하면
     로컬로 되돌리면 된다 — 위임은 정확도 개선이지 필수 경로가 아니다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "delegate" / "config.json").read_text(encoding="utf-8"))
GGUF = HERE / "model" / "qwen3-0.6b-q8_0.gguf"


def render(prompt: str) -> str:
    """헤드가 적합된 것과 같은 문자열을 만든다. 값은 config 에 박아 뒀다."""
    ct = CFG["chat_template"]
    return ct["prefix"] + prompt + ct["suffix"]


def post(url: str, body: dict, timeout: int = 300, retries: int = 3):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError):
            if k == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def alive(base: str) -> bool:
    try:
        urllib.request.urlopen(base + "/health", timeout=3).read()
        return True
    except Exception:
        return False


def start_server(binary: str, base: str, threads: int) -> subprocess.Popen:
    port = base.rsplit(":", 1)[-1]
    cmd = [binary, "-m", str(GGUF), "--host", "127.0.0.1", "--port", port,
           "--embedding", "--pooling", "last", "-t", str(threads),
           "-c", "2048", "-np", "1", "--no-webui"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if alive(base):
            return p
        time.sleep(1)
    p.kill()
    raise RuntimeError("llama-server 가 기동하지 않았다")


def load_head():
    import numpy as np
    z = np.load(HERE / "delegate" / "head.npz")
    return {k: z[k] for k in ("coef", "intercept", "mean", "scale")}


def embed(base: str, text: str, cache: bool = False):
    d = post(base + "/embedding", {"content": text, "cache_prompt": cache})
    if isinstance(d, list):
        d = d[0]
    e = d["embedding"]
    while isinstance(e[0], list):
        e = e[-1]
    return e


def route(base: str, head, prompt: str, threshold: float | None = None) -> dict:
    import numpy as np
    text = render(prompt)
    t0 = time.perf_counter()
    f = np.asarray(embed(base, text), dtype=np.float64)
    z = (f - head["mean"]) / head["scale"]
    p_fail = 1.0 / (1.0 + np.exp(-(z @ head["coef"] + float(head["intercept"][0]))))
    score = float(1.0 - p_fail)
    thr = float(CFG["default_threshold"] if threshold is None else threshold)
    return {"route": "delegate" if score < thr else "local",
            "score": score, "threshold": thr,
            "decision_ms": (time.perf_counter() - t0) * 1000.0}


def answer_local(base: str, prompt: str, n_predict: int, sampling: dict | None = None) -> dict:
    """로컬 답변. `sampling` 을 주지 않으면 그리디다.

    동결 student 와 비교할 때는 **동결 설정을 그대로 넘겨야 한다** (temperature 0.7,
    top_p 0.8, top_k 20, 256토큰). 그러지 않으면 약해진 로컬 팔을 강한 기준선과 비교하게 된다.
    """
    body = {"prompt": render(prompt), "n_predict": n_predict, "cache_prompt": False}
    body.update(sampling or {"temperature": 0})
    t0 = time.perf_counter()
    d = post(base + "/completion", body)
    return {"answer": d.get("content", ""),
            "n_tokens": int(d.get("tokens_predicted", 0)),
            "gen_ms": (time.perf_counter() - t0) * 1000.0}


def delegate_payload(pid, prompt: str, r: dict) -> dict:
    """config 의 delegate_contract 모양대로 요청을 만든다."""
    return {"id": pid, "prompt": prompt,
            "reason": {"score": r["score"], "threshold": r["threshold"]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--server-url", default="http://127.0.0.1:8081")
    ap.add_argument("--llama-bin", help="주면 서버를 직접 띄운다")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--jsonl")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--n-predict", type=int, default=256)
    a = ap.parse_args()

    base = a.server_url.rstrip("/")
    proc = None
    if not alive(base):
        if not a.llama_bin:
            print(f"서버에 닿지 않는다: {base}\n"
                  f"  --llama-bin 으로 직접 띄우거나, config 의 server_flags 로 먼저 띄워라",
                  file=sys.stderr)
            return 1
        proc = start_server(a.llama_bin, base, a.threads)

    try:
        head = load_head()
        if a.jsonl:
            for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                r = route(base, head, rec["prompt"], a.threshold)
                r["id"] = rec.get("id")
                print(json.dumps(r, ensure_ascii=False), flush=True)
            return 0

        if not a.prompt:
            ap.error("prompt 또는 --jsonl 이 필요하다")
        r = route(base, head, a.prompt, a.threshold)
        if r["route"] == "local" and not a.decide_only:
            r.update(answer_local(base, a.prompt, a.n_predict))
        elif r["route"] == "delegate":
            r["delegate_request"] = delegate_payload(None, a.prompt, r)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
