"""기준 C(지연)의 스트리밍 모드 실측 — TTFT 와 지속 토큰 속도 (SPEC 13-U -> 13-AA).

13-U 에 "TTFT = 결정 프리필 + 생성 프리필 + 1토큰 — **Stage 7 에서 실측한다, 가정하지 않는다**"
로 등록했다. 여기서 잰다.

  TTFT      = /embedding (결정) + /completion n_predict=1 (생성 프리필 + 첫 토큰)
  지속 속도  = /completion 의 timings.predicted_per_second (256토큰 강제)

두 arm 을 **순차로** 돈다(8 vCPU 를 나눠 쓰면 값이 오염된다). 프롬프트는 실제 e2e 표본에서
가져오고 캐시는 끈다 — 같은 프롬프트 반복은 캐시 적중으로 60ms 처럼 보인다(13-Q).

  .venv/bin/python scripts/bench_ttft.py [--n 24]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HOME", "$HF_HOME")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA, REPO, read_rows, require, write_json  # noqa: E402

BIN = Path("$LLAMA_CPP/build/bin/llama-server")
ARMS = {
    "qwen3-1.7b": REPO / "artifact" / "cascade-1.7b-gguf",
    "qwen3-0.6b": REPO / "artifact" / "cascade-0.6b-gguf",
}
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"
DUMP = False


def load_infer(art: Path):
    spec = importlib.util.spec_from_file_location(f"inf_{art.name}", art / "infer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def post(path: str, body: dict, timeout: int = 300) -> dict:
    rq = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())


def bench_arm(name: str, art: Path, prompts: list[str]) -> dict:
    inf = load_infer(art)
    gguf = next((art / "model").glob("*.gguf"))
    proc = subprocess.Popen(
        [str(BIN), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT),
         "--embedding", "--pooling", "last", "-t", "8", "-c", "2048", "-np", "1",
         "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if inf.alive(BASE):
                break
            time.sleep(1)
        else:
            raise SystemExit(f"{name} 서버 기동 실패")
        rss = 0
        try:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS"):
                    rss = int(line.split()[1]) / 1024
        except Exception:
            pass

        dec, first, ttft, nin = [], [], [], []
        for n, p in enumerate(prompts):
            txt = inf.render(p)
            t0 = time.perf_counter()
            inf.embed(BASE, txt, cache=False)
            d = (time.perf_counter() - t0) * 1000.0
            r = post("/completion", {"prompt": txt, "n_predict": 1,
                                     "temperature": 0, "cache_prompt": False})
            tm = r.get("timings", {})
            f = float(tm.get("prompt_ms", 0.0)) + float(tm.get("predicted_ms", 0.0))
            dec.append(d)
            first.append(f)
            ttft.append(d + f)
            nin.append(int(tm.get("prompt_n", 0)))
            if DUMP:
                print(f"    {n+1:3d}  입력 {nin[-1]:4d}토큰  결정 {d:8.1f}ms  "
                      f"프리필+1 {f:8.1f}ms", flush=True)

        # 지속 속도: 256토큰 강제
        r = post("/completion", {"prompt": inf.render(prompts[0]), "n_predict": 256,
                                 "ignore_eos": True, "temperature": 0,
                                 "cache_prompt": False})
        tps = float(r["timings"]["predicted_per_second"])
        per_tok = float(r["timings"]["predicted_per_token_ms"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        time.sleep(2)

    def s(v):
        w = sorted(v)
        return {"mean": st.mean(w), "median": st.median(w), "p90": w[int(0.9 * (len(w) - 1))]}

    out = {"arm": name, "gguf_bytes": gguf.stat().st_size, "rss_mb": rss,
           "n_prompts": len(prompts), "prompt_tokens": s(nin),
           "decision_ms": s(dec), "gen_prefill_plus_first_ms": s(first),
           "ttft_ms": s(ttft), "tokens_per_sec": tps, "per_token_ms": per_tok}
    print(f"[{name}] RSS {rss:.0f}MB / 프롬프트 토큰 중앙 {out['prompt_tokens']['median']:.0f}")
    print(f"  결정 {out['decision_ms']['mean']:.0f}ms  "
          f"생성프리필+1토큰 {out['gen_prefill_plus_first_ms']['mean']:.0f}ms  "
          f"**TTFT 평균 {out['ttft_ms']['mean']:.0f}ms / p90 {out['ttft_ms']['p90']:.0f}ms**")
    print(f"  지속 {tps:.1f} tok/s ({per_tok:.1f} ms/tok)")
    return out


def main() -> int:
    global DUMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--arm", default="", help="하나만 재려면 arm 이름")
    ap.add_argument("--dump", action="store_true", help="호출별 값을 찍는다")
    ap.add_argument("--out", default="32_ttft.json")
    a = ap.parse_args()
    DUMP = a.dump
    require(BIN.exists(), f"{BIN} 가 없다 — build_llamacpp_home.sh 를 먼저")
    for art in ARMS.values():
        require((art / "infer.py").exists(), f"{art} 산출물이 없다")

    # 실제 e2e 표본의 프롬프트를 쓴다 (합성 짧은 질의는 TTFT 를 낙관적으로 만든다)
    ids = [json.loads(x)["id"] for x in
           (DATA / "25_e2e_sc06" / "local.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip()]
    pmap = {r["id"]: r["prompt"] for r in read_rows(DATA / "01_prompts.parquet")}
    step = max(1, len(ids) // a.n)
    prompts = [pmap[i] for i in ids[::step][: a.n]]
    print(f"프롬프트 {len(prompts)}건 (e2e 표본에서 균등 추출), 캐시 끔\n")

    res = {}
    for name, art in ARMS.items():          # 순차 — 동시 기동 금지
        if a.arm and name != a.arm:
            continue
        res[name] = bench_arm(name, art, prompts)
        print()

    write_json({"arms": res,
                "criterion_c": {"nonstreaming_budget_ms": 10000,
                                "streaming_ttft_budget_ms": 1000,
                                "streaming_tps_floor": 10.0,
                                "source": "SPEC 13-U"}},
               REPO / "results" / a.out)
    print(f"-> {REPO/'results'/a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
