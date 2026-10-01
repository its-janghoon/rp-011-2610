"""공용 유틸. 파이프라인 스크립트가 서로를 import 하지 않기 위한 유일한 공유 모듈 (SPEC §3).

이름이 `_io` 가 아닌 이유: CPython 내장 `_io` 모듈과 충돌한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"
LOGS = REPO / "logs"
MODELS = REPO / "models"

MASTER_SEED = 20260916

#: 브리지가 에이전트 응답을 기다릴 때 쓰는 종료코드. 0/1 과 구분한다.
EXIT_NEED_AGENT = 2


# --------------------------------------------------------------------------- #
# 실패는 조용히 넘기지 않는다
# --------------------------------------------------------------------------- #
def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# 시드
# --------------------------------------------------------------------------- #
def derive_seed(scope: str, salt: str = "") -> int:
    """MASTER_SEED 에서 파생된 결정적 시드. 전역 RNG 는 쓰지 않는다."""
    h = hashlib.sha256(f"{MASTER_SEED}|{scope}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def nonce(scope: str, salt: str = "") -> str:
    h = hashlib.sha256(f"nonce|{MASTER_SEED}|{scope}|{salt}".encode("utf-8")).digest()
    return h[:8].hex()


# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
def load_yaml(path: Path) -> Any:
    require(path.exists(), f"설정 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scale() -> dict:
    return load_yaml(CONFIGS / "scale.yaml")


def load_categories() -> dict:
    return load_yaml(CONFIGS / "categories.yaml")


def load_student_cfg() -> dict:
    return load_yaml(CONFIGS / "student.yaml")


# --------------------------------------------------------------------------- #
# 결정적 쿼터 배분 (최대잉여법)
# --------------------------------------------------------------------------- #
def allocate(total: int, weights: dict[str, float], tie_order: Sequence[str]) -> dict[str, int]:
    """합이 정확히 total 인 정수 배분. 동률 잉여는 tie_order 순서로 깬다."""
    require(total >= 0, f"allocate: total 음수 ({total})")
    require(bool(weights), "allocate: weights 비어 있음")
    for k in weights:
        require(k in tie_order, f"allocate: '{k}' 가 tie_order 에 없음")
    wsum = float(sum(weights.values()))
    require(wsum > 0, "allocate: weight 합이 0")

    raw = {k: total * float(w) / wsum for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = total - sum(out.values())
    require(0 <= rem <= len(weights), f"allocate: 잉여 이상 ({rem})")
    order = sorted(weights, key=lambda k: (-(raw[k] - out[k]), tie_order.index(k)))
    for k in order[:rem]:
        out[k] += 1
    require(sum(out.values()) == total, f"allocate: 합 불일치 {sum(out.values())} != {total}")
    return out


# --------------------------------------------------------------------------- #
# parquet (파일 입출력은 pyarrow)
# --------------------------------------------------------------------------- #
def write_parquet(rows: list[dict], path: Path, schema: pa.Schema) -> None:
    require(len(rows) > 0, f"write_parquet: 빈 rows -> {path}")
    keys = set(schema.names)
    for i, r in enumerate(rows):
        extra = set(r) - keys
        missing = keys - set(r)
        require(not extra, f"write_parquet: row[{i}] 스키마 밖 컬럼 {sorted(extra)}")
        require(not missing, f"write_parquet: row[{i}] 누락 컬럼 {sorted(missing)}")
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_table(path: Path) -> pa.Table:
    require(path.exists(), f"read_table: 파일 없음 {path}")
    return pq.read_table(path)


def read_rows(path: Path) -> list[dict]:
    return read_table(path).to_pylist()


# --------------------------------------------------------------------------- #
# jsonl / json (에이전트 브리지)
# --------------------------------------------------------------------------- #
def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    require(path.exists(), f"read_json: 파일 없음 {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    require(path.exists(), f"read_jsonl: 파일 없음 {path}")
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:  # 조용히 건너뛰지 않는다
                raise RuntimeError(f"{path}:{ln} JSON 파싱 실패: {e}") from e
            require(isinstance(obj, dict), f"{path}:{ln} 객체가 아님")
            out.append(obj)
    return out


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 텍스트 스크리닝 (기계적 게이트)
# --------------------------------------------------------------------------- #
_HANGUL = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_ASCII_ALPHA = re.compile(r"[A-Za-z]")

# 시점 의존 표현. answer 가 시간에 따라 바뀌는 grading(exact 계열)에만 적용한다.
_TIME_DEP = re.compile(
    r"(최신|현재|올해|작년|내년|지금까지|요즘|오늘\s*(기준|현재)"
    r"|latest|current(ly)?|this year|last year|as of|so far|nowadays)",
    re.IGNORECASE,
)

# PII. 아래 placeholder 형태만 허용한다.
_EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[A-Za-z]{2,}")
_EMAIL_OK_DOMAIN = re.compile(r"@(example\.(com|org|net)|test\.com)$", re.IGNORECASE)
_PHONE = re.compile(r"\b(\+?\d{1,3}[- ]?)?\d{2,4}[- ]\d{3,4}[- ]\d{4}\b")
_RRN = re.compile(r"\b\d{6}[- ]\d{7}\b")
_LONG_DIGITS = re.compile(r"\b\d{13,19}\b")


def hangul_count(s: str) -> int:
    return len(_HANGUL.findall(s))


def ascii_alpha_count(s: str) -> int:
    return len(_ASCII_ALPHA.findall(s))


def screen_lang(prompt: str, lang: str, category: str) -> str | None:
    """언어 위반 사유 or None."""
    if lang == "ko":
        if hangul_count(prompt) < 5:
            return "lang_ko_no_hangul"
        return None
    if lang == "en":
        if ascii_alpha_count(prompt) < 20:
            return "lang_en_too_little_ascii"
        # B(번역)는 영어 지시문 안에 한국어 원문이 들어갈 수 있으므로 예외
        if category != "B" and hangul_count(prompt) > 0:
            return "lang_en_has_hangul"
        return None
    return f"lang_unknown:{lang}"


def screen_time_dependent(prompt: str, grading_default: str) -> str | None:
    """시점 의존 질문 차단. 답이 확정되어야 하는 grading 에만 적용."""
    if grading_default not in ("exact", "contains"):
        return None
    if _TIME_DEP.search(prompt):
        return "time_dependent"
    return None


def screen_pii(prompt: str) -> str | None:
    if _RRN.search(prompt):
        return "pii_rrn"
    for m in _EMAIL.finditer(prompt):
        if not _EMAIL_OK_DOMAIN.search(m.group(0)):
            return "pii_email"
    for m in _PHONE.finditer(prompt):
        digits = re.sub(r"\D", "", m.group(0))
        if not digits.endswith("0000"):
            return "pii_phone"
    if _LONG_DIGITS.search(prompt):
        return "pii_long_digits"
    return None


# --------------------------------------------------------------------------- #
# 보고 유틸
# --------------------------------------------------------------------------- #
def fmt_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cols = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            cols[i] = max(cols[i], len(c))
    line = "  ".join(h.ljust(cols[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * cols[i] for i in range(len(headers)))
    body = "\n".join("  ".join(c.ljust(cols[i]) for i, c in enumerate(r)) for r in srows)
    return f"{line}\n{sep}\n{body}"
