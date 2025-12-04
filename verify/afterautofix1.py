# fix_dataset_afterautofix.py  —  Auto-fixer for dataset JSONL (robust logs)
# -*- coding: utf-8 -*-

import json
import argparse
import sys
import io
import os
import unicodedata
import re
from typing import List, Dict, Any, Tuple, Optional

# ----------------------------- Control chars ------------------------------
# 제로폭/제어문자 제거(검증기와 동일하게 맞춤)
CTRL_RE = re.compile(r"[\u0000-\u001F\u007F\u200B\u200C\u200D\u200E\u200F]")
NBSP = "\u00A0"

# ----------------------------- IO helpers --------------------------------
def robust_read_lines(path: str) -> List[str]:
    """UTF-8/UTF-8-SIG/UTF-16/CP949 대응 + CRLF 정리."""
    with open(path, "rb") as fb:
        data = fb.read()
    if data.startswith(b"\xef\xbb\xbf"):
        text = data.decode("utf-8-sig")
    elif data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp949", errors="replace")
    return [ln.rstrip("\r\n") for ln in text.splitlines()]

def write_jsonl_safely(records_iter, out_path: str) -> int:
    """임시파일에 쓴 뒤 원자적 교체."""
    tmp = out_path + ".tmp"
    count = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fw:
        for rec in records_iter:
            fw.write(rec + "\n")
            count += 1
    os.replace(tmp, out_path)
    return count

# ------------------------ Normalization helpers ---------------------------
def remove_ctrl(s: str) -> str:
    """NBSP→space, 제어/제로폭 제거."""
    s = s.replace(NBSP, " ")
    return CTRL_RE.sub("", s)

def build_clean_map(orig: str, do_nfkc: bool) -> Tuple[str, List[int]]:
    """
    원문을 정규화+제거하여 cleaned_text 생성.
    clean_to_orig: cleaned 인덱스 -> (정규화 후) 원문 인덱스(근사).
    """
    normed = unicodedata.normalize("NFKC" if do_nfkc else "NFC", orig)
    cleaned_chars = []
    clean_to_orig = []
    for idx, ch in enumerate(normed):
        if ch == NBSP:
            ch = " "
        if CTRL_RE.match(ch):
            continue
        cleaned_chars.append(ch)
        clean_to_orig.append(idx)
    return "".join(cleaned_chars), clean_to_orig

def approx_clean_index(orig_begin: int, clean_map: List[int]) -> int:
    """원본문 인덱스를 cleaned 좌표로 근사 매핑."""
    if not clean_map:
        return 0
    best_k, best_d = 0, abs(clean_map[0] - orig_begin)
    for k, v in enumerate(clean_map):
        d = abs(v - orig_begin)
        if d < best_d:
            best_k, best_d = k, d
    return best_k

def find_best(content: str, value: str, hint_pos: int, max_window: int = 200) -> Optional[int]:
    """hint 주변 우선 탐색 후 전체 탐색."""
    L = len(content)
    vlen = len(value)
    if vlen == 0 or vlen > L:
        return None
    start = max(0, hint_pos - max_window)
    end = min(L, hint_pos + max_window)
    idx = content.find(value, start, end)
    if idx != -1:
        return idx
    idx = content.find(value)
    if idx != -1:
        return idx
    return None

# -------------------------- Overlap resolver ------------------------------
def resolve_overlaps(entities: List[Dict[str, Any]], mode: str = "trim") -> List[Dict[str, Any]]:
    """
    entities는 begin 기준 정렬 가정.
      - mode='trim': 앞 엔티티 end를 next.begin으로 잘라 겹침 제거(길이 0이면 drop)
      - mode='drop': 겹치면 짧은 엔티티 drop(동일 길이는 앞쪽 drop)
    """
    if not entities:
        return entities
    ents = sorted(entities, key=lambda e: (int(e["begin"]), int(e["end"])))
    out = [ents[0]]
    for cur in ents[1:]:
        prev = out[-1]
        if cur["begin"] < prev["end"]:  # overlap
            if mode == "trim":
                new_end = max(prev["begin"], cur["begin"])
                if new_end <= prev["begin"]:
                    # 길이 비교로 하나만 유지
                    if (prev["end"] - prev["begin"]) >= (cur["end"] - cur["begin"]):
                        # prev 유지, cur drop
                        continue
                    else:
                        out[-1] = cur
                else:
                    prev["end"] = new_end
                    prev["value"] = prev["value"][: new_end - prev["begin"]]
                    out.append(cur)
            else:  # 'drop'
                if (prev["end"] - prev["begin"]) >= (cur["end"] - cur["begin"]):
                    continue
                else:
                    out[-1] = cur
        else:
            out.append(cur)
    # 길이 0 제거
    out = [e for e in out if e["end"] > e["begin"]]
    return out

# --------------------------- Core fixer -----------------------------------
def parse_assistant_content(acont: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """assistant.content가 문자열/객체 모두 허용되도록 파싱."""
    if isinstance(acont, dict):
        return acont, None
    if isinstance(acont, str):
        try:
            obj = json.loads(acont)
            if not isinstance(obj, dict):
                return None, "assistant.content is not a JSON object"
            return obj, None
        except Exception as e:
            return None, f"assistant.content JSON parse error: {e}"
    return None, f"assistant.content has unsupported type: {type(acont).__name__}"

def fix_record(
    row: Dict[str, Any],
    *,
    nfkc: bool,
    overlap_mode: str,
    prefer_string_assistant: bool,
    max_window: int = 200
) -> Tuple[Dict[str, Any], List[str]]:
    """
    한 레코드 자동 수정:
      - 텍스트/엔티티 정규화 및 제어문자 제거
      - 오프셋 재탐색 및 slice mismatch/범위/겹침 문제 해결
      - has_sensitive 동기화
    """
    notes: List[str] = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        notes.append("bad messages shape")
        return row, notes

    acont = msgs[2].get("content", "")
    ans, perr = parse_assistant_content(acont)
    if perr:
        notes.append(perr)
        return row, notes

    text = ans.get("text")
    ents = ans.get("entities", [])
    hs = ans.get("has_sensitive", bool(ents))

    if not isinstance(text, str) or not isinstance(ents, list):
        notes.append("bad assistant json schema")
        return row, notes

    # 1) 본문 정리: 정규화 + 제어/제로폭 제거
    cleaned_text, clean_map = build_clean_map(text, do_nfkc=nfkc)

    fixed_entities: List[Dict[str, Any]] = []
    for i, e in enumerate(ents):
        if not all(k in e for k in ("value", "begin", "end", "label")):
            notes.append(f"entity[{i}] missing keys")
            continue

        try:
            b = int(e["begin"])
            en = int(e["end"])
        except Exception:
            notes.append(f"entity[{i}] begin/end not int")
            continue

        val = e.get("value", "")
        lab = e.get("label", "")

        # 엔티티 값 정리(정규화 + 제어/제로폭 제거)
        val_clean = remove_ctrl(unicodedata.normalize("NFKC" if nfkc else "NFC", str(val)))
        if not val_clean:
            notes.append(f"entity[{i}] empty after clean -> drop")
            continue

        # 원본문 인덱스를 cleaned 좌표로 근사 매핑
        hint = approx_clean_index(b, clean_map)

        # cleaned_text에서 값 재탐색
        pos = find_best(cleaned_text, val_clean, hint_pos=hint, max_window=max_window)
        if pos is None:
            # 원래 슬라이스도 클린해서 시도
            raw_slice = text[b:en] if 0 <= b < en <= len(text) else ""
            raw_slice_clean = remove_ctrl(unicodedata.normalize("NFKC" if nfkc else "NFC", raw_slice))
            if raw_slice_clean and raw_slice_clean in cleaned_text:
                pos = cleaned_text.find(raw_slice_clean)
                val_clean = raw_slice_clean
            else:
                notes.append(f"entity[{i}] not found after clean -> drop")
                continue

        fixed_entities.append({
            "value": val_clean,
            "begin": pos,
            "end": pos + len(val_clean),
            "label": lab
        })

    # 2) 겹침 해결
    fixed_entities = resolve_overlaps(fixed_entities, mode=overlap_mode)

    # 3) has_sensitive 동기화
    hs2 = bool(fixed_entities)

    # 4) 수정 결과 구성
    ans_fixed = {
        "text": cleaned_text,
        "has_sensitive": hs2,
        "entities": fixed_entities
    }

    if prefer_string_assistant:
        msgs[2]["content"] = json.dumps(ans_fixed, ensure_ascii=False, separators=(",", ":"))
    else:
        msgs[2]["content"] = ans_fixed

    row["messages"] = msgs
    return row, notes

# ------------------------------ Runner ------------------------------------
def process(
    input_path: str,
    out_path: str,
    *,
    nfkc: bool,
    overlap_mode: str,
    assistant_as_string: bool,
    max_window: int
) -> Tuple[int, int]:
    """입력→자동 수정→출력. (read count, write count) 반환."""
    lines = robust_read_lines(input_path)
    in_total = 0
    out_lines: List[str] = []
    notes_stats: Dict[str, int] = {}

    for ln, line in enumerate(lines, 1):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception as e:
            print(f"[L{ln}] JSON parse error: {e}", file=sys.stderr)
            continue

        in_total += 1
        fixed_row, notes = fix_record(
            row,
            nfkc=nfkc,
            overlap_mode=overlap_mode,
            prefer_string_assistant=assistant_as_string,
            max_window=max_window
        )
        if notes:
            for n in notes:
                notes_stats[n] = notes_stats.get(n, 0) + 1

        out_lines.append(json.dumps(fixed_row, ensure_ascii=False, separators=(",", ":")))

    wrote = write_jsonl_safely(iter(out_lines), out_path)

    # stderr에 요약 노트
    if notes_stats:
        sys.stderr.write("notes:\n")
        for k, v in sorted(notes_stats.items(), key=lambda x: -x[1]):
            sys.stderr.write(f"  - {k}: {v}\n")
        sys.stderr.flush()

    return in_total, wrote

def main():
    ap = argparse.ArgumentParser(description="Auto-fix dataset JSONL (control chars / offsets / overlaps)")
    ap.add_argument("--input", required=True, help="입력 JSONL (messages 구조)")
    ap.add_argument("--out", required=True, help="출력 JSONL")
    ap.add_argument("--nfkc", action="store_true", help="NFKC 정규화 사용(기본 NFC)")
    ap.add_argument("--overlap-mode", choices=["trim", "drop"], default="trim", help="엔티티 겹침 처리 방식")
    ap.add_argument("--assistant-as-string", action="store_true", help="assistant.content를 JSON 문자열로 저장")
    ap.add_argument("--max-window", type=int, default=200, help="근접 탐색 윈도 크기")
    args = ap.parse_args()

    try:
        read_n, write_n = process(
            args.input,
            args.out,
            nfkc=args.nfkc,
            overlap_mode=args.overlap_mode,
            assistant_as_string=args.assistant_as_string,
            max_window=args.max_window,
        )
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise

    abs_in = os.path.abspath(args.input)
    abs_out = os.path.abspath(args.out)
    print(f"[OK] Input:  {abs_in}")
    print(f"[OK] Output: {abs_out}")
    print(f"[OK] Lines:  read={read_n}, wrote={write_n}")

if __name__ == "__main__":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    main()
