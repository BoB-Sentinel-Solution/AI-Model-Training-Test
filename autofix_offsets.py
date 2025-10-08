# autofix_offsets.py
# -*- coding: utf-8 -*-

import json
import sys
import argparse
import unicodedata
import re
import io
import os

# --- (옵션) Windows 콘솔에서 메시지 깨짐 방지 ---
try:
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# -------------------------------------------------

def norm(s: str, use_nfkc: bool) -> str:
    """NFC(기본) 또는 NFKC로 정규화."""
    return unicodedata.normalize("NFKC" if use_nfkc else "NFC", s)

def find_all_exact(text: str, value: str):
    """text에서 value가 정확히 일치하는 모든 시작 인덱스 반환."""
    out = []
    start = 0
    while True:
        i = text.find(value, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out

def best_occurrence(candidates, prefer_begin):
    """여러 후보 중 기존 begin에 가장 가까운 시작 인덱스 선택."""
    if not candidates:
        return None
    return min(candidates, key=lambda b: abs(b - prefer_begin))

def brute_force_norm_match(text, value, prefer_begin, use_nfkc):
    """
    정규화 기반 근사 탐색:
      - 첫 글자가 같은 지점들을 후보 b로 삼고,
      - e를 b+1..b+len(value)+max_extra 범위에서 확장,
      - norm(text[b:e]) == norm(value) 가장 먼저 만족하는 (b,e) 채택.
      - 최종 선택은 prefer_begin과의 거리, 그리고 span 길이로 판정.
    """
    if not value:
        return None
    nv = norm(value, use_nfkc)
    n = len(text)
    max_extra = 8
    b_hits = []

    first = value[0]
    candidate_bs = []
    pos = -1
    while True:
        pos = text.find(first, pos + 1)
        if pos == -1:
            break
        candidate_bs.append(pos)

    for b in candidate_bs:
        min_e = b + 1
        max_e = min(n, b + max(len(value), 1) + max_extra)
        for e in range(min_e, max_e + 1):
            if e - b < max(1, len(value) - max_extra):
                continue
            slice_norm = norm(text[b:e], use_nfkc)
            if slice_norm == nv:
                b_hits.append((b, e))
                break  # 동일 b에서는 가장 짧은 e만 수집

    if not b_hits:
        return None

    b, e = min(b_hits, key=lambda t: (abs(t[0] - prefer_begin), (t[1] - t[0])))
    return b, e

def fix_entity_offsets(text: str, entity: dict, use_nfkc: bool):
    """
    엔티티의 (begin,end)를 value에 맞게 자동 보정.
    우선순위:
      1) 원문(raw) 정확 매칭
      2) 정규화 기반 근사 탐색
    성공 시 (begin, end) 반환, 실패 시 None.
    """
    value = entity.get("value")
    b_old = entity.get("begin")
    e_old = entity.get("end")
    if not isinstance(value, str) or not isinstance(b_old, int) or not isinstance(e_old, int):
        return None

    # 1) 원문 정확 매칭
    exacts = find_all_exact(text, value)
    if exacts:
        b_new = best_occurrence(exacts, b_old)
        if b_new is not None:
            return b_new, b_new + len(value)

    # 2) 정규화 근사 탐색
    bf = brute_force_norm_match(text, value, b_old, use_nfkc)
    if bf:
        return bf

    return None

# ---------- 입출력 인코딩 도우미 ----------

def detect_bom_encoding(path: str):
    """파일 BOM을 보고 적절한 텍스트 인코딩을 결정."""
    with open(path, "rb") as fb:
        head = fb.read(4)
    if head.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    if head.startswith(b'\xff\xfe'):
        return 'utf-16'      # LE
    if head.startswith(b'\xfe\xff'):
        return 'utf-16-be'   # BE
    return 'utf-8'           # 기본 가정

def open_text_auto(path: str):
    """BOM 감지로 텍스트 모드 오픈(스트리밍). 실패 시 CP949 폴백."""
    enc = detect_bom_encoding(path)
    try:
        return open(path, "r", encoding=enc, newline=None)
    except UnicodeError:
        # 희귀 케이스 폴백
        return open(path, "r", encoding="cp949", errors="replace", newline=None)

# -----------------------------------------

def process_row(row, use_nfkc: bool, stats):
    """
    각 JSONL 라인의 assistant 메시지 엔티티 오프셋을 보정.
    변경 시 row 내 assistant.content(JSON 문자열)를 갱신.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return row  # 구조가 다르면 그대로 통과(검증기는 따로 잡음)

    ac = msgs[2].get("content", "")
    try:
        ans = json.loads(ac)
    except Exception:
        return row

    text = ans.get("text")
    ents = ans.get("entities")
    if not isinstance(text, str) or not isinstance(ents, list):
        return row

    changed = False
    for i, ent in enumerate(ents):
        if not isinstance(ent, dict):
            continue
        b = ent.get("begin"); e = ent.get("end"); v = ent.get("value")
        ok = (isinstance(b, int) and isinstance(e, int) and isinstance(v, str)
              and 0 <= b < e <= len(text) and text[b:e] == v)
        if ok:
            continue

        fixed = fix_entity_offsets(text, ent, use_nfkc)
        if fixed:
            b2, e2 = fixed
            # 최종 가드: 정규화 동등성 검사
            if norm(text[b2:e2], use_nfkc) == norm(v, use_nfkc):
                ent["begin"], ent["end"] = b2, e2
                changed = True
                stats["fixed"] += 1
            else:
                stats["unmatched"] += 1
        else:
            stats["unmatched"] += 1

    if changed:
        msgs[2]["content"] = json.dumps(ans, ensure_ascii=False)
    return row

def main():
    ap = argparse.ArgumentParser(
        description="Fix entity (begin,end) to match value in assistant JSON; always write UTF-8."
    )
    ap.add_argument("input", help="입력 JSONL (각 줄: messages[system,user,assistant])")
    ap.add_argument("output", help="출력 JSONL (항상 UTF-8로 저장)")
    ap.add_argument("--nfkc", action="store_true", help="NFKC 정규화 비교 사용(기본은 NFC)")
    args = ap.parse_args()

    stats = {"lines": 0, "fixed": 0, "unmatched": 0}

    # 입력 스트리밍 열기(인코딩 자동 판별)
    with open_text_auto(args.input) as fin, open(args.output, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write(raw + "\n")
                continue

            stats["lines"] += 1
            try:
                row = json.loads(raw)
            except Exception:
                # 라인 자체가 JSON이 아니면 그대로 통과
                fout.write(raw + "\n")
                continue

            row2 = process_row(row, args.nfkc, stats)
            out = json.dumps(row2, ensure_ascii=False)
            fout.write(out + "\n")

    sys.stderr.write(
        f"[autofix] lines={stats['lines']} fixed={stats['fixed']} unmatched={stats['unmatched']}\n"
    )

if __name__ == "__main__":
    main()
