# autofix_offsets.py
# -*- coding: utf-8 -*-
import json
import sys
import argparse
import unicodedata
import re

def norm(s: str, use_nfkc: bool) -> str:
    return unicodedata.normalize("NFKC" if use_nfkc else "NFC", s)

def find_all_exact(text: str, value: str):
    """text에서 value의 모든 정확 일치 시작 인덱스를 반환"""
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
    """여러 시작 인덱스 중 prefer_begin(기존 begin)과 가장 가까운 것을 고름"""
    if not candidates:
        return None
    return min(candidates, key=lambda b: abs(b - prefer_begin))

def brute_force_norm_match(text, value, prefer_begin, use_nfkc):
    """
    정규화 비교 기반의 근사 탐색:
    - 시작점 b 후보들을 전역으로 훑되, 비용 줄이기 위해 value 길이±8 범위로만 e를 확장
    - norm(text[b:e]) == norm(value) 인 최소 e를 찾음
    - 여러 후보면 prefer_begin과 가장 가까운 b를 선택
    """
    nv = norm(value, use_nfkc)
    n = len(text)
    max_extra = 8  # value 길이와 약간 다를 수 있는 여유
    b_hits = []

    # 빠른 프리필터: 첫 글자 매칭 위치들
    first_chars = {text[i] for i in range(n)}
    # 전 탐색은 비용이 크니, value[0] 와 같은 지점들만 후보로
    first = value[0] if value else None
    candidate_bs = []
    if first is None:
        return None
    pos = -1
    while True:
        pos = text.find(first, pos + 1)
        if pos == -1: break
        candidate_bs.append(pos)

    for b in candidate_bs:
        # e는 b+1 .. b+len(value)+max_extra 사이에서 탐색
        min_e = b + 1
        max_e = min(n, b + max(len(value), 1) + max_extra)
        # 빠른 실패 방지: 너무 짧으면 skip
        for e in range(min_e, max_e + 1):
            # 조기 중단: 대충 길이가 value보다 훨씬 작으면 패스
            if e - b < max(1, len(value) - max_extra):
                continue
            slice_norm = norm(text[b:e], use_nfkc)
            if slice_norm == nv:
                b_hits.append((b, e))
                break  # 같은 b에서 가장 짧은 e만 수집
    if not b_hits:
        return None
    # 기존 begin과 가까운 b 먼저, 그 다음으로 span 길이가 짧은 것 선호
    b, e = min(b_hits, key=lambda t: (abs(t[0]-prefer_begin), (t[1]-t[0])))
    return b, e

def fix_entity_offsets(text: str, entity: dict, use_nfkc: bool):
    """
    entity의 begin/end를 value에 맞게 자동 수정.
    우선순위:
      1) raw 정확매칭
      2) 정규화 후 정확매칭
      3) 정규화 기반 근사탐색(윈도우)
    실패 시 None
    """
    value = entity.get("value")
    b_old = entity.get("begin")
    e_old = entity.get("end")
    if not isinstance(value, str) or not isinstance(b_old, int) or not isinstance(e_old, int):
        return None

    # 1) raw 정확매칭들 중 가장 가까운 것
    exacts = find_all_exact(text, value)
    if exacts:
        b_new = best_occurrence(exacts, b_old)
        if b_new is not None:
            return b_new, b_new + len(value)

    # 2) 정규화 후 정확매칭
    ntext = norm(text, use_nfkc)
    nvalue = norm(value, use_nfkc)
    exacts_norm = find_all_exact(ntext, nvalue)
    if exacts_norm:
        # 정규화 인덱스를 원문 인덱스로 정확히 역매핑 하기는 어렵지만,
        # 근사적으로 원문에서도 같은 value 길이로 슬라이스 비교해본다(슬라이딩 매칭).
        # 역탐색: 원문 전역에서 후보 b를 찾고 norm(text[b:b+L±8]) == nvalue 인 지점을 선택
        bf = brute_force_norm_match(text, value, b_old, use_nfkc)
        if bf:
            return bf

    # 3) 근사 탐색(정규화 비교, 짧은 윈도우)
    bf = brute_force_norm_match(text, value, b_old, use_nfkc)
    if bf:
        return bf

    return None

def process_row(row, use_nfkc: bool, stats):
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return row  # 구조가 다르면 패스(검증기는 따로 잡을 것)
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
            # 최종 검증: 정규화 비교라도 맞는지 한 번 더 확인
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
    ap = argparse.ArgumentParser(description="Auto-fix (begin,end) offsets to match value")
    ap.add_argument("input", help="input JSONL (messages/system,user,assistant)")
    ap.add_argument("--nfkc", action="store_true",
                    help="compare with NFKC (default NFC)")
    args = ap.parse_args()

    stats = {"lines": 0, "fixed": 0, "unmatched": 0}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                print(raw)
                continue
            stats["lines"] += 1
            try:
                row = json.loads(raw)
            except Exception:
                # 깨진 라인은 그대로 통과(검증기는 따로 에러를 낼 것)
                print(raw)
                continue
            row2 = process_row(row, args.nfkc, stats)
            print(json.dumps(row2, ensure_ascii=False))
    # stderr로 요약
    sys.stderr.write(
        f"[autofix] lines={stats['lines']} fixed={stats['fixed']} unmatched={stats['unmatched']}\n"
    )

if __name__ == "__main__":
    main()
