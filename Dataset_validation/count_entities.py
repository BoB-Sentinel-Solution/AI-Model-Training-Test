# count_entities.py
# -*- coding: utf-8 -*-
import sys, json, argparse, io, unicodedata

def read_text_safely(path: str) -> str:
    with open(path, "rb") as fb:
        data = fb.read()
    # BOM 감지
    if data.startswith(b'\xef\xbb\xbf'):
        return data.decode('utf-8-sig')
    if data.startswith(b'\xff\xfe'):
        return data.decode('utf-16')      # LE
    if data.startswith(b'\xfe\xff'):
        return data.decode('utf-16-be')   # BE
    # 기본 UTF-8, 실패 시 CP949 폴백
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('cp949', errors='replace')

def main():
    ap = argparse.ArgumentParser(description="JSONL에서 id별 중요정보(entities) 개수 집계")
    ap.add_argument("input", help="입력 JSONL 파일 경로")
    args = ap.parse_args()

    text = read_text_safely(args.input)

    per_id = {}           # id -> count
    groups = {}           # count -> [ids]
    total_entities = 0
    total_rows = 0
    bad_lines = 0

    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            bad_lines += 1
            continue

        rid = row.get("id")
        msgs = row.get("messages")
        if rid is None or not isinstance(msgs, list) or len(msgs) < 3:
            bad_lines += 1
            continue

        ac = msgs[2].get("content", "")
        try:
            ans = json.loads(ac)
        except Exception:
            bad_lines += 1
            continue

        ents = ans.get("entities")
        if not isinstance(ents, list):
            bad_lines += 1
            continue

        cnt = len(ents)
        per_id[rid] = cnt
        groups.setdefault(cnt, []).append(rid)
        total_entities += cnt
        total_rows += 1

    # 1) id별 개수 출력
    print("# id별 중요정보 엔티티 개수")
    for rid in sorted(per_id):
        print(f"id {rid}: {per_id[rid]}")

    # 2) 요약
    print("\n# 요약")
    print(f"총 라인 수: {total_rows}, 총 엔티티 수: {total_entities}, 평균: { (total_entities/total_rows) if total_rows else 0:.2f}")
    if bad_lines:
        print(f"(무시된/깨진 라인: {bad_lines})")

    # 3) 5개/4개 및 기타 그룹 출력
    def show_group(k):
        ids = sorted(groups.get(k, []))
        print(f"\n엔티티 {k}개: {len(ids)}개 라인")
        if ids:
            print("ids:", ", ".join(map(str, ids)))

    show_group(5)
    show_group(4)

    # 필요하다면 다른 개수들도 함께 보고 싶을 때:
    others = sorted([k for k in groups.keys() if k not in (4,5)])
    if others:
        print("\n기타 엔티티 개수별:")
        for k in others:
            ids = sorted(groups[k])
            print(f"- {k}개: {len(ids)}개 라인 | ids: {', '.join(map(str, ids))}")

if __name__ == "__main__":
    # Windows 콘솔에서 출력 깨짐 방지(옵션)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
