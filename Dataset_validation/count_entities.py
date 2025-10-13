# count_entities.py (revised)
# -*- coding: utf-8 -*-
import sys, json, argparse, io

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

def parse_answer_field(answer_field):
    """
    answer 필드에서 dict를 파싱해 반환.
    - 문자열이면 JSON으로 로드
    - dict이면 그대로
    반환: (ans_dict or None)
    """
    if isinstance(answer_field, str):
        try:
            return json.loads(answer_field)
        except Exception:
            return None
    elif isinstance(answer_field, dict):
        return answer_field
    else:
        return None

def extract_entities_from_row(row, ln, strict_id=False):
    """
    한 줄(row)에서 (rid, entities or None) 추출.
    지원 포맷:
      - messages[2].content(JSON 문자열)
      - id/answer (answer는 JSON 문자열 또는 객체)
    rid가 없을 때:
      - strict_id=True면 None 반환(무시 대상)
      - strict_id=False면 '@L<ln>' 사용
    """
    rid = row.get("id")
    # (A) messages 포맷
    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 3 and isinstance(msgs[2], dict):
        ac = msgs[2].get("content", "")
        try:
            ans = json.loads(ac)
        except Exception:
            return (rid if rid is not None else (None if strict_id else f"@L{ln}"), None)
        ents = ans.get("entities")
        return (rid if rid is not None else (None if strict_id else f"@L{ln}"),
                ents if isinstance(ents, list) else None)

    # (B) id/answer 포맷
    if "answer" in row:
        ans = parse_answer_field(row["answer"])
        if not isinstance(ans, dict):
            return (rid if rid is not None else (None if strict_id else f"@L{ln}"), None)
        ents = ans.get("entities")
        return (rid if rid is not None else (None if strict_id else f"@L{ln}"),
                ents if isinstance(ents, list) else None)

    # 지원하지 않는 포맷
    return (rid if rid is not None else (None if strict_id else f"@L{ln}"), None)

def main():
    ap = argparse.ArgumentParser(description="JSONL에서 레코드별 중요정보(entities) 개수 집계 (messages 또는 id/answer 포맷 지원)")
    ap.add_argument("input", help="입력 JSONL 파일 경로")
    ap.add_argument("--strict-id", action="store_true", help="id가 없는 라인은 무시(기본은 @L<라인번호>로 대체)")
    args = ap.parse_args()

    text = read_text_safely(args.input)

    per_id = {}           # rid -> count
    groups = {}           # count -> [rids]
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

        rid, ents = extract_entities_from_row(row, ln, strict_id=args.strict_id)
        if rid is None:
            bad_lines += 1
            continue
        if ents is None:
            bad_lines += 1
            continue

        cnt = len(ents)
        per_id[rid] = cnt
        groups.setdefault(cnt, []).append(rid)
        total_entities += cnt
        total_rows += 1

    # 1) id별 개수 출력
    print("# id별 중요정보 엔티티 개수")
    # 정렬: 숫자형 id도, 문자열 id도 무리 없이 정렬되도록 tuple 키 사용
    def sort_key(k):
        try:
            return (0, int(k))
        except Exception:
            return (1, str(k))
    for rid in sorted(per_id, key=sort_key):
        print(f"id {rid}: {per_id[rid]}")

    # 2) 요약
    print("\n# 요약")
    avg = (total_entities/total_rows) if total_rows else 0.0
    print(f"총 라인 수(집계 성공): {total_rows}, 총 엔티티 수: {total_entities}, 평균: {avg:.2f}")
    if bad_lines:
        print(f"(무시된/깨진 라인: {bad_lines})")

    # 3) 5개/4개 및 기타 그룹 출력
    def show_group(k):
        ids = sorted(groups.get(k, []), key=sort_key)
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
            ids = sorted(groups[k], key=sort_key)
            print(f"- {k}개: {len(ids)}개 라인 | ids: {', '.join(map(str, ids))}")

if __name__ == "__main__":
    # Windows 콘솔에서 출력 깨짐 방지(옵션)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
