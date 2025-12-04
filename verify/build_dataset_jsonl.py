# build_dataset_jsonl.py  (robust merge, no validation)
# -*- coding: utf-8 -*-
import json, csv, argparse, sys
from typing import Any, Dict, Iterable, Union, Optional

SYSTEM_TEXT = (
"You are a strict whitelist-only detector for specific entities.\n"
"Given the user's text, return ONLY a JSON with keys\n"
"You must output text in JSON format.\n"
"Input : You receive an arbitrary text\n"
"Output : \n"
"{\n"
"  \"text\": \"<original input text verbatim>\",\n"
"  \"has_sensitive\": <boolean>,\n"
"  \"entities\": [\n"
"    {\n"
"      \"value\": \"<exact substring as it appears>\",\n"
"      \"begin\": <integer>,   // 0-based char offset (inclusive)\n"
"      \"end\": <integer>,     // 0-based char offset (exclusive)\n"
"      \"label\": \"<UPPER_SNAKE_CASE category>\"\n"
"    }\n"
"  ]\n"
"}\n"
"Example:\n"
"Input text:\n"
"{\"AI야, 우리 회사 새 지사 네트워크 설계 도와줘. 본사 게이트웨이는 10.25.30.1이고, 지사는 10.25.31.1로 구성할 거야.\" }\n"
"Expected output (offsets must match the exact input you receive):\n"
"{\n"
"  \"text\": \"AI야, 우리 회사 새 지사 네트워크 설계 도와줘. 본사 게이트웨이는 10.25.30.1이고, 지사는 10.25.31.1로 구성할 거야.\",\n"
"  \"has_sensitive\": true,\n"
"  \"entities\": [\n"
"    {\"value\": \"10.25.30.1\", \"begin\": 39, \"end\": 49, \"label\": \"IPV4\"}, {\"value\": \"10.25.31.1\", \"begin\": 57, \"end\": 67, \"label\": \"IPV4\"}\n"
"  ]\n"
"}\n"
"Example 2:\n"
"Input text:\n"
"{\"고객 불만 사항에 대응하기 위한 표준 절차를 정리해줘.\"}\n"
"Expected Output:\n"
"{\n"
"  \"text\": \"고객 불만 사항에 대응하기 위한 표준 절차를 정리해줘.\",\n"
"  \"has_sensitive\": false,\n"
"  \"entities\": []\n"
"}"
)
SYSTEM_FIXED = {"role": "system", "content": SYSTEM_TEXT}

def read_csv_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row

def read_jsonl_rows(path: str):
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            # \r\n 모두 제거
            s = line.rstrip("\r\n")
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                # 위치 정보(라인/열/오프셋) 확보
                pos    = getattr(e, "pos", None)
                lno    = getattr(e, "lineno", lineno)  # 보통 1
                col    = getattr(e, "colno", None)

                # 주변 문맥 120자
                if pos is not None:
                    start = max(0, pos - 60)
                    end   = min(len(s), pos + 60)
                    snippet = s[start:end]
                    pointer = " " * (pos - start) + "^"
                else:
                    snippet = s[:120]
                    pointer = ""

                # stderr로 즉시 출력 (버퍼링 방지)
                sys.stderr.write(
                    f"\n[JSON ERROR] {path}:{lineno} "
                    f"(lineno={lno}, colno={col}, pos={pos})\n"
                    f"error: {e}\n"
                    f"context: {snippet}\n"
                    f"         {pointer}\n"
                )
                sys.stderr.flush()
                # 더 이상 진행하지 않고 즉시 종료 (상위에서 메시지 누락 방지)
                raise SystemExit(1)
            except Exception as e:
                # 다른 예외도 위치와 함께 노출
                sys.stderr.write(
                    f"\n[READ ERROR] {path}:{lineno}\n{type(e).__name__}: {e}\n"
                    f"line(raw): {s[:120]}\n"
                )
                sys.stderr.flush()
                raise SystemExit(1)


def parse_json_maybe(x: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(x, dict): return x
    return json.loads(x)

def extract_from_messages(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    msgs = row.get("messages")
    if not isinstance(msgs, list): return None
    user_text = None
    assistant_payload = None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = m.get("content"); break
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            assistant_payload = m.get("content"); break
    if user_text is None or assistant_payload is None: return None
    assist_obj = json.loads(assistant_payload) if isinstance(assistant_payload, str) else assistant_payload
    return {"user": user_text, "assistant_json": assist_obj}

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return: {'id': int|None, 'user': str, 'assistant_json': dict}"""
    rec_id = int(row["id"]) if "id" in row and str(row["id"]).strip() != "" else None

    if "user" in row and ("assistant_json" in row or "assistant" in row or "assistant_obj" in row):
        user_text = row["user"]
        assist_raw = row.get("assistant_json") or row.get("assistant") or row.get("assistant_obj")
        assist_obj = parse_json_maybe(assist_raw)
        return {"id": rec_id, "user": user_text, "assistant_json": assist_obj}

    if "content" in row and "has_sensitive" in row and "entities" in row:
        user_text = row["content"]
        assist_obj = {"text": row["content"], "has_sensitive": row["has_sensitive"], "entities": row["entities"]}
        return {"id": rec_id, "user": user_text, "assistant_json": assist_obj}

    from_msgs = extract_from_messages(row)
    if from_msgs is not None:
        return {"id": rec_id, **from_msgs}

    if "assistant" in row and isinstance(row["assistant"], dict) and "content" in row:
        return {"id": rec_id, "user": row["content"], "assistant_json": row["assistant"]}

    raise RuntimeError("Unrecognized input schema for row.")

def build_record(rec_id: int, user_text: str, assist_obj: Dict[str, Any], assistant_as_string: bool) -> Dict[str, Any]:
    assist_payload = json.dumps(assist_obj, ensure_ascii=False, separators=(",",":")) if assistant_as_string else assist_obj
    return {
        "id": rec_id,
        "messages": [
            SYSTEM_FIXED,
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assist_payload}
        ]
    }

def build_from_rows(rows: Iterable[Dict[str, Any]], start_id: int, force_start: bool, assistant_as_string: bool) -> Iterable[Dict[str, Any]]:
    cur_id = start_id
    for idx, row in enumerate(rows, 1):  # ← 라인 인덱스 추적
        try:
            norm = normalize_row(row)
        except Exception as e:
            # 문제 라인 디버그 정보 최대한 노출
            import sys, json
            sys.stderr.write("\n[SCHEMA ERROR] at input row #{idx}\n".format(idx=idx))
            try:
                sys.stderr.write("keys: {keys}\n".format(keys=list(row.keys())))
            except Exception:
                pass
            try:
                sys.stderr.write("row-json: {j}\n".format(j=json.dumps(row, ensure_ascii=False)))
            except Exception:
                sys.stderr.write("row-str: {s}\n".format(s=str(row)))
            sys.stderr.write("error: {e}\n\n".format(e=e))
            sys.stderr.flush()
            raise  # 그대로 중단

        if force_start:
            if cur_id is None:
                raise RuntimeError("--force-start requires --start-id")
            rec_id, cur_id = cur_id, cur_id + 1
        else:
            rec_id = norm["id"]
            if rec_id is None:
                if cur_id is None:
                    raise RuntimeError("Provide --start-id or include 'id' in input rows")
                rec_id, cur_id = cur_id, cur_id + 1

        yield build_record(rec_id, norm["user"], norm["assistant_json"], assistant_as_string)


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",",":")) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Merge fixed SYSTEM + user/assistant into JSONL (robust, no validation)")
    ap.add_argument("--input", required=True, help="입력 파일 (CSV or JSONL)")
    ap.add_argument("--out", required=True, help="출력 JSONL")
    ap.add_argument("--format", choices=["csv","jsonl"], default=None, help="입력 포맷 (미지정 시 확장자로 추론)")
    ap.add_argument("--start-id", type=int, default=None, help="시작 id (입력에 id 없거나 --force-start일 때 사용)")
    ap.add_argument("--force-start", action="store_true", help="입력의 기존 id를 무시하고 --start-id부터 재부여")
    ap.add_argument("--assistant-as-string", action="store_true", help="assistant JSON을 문자열로 저장(내부 \\\" 이스케이프 표시)")
    args = ap.parse_args()

    fmt = args.format or ("csv" if args.input.lower().endswith(".csv") else "jsonl")
    rows = read_csv_rows(args.input) if fmt == "csv" else read_jsonl_rows(args.input)

    write_jsonl(
        build_from_rows(
            rows,
            start_id=args.start_id,
            force_start=args.force_start,
            assistant_as_string=args.assistant_as_string
        ),
        args.out
    )
    print(f"[OK] Wrote -> {args.out}")

if __name__ == "__main__":
    main()
