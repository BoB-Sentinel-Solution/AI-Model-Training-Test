# fix_content.py
# 사용법:
#   python fix_content.py dataset_full_build_1-56000.jsonl
#   -> dataset_full_build_1-56000.jsonl.fixed.jsonl 으로 저장

import json
import argparse
from pathlib import Path

def fix_jsonl(in_path: str, out_path: str | None = None):
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(in_path.suffix + ".fixed.jsonl")
    else:
        out_path = Path(out_path)

    total = 0
    patched = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for ln, line in enumerate(fin, 1):
            raw = line.strip()
            if not raw:
                # 빈 줄은 그냥 스킵 (원래 check_dataset.py도 이렇게 함)
                continue

            total += 1
            try:
                row = json.loads(raw)
            except Exception as e:
                print(f"[L{ln}] JSON parse error: {e}")
                continue

            msgs = row.get("messages")
            if isinstance(msgs, list) and len(msgs) == 3:
                assistant = msgs[2]
                if isinstance(assistant, dict) and "content" in assistant:
                    ac = assistant["content"]
                    # ✅ content가 dict이면 문자열 JSON으로 변환
                    if isinstance(ac, dict):
                        assistant["content"] = json.dumps(ac, ensure_ascii=False)
                        patched += 1

            # 한 줄씩 다시 JSON 문자열로 써줌
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Total lines: {total}, patched: {patched}")
    print(f"Output -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Fix messages[2].content dict -> JSON string")
    ap.add_argument("path", help="input JSONL file")
    ap.add_argument("--out", help="output JSONL file (default: *.fixed.jsonl)", default=None)
    args = ap.parse_args()
    fix_jsonl(args.path, args.out)

if __name__ == "__main__":
    main()
