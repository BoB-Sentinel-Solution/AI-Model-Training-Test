# -*- coding: utf-8 -*-
import json, sys, unicodedata

ALLOWED = {"USERNAME","PASSWORD","EMAIL","ADDRESS","NAME","BANK_ACCOUNT"}

def parse_assistant_json(s):
    # assistant.content 안의 문자열(JSON)을 실제 객체로 파싱
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None, "assistant.content is not a JSON object"
        return obj, None
    except Exception as e:
        return None, f"assistant.content JSON parse error: {e}"

def check_offsets(text, ents):
    errs = []
    for i, e in enumerate(ents):
        for k in ("value","begin","end","label"):
            if k not in e:
                errs.append(f"entity[{i}] missing key {k}")
                continue
        val, b, en, lab = e.get("value"), e.get("begin"), e.get("end"), e.get("label")
        if not isinstance(b, int) or not isinstance(en, int) or not isinstance(val, str):
            errs.append(f"entity[{i}] bad types (begin/end/value)")
            continue
        if not (0 <= b < en <= len(text)):
            errs.append(f"entity[{i}] span out of range: [{b},{en}) vs len={len(text)}")
            continue
        if text[b:en] != val:
            errs.append(f"entity[{i}] slice mismatch: text[{b}:{en}] != value")
        if lab not in ALLOWED:
            errs.append(f"entity[{i}] label not allowed: {lab}")
    return errs

def main(path):
    bad = 0; total = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"[L{ln}] JSON parse error: {e}")
                bad += 1
                continue

            # messages 구조
            msgs = row.get("messages")
            if not isinstance(msgs, list) or len(msgs) != 3:
                print(f"[L{ln}] messages must be list of length 3")
                bad += 1; continue

            roles = [m.get("role") for m in msgs]
            if roles != ["system","user","assistant"]:
                print(f"[L{ln}] role order must be system,user,assistant (got {roles})")
                bad += 1

            for ri, m in enumerate(msgs):
                if "content" not in m or not isinstance(m["content"], str):
                    print(f"[L{ln}] messages[{ri}] missing content or not string")
                    bad += 1

            # assistant.content 파싱
            ac = msgs[2].get("content","")
            ans, err = parse_assistant_json(ac)
            if err:
                print(f"[L{ln}] {err}")
                bad += 1
                continue

            # 정답 JSON 스키마 검사
            exp_keys = {"text","has_sensitive","entities"}
            if set(ans.keys()) != exp_keys:
                print(f"[L{ln}] assistant JSON keys must be {exp_keys} (got {set(ans.keys())})")
                bad += 1

            text = ans.get("text")
            hs = ans.get("has_sensitive")
            ents = ans.get("entities")

            if not isinstance(text, str):
                print(f"[L{ln}] 'text' must be string")
                bad += 1
            if not isinstance(hs, bool):
                print(f"[L{ln}] 'has_sensitive' must be boolean")
                bad += 1
            if not isinstance(ents, list):
                print(f"[L{ln}] 'entities' must be list")
                bad += 1
                continue

            # 오프셋/라벨 검사
            errs = check_offsets(text, ents)
            for e in errs:
                print(f"[L{ln}] {e}")
            if errs:
                bad += 1

            # has_sensitive 논리 일치
            if (len(ents) > 0) != bool(hs):
                print(f"[L{ln}] has_sensitive mismatch: entities={len(ents)} hs={hs}")
                bad += 1

            # 중복 엔티티 검사
            seen = set()
            for e in ents:
                k = (e.get("label"), e.get("begin"), e.get("end"))
                if k in seen:
                    print(f"[L{ln}] duplicate entity (label,begin,end)={k}")
                    bad += 1
                    break
                seen.add(k)

            # 유니코드 정규화 경고(선택)
            if text != unicodedata.normalize("NFC", text):
                print(f"[L{ln}] WARNING: text not NFC-normalized (may cause offset drift)")

    print(f"\nChecked {total} lines. Problems: {bad}")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv)>1 else (print("Usage: python validate_messages_jsonl.py <file.jsonl>") or 2))
