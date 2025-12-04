# fix_trim_spaces.py
# -*- coding: utf-8 -*-
import json, sys

def fix_line(obj):
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return obj
    ac = msgs[2].get("content", "")
    ans = json.loads(ac) if isinstance(ac, str) else ac
    if not isinstance(ans, dict): 
        return obj

    text = ans.get("text", "")
    ents = ans.get("entities", [])
    changed = False

    new_ents = []
    for e in ents:
        v = e.get("value", "")
        b = int(e.get("begin", 0))
        en = int(e.get("end", 0))
        if not (0 <= b <= en <= len(text)):
            new_ents.append(e)
            continue

        # 앞/뒤 공백을 본문 기준으로 잘라서 begin/end를 조정
        b2, en2 = b, en
        while b2 < en2 and text[b2].isspace():
            b2 += 1
        while b2 < en2 and text[en2 - 1].isspace():
            en2 -= 1

        if (b2, en2) != (b, en):
            nv = text[b2:en2]
            new_ents.append({"value": nv, "begin": b2, "end": en2, "label": e.get("label")})
            changed = True
        else:
            # value 자체에만 공백이 있었던 경우도 본문 슬라이스로 동기화
            if v != text[b:en]:
                new_ents.append({"value": text[b:en], "begin": b, "end": en, "label": e.get("label")})
                changed = True
            else:
                new_ents.append(e)

    if changed:
        ans["entities"] = new_ents
        # has_sensitive 동기화
        ans["has_sensitive"] = bool(new_ents)
        # assistant.content가 문자열(JSON-in-JSON)일 수도 있으니 동일 형태 유지
        if isinstance(ac, str):
            msgs[2]["content"] = json.dumps(ans, ensure_ascii=False, separators=(",", ":"))
        else:
            msgs[2]["content"] = ans
        obj["messages"] = msgs
    return obj

def main(in_path, out_path):
    n_in = n_out = 0
    with open(in_path, encoding="utf-8") as fr, open(out_path, "w", encoding="utf-8", newline="\n") as fw:
        for line in fr:
            s = line.strip()
            if not s:
                continue
            n_in += 1
            try:
                obj = json.loads(s)
            except Exception:
                # 그대로 통과(필요시 건너뛰기)
                fw.write(line if line.endswith("\n") else line + "\n")
                n_out += 1
                continue
            fixed = fix_line(obj)
            fw.write(json.dumps(fixed, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_out += 1
    print(f"[OK] read={n_in}, wrote={n_out}, out={out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_trim_spaces.py <in.jsonl> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
