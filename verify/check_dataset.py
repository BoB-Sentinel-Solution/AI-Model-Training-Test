# check_dataset.py
# -*- coding: utf-8 -*-

import json
import sys
import unicodedata
import argparse
import io
import re

# -------------------- ALLOWED 라벨 (사용자 제공 버전) --------------------
ALLOWED = {
    # 기본 신원 정보
    "NAME",                     # 성명
    "PHONE",                    # 전화번호
    "EMAIL",                    # Email
    "ADDRESS",                  # 주소(시·구·동/도로명·상세)
    "POSTAL_CODE",              # 우편번호
    
	# 공적 식별번호
	"PERSONAL_CUSTOMS_ID",      # 개인통관고유번호
	"RESIDENT_ID",              # 주민등록번호
    "PASSPORT",                 # 여권번호
    "DRIVER_LICENSE",           # 운전면허번호
    "FOREIGNER_ID",             # 외국인 등록번호
    "HEALTH_INSURANCE_ID",      # 건강보험증번호
    "BUSINESS_ID",              # 사업자 등록번호
    "MILITARY_ID",              # 군번
    
    # 인증 정보
    "JWT",                      # JWT (eyJ…)
    "API_KEY",                  # API 키 (Google/OpenAI/Kakao 등)
    "GITHUB_PAT",               # GitHub Personal Access Token
    "PRIVATE_KEY",              # SSH TLS/SSL PGP 개인키

    # 금융 정보
    "CARD_NUMBER",              # 카드 번호 (Luhn/모듈러스10 검증)
    "CARD_EXPIRY",              # 카드 유효 기간 (MM/YY)
    "BANK_ACCOUNT",             # 계좌번호
    "CARD_CVV",                 # CVC/CVV
    "PAYMENT_PIN",              # 통장/카드/ATM 비밀번호 (4자리 PIN)
    "MOBILE_PAYMENT_PIN",       # 모바일 결제 비밀번호 (6자리 PIN)

    # 가상화폐 정보
    "MNEMONIC",                 # 복구 시드/니모닉
    "CRYPTO_PRIVATE_KEY",       # (가상자산) 개인 키
    "HD_WALLET",                # HD 지갑 확장키 
	"PAYMENT_URI_QR",           # 결제 URI/QR (BTC, ETH, XRP, SOLANA, TRON)
		
    # 네트워크 정보 / 기타
    "IPV4",                     # ip v4
    "IPV6",                     # ip v6
    "MAC_ADDRESS",              # mac 주소
    "IMEI",                     # IMEI
}
# ------------------------------------------------------------------------

CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")

def normalize_text(s: str, use_nfkc: bool) -> str:
    return unicodedata.normalize("NFKC" if use_nfkc else "NFC", s)

def parse_assistant_json(s):
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None, "assistant.content is not a JSON object"
        return obj, None
    except Exception as e:
        return None, f"assistant.content JSON parse error: {e}"

def read_text_safely(path: str) -> str:
    """UTF-8 / UTF-8-SIG / UTF-16LE/BE 자동 인식, 실패시 CP949 폴백."""
    with open(path, "rb") as fb:
        data = fb.read()
    if data.startswith(b'\xef\xbb\xbf'):
        return data.decode('utf-8-sig')
    if data.startswith(b'\xff\xfe'):
        return data.decode('utf-16')      # LE
    if data.startswith(b'\xfe\xff'):
        return data.decode('utf-16-be')   # BE
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('cp949', errors='replace')

def check_offsets(text, ents, *, use_nfkc=False, allow_overlap=False, strict_entity_keys=False, warn_sort=True):
    errs = []
    norm_text = normalize_text(text, use_nfkc)

    prev_begin = -1
    seen = set()  # (label, begin, end)
    spans_sorted = []

    for i, e in enumerate(ents):
        # 스키마 키 검사
        req = {"value","begin","end","label"}
        if strict_entity_keys:
            extra = set(e.keys()) - req
            missing = req - set(e.keys())
            if missing:
                errs.append(f"entity[{i}] missing keys {sorted(missing)}")
            if extra:
                errs.append(f"entity[{i}] unexpected keys {sorted(extra)}")
        else:
            for k in ("value","begin","end","label"):
                if k not in e:
                    errs.append(f"entity[{i}] missing key {k}")
                    # 다음 검사 최소화
                    continue

        val, b, en, lab = e.get("value"), e.get("begin"), e.get("end"), e.get("label")

        # 타입 검사
        if not isinstance(b, int) or not isinstance(en, int) or not isinstance(val, str) or not isinstance(lab, str):
            errs.append(f"entity[{i}] bad types (begin/end/value/label)")
            continue

        # 범위 검사
        if not (0 <= b < en <= len(text)):
            errs.append(f"entity[{i}] span out of range: [{b},{en}) vs len={len(text)}")
            continue

        # slice 일치(정규화 기준 선택 가능)
        raw_slice = text[b:en]
        if normalize_text(raw_slice, use_nfkc) != normalize_text(val, use_nfkc):
            errs.append(f"entity[{i}] slice mismatch: text[{b}:{en}] != value")

        # 허용 라벨
        if lab not in ALLOWED:
            errs.append(f"entity[{i}] label not allowed: {lab}")

        # 공백/제어문자 경고
        if val != val.strip():
            errs.append(f"entity[{i}] value has leading/trailing spaces")
        if CTRL_RE.search(val) or CTRL_RE.search(raw_slice):
            errs.append(f"entity[{i}] value contains control chars")

        # 정렬 경고
        if warn_sort and b < prev_begin:
            errs.append("entities not sorted by begin offset")
        prev_begin = b

        # 중복/겹침 검사 준비
        key = (lab, b, en)
        if key in seen:
            errs.append(f"duplicate entity (label,begin,end)={key}")
        seen.add(key)
        spans_sorted.append((b, en))

    # 겹침 검사
    spans_sorted.sort()
    if not allow_overlap:
        for j in range(len(spans_sorted) - 1):
            b1, e1 = spans_sorted[j]
            b2, e2 = spans_sorted[j + 1]
            if b2 < e1:
                errs.append(f"overlapping spans: [{b1},{e1}) & [{b2},{e2})")
                break

    # 본문 정규화 경고
    if text != unicodedata.normalize("NFC", text):
        errs.append("WARNING: text not NFC-normalized (may cause offset drift)")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Dataset validator for messages JSONL")
    ap.add_argument("path", nargs=1, help="input JSONL file")
    ap.add_argument("--nfkc", action="store_true", help="use NFKC normalization for slice comparison (default NFC)")
    ap.add_argument("--allow-overlap", action="store_true", help="do not error on overlapping entity spans")
    ap.add_argument("--strict-entity-keys", action="store_true", help="error on extra/missing keys in entity objects")
    ap.add_argument("--no-sort-warn", action="store_true", help="disable sorted-by-begin warning")
    args = ap.parse_args()

    path = args.path[0]
    bad = 0
    total = 0

    # Windows 콘솔 안전(stderr)
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    text = read_text_safely(path)
    for ln, line in enumerate(text.splitlines(), 1):
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
            bad += 1
            continue

        roles = [m.get("role") for m in msgs]
        if roles != ["system","user","assistant"]:
            print(f"[L{ln}] role order must be system,user,assistant (got {roles})")
            bad += 1

        for ri, m in enumerate(msgs):
            if "content" not in m or not isinstance(m["content"], str):
                print(f"[L{ln}] messages[{ri}] missing content or not string")
                bad += 1

        # assistant.content 파싱
        ac = msgs[2].get("content", "")
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

        text_body = ans.get("text")
        hs = ans.get("has_sensitive")
        ents = ans.get("entities")

        if not isinstance(text_body, str):
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
        errs = check_offsets(
            text_body, ents,
            use_nfkc=args.nfkc,
            allow_overlap=args.allow_overlap,
            strict_entity_keys=args.strict_entity_keys,
            warn_sort=not args.no_sort_warn
        )
        for e in errs:
            print(f"[L{ln}] {e}")
        if errs:
            bad += 1

        # has_sensitive 논리 일치
        if (len(ents) > 0) != bool(hs):
            print(f"[L{ln}] has_sensitive mismatch: entities={len(ents)} hs={hs}")
            bad += 1

    print(f"\nChecked {total} lines. Problems: {bad}")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
