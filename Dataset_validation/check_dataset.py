# check_dataset.py (revised)
# -*- coding: utf-8 -*-

import json
import sys
import unicodedata
import argparse
import io
import re

# -------------------- ALLOWED 라벨 (사용자 제공 버전) --------------------
ALLOWED = {
    # 개인 식별·연락
    "NAME","PHONE","EMAIL","ADDRESS","POSTAL_CODE","DATE_OF_BIRTH","RESIDENT_ID",
    "PASSPORT","DRIVER_LICENSE","FOREIGNER_ID","HEALTH_INSURANCE_ID","BUSINESS_ID",
    "TAX_ID","SSN","EMERGENCY_CONTACT","EMERGENCY_PHONE",

    # 계정·인증
    "USERNAME","NICKNAME","ROLE","GROUP","PASSWORD","PASSWORD_HASH","SECURITY_QA",
    "MFA_SECRET","BACKUP_CODE","LAST_LOGIN_IP","LAST_LOGIN_DEVICE","LAST_LOGIN_BROWSER",
    "SESSION_ID","COOKIE","JWT","ACCESS_TOKEN","REFRESH_TOKEN","OAUTH_CLIENT_ID",
    "OAUTH_CLIENT_SECRET","API_KEY","SSH_PRIVATE_KEY","TLS_PRIVATE_KEY","PGP_PRIVATE_KEY",
    "MNEMONIC","TEMP_CLOUD_CREDENTIAL","DEVICE_ID","IMEI","SERIAL_NUMBER",
    "BROWSER_FINGERPRINT","SAML_ASSERTION","OIDC_ID_TOKEN","INTERNAL_URL",
    "CONNECTION_STRING","LAST_LOGIN_AT",

    # 금융·결제
    "BANK_ACCOUNT","BANK_NAME","BANK_BRANCH","ACCOUNT_HOLDER","BALANCE","CURRENCY",
    "CARD_NUMBER","CARD_EXPIRY","CARD_HOLDER","CARD_CVV","PAYMENT_PIN",
    "SECURITIES_ACCOUNT","VIRTUAL_ACCOUNT","WALLET_ADDRESS","IBAN","SWIFT_BIC",
    "ROUTING_NUMBER","PAYMENT_APPROVAL_CODE","GATEWAY_CUSTOMER_ID","PAYMENT_PROFILE_ID",

    # 고객·거래·지원
    "COMPANY_NAME","BUYER_NAME","CUSTOMER_ID","MEMBERSHIP_ID","ORDER_ID","INVOICE_ID",
    "REFUND_ID","EXCHANGE_ID","SHIPPING_ADDRESS","TRACKING_ID","CRM_RECORD_ID",
    "TICKET_ID","RMA_ID","COUPON_CODE","VOUCHER_CODE","BILLING_ADDRESS",
    "TAX_INVOICE_ID","CUSTOMER_NOTE_ID",

    # 조직
    "EMPLOYEE_ID","ORG_NAME","DEPARTMENT_NAME","JOB_TITLE","EMPLOYMENT_TYPE",
    "HIRE_DATE","LEAVE_DATE","SALARY","BENEFIT_INFO","INSURANCE_INFO","PROFILE_INFO",
    "OFFICE_EXT","ACCESS_CARD_ID","READER_ID","WORKSITE","OFFICE_LOCATION",
    "PERFORMANCE_GRADE","EDUCATION_CERT","ACCESS_LOG","DUTY_ASSIGNMENT","MANAGER_FLAG",
    "TRAINING_COMPLETION_DATE","TRAINING_EXPIRY"
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

def extract_payload(row, *, ln, strict_messages=True):
    """
    row에서 유효한 payload(text, has_sensitive, entities)를 뽑아온다.
    두 포맷 지원:
      (A) {"messages":[{"role":"system"...},{"role":"user"...},{"role":"assistant","content":"{...}"}]}
      (B) {"id":"...", "answer":"{...}"}  또는 {"id":"...", "answer":{...}}
    반환: (payload_dict or None, errors[list of str])
    """
    errs = []

    # (A) messages 포맷
    if "messages" in row:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            errs.append(f"[L{ln}] messages must be list of length 3")
            return None, errs

        roles = [m.get("role") for m in msgs]
        if roles != ["system","user","assistant"]:
            errs.append(f"[L{ln}] role order must be system,user,assistant (got {roles})")

        for ri, m in enumerate(msgs):
            if "content" not in m or not isinstance(m["content"], str):
                errs.append(f"[L{ln}] messages[{ri}] missing content or not string")

        ac = msgs[2].get("content", "")
        ans, err = parse_assistant_json(ac)
        if err:
            errs.append(f"[L{ln}] {err}")
            return None, errs
        return ans, errs

    # (B) id/answer 포맷
    if "answer" in row:
        ans_raw = row["answer"]
        if isinstance(ans_raw, str):
            try:
                ans = json.loads(ans_raw)
            except Exception as e:
                errs.append(f"[L{ln}] answer JSON parse error: {e}")
                return None, errs
        elif isinstance(ans_raw, dict):
            ans = ans_raw
        else:
            errs.append(f"[L{ln}] 'answer' must be JSON string or object")
            return None, errs

        if not isinstance(ans, dict):
            errs.append(f"[L{ln}] answer is not a JSON object")
            return None, errs
        return ans, errs

    # 인식 불가 포맷
    errs.append(f"[L{ln}] unsupported format: expected 'messages' or 'answer' key")
    return None, errs

def validate_payload(ans, *, ln):
    """payload(dict)에서 키/타입 검사 후 (text, has_sensitive, entities, errors) 반환"""
    errs = []
    exp_keys = {"text","has_sensitive","entities"}
    if set(ans.keys()) != exp_keys:
        errs.append(f"[L{ln}] assistant JSON keys must be {exp_keys} (got {set(ans.keys())})")

    text_body = ans.get("text")
    hs = ans.get("has_sensitive")
    ents = ans.get("entities")

    if not isinstance(text_body, str):
        errs.append(f"[L{ln}] 'text' must be string")
    if not isinstance(hs, bool):
        errs.append(f"[L{ln}] 'has_sensitive' must be boolean")
    if not isinstance(ents, list):
        errs.append(f"[L{ln}] 'entities' must be list")

    return text_body, hs, ents, errs

def main():
    ap = argparse.ArgumentParser(description="Dataset validator for messages/answer JSONL")
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

        # 공통 payload 추출 (messages 또는 answer)
        ans, errs = extract_payload(row, ln=ln)
        if errs:
            for e in errs:
                print(e)
            if ans is None:
                bad += 1
                continue  # 다음 라인

        # payload 스키마 검사
        text_body, hs, ents, keyerrs = validate_payload(ans, ln=ln)
        for e in keyerrs:
            print(e)
        if keyerrs:
            bad += 1
            # entities 타입이 아니면 이후 오프셋 검사 불가
            if not isinstance(ents, list) or not isinstance(text_body, str):
                continue

        # 오프셋/라벨 검사
        offs = check_offsets(
            text_body, ents,
            use_nfkc=args.nfkc,
            allow_overlap=args.allow_overlap,
            strict_entity_keys=args.strict_entity_keys,
            warn_sort=not args.no_sort_warn
        )
        for e in offs:
            print(f"[L{ln}] {e}")
        if offs:
            bad += 1

        # has_sensitive 논리 일치
        if isinstance(hs, bool) and ((len(ents) > 0) != bool(hs)):
            print(f"[L{ln}] has_sensitive mismatch: entities={len(ents)} hs={hs}")
            bad += 1

    print(f"\nChecked {total} lines. Problems: {bad}")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
