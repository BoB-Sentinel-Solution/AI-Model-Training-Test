#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
민감정보 탐지 DLP 모델 성능평가 (튜닝 전/후 비교, 향상률 포함)
- 공통 시스템 프롬프트(SYS_PROMPT) 강제 적용 (모델 무관)
- 모델 출력(JSON: has_sensitive/entities[type,value]) → 평가 포맷(begin/end/label) 자동 변환
- Precision / Recall / F1 (Micro/Macro), 라벨별 상세, Latency/Throughput, 향상률, 그래프, PDF
- (--save-steps) 단계별 디버그 로그: <모델명>_debug.jsonl (raw_output/parsed/filter_actions/filtered_entities/std_entities)
- (--span-only) 라벨 무시 스팬 매칭 평가(탐지력만 평가)
"""

import argparse
import json
import os
import time
import difflib
from collections import Counter
from typing import Dict, List, Tuple, Optional
import gc
import torch

# ==== 전역 Matplotlib 한글 폰트 보장 ====
from matplotlib import font_manager as fm, pyplot as plt
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
except Exception:
    pass
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
# ======================================

def free_cuda():
    """강제 GC + CUDA 캐시/IP C 청소 (모델 전환 전/후 호출)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

# =========================
# 1) 공통 시스템 프롬프트
# =========================
SYS_PROMPT = """
You are a strict detector for sensitive entities (PII and secrets).

Return ONLY a compact JSON with these keys:
- has_sensitive: true or false
- entities: list of {"type": <LABEL>, "value": <exact substring>}

HARD RULES
- Allowed labels ONLY (uppercase, exact match). If a label is not in the list below, DO NOT invent or output it.
- If the text contains none of the allowed entities: return exactly {"has_sensitive": false, "entities": []}.
- `value` must be the exact substring from the user text (no masking, no redaction, no normalization).
- Output JSON only — no explanations, no extra text, no code fences, no trailing commas.
- The JSON must be valid and parseable.

ALLOWED LABELS
# 1) Personal Identification & Contact
NAME, PHONE, EMAIL, ADDRESS, POSTAL_CODE, DATE_OF_BIRTH, RESIDENT_ID, PASSPORT, DRIVER_LICENSE,
FOREIGNER_ID, HEALTH_INSURANCE_ID, BUSINESS_ID, TAX_ID, SSN, EMERGENCY_CONTACT, EMERGENCY_PHONE,

# 2) Account & Auth
USERNAME, NICKNAME, ROLE, GROUP, PASSWORD, PASSWORD_HASH, SECURITY_QA, MFA_SECRET, BACKUP_CODE,
LAST_LOGIN_IP, LAST_LOGIN_DEVICE, LAST_LOGIN_BROWSER, SESSION_ID, COOKIE, JWT, ACCESS_TOKEN,
REFRESH_TOKEN, OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, API_KEY, SSH_PRIVATE_KEY, TLS_PRIVATE_KEY,
PGP_PRIVATE_KEY, MNEMONIC, TEMP_CLOUD_CREDENTIAL, DEVICE_ID, IMEI, SERIAL_NUMBER,
BROWSER_FINGERPRINT, SAML_ASSERTION, OIDC_ID_TOKEN, INTERNAL_URL, CONNECTION_STRING, LAST_LOGIN_AT,

# 3) Finance & Payment
BANK_ACCOUNT, BANK_NAME, BANK_BRANCH, ACCOUNT_HOLDER, BALANCE, CURRENCY, CARD_NUMBER, CARD_EXPIRY,
CARD_HOLDER, CARD_CVV, PAYMENT_PIN, SECURITIES_ACCOUNT, VIRTUAL_ACCOUNT, WALLET_ADDRESS, IBAN,
SWIFT_BIC, ROUTING_NUMBER, PAYMENT_APPROVAL_CODE, GATEWAY_CUSTOMER_ID, PAYMENT_PROFILE_ID,

# 4) Customer / Order / Support
COMPANY_NAME, BUYER_NAME, CUSTOMER_ID, MEMBERSHIP_ID, ORDER_ID, INVOICE_ID, REFUND_ID, EXCHANGE_ID,
SHIPPING_ADDRESS, TRACKING_ID, CRM_RECORD_ID, TICKET_ID, RMA_ID, COUPON_CODE, VOUCHER_CODE,
BILLING_ADDRESS, TAX_INVOICE_ID, CUSTOMER_NOTE_ID,

# 5) Organization
EMPLOYEE_ID, ORG_NAME, DEPARTMENT_NAME, JOB_TITLE, EMPLOYMENT_TYPE, HIRE_DATE, LEAVE_DATE, SALARY,
BENEFIT_INFO, INSURANCE_INFO, PROFILE_INFO, OFFICE_EXT, ACCESS_CARD_ID, READER_ID, WORKSITE,
OFFICE_LOCATION, PERFORMANCE_GRADE, EDUCATION_CERT, ACCESS_LOG, DUTY_ASSIGNMENT, MANAGER_FLAG,
TRAINING_COMPLETION_DATE, TRAINING_EXPIRY
"""

# ALLOWED_LABELS: 프롬프트와 동일 집합(대문자, 공백 제거)
ALLOWED_LABELS = [
    "NAME","PHONE","EMAIL","ADDRESS","POSTAL_CODE","DATE_OF_BIRTH","RESIDENT_ID","PASSPORT","DRIVER_LICENSE",
    "FOREIGNER_ID","HEALTH_INSURANCE_ID","BUSINESS_ID","TAX_ID","SSN","EMERGENCY_CONTACT","EMERGENCY_PHONE",
    "USERNAME","NICKNAME","ROLE","GROUP","PASSWORD","PASSWORD_HASH","SECURITY_QA","MFA_SECRET","BACKUP_CODE",
    "LAST_LOGIN_IP","LAST_LOGIN_DEVICE","LAST_LOGIN_BROWSER","SESSION_ID","COOKIE","JWT","ACCESS_TOKEN",
    "REFRESH_TOKEN","OAUTH_CLIENT_ID","OAUTH_CLIENT_SECRET","API_KEY","SSH_PRIVATE_KEY","TLS_PRIVATE_KEY",
    "PGP_PRIVATE_KEY","MNEMONIC","TEMP_CLOUD_CREDENTIAL","DEVICE_ID","IMEI","SERIAL_NUMBER","BROWSER_FINGERPRINT",
    "SAML_ASSERTION","OIDC_ID_TOKEN","INTERNAL_URL","CONNECTION_STRING","LAST_LOGIN_AT",
    "BANK_ACCOUNT","BANK_NAME","BANK_BRANCH","ACCOUNT_HOLDER","BALANCE","CURRENCY","CARD_NUMBER","CARD_EXPIRY",
    "CARD_HOLDER","CARD_CVV","PAYMENT_PIN","SECURITIES_ACCOUNT","VIRTUAL_ACCOUNT","WALLET_ADDRESS","IBAN",
    "SWIFT_BIC","ROUTING_NUMBER","PAYMENT_APPROVAL_CODE","GATEWAY_CUSTOMER_ID","PAYMENT_PROFILE_ID",
    "COMPANY_NAME","BUYER_NAME","CUSTOMER_ID","MEMBERSHIP_ID","ORDER_ID","INVOICE_ID","REFUND_ID","EXCHANGE_ID",
    "SHIPPING_ADDRESS","TRACKING_ID","CRM_RECORD_ID","TICKET_ID","RMA_ID","COUPON_CODE","VOUCHER_CODE",
    "BILLING_ADDRESS","TAX_INVOICE_ID","CUSTOMER_NOTE_ID",
    "EMPLOYEE_ID","ORG_NAME","DEPARTMENT_NAME","JOB_TITLE","EMPLOYMENT_TYPE","HIRE_DATE","LEAVE_DATE","SALARY",
    "BENEFIT_INFO","INSURANCE_INFO","PROFILE_INFO","OFFICE_EXT","ACCESS_CARD_ID","READER_ID","WORKSITE",
    "OFFICE_LOCATION","PERFORMANCE_GRADE","EDUCATION_CERT","ACCESS_LOG","DUTY_ASSIGNMENT","MANAGER_FLAG",
    "TRAINING_COMPLETION_DATE","TRAINING_EXPIRY"
]

# =========================
# 2) 공통 유틸/평가
# =========================
def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_jsonl_by_id(path: str) -> Dict[str, dict]:
    """
    표준 포맷과 기존(문자열로 감싼) 포맷을 모두 지원:
      1) 표준: {"id":"...","entities":[{"begin":..,"end":..,"label":"..."}]}
      2) 기존: {"id":"...","answer":"{\"text\":\"...\",\"entities\":[...]}"}  # answer 안에 JSON 문자열
    반환값은 항상 {"id":..., "entities":[...]} 형태로 정규화해서 돌려준다.
    """
    def _sanitize_entities(ents):
        out = []
        for e in ents or []:
            try:
                b = int(e.get("begin"))
                ed = int(e.get("end"))
                lab = str(e.get("label"))
                out.append({"begin": b, "end": ed, "label": lab})
            except Exception:
                continue
        return out

    data: Dict[str, dict] = {}
    for obj in load_jsonl(path):
        sid = str(obj.get("id"))

        if isinstance(obj.get("entities"), list):
            data[sid] = {"id": sid, "entities": _sanitize_entities(obj["entities"])}
            continue

        ans = obj.get("answer")
        if isinstance(ans, str):
            try:
                inner = json.loads(ans)  # 문자열 → dict
                ents = inner.get("entities", [])
                data[sid] = {"id": sid, "entities": _sanitize_entities(ents)}
                continue
            except Exception:
                print(f"[WARN] answers parse failed for id={sid}: invalid JSON in 'answer'")
                data[sid] = {"id": sid, "entities": []}
                continue

        data[sid] = {"id": sid, "entities": []}

    return data

def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0

def iou_span(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    (a0,a1), (b0,b1) = a, b
    inter = max(0, min(a1,b1) - max(a0,b0))
    if inter == 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def chunk_text(text: str, tokenizer, stride_chars: int = 50) -> List[Tuple[int, str]]:
    if not text:
        return [(0, "")]
    max_chars = 4096
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        chunk = text[i:end]
        chunks.append((i, chunk))
        if end == n:
            break
        i = end - min(stride_chars, end - i)
    return chunks

def match_entities(preds, golds, mode="exact", iou_th=0.5, span_only: bool=False):
    """
    preds, golds: [(begin, end, label), ...]
    span_only=True 이면 라벨을 무시하고 스팬 겹침으로만 매칭한다.
    TP/FN은 '골드 라벨' 기준으로 누적하여 per-label 리포트가 의미있게 보이도록 한다.
    """
    used_g = [False]*len(golds)
    tp = 0
    label_counts = Counter()

    for pb, pe, pl in preds:
        match_idx, best_score = -1, -1.0
        for gi, (gb, ge, gl) in enumerate(golds):
            if used_g[gi]:
                continue

            # 라벨 비교 여부
            if not span_only and (pl != gl):
                continue

            if mode == "exact":
                ok = (pb == gb) and (pe == ge)
                score = 1.0 if ok else -1.0
            else:
                score = iou_span((pb, pe), (gb, ge))
                ok = score >= iou_th

            if ok and score > best_score:
                best_score = score
                match_idx = gi

        if match_idx >= 0:
            used_g[match_idx] = True
            gold_label = golds[match_idx][2]
            label_counts[("TP", gold_label)] += 1
            tp += 1
        else:
            # FP는 예측 라벨 기준으로 집계(분석 편의). span_only여도 그대로 둔다.
            label_counts[("FP", pl)] += 1

    for gi, (gb, ge, gl) in enumerate(golds):
        if not used_g[gi]:
            label_counts[("FN", gl)] += 1

    fp = sum(v for (t, _), v in label_counts.items() if t == "FP")
    fn = sum(v for (t, _), v in label_counts.items() if t == "FN")
    return tp, fp, fn, label_counts

def evaluate_core(answers_path: str, predictions_path: str, match: str, iou: float,
                  verbose: bool=True, span_only: bool=False):
    gold_by_id = load_jsonl_by_id(answers_path)
    pred_by_id = load_jsonl_by_id(predictions_path)
    common_ids = sorted(set(gold_by_id.keys()) & set(pred_by_id.keys()))
    if not common_ids:
        raise SystemExit("정답과 예측 간에 공통 id가 없습니다.")
    total_tp=total_fp=total_fn=0; per_label=Counter()
    for sid in common_ids:
        golds=[(int(e["begin"]),int(e["end"]),str(e["label"])) for e in gold_by_id[sid].get("entities",[])]
        preds=[(int(e["begin"]),int(e["end"]),str(e["label"])) for e in pred_by_id[sid].get("entities",[])]
        tp,fp,fn,lc=match_entities(preds,golds,mode=match,iou_th=iou,span_only=span_only)
        total_tp+=tp; total_fp+=fp; total_fn+=fn; per_label.update(lc)
    precision=safe_div(total_tp,total_tp+total_fp)
    recall=safe_div(total_tp,total_tp+total_fn)
    f1=safe_div(2*precision*recall,precision+recall) if (precision+recall) else 0.0
    labels=sorted({lab for (_,lab) in {(t,l) for (t,l) in per_label.keys()}})
    p_list=[]; r_list=[]; f1_list=[]
    per_label_metrics=[]
    for lab in labels:
        tp_l=per_label.get(("TP",lab),0); fp_l=per_label.get(("FP",lab),0); fn_l=per_label.get(("FN",lab),0)
        p_l=safe_div(tp_l,tp_l+fp_l); r_l=safe_div(tp_l,tp_l+fn_l); f1_l=safe_div(2*p_l*r_l,p_l+r_l) if (p_l+r_l) else 0.0
        per_label_metrics.append({"label":lab,"tp":tp_l,"fp":fp_l,"fn":fn_l,"precision":p_l,"recall":r_l,"f1":f1_l})
        p_list.append(p_l); r_list.append(r_l); f1_list.append(f1_l)
    macro_p=sum(p_list)/len(p_list) if p_list else 0.0
    macro_r=sum(r_list)/len(r_list) if r_list else 0.0
    macro_f1=sum(f1_list)/len(f1_list) if f1_list else 0.0
    if verbose:
        tag = "SPAN-ONLY" if span_only else "LABEL-AWARE"
        print(f"■ 전체 요약 ({tag}, Micro)"); print(f"  Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
        print(f"■ 매크로 평균 ({tag}, Macro)"); print(f"  Precision={macro_p:.4f} Recall={macro_r:.4f} F1={macro_f1:.4f}")
    return {
        "precision_micro":precision, "recall_micro":recall, "f1_micro":f1,
        "precision_macro":macro_p, "recall_macro":macro_r, "f1_macro":macro_f1,
        "per_label":per_label_metrics, "tp":total_tp, "fp":total_fp, "fn":total_fn
    }

# =========================
# 3) 향상률
# =========================
def compute_improvement(before_model: dict, after_model: dict) -> dict:
    metrics = ["precision_micro","recall_micro","f1_micro","f1_macro"]
    out={}
    for m in metrics:
        b=before_model.get(m,0.0); a=after_model.get(m,0.0)
        out[m]=safe_div((a-b),b)*100 if b>0 else 0.0
    return out

def print_improvement_summary(summary_list: List[dict]):
    pairs=[("3B_before","3B_after"),("7B_before","7B_after")]
    print("\n📈 [튜닝 전/후 향상률 (%)]")
    print(f"{'모델쌍':<20} {'ΔP_micro':>10} {'ΔR_micro':>10} {'ΔF1_micro':>10} {'ΔF1_macro':>10}")
    print("-"*60)
    for b,a in pairs:
        mb=next((x for x in summary_list if x["model"]==b),None)
        ma=next((x for x in summary_list if x["model"]==a),None)
        if not (mb and ma): continue
        imp=compute_improvement(mb,ma)
        print(f"{b}->{a:<12} {imp['precision_micro']:+.1f}% {imp['recall_micro']:+.1f}% {imp['f1_micro']:+.1f}% {imp['f1_macro']:+.1f}%")

# =========================
# 4) 공통 프롬프트 강제 & 생성 추론
# =========================
import json as _json

def extract_first_json(text: str) -> Optional[dict]:
    # 텍스트 전체에서 { ... } 블록 후보를 전부 추출
    candidates = []
    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append('{')  # ✅ 오타 수정
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start != -1:
                    candidates.append(text[start:i+1])
                    start = -1

    # 뒤에서부터(가장 최근) 유효 JSON인지 검사 + 키 확인
    for raw in reversed(candidates):
        try:
            obj = _json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and "entities" in obj and "has_sensitive" in obj:
            ents = obj.get("entities")
            hs = obj.get("has_sensitive")
            if isinstance(ents, list) and isinstance(hs, (bool, int)):
                return obj
    return None

def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    # 모델이 chat_template 제공하면 적용, 아니면 fallback
    try:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return f"<<SYS>>\n{system_prompt.strip()}\n<</SYS>>\n<<USER>>\n{user_prompt.strip()}\n<</USER>>\n<<ASSISTANT>>"

def closest_label(label: str, allowed: List[str]) -> Optional[str]:
    cand = difflib.get_close_matches(label, allowed, n=1, cutoff=0.6)
    return cand[0] if cand else None

def find_all_spans(text: str, sub: str) -> List[Tuple[int,int]]:
    spans=[]; start=0
    if not sub: return spans
    while True:
        i=text.find(sub, start)
        if i==-1: break
        spans.append((i, i+len(sub)))
        start=i+len(sub)
    return spans

def infer_token_classification(
    prompts_path: str, model_id: str, out_path: str,
    aggregation: str = "simple",
    device: Optional[str] = None,
    label_map: Optional[Dict[str,str]] = None,
    batch_size: int = 8
):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    print(f"[INFO] token-classification 모델 로딩: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    try:
        import torch
        device_id = 0 if (device=="cuda" or (device is None and torch.cuda.is_available())) else -1
    except Exception:
        device_id = -1
    nlp = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy=aggregation, device=device_id)
    prompts = load_jsonl(prompts_path)
    def map_label(lab: str) -> str:
        if label_map and lab in label_map: return label_map[lab]
        for pref in ("B-","I-","S-","E-","U-","L-"):
            if lab.startswith(pref): return lab[len(pref):]
        return lab
    with open(out_path,"w",encoding="utf-8") as w:
        for obj in prompts:
            sid=str(obj.get("id")); text=obj.get("text","")
            if not text:
                w.write(json.dumps({"id":sid,"entities":[]},ensure_ascii=False)+"\n"); continue
            entities=[]
            for base_off, chunk in chunk_text(text, tok):
                out = nlp(chunk, batch_size=batch_size)
                for ent in out:
                    b=int(ent.get("start",0))+base_off; e=int(ent.get("end",0))+base_off
                    lab=map_label(str(ent.get("entity_group") or ent.get("entity") or ""))
                    entities.append({"begin":b,"end":e,"label":lab})
            w.write(json.dumps({"id":sid,"entities":entities},ensure_ascii=False)+"\n")
    print(f"[INFO] 예측 저장: {out_path}")

def infer_generation(
    prompts_path: str, model_id: str, out_path: str,
    device: Optional[str] = None, max_new_tokens: int = 256,
    temperature: float = 0.2, top_p: float = 0.9,
    strict_policy: str = "drop", save_steps: bool = False
):
    """
    모델 출력: {"has_sensitive": bool, "entities":[{"type":LABEL,"value":SUBSTR}]}
    → 평가 포맷: {"id":..., "entities":[{"begin":int,"end":int,"label":LABEL}]}
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ✅ 새 모델 올리기 전에 남은 VRAM 깔끔히 청소
    free_cuda()

    print(f"[INFO] generation 모델 로딩: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    # pad 토큰 안전 설정 (없으면 eos로 대체)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    if device == "cuda" or (device is None and torch.cuda.is_available()):
        model.to("cuda"); dev = "cuda"
    else:
        dev = "cpu"

    try:
        prompts = load_jsonl(prompts_path)

        # 디버그 파일 준비
        debug_path = os.path.join(
            os.path.dirname(out_path),
            f"{os.path.splitext(os.path.basename(out_path))[0].replace('_predictions','')}_debug.jsonl"
        )
        dbg_f = open(debug_path, "w", encoding="utf-8") if save_steps else None

        with open(out_path, "w", encoding="utf-8") as w:
            for obj in prompts:
                sid = str(obj.get("id"))
                text = obj.get("text","")

                # 디버그 레코드 시작
                debug_rec = {"id": sid}

                if not text:
                    w.write(json.dumps({"id": sid, "entities": []}, ensure_ascii=False)+"\n")
                    if dbg_f:
                        debug_rec.update({
                            "raw_output": "",
                            "parsed": None,
                            "filter_actions": [],
                            "filtered_entities": [],
                            "std_entities": [],
                            "note": "empty text"
                        })
                        dbg_f.write(json.dumps(debug_rec, ensure_ascii=False) + "\n")
                    continue

                user_prompt = (
                    "Analyze the input text. Use ONLY the allowed labels defined above and return JSON ONLY.\n"
                    f"Text:\n{text}"
                )
                rendered = render_chat_prompt(tok, SYS_PROMPT, user_prompt)

                max_len = getattr(tok, "model_max_length", None) or 4096  # fallback
                inputs = tok(
                    rendered,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len
                )
                inputs = {k: v.to(dev) for k, v in inputs.items()}

                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,          # 완전 결정적
                        temperature=0.0,
                        top_p=1.0,
                        repetition_penalty=1.05,  # 가벼운 반복 억제
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True
                    )

                out_text = tok.decode(out_ids[0], skip_special_tokens=True)
                if dbg_f:
                    debug_rec["raw_output"] = out_text

                parsed = extract_first_json(out_text)
                if dbg_f:
                    debug_rec["parsed"] = parsed

                if not parsed:
                    # 파싱 실패: 예측은 빈 엔티티로
                    w.write(json.dumps({"id": sid, "entities": []}, ensure_ascii=False)+"\n")
                    if dbg_f:
                        debug_rec.update({
                            "filter_actions": [],
                            "filtered_entities": [],
                            "std_entities": [],
                            "note": "json parse failed"
                        })
                        dbg_f.write(json.dumps(debug_rec, ensure_ascii=False) + "\n")
                    continue

                has_sensitive = bool(parsed.get("has_sensitive", False))
                ents_in = parsed.get("entities", []) if has_sensitive else []

                # 허용 라벨 필터링/보정 + 행동 로그
                filter_actions = []
                filtered_entities = []
                for e in ents_in:
                    orig_type = str(e.get("type","")).strip()
                    norm_type = orig_type.upper()
                    value = str(e.get("value","")).strip()
                    if not norm_type or not value:
                        continue

                    action = "kept"
                    final_type = norm_type
                    if norm_type not in ALLOWED_LABELS:
                        if strict_policy == "closest":
                            mapped = closest_label(norm_type, ALLOWED_LABELS)
                            if mapped:
                                action = f"mapped:{norm_type}->{mapped}"
                                final_type = mapped
                            else:
                                action = "dropped"
                        else:
                            action = "dropped"

                    filter_actions.append({
                        "source_type": orig_type,
                        "normalized": norm_type,
                        "final_type": (final_type if action!="dropped" else None),
                        "value": value,
                        "action": action
                    })
                    if action != "dropped":
                        filtered_entities.append({"type": final_type, "value": value})

                # 스팬 매핑(begin/end)
                std_entities = []
                for fe in filtered_entities:
                    final_type = fe["type"]; value = fe["value"]
                    for b, e_ in find_all_spans(text, value):
                        std_entities.append({"begin": b, "end": e_, "label": final_type})

                # 최종 예측 저장 (평가 포맷)
                w.write(json.dumps({"id": sid, "entities": std_entities}, ensure_ascii=False)+"\n")

                # 디버그 저장
                if dbg_f:
                    debug_rec.update({
                        "filter_actions": filter_actions,
                        "filtered_entities": filtered_entities,
                        "std_entities": std_entities
                    })
                    dbg_f.write(json.dumps(debug_rec, ensure_ascii=False) + "\n")

        print(f"[INFO] 예측 저장: {out_path}")
        if dbg_f:
            dbg_f.close()
            print(f"[INFO] 디버그 저장: {debug_path}")

    finally:
        # ✅ 이 모델 사용 끝: 참조 해제 + VRAM 정리
        try:
            del model
        except Exception:
            pass
        try:
            del tok
        except Exception:
            pass
        free_cuda()

# =========================
# 5) 시각화 & PDF
# =========================
def setup_korean_font_for_matplotlib():
    # (보조) 그래프 한글 깨짐 방지: 설치된 한글 폰트(NanumGothic 등) 등록/지정
    import matplotlib
    import matplotlib.font_manager as fm
    nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    try:
        if os.path.exists(nanum_path):
            try:
                fm.fontManager.addfont(nanum_path)
            except Exception:
                pass
            matplotlib.rcParams["font.family"] = "NanumGothic"
            matplotlib.rcParams["axes.unicode_minus"] = False
            print("[INFO] Matplotlib font set to NanumGothic")
            return
    except Exception:
        pass
    preferred = ["NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] Matplotlib font set to: {name}")
            return
    print("[WARN] No Korean font found for Matplotlib; Korean text in charts may be broken.")

def save_bar_f1(summary_list, out_png):
    setup_korean_font_for_matplotlib()
    import matplotlib.pyplot as plt
    models=[m["model"] for m in summary_list]; f1=[m["f1_micro"] for m in summary_list]
    plt.figure(figsize=(9,5)); plt.bar(models,f1); plt.ylim(0,1)
    plt.title("튜닝 전/후 모델별 F1 (Micro)"); plt.ylabel("F1 Score")
    for i,v in enumerate(f1): plt.text(i, v+0.015, f"{v:.3f}", ha='center', fontweight='bold')
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close()

def save_radar_per_label(models_pl, labels, out_png, title="라벨별 F1 비교(전/후)"):
    setup_korean_font_for_matplotlib()
    import math, matplotlib.pyplot as plt
    def f1_vec(per_label):
        d={x["label"]:x["f1"] for x in per_label}; return [d.get(l,0.0) for l in labels]
    angles=[n/float(len(labels))*2*math.pi for n in range(len(labels))]; angles+=angles[:1]
    plt.figure(figsize=(7,7)); ax=plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels, fontsize=8); ax.set_rlabel_position(0); ax.set_ylim(0,1)
    for m in models_pl:
        vals=f1_vec(m["per_label"]); vals+=vals[:1]
        ax.plot(angles, vals, linewidth=1); ax.fill(angles, vals, alpha=0.1, label=m["name"])
    plt.title(title); plt.legend(loc="lower left", bbox_to_anchor=(0.0,-0.15), ncol=2)
    plt.tight_layout(); plt.savefig(out_png,dpi=150,bbox_inches="tight"); plt.close()

def build_pdf_report(out_pdf, summary_list, per_label_union):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

    # 🔤 ReportLab 한글 폰트 등록 (NanumGothic)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    doc=SimpleDocTemplate(out_pdf,pagesize=A4)
    styles=getSampleStyleSheet()

    try:
        if os.path.exists(nanum_path):
            pdfmetrics.registerFont(TTFont("NanumGothic", nanum_path))
            styles['Normal'].fontName = "NanumGothic"
            styles['Title'].fontName = "NanumGothic"
            styles['Heading2'].fontName = "NanumGothic"
            styles['Heading3'].fontName = "NanumGothic"
            print("[INFO] ReportLab font set to NanumGothic")
        else:
            print("[WARN] NanumGothic.ttf not found; Korean in PDF may be broken.")
    except Exception as e:
        print(f"[WARN] ReportLab font registration failed: {e}")

    story=[]
    story.append(Paragraph("<b>민감정보 탐지 모델 성능 비교 리포트</b>", styles['Title'])); story.append(Spacer(1, 12))
    data=[["모델","P_micro","R_micro","F1_micro","F1_macro","Latency(s)","Throughput"]]
    for s in summary_list:
        data.append([s["model"],f"{s['precision_micro']:.3f}",f"{s['recall_micro']:.3f}",
                     f"{s['f1_micro']:.3f}",f"{s['f1_macro']:.3f}",
                     f"{s.get('latency',0.0):.3f}",f"{s.get('throughput',0.0):.2f}"])
    tbl=Table(data,hAlign="LEFT"); tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eeeeee")),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"CENTER"),
    ]))
    story.append(Paragraph("<b>요약 지표</b>", styles['Heading2'])); story.append(tbl); story.append(Spacer(1,12))
    pairs=[("3B_before","3B_after"),("7B_before","7B_after")]
    story.append(Paragraph("<b>튜닝 전/후 향상률 (%)</b>", styles['Heading2']))
    data3=[["모델쌍","ΔP_micro","ΔR_micro","ΔF1_micro","ΔF1_macro"]]
    for b,a in pairs:
        mb=next((x for x in summary_list if x["model"]==b),None)
        ma=next((x for x in summary_list if x["model"]==a),None)
        if not (mb and ma): continue
        imp=compute_improvement(mb,ma)
        data3.append([f"{b}→{a}",f"{imp['precision_micro']:+.1f}%",f"{imp['recall_micro']:+.1f}%",
                      f"{imp['f1_micro']:+.1f}%",f"{imp['f1_macro']:+.1f}%"])
    tbl3=Table(data3,hAlign="LEFT"); tbl3.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f0f0")),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"CENTER"),
    ])); story.append(tbl3); story.append(Spacer(1,12))
    bar_png=os.path.join(os.path.dirname(out_pdf),"compare_F1_micro.png")
    if os.path.exists(bar_png):
        story.append(Paragraph("<b>모델별 F1(Micro) 비교</b>", styles['Heading2']))
        story.append(Image(bar_png,width=420,height=260)); story.append(Spacer(1,12))
    story.append(Paragraph("<b>라벨별 상세 성능</b>", styles['Heading2']))
    for s in summary_list:
        story.append(Paragraph(f"<b>{s['model']}</b>", styles['Heading3']))
        data2=[["Label","TP","FP","FN","Precision","Recall","F1"]]
        d={x["label"]:x for x in s["per_label"]}
        for lab in per_label_union:
            x=d.get(lab,{"tp":0,"fp":0,"fn":0,"precision":0,"recall":0,"f1":0})
            data2.append([lab,str(x["tp"]),str(x["fp"]),str(x["fn"]),
                          f"{x['precision']:.3f}",f"{x['recall']:.3f}",f"{x['f1']:.3f}"])
        tbl2=Table(data2,hAlign="LEFT"); tbl2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f6f6f6")),("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"CENTER"),
        ]))
        story.append(tbl2); story.append(Spacer(1,10))
    radar_png=os.path.join(os.path.dirname(out_pdf),"per_label_radar.png")
    if os.path.exists(radar_png):
        story.append(Paragraph("<b>라벨별 F1 레이더(전/후 비교)</b>", styles['Heading2']))
        story.append(Image(radar_png,width=420,height=420)); story.append(Spacer(1,12))
    doc.build(story)

# =========================
# 6) 실행 (모델 루프)
# =========================
def run_one_model(task, prompts_path, answers_path, model_name, model_id,
                  outdir, device, match, iou, span_only=False, **gen_kwargs):
    os.makedirs(outdir, exist_ok=True)
    pred_path=os.path.join(outdir,f"{model_name}_predictions.jsonl")  # ✅ 여분의 } 제거
    t0=time.time()
    if task=="token":
        infer_token_classification(prompts_path, model_id, pred_path,
                                   aggregation=gen_kwargs.get("aggregation","simple"),
                                   device=device, label_map=None,
                                   batch_size=gen_kwargs.get("batch_size",8))
    else:
        infer_generation(prompts_path, model_id, pred_path,
                         device=device,
                         max_new_tokens=gen_kwargs.get("max_new_tokens",256),
                         temperature=gen_kwargs.get("temperature",0.2),
                         top_p=gen_kwargs.get("top_p",0.9),
                         strict_policy=gen_kwargs.get("strict_policy","drop"),
                         save_steps=gen_kwargs.get("save_steps", False))
    t1=time.time()
    n_items=len(load_jsonl(prompts_path)) or 1
    latency=(t1-t0)/n_items; throughput=n_items/max(1e-9,(t1-t0))
    metrics=evaluate_core(answers_path, pred_path, match, iou, verbose=True, span_only=span_only)
    metrics["model"]=model_name; metrics["latency"]=latency; metrics["throughput"]=throughput
    free_cuda()
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["token","generation"], default="generation")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--models-json", required=True,
                    help='{"3B_before":"Qwen/...","3B_after":"./ft", ...}')
    ap.add_argument("--outdir", default="eval_results")
    ap.add_argument("--device", choices=["cpu","cuda","mps"])
    ap.add_argument("--match", choices=["exact","overlap"], default="exact")
    ap.add_argument("--iou", type=float, default=0.5)
    # generation
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--strict-policy", choices=["drop","closest"], default="drop",
                    help="허용외 라벨 처리: drop=버림, closest=가장 가까운 허용라벨로 보정")
    ap.add_argument("--save-steps", action="store_true",
                    help="단계별 결과를 하나의 debug.jsonl로 함께 저장")
    # span-only (라벨 무시)
    ap.add_argument("--span-only", action="store_true",
                    help="라벨을 무시하고 begin/end 스팬 매칭만으로 평가(탐지력만)")
    # token
    ap.add_argument("--aggregation", default="simple")
    ap.add_argument("--batch-size", type=int, default=8)
    args=ap.parse_args()

    models: Dict[str,str]=json.loads(args.models_json)
    os.makedirs(args.outdir, exist_ok=True)

    print("\n===== 모델 목록 =====")
    for k,v in models.items(): print(f" - {k}: {v}")
    print("=====================\n")

    summaries=[]
    for name,mid in models.items():
        print(f"\n\n==============================\n[MODEL] {name}  ({mid})\n==============================")
        m=run_one_model(task=args.task, prompts_path=args.prompts, answers_path=args.answers,
                        model_name=name, model_id=mid, outdir=args.outdir, device=args.device,
                        match=args.match, iou=args.iou, span_only=args.span_only,
                        aggregation=args.aggregation, batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_p=args.top_p, strict_policy=args.strict_policy, save_steps=args.save_steps)
        summaries.append(m)

    print("\n\n📋 [성능 요약표]")
    print(f"{'모델명':<15} {'P_micro':>9} {'R_micro':>9} {'F1_micro':>9} {'F1_macro':>9} {'Latency(s)':>12} {'Throughput':>12}")
    print("-"*80)
    for s in summaries:
        print(f"{s['model']:<15} {s['precision_micro']:.4f} {s['recall_micro']:.4f} {s['f1_micro']:.4f} {s['f1_macro']:.4f} {s['latency']:.3f} {s['throughput']:.2f}")

    print_improvement_summary(summaries)

    bar_png=os.path.join(args.outdir,"compare_F1_micro.png"); save_bar_f1(summaries, bar_png)
    all_labels=sorted({x["label"] for s in summaries for x in s["per_label"]})
    radar_models=[]
    for key in ["3B_before","3B_after","7B_before","7B_after"]:
        for s in summaries:
            if s["model"]==key:
                radar_models.append({"name":key,"per_label":s["per_label"]}); break
    radar_png=os.path.join(args.outdir,"per_label_radar.png")
    if len(radar_models)>=2 and len(all_labels)>=3:
        save_radar_per_label(radar_models, all_labels, radar_png, title="라벨별 F1 레이더(전/후 비교)")
    out_pdf=os.path.join(args.outdir,"evaluation_report.pdf")
    build_pdf_report(out_pdf, summaries, all_labels)
    print(f"\n✅ 완료: 결과 폴더 = {args.outdir}")
    print(f"   - 예측 파일: <모델명>_predictions.jsonl")
    print(f"   - 디버그 파일(옵션): <모델명>_debug.jsonl  (--save-steps 사용 시)")
    print(f"   - 그래프: compare_F1_micro.png, per_label_radar.png")
    print(f"   - PDF 리포트: evaluation_report.pdf")

if __name__=="__main__":
    main()
