# test_print_only.py
# -*- coding: utf-8 -*-
import argparse, json, time, threading
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

SYS_PROMPT = (
    "You are a strict detector for sensitive entities (PII and secrets). "
    "Given the user's text, return ONLY a compact JSON with keys: "
    "`has_sensitive` (true/false) and `entities` (list of objects with `type` and `value`). "
    "If none, return {\"has_sensitive\": false, \"entities\": []}. "
)

DEFAULT_PROMPTS = [
"주문ID ORDER-25-0912-ABCD-7788, 수령인 김민서, 연락처 010-1234-5678, 이메일 minseo.kim@example.com, 주소 서울 강남구 테헤란로 12 5층(06234)",
"계정ID ms.park, 비밀번호 SuperSecret!234, 2FA 코드 ABCD-EFGH-IJKL-MNOP, 최근 로그인 IP 192.168.10.45",
"OpenAI API 키 sk-EXAMPLEFAKEKEY1234567890abcd, GitHub PAT ghp_EXAMPLEExample1234, Slack 토큰 xoxb-12345-EXAMPLE-abcdefghijkl",
"Authorization 토큰 eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.EXAMPLE.zzZEXAMPLE, refresh_token eyJraWQiOiJLT0RFIiJ9.EXAMPLE.SIGN",
"IBAN DE89370400440532013000, BIC DEUTDEFF, 수취인 Hans Müller, 금액 EUR 1,250.00",
"직원번호 EMP-00912, 이름 박지훈, 주민등록번호 900101-1234567, 부서 영업부",
"사업자등록번호 123-45-67890, 상호 주식회사 알파, 대표자 김은정",
"사용자ID lee.admin, 접속IP 10.20.30.40, 접속시간 2025-09-15 10:22:33",
"이메일 user@example.com, 백업 이메일 alt@example.org, 전화번호 +82-10-9876-5432",
"계약서 열람 기록 CT-2025-0915-XYZ",
"InvoiceNo INV-887766, 고객 LG Electronics, 금액 KRW 5,400,000",
"Host db.internal.local, User sa, Password P@ssw0rd2025!, Port 1433",
"GitLab PAT glpat-EXAMPLE1234567890",
"DriveID drv-998877, 파일명 Confidential_Report.pdf",
"사용자ID yoon_choi, 메시지 '회의 2025/09/20 14:00', session_token EXAMPLETOKEN",
"고객번호 CUST-002931, CRM 레코드ID CRM-7F2A-11EE-BC12, 연락처 010-2233-4455",
"송장번호 INV-2025-000123, 청구지 부산 해운대구 A로 77 1203호, 고객ID cus_FAKE12345",
"계정 park.min@example.com, MFA 556677",
"사번 EMP-7733, 이름 이영호, 급여 KRW 4,200,000, 계좌 100-222-333444",
"프로젝트ID PRJ-56789, PM 김지후",
"이름 김민아, 주민등록번호 920505-2345678",
"API Key AIzaSyEXAMPLE1234",
"카드번호 5555 4444 3333 2222",
"세션 토큰 sessionid=s%3AEXAMPLE.SIG.",
"비밀번호 StrongPass2025!",
"주문번호 ORD-2025-5555, 수령인 이나래, 연락처 010-1122-3344",
"UserID sales_admin, Token 00Dxx0000001gPz!AQ4AQFakeToken",
"CouponID CP-2025-7788, 고객ID cust_8899, 할인 KRW 20,000",
"PartnerID PRT-2025-88, 회사 Beta Solutions, 계약금액 USD 120,000",
"Slack Webhook URL https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
"사번 EMP-2025-123, 이름 한지원, 부서 재무팀, 내선 023-456-7890",
"환자ID PT-777, 이름 최유리, 보험번호 INS-889977, 병명 Hypertension",
"From ceo@company.com, To cfo@company.com, Subject 'M&A Plan', Attachment secret.xlsx",
"카드번호 CARD-991122, 이름 장수빈, 출입시간 2025-09-14 09:05:00",
"SecretName db-password, SecretValue P@ssExample123",
"고객명 (미지정), 주소 부산 해운대구 A로 77, 금액 KRW 1,980,000",
"csrftoken=xyz123, XSRF-TOKEN=EXAMPLETOKEN123",
"SecretName DOCKER_PASSWORD, Value ExAmpl3D0ckerP@ss",
"TX-112233, 카드번호 5555 4444 3333 2222, 금액 USD 99.99",
"User root, Host 192.168.1.100, PrivateKey -----BEGIN OPENSSH PRIVATE KEY-----FAKEKEY-----END OPENSSH PRIVATE KEY-----",
"이름 김민수, 계좌번호 110-222-333444, 이메일 minsu@example.com",
"고객ID CUST-8899, 연락처 010-4455-6677, 주소 서울시 강남구",
"주민등록번호 950101-1234567",
"AWS AccessKey AKIAEXAMPLE, Secret abc123xyz456",
"GOOGLE_API_KEY AIzaSyFakeKey123, MAPBOX_KEY pk.EXAMPLEKEY",
"주문ID ORDER-9988, 연락처 010-5566-7788",
"환자ID P-9981, 이름 이지은, 나이 45, 진단 Hypertension",
"IP 192.168.77.10, session abcdef12345",
"From alice@company.com, To bob@partner.org",
"카드번호 4444 3333 2222 1111, CVV 987, 만료일 09/27"
]

def infer_once(tok, model, prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt}
    ]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})
    gen_kwargs = dict(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )

    # run generate on a background thread to use the streamer
    first_token_time: Optional[float] = None
    raw_chunks = []
    start = time.perf_counter()

    def _consume():
        nonlocal first_token_time
        for i, token_text in enumerate(streamer):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            raw_chunks.append(token_text)

    t = threading.Thread(target=_consume)
    t.start()

    with torch.no_grad():
        model.generate(**gen_kwargs)

    t.join()
    end = time.perf_counter()

    raw_out = "".join(raw_chunks).strip()
    total_ms = (end - start) * 1000.0
    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time else None

    # token/s 계산 (생성 토큰 기준)
    # 입력/출력 토큰 길이 측정
    out_ids = tok(raw_out, return_tensors="pt")["input_ids"][0]
    gen_tokens = out_ids.size(0)
    tok_s = None
    if first_token_time:
        duration_gen = end - first_token_time
        if duration_gen > 0:
            tok_s = gen_tokens / duration_gen

    # JSON 파싱 시도
    parsed = None
    try:
        # 모델이 앞말 덧붙였을 수 있으니 중괄호 첫/끝 구간만 salvage
        s = raw_out
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            parsed = json.loads(s[l:r+1])
        else:
            parsed = json.loads(s)  # 혹시 순수 JSON이면
    except Exception:
        parsed = None

    return {
        "raw": raw_out,
        "json": parsed,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tok_s": tok_s
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Merged model dir (e.g., runs/qwen7b_sft_merged)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="테스트할 프롬프트 개수 제한(0=전체)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype="auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = DEFAULT_PROMPTS[: args.limit or None]

    for i, p in enumerate(prompts, 1):
        r = infer_once(tok, model, p, max_new_tokens=args.max_new_tokens)
        print(f"\n--- TEST #{i} ---")
        print("prompt:", p)
        print(
            "ttft_ms:", f"{r['ttft_ms']:.2f}" if r['ttft_ms'] else "NA",
            "| tok/s:", f"{r['tok_s']:.2f}" if r['tok_s'] else "NA",
            "| total_ms:", f"{r['total_ms']:.2f}"
        )
        print("output:", r["raw"])
        print("parsed_json:", json.dumps(r["json"], ensure_ascii=False) if r["json"] is not None else "None")

if __name__ == "__main__":
    main()
