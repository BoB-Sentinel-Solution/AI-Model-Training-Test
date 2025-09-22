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
"ORDER-25-0912-ABCD-7788, 김민서, 010-1234-5678, minseo.kim@example.com, 서울 강남구 테헤란로 12 5층(06234)",
"ms.park, SuperSecret!234, ABCD-EFGH-IJKL-MNOP, 192.168.10.45",
"sk-EXAMPLEFAKEKEY1234567890abcd, ghp_EXAMPLEExample1234, xoxb-12345-EXAMPLE-abcdefghijkl",
"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.EXAMPLE.zzZEXAMPLE, eyJraWQiOiJLT0RFIiJ9.EXAMPLE.SIGN",
"DE89370400440532013000, DEUTDEFF, Hans Müller, EUR 1,250.00",
"EMP-00912, 박지훈, 900101-1234567, 영업부",
"123-45-67890, 주식회사 알파, 김은정",
"lee.admin, 10.20.30.40, 2025-09-15 10:22:33",
"user@example.com, alt@example.org, +82-10-9876-5432",
"CT-2025-0915-XYZ",
"INV-887766, LG Electronics, KRW 5,400,000",
"db.internal.local, sa, P@ssw0rd2025!",
"glpat-EXAMPLE1234567890",
"drv-998877, Confidential_Report.pdf",
"yoon_choi, '회의 2025/09/20 14:00', EXAMPLETOKEN",
"CUST-002931, CRM-7F2A-11EE-BC12, 010-2233-4455",
"INV-2025-000123, 부산 해운대구 A로 77 1203호, cus_FAKE12345",
"park.min@example.com, 556677",
"EMP-7733, 이영호, KRW 4,200,000, 100-222-333444",
"PRJ-56789, 김지후",
"김민아, 920505-2345678",
"AIzaSyEXAMPLE1234",
"5555 4444 3333 2222",
"sessionid=s%3AEXAMPLE.SIG.",
"StrongPass2025!",
"ORD-2025-5555, 이나래, 010-1122-3344",
"sales_admin, 00Dxx0000001gPz!AQ4AQFakeToken",
"CP-2025-7788, cust_8899, KRW 20,000",
"PRT-2025-88, Beta Solutions, USD 120,000",
"https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
"EMP-2025-123, 한지원, 재무팀, 023-456-7890",
"PT-777, 최유리, INS-889977, Hypertension",
"ceo@company.com, cfo@company.com, M&A Plan, secret.xlsx",
"CARD-991122, 장수빈, 2025-09-14 09:05:00",
"db-password, P@ssExample123",
"부산 해운대구 A로 77, KRW 1,980,000",
"csrftoken=xyz123, XSRF-TOKEN=EXAMPLETOKEN123",
"DOCKER_PASSWORD, ExAmpl3D0ckerP@ss",
"TX-112233, 5555 4444 3333 2222, USD 99.99",
"root, 192.168.1.100, -----BEGIN OPENSSH PRIVATE KEY-----FAKEKEY-----END OPENSSH PRIVATE KEY-----",
"김민수, 110-222-333444, minsu@example.com",
"CUST-8899, 010-4455-6677, 서울시 강남구",
"950101-1234567",
"AKIAEXAMPLE, abc123xyz456",
"AIzaSyFakeKey123, pk.EXAMPLEKEY",
"ORDER-9988, 010-5566-7788",
"P-9981, 이지은, 45, Hypertension",
"192.168.77.10, abcdef12345",
"alice@company.com, bob@partner.org",
"4444 3333 2222 1111, 987, 09/27"
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
