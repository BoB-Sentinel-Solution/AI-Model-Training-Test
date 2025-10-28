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
"자 이 프롬프트 다음에 많은 고객 정보를 줄꺼야. 정보들을 주면 니가 같은것끼리 정리해줘.",
"김서연, 1998-04-12, 980412-2345678, 여, 010-1234-5678, sy.kim@gmail.com, 서울 강남구 논현동, USR-1001, 4539-1122-3344-5566, 대한민국, 디자이너, 패션, 박민수, 1995-11-30, 951130-1234567, 남, 010-9876-5432, ms.park@naver.com, 부산 해운대구 좌동, CLI-2001, 4929-2233-4455-6677, 대한민국, 개발자, 게임, 이하늘, 2001-07-25, 010725-4234567, 여, 010-2233-4455, hn.lee@yahoo.co.jp, 인천 연수구 송도동, ID-3001, 4024-3344-5566-7788, 대한민국, 대학생, 여행, 최강호, 1990-02-18, 900218-1234567, 남, 010-3344-5566, gh.choi@daum.net, 대구 수성구 만촌동, ACC-4001, 4712-4455-6677-8899, 대한민국, 의사, 등산, 정유라, 1997-12-09, 971209-2234567, 여, 010-4455-6677, yr.jung@hotmail.com, 경기 성남시 분당구, CLI-2002, 4532-5566-7788-9900, 대한민국, 연구원, 독서, 佐藤健, 1991-03-15, 910315-1234567, 남, 090-1122-3344, takeshi.sato@yahoo.co.jp, 도쿄 시부야구, JP-5001, 4021-6677-8899-0011, 일본, 엔지니어, 애니메이션, 山田花, 1999-06-07, 990607-4234567, 여, 080-5566-7788, hana.yamada@gmail.com, 오사카 주오구, JP-5002, 4710-7788-9900-1122, 일본, 대학원생, 패션, John Doe, 1988-09-22, 880922-1234567, 남, +1-202-555-0172, john.doe@outlook.com, New York, US-6001, 4921-8899-0011-2233, 미국, 회계사, 농구, Emily Lin, 1996-11-03, 961103-4234567, 여, +1-415-555-2299, em.lin@gmail.com, San Francisco, US-6002, 4538-9900-1122-3344, 미국, 디자이너, 음악, Max Müller, 1992-08-14, 920814-1234567, 남, +49-151-2345678, max.mueller@web.de, Berlin, DE-7001, 4022-1122-2233-4455, 독일, 연구원, 축구, Anna Schulz, 1994-10-05, 941005-4234567, 여, +49-170-8765432, anna.schulz@outlook.de, München, DE-7002, 4713-2233-3344-5566, 독일, 간호사, 요가, Carlos Silva, 1987-12-01, 871201-1234567, 남, +55-21-99887766, carlos.silva@gmail.com, Rio de Janeiro, BR-8001, 4928-3344-4455-6677, 브라질, 교사, 축구, Ana Costa, 1993-05-19, 930519-4234567, 여, +55-11-98765432, ana.costa@yahoo.com, São Paulo, BR-8002, 4537-4455-5566-7788, 브라질, 변호사, 음악, Nguyen An, 1999-03-27, 990327-1234567, 남, +84-90-1234567, an.nguyen@gmail.com, Hanoi, VN-9001, 4029-5566-6677-8899, 베트남, 학생, 영화, Tran Hoa, 1997-07-16, 970716-4234567, 여, +84-91-7654321, hoa.tran@outlook.vn, Ho Chi Minh City, VN-9002, 4714-6677-7788-9900, 베트남, 디자이너, 사진, 王伟, 1990-01-09, 900109-1234567, 남, +86-138-00110022, wang.wei@qq.com, Beijing, CN-10001, 4920-7788-8899-0011, 중국, 개발자, 바둑, 刘芳, 1995-11-20, 951120-4234567, 여, +86-139-22334455, liu.fang@163.com, Shanghai, CN-10002, 4536-8899-9900-1122, 중국, 교사, 피아노, Michael Smith, 1985-04-14, 850414-1234567, 남, +44-20-7946-1234, m.smith@hotmail.co.uk, London, UK-11001, 4023-9900-1111-2233, 영국, 엔지니어, 럭비, Sarah Brown, 1992-09-02, 920902-4234567, 여, +44-161-234-5678, sarah.brown@gmail.com, Manchester, UK-11002, 4711-1111-2222-3344, 영국, 회계사, 여행, Иван Петров, 1986-07-11, 860711-1234567, 남, +7-901-123-4567, ivan.petrov@yandex.ru, Moscow, RU-12001, 4924-2222-3333-4455, 러시아, 연구원, 체스, Ольга Иванова, 1991-02-28, 910228-4234567, 여, +7-921-987-6543, olga.ivanova@mail.ru, St. Petersburg, RU-12002, 4535-3333-4444-5566, 러시아, 간호사, 발레, Ali Khan, 1989-08-30, 890830-1234567, 남, +92-300-1234567, ali.khan@gmail.com, Karachi, PK-13001, 4026-4444-5555-6677, 파키스탄, 의사, 크리켓, Fatima Noor, 1996-04-22, 960422-4234567, 여, +92-321-9876543, fatima.noor@yahoo.com, Lahore, PK-13002, 4715-5555-6666-7788, 파키스탄, 변호사, 요리, David Kim, 1993-11-05, 931105-1234567, 남, +1-213-555-9988, david.kim@gmail.com, Los Angeles, US-14001, 4925-6666-7777-8899, 미국, 개발자, 농구, Jessica Lee, 1997-01-18, 970118-4234567, 여, +1-646-555-7788, jessica.lee@outlook.com, New York, US-14002, 4534-7777-8888-9900, 미국, 마케터, 패션, 김도윤, 1996-03-25, 960325-1234567, 남, 010-7777-8888, dy.kim@kakao.com, 서울 서초구 서초동, ACC-15001, 4027-8888-9999-0000, 대한민국, 연구원, 독서, 오하람, 1999-12-12, 991212-4234567, 여, 010-9999-0000, hr.oh@gmail.com, 서울 영등포구 여의도동, CLI-15002, 4716-9999-0000-1111, 대한민국, 대학생, 댄스, 박서우, 1998-08-09, 980809-2234567, 여, 010-2222-3333, sw.park@naver.com, 대전 서구 둔산동, USR-15003, 4927-0000-1111-2222, 대한민국, 간호사, 피아노, 이승현, 1994-05-01, 940501-1234567, 남, 010-3333-4444, sh.lee@gmail.com, 광주 북구 충장로, ID-15004, 4533-1111-2222-3333, 대한민국, 회계사, 여행, Tan Wei, 1991-06-06, 910606-1234567, 남, +65-8123-4567, tan.wei@yahoo.com, Singapore, SG-16001, 4028-2222-3333-4444, 싱가포르, 엔지니어, 게임, Lim Mei, 1995-10-15, 951015-4234567, 여, +65-8234-5678, lim.mei@gmail.com, Singapore, SG-16002, 4717-3333-4444-5555, 싱가포르, 디자이너, 요리, Alex Johnson, 1990-12-24, 901224-1234567, 남, +61-412-345-678, alex.johnson@hotmail.com, Sydney, AU-17001, 4926-4444-5555-6666, 호주, 교사, 서핑, Sophie Taylor, 1994-11-02, 941102-4234567, 여, +61-423-987-654, sophie.taylor@gmail.com, Melbourne, AU-17002, 4531-5555-6666-7777, 호주, 간호사, 영화, Martín López, 1988-03-30, 880330-1234567, 남, +34-600-123-456, martin.lopez@yahoo.es, Madrid, ES-18001, 4025-6666-7777-8888, 스페인, 변호사, 축구, Carmen Ruiz, 1992-07-19, 920719-4234567, 여, +34-699-987-654, carmen.ruiz@gmail.com, Barcelona, ES-18002, 4718-7777-8888-9999, 스페인, 연구원, 플라멩코, Francesco Rossi, 1987-09-25, 870925-1234567, 남, +39-320-123-4567, f.rossi@libero.it, Rome, IT-19001, 4922-8888-9999-0000, 이탈리아, 셰프, 축구, Giulia Bianchi, 1993-02-13, 930213-4234567, 여, +39-331-987-6543, g.bianchi@gmail.com, Milan, IT-19002, 4530-9999-0000-1111, 이탈리아, 디자이너, 패션, محمد علي, 1989-04-20, 890420-1234567, 남, +20-100-123-4567, mohamed.ali@gmail.com, Cairo, EG-20001, 4020-0000-1111-2222, 이집트, 의사, 축구, فاطمة حسن, 1996-09-01, 960901-4234567, 여, +20-101-987-6543, fatima.hassan@yahoo.com, Giza, EG-20002, 4719-1111-2222-3333, 이집트, 교사, 요리, Lucas Brown, 1991-11-07, 911107-1234567, 남, +1-305-555-1234, lucas.brown@hotmail.com, Miami, US-21001, 4923-2222-3333-4444, 미국, 엔지니어, 음악, Olivia Davis, 1998-01-23, 980123-4234567, 여, +1-702-555-5678, olivia.davis@gmail.com, Las Vegas, US-21002, 4539-3333-4444-5555, 미국, 대학생, 영화, 김하린, 1997-06-29, 970629-4234567, 여, 010-5555-6666, hr.kim@naver.com, 서울 송파구 잠실동, ACC-22001, 4021-4444-5555-6666, 대한민국, 연구원, 피아노, 정도윤, 1995-08-15, 950815-1234567, 남, 010-6666-7777, dy.jung@daum.net, 서울 서대문구 연희동, CLI-22002, 4710-5555-6666-7777, 대한민국, 회계사, 자전거, 이채린, 1999-02-11, 990211-4234567, 여, 010-7777-8888, cr.lee@gmail.com, 서울 동작구 흑석동, USR-22003, 4929-6666-7777-8888, 대한민국, 대학생, 패션, 박지후, 1996-10-05, 961005-1234567, 남, 010-8888-9999, jh.park@hotmail.com, 서울 마포구 합정동, ID-22004, 4538-7777-8888-9999, 대한민국, 엔지니어, 농구, Lucas Wang, 1992-04-12, 920412-1234567, 남, +852-9123-4567, lucas.wang@gmail.com, Hong Kong, HK-23001, 4024-8888-9999-0000, 홍콩, 금융분석가, 영화, Mei Chen, 1997-09-08, 970908-4234567, 여, +852-9234-5678, mei.chen@yahoo.com, Kowloon, HK-23002, 4712-9999-0000-1111, 홍콩, 회계사, 패션, Raj Patel, 1991-05-17, 910517-1234567, 남, +91-98123-45678, raj.patel@gmail.com, Mumbai, IN-24001, 4921-0000-1111-2222, 인도, 개발자, 크리켓, Priya Sharma, 1995-12-22, 951222-4234567, 여, +91-98234-56789, priya.sharma@yahoo.com, Delhi, IN-24002, 4537-1111-2222-3333, 인도, 연구원, 요가"
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
