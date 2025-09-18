# -*- coding: utf-8 -*-
"""
Qwen 3B/7B용 다중 데이터 학습 + LoRA 병합(단일 모델 저장) 스크립트
- 입력 데이터 형식:
  * JSONL: 한 줄에 하나의 샘플(dict: {"id","text","has_sensitive","entities":[...]} )
  * JSON 배열: [ {...}, {...}, ... ]
  * 여러 파일일 경우 --data 를 반복 지정하거나, JSON(배열)파일로도 가능
- 출력:
  * --out_dir: 학습 결과(LoRA 어댑터 포함)
  * --merged_out: LoRA를 베이스에 병합한 "단일 모델" 디렉터리

사용 예:
  python train_and_merge_qwen_multi.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data test_dataset.json \
    --out_dir runs/qwen3b_sft \
    --merged_out runs/qwen3b_sft_merged \
    --epochs 3 --bf16

  # 여러 파일
  python train_and_merge_qwen_multi.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data ds1.jsonl --data ds2.json --data ds3.jsonl \
    --out_dir runs/qwen7b_multi \
    --merged_out runs/qwen7b_multi_merged \
    --epochs 3 --bf16 --samples_per_epoch 8000 --temperature 1.3
"""
import os, json, argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

SEED = 42
random.seed(SEED); np.random.seed(SEED)

SYS_PROMPT = """You are "Sensitive-Info Detector", a precision assistant that outputs ONLY one JSON object per request.

TASK
- Given a single user sentence (Korean or English), analyze it and return:
  {
    "has_sensitive": <boolean>,
    "entities": [
      {"label": "<LABEL>", "text": "<exact substring>", "begin": <int>, "end": <int>},
      ...
    ]
  }

OUTPUT RULES (STRICT)
1) Output exactly one JSON object. No code fences, no extra text, no explanations.
2) JSON keys MUST appear in this order:
   - has_sensitive
   - entities
3) Each entity object MUST have keys in this order: label, text, begin, end.
4) Use UTF-8 JSON escaping when required. Do not alter input text’s characters.
5) Offsets are 0-based character indices in the ORIGINAL input string; end is exclusive.
6) Sort entities by (begin ASC, end ASC, then label ASC).
7) If two entities have identical (label, begin, end), keep only one.
8) If an entity’s substring doesn’t exactly match the input (e.g., spacing differs), adjust begin/end so that text == input[begin:end].
9) If nothing is extractable, entities = [] and has_sensitive = false.
10) NEVER invent content. Do not infer hidden values from context. Only label substrings that appear verbatim in the input.

SENSITIVITY POLICY
- Set has_sensitive = true if AND ONLY IF at least one labeled entity is extracted.
- Otherwise, has_sensitive = false.

SUPPORTED LABELS (UPPERCASE, FIXED SET)
- USERNAME      : Account or login handle (e.g., hong_gildong, user_han99).
- PASSWORD      : Any password-like token exactly written in text (e.g., Abc1234!).
- EMAIL         : RFC-5322-like email patterns (e.g., yujin.kim@company.example).
- ADDRESS       : Postal address strings (e.g., "부산광역시 해운대구 센텀동로 25").
- NAME          : Full personal names when explicitly present (e.g., "이지훈").
- BANK_ACCOUNT  : Bank account numbers (e.g., 123-456-789012).
- ORDER_NUMBER  : Order IDs (e.g., ORD-20250918-7788).

DETECTION GUIDELINES
- USERNAME: Alphanumerics + `_`/`.` allowed; label only the handle itself, not surrounding words.
- PASSWORD: Label only if the literal password string is present; do NOT label “password” as a word unless the actual secret appears.
- EMAIL: Match common local@domain patterns; include the entire email address.
- ADDRESS: Label the full address substring, including numbers/floor markers if present.
- NAME: Label the exact name substring (Korean/English). Do not include role words (e.g., "직원").
- BANK_ACCOUNT: Label the numeric/hyphen account token only.
- ORDER_NUMBER: Label the ID token only (keep any fixed prefix like ORD- if part of the ID).

BOUNDARY & QUOTING
- Do not include trailing spaces or punctuation unless they are part of the entity.
- For surrounding quotes or brackets, include them ONLY if they are part of the true entity token itself.

AMBIGUITY & NEGATION
- Requests that talk ABOUT sensitive info without showing it (e.g., “Please send an email”) are NOT sensitive unless an actual entity substring appears.

ROBUSTNESS
- Preserve original casing and Unicode exactly in "text".
- If the same sensitive value appears multiple times, output each occurrence with correct offsets.
- If the input contains masked placeholders (e.g., ****), do NOT label them.

SCHEMA EXAMPLES

(1) Contains username and password
Input: "로그인 계정명: hong_gildong, 패스워드: Abc1234! 입력 시 실패 원인을 분석해줘."
Output:
{
  "has_sensitive": true,
  "entities": [
    {"label": "USERNAME", "text": "hong_gildong", "begin": 9, "end": 21},
    {"label": "PASSWORD", "text": "Abc1234!", "begin": 33, "end": 41}
  ]
}

(2) Address
Input: "본사 주소는 부산광역시 해운대구 센텀동로 25입니다."
Output:
{
  "has_sensitive": true,
  "entities": [
    {"label": "ADDRESS", "text": "부산광역시 해운대구 센텀동로 25", "begin": 6, "end": 27}
  ]
}

(3) Email
Input: "담당자 이메일은 yujin.kim@company.example인데 이메일 보내줘"
Output:
{
  "has_sensitive": true,
  "entities": [
    {"label": "EMAIL", "text": "yujin.kim@company.example", "begin": 10, "end": 38}
  ]
}

(4) Non-sensitive
Input: "IT 부서에서 사용할 신규 장비 리스트를 표로 정리해줘."
Output:
{
  "has_sensitive": false,
  "entities": []
}

VALIDATION CHECKS (BEFORE YOU OUTPUT)
- The JSON parses.
- Every entity.text equals input[begin:end].
- begin and end are integers with 0 <= begin < end <= len(input).
- Label ∈ {USERNAME, PASSWORD, EMAIL, ADDRESS, NAME, BANK_ACCOUNT, ORDER_NUMBER}.
- Entities sorted and deduplicated.

FAIL-SAFE
- If any rule would be violated by adding an entity, drop that entity instead of guessing.
- If nothing valid remains, output has_sensitive=false and entities=[].
"""

# ------------------ IO ------------------
def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """JSONL 또는 JSON 배열 파일을 로드하여 리스트로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return []
    # JSONL 추정
    if "\n" in s and s[0] != "[":
        rows = []
        for line in s.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    # JSON 배열
    obj = json.loads(s)
    if isinstance(obj, list):
        return obj
    # JSON manifest(경로 리스트)도 지원
    if isinstance(obj, dict) and "files" in obj and isinstance(obj["files"], list):
        records = []
        for p in obj["files"]:
            records.extend(_read_json_or_jsonl(p))
        return records
    raise ValueError(f"알 수 없는 JSON 형식: {path}")

def load_datasets(paths: List[str]) -> List[List[Dict[str, Any]]]:
    """여러 파일을 개별 데이터셋(list of samples)으로 로드"""
    datasets = []
    for p in paths:
        rows = _read_json_or_jsonl(p)
        datasets.append(rows)
    return datasets

def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """엔티티 정규화/경계 보정"""
    text = r.get("text", "")
    ents = []
    for e in r.get("entities", []) or []:
        label = e.get("label") or e.get("type")
        begin = int(e.get("begin", -1))
        end = int(e.get("end", -1))
        if 0 <= begin < end <= len(text):
            ent_text = e.get("text") or e.get("value") or text[begin:end]
            ents.append({"label": label, "text": ent_text, "begin": begin, "end": end})
    return {"text": text, "has_sensitive": bool(r.get("has_sensitive", False)), "entities": ents}

def to_target_json(r: Dict[str, Any]) -> str:
    obj = {"has_sensitive": bool(r.get("has_sensitive", False)), "entities": r.get("entities", [])}
    return json.dumps(obj, ensure_ascii=False)

def to_chat_example(text: str, tgt_json: str) -> Dict[str, Any]:
    return {"messages": [
        {"role":"system","content":SYS_PROMPT},
        {"role":"user","content":text},
        {"role":"assistant","content":tgt_json}
    ]}

# ------------------ split & mixing ------------------
def split_train_val(rows: List[Dict[str, Any]], val_ratio: float=0.2) -> Tuple[List, List]:
    n = len(rows)
    idx = list(range(n)); random.shuffle(idx)
    k = max(1, int(n*val_ratio)) if n>1 else 1
    val_idx = set(idx[:k]); tr, va = [], []
    for i in range(n):
        (va if i in val_idx else tr).append(rows[i])
    return tr, va

def temp_mix_weights(sizes: List[int], temperature: float) -> List[float]:
    if temperature <= 0: temperature = 1e-6
    w = [(s ** (1.0/temperature)) if s>0 else 0.0 for s in sizes]
    sm = sum(w) or 1.0
    return [x/sm for x in w]

def pack_epoch(per_ds_train_msgs: List[List[Dict[str,Any]]], samples_per_epoch: int, temperature: float) -> List[Dict[str,Any]]:
    sizes = [len(x) for x in per_ds_train_msgs]
    weights = temp_mix_weights(sizes, temperature)
    counts = [int(round(samples_per_epoch*w)) for w in weights]
    # 보정
    diff = samples_per_epoch - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: weights[i], reverse=True)
    i=0
    while diff != 0 and order:
        counts[order[i % len(order)]] += 1 if diff>0 else -1
        diff += -1 if diff>0 else 1
        i += 1
    # 샘플링
    bucket = []
    for ds_rows, c in zip(per_ds_train_msgs, counts):
        if c<=0: continue
        rows = ds_rows[:]
        random.shuffle(rows)
        if c <= len(rows):
            bucket.extend(rows[:c])
        else:
            bucket.extend(rows)
            # 부족분은 중복허용
            for _ in range(c-len(rows)):
                bucket.append(random.choice(ds_rows))
    random.shuffle(bucket)
    return bucket

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="베이스 모델(예: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--data", required=True, nargs="+", help="하나 이상의 데이터 파일(JSONL/JSON). JSON manifest도 지원")
    ap.add_argument("--out_dir", required=True, help="학습 산출물(LoRA 어댑터 포함) 저장 폴더")
    ap.add_argument("--merged_out", required=True, help="병합(단일 모델) 저장 폴더")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--samples_per_epoch", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    # LoRA 하이퍼파라미터 옵션
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.merged_out, exist_ok=True)

    # ----- 데이터 로드 -----
    # 각 파일을 "하나의 데이터셋"으로 취급 (파일 하나만 주면 단일 데이터셋)
    raw_datasets = load_datasets(args.data)
    # 정규화 + 중복 텍스트 제거 + split
    per_ds_train_msgs, all_val_msgs = [], []
    total_train = 0
    for rows in raw_datasets:
        rows = [normalize_row(r) for r in rows if r.get("text")]
        # 내부 중복 제거
        seen=set(); uniq=[]
        for r in rows:
            t=r["text"]
            if t not in seen:
                uniq.append(r); seen.add(t)
        rows = uniq
        if not rows: 
            continue
        tr, va = split_train_val(rows, args.val_ratio)
        tr_msgs = [to_chat_example(r["text"], to_target_json(r)) for r in tr]
        va_msgs = [to_chat_example(r["text"], to_target_json(r)) for r in va]
        per_ds_train_msgs.append(tr_msgs)
        all_val_msgs.extend(va_msgs)
        total_train += len(tr_msgs)

    if not per_ds_train_msgs:
        raise RuntimeError("학습할 샘플이 없습니다. --data 파일을 확인하세요.")
    if not all_val_msgs:
        # 최소 1개 보장
        all_val_msgs = per_ds_train_msgs[0][:1]

    print(f"[INFO] 데이터셋 수: {len(per_ds_train_msgs)}, train 합계: {total_train}, val 합계: {len(all_val_msgs)}")

    # ----- 토크나이저/모델/LoRA(QLoRA) -----
    use_bnb = True
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=bnb_cfg
    )
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    # assistant 응답만 학습
    resp_tmpl = tok.apply_chat_template([{"role":"assistant","content":""}], tokenize=False, add_generation_prompt=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=resp_tmpl, tokenizer=tok)

    sft_cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=1,  # 아래에서 epoch-by-epoch 루프 수행
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch//2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        max_seq_length=args.max_len,
        packing=True,
        dataset_num_proc=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        push_to_hub=False
    )

    def formatting_func(example):  # TRL이 chat_template 적용
        return example["messages"]

    # 최초 에폭 버킷
    epoch_bucket = pack_epoch(per_ds_train_msgs, args.samples_per_epoch, args.temperature)
    ds_train = Dataset.from_list(epoch_bucket)
    ds_val = Dataset.from_list(all_val_msgs)

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=ds_train, eval_dataset=ds_val,
        peft_config=peft_cfg,
        packing=sft_cfg.packing,
        dataset_text_field=None,
        max_seq_length=sft_cfg.max_seq_length,
        args=sft_cfg,
        formatting_func=formatting_func,
        data_collator=collator
    )

    total_epochs = args.epochs
    for ep in range(total_epochs):
        if ep > 0:
            epoch_bucket = pack_epoch(per_ds_train_msgs, args.samples_per_epoch, args.temperature)
            trainer.train_dataset = Dataset.from_list(epoch_bucket)
        print(f"\n[TRAIN] Epoch {ep+1}/{total_epochs} - samples: {len(trainer.train_dataset)}")
        trainer.train(resume_from_checkpoint=False)
        metrics = trainer.evaluate()
        print(f"[EVAL] Epoch {ep+1} metrics: {metrics}")
        # 중간 체크포인트(선택): trainer.save_model(os.path.join(args.out_dir, f"ep{ep+1}"))

    # ------ LoRA 병합 → 단일 모델 저장 ------
    print("\n[MERGE] LoRA를 베이스에 병합합니다...")
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    merged = PeftModel.from_pretrained(base, args.out_dir).merge_and_unload()
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)

    print(f"\n[완료] 어댑터 저장: {args.out_dir}")
    print(f"[완료] 병합 단일 모델 저장: {args.merged_out}")
    print("→ 배포 시 merged_out 경로를 바로 from_pretrained로 로드")

if __name__ == "__main__":
    main()
