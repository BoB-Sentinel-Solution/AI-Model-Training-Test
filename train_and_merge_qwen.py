# -*- coding: utf-8 -*-
"""
Qwen 3B/7B SFT (QLoRA) for Sensitive-Info Detection
- Input dataset schema (JSON/JSONL):
  {
    "text": "<string>",
    "has_sensitive": <bool>,
    "entities": [
      {"value": "<exact substring>", "begin": <int>, "end": <int>, "label": "<LABEL>"},
      ...
    ]
  }
- Supported LABEL set (dataset-aligned):
  {USERNAME, PASSWORD, EMAIL, ADDRESS, NAME, BANK_ACCOUNT}

Features:
* Multiple --data files (JSONL or JSON array) -> per-file train/val split -> temperature-mixed sampling per epoch
* QLoRA SFT on Qwen 3B/7B
* Merge LoRA into base → single model at --merged_out

Usage examples:
  python train_and_merge_qwen_multi.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data test_dataset.json \
    --out_dir runs/qwen3b_sft \
    --merged_out runs/qwen3b_sft_merged \
    --epochs 3 --bf16

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

# =========================
# Ultra-Strict System Prompt
# (dataset-aligned label set)
# =========================
SYS_PROMPT = """You are "Sensitive-Info Detector", a precision assistant that outputs ONLY one JSON object per request.

TASK
- Given exactly one user sentence (Korean or English), analyze it and return:
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

DETECTION GUIDELINES
- USERNAME: Alphanumerics plus `_` or `.` allowed; label only the handle string, not surrounding words.
- PASSWORD: Label only if the literal secret string is present; do NOT label the word “password” by itself.
- EMAIL: Match common local@domain patterns; include the entire address.
- ADDRESS: Label the full postal substring, including numbers/floor markers if present.
- NAME: Label the exact personal name substring (Korean/English). Do not include role words (e.g., "직원").
- BANK_ACCOUNT: Label the numeric/hyphenated account token only.

BOUNDARY & QUOTING
- Do not include trailing spaces or punctuation unless they are part of the entity.
- For surrounding quotes/brackets, include them ONLY if they are part of the true entity token.

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
- Label ∈ {USERNAME, PASSWORD, EMAIL, ADDRESS, NAME, BANK_ACCOUNT}.
- Entities sorted and deduplicated.

FAIL-SAFE
- If any rule would be violated by adding an entity, drop that entity instead of guessing.
- If nothing valid remains, output has_sensitive=false and entities=[].
"""

# ------------------ IO helpers ------------------
def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL or JSON array → list[dict]. Also supports a JSON manifest: {"files": ["a.jsonl", ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return []
    # JSONL heuristic
    if "\n" in s and not s.lstrip().startswith("["):
        rows = []
        for line in s.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    obj = json.loads(s)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("files"), list):
        out = []
        for p in obj["files"]:
            out.extend(_read_json_or_jsonl(p))
        return out
    raise ValueError(f"Unknown JSON format: {path}")

def load_datasets(paths: List[str]) -> List[List[Dict[str, Any]]]:
    return [_read_json_or_jsonl(p) for p in paths]

# ------------------ normalization ------------------
ALLOWED = {"USERNAME","PASSWORD","EMAIL","ADDRESS","NAME","BANK_ACCOUNT"}

def _safe_entity(text: str, value: str, begin: int, end: int, label: str):
    """Map dataset entity(value/begin/end/label) → model target entity(text/begin/end/label)."""
    if label not in ALLOWED:
        return None
    if not isinstance(begin, int) or not isinstance(end, int):
        return None
    if not (0 <= begin < end <= len(text)):
        return None
    # Prefer dataset 'value', but force exact match with input range
    span = text[begin:end]
    if value is None or value == "":
        value = span
    # If mismatch, trust offsets and replace to keep schema invariant text == input[begin:end]
    if span != value:
        value = span
    return {"label": label, "text": value, "begin": begin, "end": end}

def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Return training target object: text + has_sensitive(bool) + entities([{label,text,begin,end}])."""
    text = r.get("text", "")
    ents_in = r.get("entities") or []
    ents_out = []
    for e in ents_in:
        ent = _safe_entity(text, e.get("value"), int(e.get("begin", -1)), int(e.get("end", -1)), (e.get("label") or "").upper())
        if ent:
            ents_out.append(ent)
    # sort & dedup
    ents_out.sort(key=lambda x: (x["begin"], x["end"], x["label"]))
    dedup = []
    seen = set()
    for e in ents_out:
        key = (e["label"], e["begin"], e["end"])
        if key not in seen:
            dedup.append(e); seen.add(key)
    has_sensitive = bool(dedup) if r.get("has_sensitive") is None else bool(r.get("has_sensitive"))
    return {"text": text, "has_sensitive": has_sensitive, "entities": dedup}

def to_target_json(r: Dict[str, Any]) -> str:
    obj = {"has_sensitive": bool(r.get("has_sensitive", False)), "entities": r.get("entities", [])}
    return json.dumps(obj, ensure_ascii=False)

def to_chat_example(text: str, target_json: str) -> Dict[str, Any]:
    return {"messages": [
        {"role":"system","content": SYS_PROMPT},
        {"role":"user","content": text},
        {"role":"assistant","content": target_json}
    ]}

# ------------------ split & mixing ------------------
def split_train_val(rows: List[Dict[str, Any]], val_ratio: float=0.2) -> Tuple[List, List]:
    n = len(rows)
    idx = list(range(n)); random.shuffle(idx)
    k = max(1, int(n*val_ratio)) if n>1 else 1
    vset = set(idx[:k]); tr, va = [], []
    for i in range(n):
        (va if i in vset else tr).append(rows[i])
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
    diff = samples_per_epoch - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: weights[i], reverse=True)
    i=0
    while diff != 0 and order:
        counts[order[i % len(order)]] += 1 if diff>0 else -1
        diff += -1 if diff>0 else 1
        i += 1
    bucket = []
    for ds_rows, c in zip(per_ds_train_msgs, counts):
        if c<=0: continue
        rows = ds_rows[:]; random.shuffle(rows)
        if c <= len(rows):
            bucket.extend(rows[:c])
        else:
            bucket.extend(rows)
            for _ in range(c-len(rows)):
                bucket.append(random.choice(ds_rows))
    random.shuffle(bucket)
    return bucket

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model (e.g., Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--data", required=True, nargs="+", help="One or more dataset files (JSONL/JSON). Manifest JSON {'files':[...]} supported.")
    ap.add_argument("--out_dir", required=True, help="Directory to save SFT (LoRA adapter)")
    ap.add_argument("--merged_out", required=True, help="Directory to save merged single model")
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
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.merged_out, exist_ok=True)

    # Load & normalize per dataset
    raw_sets = load_datasets(args.data)
    per_ds_train_msgs, all_val_msgs = [], []
    total_train = 0
    for rows in raw_sets:
        rows = [normalize_row(r) for r in rows if r.get("text")]
        # per-dataset de-dup by text
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
        raise RuntimeError("No training samples. Check --data files.")
    if not all_val_msgs:
        all_val_msgs = per_ds_train_msgs[0][:1]

    print(f"[INFO] datasets: {len(per_ds_train_msgs)}, train total: {total_train}, val total: {len(all_val_msgs)}")

    # Tokenizer/Model with QLoRA
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
        args.model, device_map="auto", torch_dtype="auto", quantization_config=bnb_cfg
    )
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    resp_tmpl = tok.apply_chat_template([{"role":"assistant","content":""}], tokenize=False, add_generation_prompt=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=resp_tmpl, tokenizer=tok)

    sft_cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=1,                     # loop epochs manually
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

    def formatting_func(example):  # TRL will apply chat template
        return example["messages"]

    # initial epoch bucket
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

    # Merge LoRA → single model
    print("\n[MERGE] Merging LoRA into base...")
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    merged = PeftModel.from_pretrained(base, args.out_dir).merge_and_unload()
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)

    # quick inference helper
    infer_py = f'''# -*- coding: utf-8 -*-
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"{os.path.abspath(args.merged_out)}"   # merged single model

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype="auto")
if tok.pad_token is None: tok.pad_token = tok.eos_token

SYS = {json.dumps(SYS_PROMPT, ensure_ascii=False)}

def ask(text: str, max_new_tokens=256):
    msgs = [{{"role":"system","content":SYS}}, {{"role":"user","content":text}}]
    x = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tok.eos_token_id)
    return tok.decode(y[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        try:
            q = input("text> ").strip()
            if not q: continue
            print(ask(q))
        except (EOFError, KeyboardInterrupt):
            break
'''
    with open(os.path.join(args.merged_out, "infer.py"), "w", encoding="utf-8") as f:
        f.write(infer_py)

    print(f"\n[Done] Adapter saved at: {args.out_dir}")
    print(f"[Done] Merged single model at: {args.merged_out}")
    print(f"Try: python {os.path.join(args.merged_out, 'infer.py')}")

if __name__ == "__main__":
    main()
