# -*- coding: utf-8 -*-
"""
Qwen 3B/7B — QLoRA/LoRA SFT with CUSTOM MASKING using transformers.Trainer (TRL 없이도 동작)
- Dataset schema (JSON/JSONL):
  {
    "text": "<string>",
    "has_sensitive": <bool>,
    "entities": [
      {"value": "<substring>", "begin": <int>, "end": <int>, "label": "<LABEL>"}
    ]
  }
- Labels used in target JSON: {USERNAME, PASSWORD, EMAIL, ADDRESS, NAME, BANK_ACCOUNT}
"""

import os, json, argparse, random
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ========================= System Prompt (엄격) =========================
SYS_PROMPT = """You are "Sensitive-Info Detector", a precision assistant that outputs ONLY one JSON object per request.

TASK
- Given exactly one user sentence (Korean or English), analyze it and return:
  {
    "has_sensitive": <boolean>,
    "entities": [
      {"label": "<LABEL>", "text": "<exact substring>", "begin": <int>, "end": <int>}
    ]
  }

OUTPUT RULES (STRICT)
1) Output exactly one JSON object. No code fences, no extra text, no explanations. Trim leading/trailing whitespace.
2) JSON keys MUST appear in this order: has_sensitive, entities.
3) Each entity object MUST have keys in this order: label, text, begin, end.
4) Use UTF-8 JSON escaping when required. Do not alter or normalize input text; keep Unicode exactly as given.
5) Offsets are 0-based character indices in the ORIGINAL input string; end is exclusive.
6) Sort entities by (begin ASC, end ASC, then label ASC). If duplicates (same label, begin, end) exist, keep only one.
7) If an entity’s substring doesn’t exactly match the input, adjust begin/end so that text == input[begin:end].
8) If nothing is extractable, entities = [] and has_sensitive = false.
9) NEVER invent content. Do not infer hidden values; only label substrings that appear verbatim in the input.

SENSITIVITY POLICY
- Set has_sensitive = true iff at least one labeled entity is extracted; otherwise false.

SUPPORTED LABELS (UPPERCASE, FIXED SET)
- USERNAME, PASSWORD, EMAIL, ADDRESS, NAME, BANK_ACCOUNT

BOUNDARY & QUOTING
- Do not include trailing spaces/punctuation unless part of the entity.
- Include quotes/brackets only if they are part of the true token.

AMBIGUITY & NEGATION
- Talking ABOUT sensitive info without showing a concrete value is NOT sensitive.

ROBUSTNESS
- Preserve original casing and Unicode in "text".
- If the same value appears multiple times, output each occurrence with correct offsets.
- Ignore masked placeholders (e.g., ****).
"""

ALLOWED = {"USERNAME","PASSWORD","EMAIL","ADDRESS","NAME","BANK_ACCOUNT"}

# ========================= IO =========================
def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return []
    if "\n" in s and not s.lstrip().startswith("["):  # JSONL heuristic
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

# ========================= normalize to target JSON =========================
def _safe_entity(text: str, value: str, begin: int, end: int, label: str):
    if label not in ALLOWED:
        return None
    if not isinstance(begin, int) or not isinstance(end, int) or not (0 <= begin < end <= len(text)):
        return None
    span = text[begin:end]
    if not value:
        value = span
    if value != span:  # offsets authoritative
        value = span
    return {"label": label, "text": value, "begin": begin, "end": end}

def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    text = r.get("text", "")
    ents_in = r.get("entities") or []
    ents_out = []
    for e in ents_in:
        ent = _safe_entity(text, e.get("value"), int(e.get("begin", -1)), int(e.get("end", -1)), (e.get("label") or "").upper())
        if ent:
            ents_out.append(ent)
    ents_out.sort(key=lambda x: (x["begin"], x["end"], x["label"]))
    # dedup
    dedup, seen = [], set()
    for e in ents_out:
        k = (e["label"], e["begin"], e["end"])
        if k not in seen:
            dedup.append(e); seen.add(k)
    has_sensitive = bool(dedup) if r.get("has_sensitive") is None else bool(r.get("has_sensitive"))
    return {"text": text, "has_sensitive": has_sensitive, "entities": dedup}

def to_target_json(r: Dict[str, Any]) -> str:
    obj = {"has_sensitive": bool(r.get("has_sensitive", False)), "entities": r.get("entities", [])}
    return json.dumps(obj, ensure_ascii=False)

def to_chat_example(text: str, tgt_json: str) -> Dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": tgt_json}
    ]}

# ========================= split & mixing =========================
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

# ========================= CUSTOM MASKING =========================
def build_sample(tok, messages, max_len=1024):
    """
    Make tensors with labels on ASSISTANT segment only; system/user are masked (-100).
    """
    # full convo
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # assistant-only
    assistant_text = tok.apply_chat_template(
        [{"role":"assistant","content":messages[-1]["content"]}],
        tokenize=False, add_generation_prompt=False
    )
    start_char = full_text.rfind(assistant_text)
    if start_char == -1:
        start_char = len(full_text) - len(assistant_text)

    enc_full = tok(full_text, return_tensors="pt", truncation=False)
    input_ids = enc_full["input_ids"][0]
    attn = enc_full["attention_mask"][0]

    # token start index estimation via prefix tokenization
    prefix = full_text[:start_char]
    enc_prefix = tok(prefix, return_tensors="pt", truncation=False)
    start_tok = enc_prefix["input_ids"].shape[1]

    enc_asst = tok(assistant_text, return_tensors="pt", truncation=False)
    asst_len = enc_asst["input_ids"].shape[1]
    end_tok = start_tok + asst_len

    # truncate head if too long
    if input_ids.shape[0] > max_len:
        cut = input_ids.shape[0] - max_len
        input_ids = input_ids[cut:]
        attn = attn[cut:]
        start_tok = max(0, start_tok - cut)
        end_tok = max(0, end_tok - cut)
        end_tok = min(end_tok, input_ids.shape[0])

    labels = torch.full_like(input_ids, fill_value=-100)
    start_tok = max(0, min(start_tok, input_ids.shape[0]))
    end_tok   = max(0, min(end_tok,   input_ids.shape[0]))
    if end_tok > start_tok:
        labels[start_tok:end_tok] = input_ids[start_tok:end_tok]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

class ChatMaskedDataset(TorchDataset):
    def __init__(self, tok, list_of_messages: List[Dict[str,Any]], max_len=1024):
        self.tok = tok
        self.rows = list_of_messages
        self.max_len = max_len
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        return build_sample(self.tok, msgs, self.max_len)

# --- collate: pad batch tensors (input_ids, attention_mask, labels) ---
def pad_collate_fn(batch, pad_id: int, label_pad_id: int = -100):
    input_ids = [b["input_ids"] for b in batch]
    attn      = [b["attention_mask"] for b in batch]
    labels    = [b["labels"] for b in batch]
    maxlen = max(x.size(0) for x in input_ids)
    def _pad(seq, value):
        return torch.cat([seq, torch.full((maxlen - seq.size(0),), value, dtype=seq.dtype)], dim=0)
    input_ids = torch.stack([_pad(x, pad_id) for x in input_ids], dim=0)
    attn      = torch.stack([_pad(x, 0) for x in attn], dim=0)
    labels    = torch.stack([_pad(x, label_pad_id) for x in labels], dim=0)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# ========================= main =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model, e.g., Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True, nargs="+", help="One or more dataset files (JSONL/JSON).")
    ap.add_argument("--out_dir", required=True, help="Where to save LoRA adapter")
    ap.add_argument("--merged_out", required=True, help="Where to save merged single model")
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
    ap.add_argument("--no_qlora", action="store_true", help="Disable 4-bit quant; use normal LoRA")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.merged_out, exist_ok=True)

    # Load & normalize
    raw_sets = load_datasets(args.data)
    per_ds_train_msgs, all_val_msgs = [], []
    total_train = 0
    for rows in raw_sets:
        rows = [normalize_row(r) for r in rows if r.get("text")]
        # de-dup by text within each dataset
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

    # Tokenizer & Model
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if not args.no_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype="auto",
        quantization_config=quant_cfg
    )

    # Apply LoRA (PEFT) directly
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    # Trainer args
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch//2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=1,          # 에폭은 아래 수동 루프
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        remove_unused_columns=False,  # 커스텀 텐서 컬럼 유지
        optim="paged_adamw_8bit" if (quant_cfg is not None) else "adamw_torch"
    )

    # epoch 1개 분량 버킷 → Dataset 생성
    epoch_bucket = pack_epoch(per_ds_train_msgs, args.samples_per_epoch, args.temperature)
    ds_train = ChatMaskedDataset(tok, epoch_bucket, max_len=args.max_len)
    ds_val   = ChatMaskedDataset(tok, all_val_msgs, max_len=args.max_len)

    collate = lambda batch: pad_collate_fn(batch, pad_id=tok.pad_token_id, label_pad_id=-100)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collate
    )

    total_epochs = args.epochs
    for ep in range(total_epochs):
        if ep > 0:
            epoch_bucket = pack_epoch(per_ds_train_msgs, args.samples_per_epoch, args.temperature)
            trainer.train_dataset = ChatMaskedDataset(tok, epoch_bucket, max_len=args.max_len)
        print(f"\n[TRAIN] Epoch {ep+1}/{total_epochs} - samples: {len(trainer.train_dataset)}")
        trainer.train()
        metrics = trainer.evaluate()
        print(f"[EVAL] Epoch {ep+1} metrics: {metrics}")

    # Merge LoRA → single model
    print("\n[MERGE] Merging LoRA into base...")
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype="auto")
    merged = PeftModel.from_pretrained(base, args.out_dir).merge_and_unload()
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)

    # quick inference helper
    infer_py = f'''# -*- coding: utf-8 -*-
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"{os.path.abspath(args.merged_out)}"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", dtype="auto")
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
