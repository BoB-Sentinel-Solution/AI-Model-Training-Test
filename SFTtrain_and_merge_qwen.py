# -*- coding: utf-8 -*-
"""
Qwen/Mistral 등 LLaMA계열 — QLoRA/LoRA SFT for JSONL(messages) with STABLE MASKING
-----------------------------------------------------------------------------------
Dataset format (JSONL only; one JSON object per line):

{"id": 1, "messages": [
  {"role":"system","content":"<system prompt text>"},
  {"role":"user","content":"<user text>"},
  {"role":"assistant","content":"<assistant target text>"}
]}

- Exactly 3 messages per sample: system, user, assistant (in this order).
- The assistant content is the supervision target (we mask out system/user).

Usage (examples):
  python train_qwen_jsonl.py --model Qwen/Qwen2.5-7B-Instruct \
    --data Learning_Test_Dataset_10.jsonl \
    --out_dir runs/qwen_sft_adapter \
    --merged_out runs/qwen_sft_merged \
    --epochs 3 --bf16 --batch 1 --grad_accum 16 --max_len 1024

  # Disable 4-bit (use full-precision LoRA):
  python train_qwen_jsonl.py ... --no_qlora
"""

import os, json, argparse, random
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------------------------------------------------------------------
# Data loading: JSONL (strict). Each line -> {"id":..., "messages":[{sys},{usr},{asst}]}
# ----------------------------------------------------------------------------------------
def read_jsonl_messages(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"[{p}:L{ln}] JSON parse error: {e}")
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or len(msgs) != 3:
                    raise ValueError(f"[{p}:L{ln}] 'messages' must be list of length 3 (system,user,assistant).")
                roles = [m.get("role") for m in msgs]
                if roles != ["system","user","assistant"]:
                    raise ValueError(f"[{p}:L{ln}] roles must be ['system','user','assistant'], got {roles}.")
                for i, m in enumerate(msgs):
                    if "content" not in m or not isinstance(m["content"], str):
                        raise ValueError(f"[{p}:L{ln}] messages[{i}].content must be a string.")
                rows.append(obj)
    return rows

# ----------------------------------------------------------------------------------------
# Masking: system/user masked; assistant segment labeled
# ----------------------------------------------------------------------------------------
def build_sample(tok, messages: List[Dict[str,str]], max_len: int = 1024):
    """
    안정적 라벨링:
    - full: system+user+assistant(content 포함)
    - prefix: system+user + assistant 헤더만 (add_generation_prompt=True)
    → 두 토큰 길이 차이로 assistant 라벨 범위를 계산.
    """
    # full
    input_ids = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=False
    )[0]
    attn = torch.ones_like(input_ids)

    # prefix (assistant 헤더까지만)
    prefix_ids = tok.apply_chat_template(
        messages[:-1], return_tensors="pt", add_generation_prompt=True
    )[0]

    start_tok = prefix_ids.shape[0]
    end_tok   = input_ids.shape[0]

    # Truncate head to fit max_len (keep tail; adjust offsets)
    if input_ids.shape[0] > max_len:
        cut = input_ids.shape[0] - max_len
        input_ids = input_ids[cut:]
        attn      = attn[cut:]
        start_tok = max(0, start_tok - cut)
        end_tok   = max(0, end_tok   - cut)
        end_tok   = min(end_tok, input_ids.shape[0])

    labels = torch.full_like(input_ids, fill_value=-100)
    start_tok = max(0, min(start_tok, input_ids.shape[0]))
    end_tok   = max(0, min(end_tok,   input_ids.shape[0]))
    if end_tok > start_tok:
        labels[start_tok:end_tok] = input_ids[start_tok:end_tok]
    # else: 어시스턴트 라벨이 모두 잘렸다면 상위에서 필터링 권장

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

class ChatMaskedDataset(Dataset):
    def __init__(self, tok, list_of_msg_objs: List[Dict[str, Any]], max_len=1024):
        self.tok = tok
        self.rows = list_of_msg_objs
        self.max_len = max_len
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        return build_sample(self.tok, msgs, self.max_len)

def pad_collate_fn(batch, pad_id: int, label_pad_id: int = -100):
    input_ids = [b["input_ids"] for b in batch]
    attn      = [b["attention_mask"] for b in batch]
    labels    = [b["labels"] for b in batch]
    maxlen = max(x.size(0) for x in input_ids)
    def _pad(seq, value):
        if seq.size(0) < maxlen:
            pad = torch.full((maxlen - seq.size(0),), value, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)
        return seq
    input_ids = torch.stack([_pad(x, pad_id) for x in input_ids], dim=0)
    attn      = torch.stack([_pad(x, 0) for x in attn], dim=0)
    labels    = torch.stack([_pad(x, label_pad_id) for x in labels], dim=0)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# ----------------------------------------------------------------------------------------
# TrainingArguments
# ----------------------------------------------------------------------------------------
def make_training_args(args):
    try:
        return TrainingArguments(
            output_dir=args.out_dir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=max(1, args.batch//2),
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=1,  # we loop epochs manually
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            gradient_checkpointing=False,
            bf16=args.bf16,
            fp16=(args.fp16 and not args.bf16),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            remove_unused_columns=False,
            optim=("paged_adamw_8bit" if not args.no_qlora else "adamw_torch"),
            report_to="none",
            save_safetensors=True,
            logging_first_step=True,
            dataloader_num_workers=2,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        kwargs = dict(
            output_dir=args.out_dir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=max(1, args.batch//2),
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=1,
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            save_safetensors=True,
        )
        if args.bf16 or args.fp16:
            kwargs["fp16"] = True
        return TrainingArguments(**kwargs)

# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model, e.g., Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True, nargs="+", help="JSONL file(s); each line has {id, messages[sys,user,assistant]}")
    ap.add_argument("--out_dir", required=True, help="Where to save LoRA adapter")
    ap.add_argument("--merged_out", required=True, help="Where to save merged single model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_qlora", action="store_true", help="Disable 4-bit quantization; use standard LoRA")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.merged_out, exist_ok=True)

    # Load JSONL messages
    all_rows = read_jsonl_messages(args.data)
    n = len(all_rows)
    if n == 0:
        raise RuntimeError("No samples found in the given JSONL file(s).")

    # Split
    idx = list(range(n))
    random.shuffle(idx)
    k = max(1, int(n * args.val_ratio)) if n > 1 else 1
    val_idx = set(idx[:k])
    train_rows = [all_rows[i] for i in range(n) if i not in val_idx]
    val_rows   = [all_rows[i] for i in range(n) if i in val_idx]

    print(f"[INFO] Loaded samples: {n} (train {len(train_rows)}, val {len(val_rows)})")

    # Tokenizer (fast → slow 폴백 + chat_template 가드)
    try:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] fast tokenizer failed: {e}\n-> retry with use_fast=False")
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{{ eos_token }}"
        )

    # QLoRA config
    quant_cfg = None
    if not args.no_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(torch.bfloat16 if args.bf16 else torch.float16),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )

    # (옵션) 속도 최적화
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    # 캐시는 활성화 유지 (요청사항)

    # QLoRA 준비
    if quant_cfg is not None:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except Exception as e:
            print(f"[WARN] prepare_model_for_kbit_training skipped: {e}")

    # LoRA adapter
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    # Datasets
    ds_train = ChatMaskedDataset(tok, train_rows, max_len=args.max_len)
    ds_val   = ChatMaskedDataset(tok, val_rows if val_rows else train_rows[:1], max_len=args.max_len)
    collate  = lambda batch: pad_collate_fn(batch, pad_id=tok.pad_token_id, label_pad_id=-100)

    # Trainer
    train_args = make_training_args(args)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collate
    )

    # Manual epochs
    total_epochs = args.epochs
    for ep in range(total_epochs):
        print(f"\n[TRAIN] Epoch {ep+1}/{total_epochs} - samples: {len(ds_train)}")
        trainer.train()
        try:
            metrics = trainer.evaluate()
            print(f"[EVAL] Epoch {ep+1} metrics: {metrics}")
        except Exception as e:
            print(f"[WARN] evaluate skipped: {e}")

    # Save LoRA adapter
    model.save_pretrained(args.out_dir)

    # Merge LoRA → single model
    print("\n[MERGE] Merging LoRA into base...")
    # GPU 여유가 넉넉하지 않으면 device_map="cpu"로 병합 권장
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, args.out_dir).merge_and_unload()
    merged.save_pretrained(args.merged_out, safe_serialization=True)
    tok.save_pretrained(args.merged_out)

    # Quick inference script
    infer_py = f'''# -*- coding: utf-8 -*-
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"{os.path.abspath(args.merged_out)}"
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16 if {args.bf16} else (torch.float16 if {args.fp16} else None), trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def chat_once(system_text: str, user_text: str, max_new_tokens=256):
    msgs = [{{"role":"system","content":system_text}},
            {{"role":"user","content":user_text}}]
    x = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tok.eos_token_id)
    print(tok.decode(y[0], skip_special_tokens=True))

if __name__ == "__main__":
    sys = "You are a strict detector for sensitive entities. Output ONLY one JSON object."
    while True:
        try:
            q = input("text> ").strip()
            if not q: continue
            chat_once(sys, q)
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
