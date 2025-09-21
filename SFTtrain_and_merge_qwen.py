# -*- coding: utf-8 -*-
"""
Qwen 3B/7B — QLoRA/LoRA SFT for JSONL(messages) with CUSTOM MASKING (transformers.Trainer)
-------------------------------------------------------------------------------------------
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
# Expanded label set (for consistency across project; not directly used by training loop)
# ----------------------------------------------------------------------------------------
ALLOWED = {
    # 개인 식별·연락
    "NAME","PHONE","EMAIL","ADDRESS","BILLING_ADDRESS","SHIPPING_ADDRESS","POSTAL_CODE",
    "DATE_OF_BIRTH","RESIDENT_ID","FOREIGNER_ID","PASSPORT","DRIVER_LICENSE","BUSINESS_ID",
    "TAX_ID","SSN","HEALTH_INSURANCE_ID","EMERGENCY_CONTACT","EMERGENCY_PHONE",
    # 계정·인증
    "USERNAME","NICKNAME","ROLE","DEPARTMENT","GROUP","PERMISSION","PASSWORD","PASSWORD_HASH",
    "SECURITY_QA","MFA_SECRET","BACKUP_CODE","SESSION_ID","COOKIE","JWT","ACCESS_TOKEN",
    "REFRESH_TOKEN","OAUTH_CLIENT_ID","OAUTH_CLIENT_SECRET","API_KEY","SSH_PRIVATE_KEY",
    "TLS_PRIVATE_KEY","PGP_PRIVATE_KEY","MNEMONIC","TEMP_CLOUD_CREDENTIAL","DEVICE_ID","IMEI",
    "SERIAL_NUMBER","BROWSER_FINGERPRINT","SAML_ASSERTION","OIDC_ID_TOKEN","CONNECTION_STRING",
    "INTERNAL_URL","LAST_LOGIN_IP","LAST_LOGIN_DEVICE","LAST_LOGIN_BROWSER","LAST_LOGIN_AT",
    # 금융·결제
    "BANK_NAME","BANK_BRANCH","BANK_ACCOUNT","ACCOUNT_HOLDER","IBAN","SWIFT_BIC","ROUTING_NUMBER",
    "VIRTUAL_ACCOUNT","CURRENCY","BALANCE","CARD_NUMBER","CARD_EXPIRY","CARD_CVV","CARD_HOLDER",
    "PAYMENT_PIN","SECURITIES_ACCOUNT","WALLET_ADDRESS","LOYALTY_ID","LOYALTY_BALANCE",
    # 고객·거래·지원
    "CUSTOMER_ID","MEMBERSHIP_ID","ORDER_ID","INVOICE_ID","TAX_INVOICE_ID","BILL_ID","REFUND_ID",
    "EXCHANGE_ID","RMA_ID","TICKET_ID","TRACKING_ID","COUPON_CODE","VOUCHER_CODE",
    "GATEWAY_CUSTOMER_ID","PAYMENT_PROFILE_ID","BUYER_NAME","RECIPIENT_NAME","CRM_RECORD_ID",
    "CUSTOMER_NOTE_ID",
    # 조직
    "COMPANY_NAME","ORG_NAME","DEPARTMENT_NAME","EMPLOYEE_ID","JOB_TITLE","EMPLOYMENT_TYPE",
    "HIRE_DATE","LEAVE_DATE","SALARY","BENEFIT_INFO","INSURANCE_INFO","OFFICE_EXT","OFFICE_LOCATION",
    "WORKSITE","MANAGER_FLAG","ACCESS_CARD_ID","READER_ID","DUTY_ASSIGNMENT","TRAINING_COMPLETION_DATE",
    "TRAINING_EXPIRY","EDUCATION_CERT"
}

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
    - Compose chat with tokenizer's chat template.
    - Find assistant segment and label only that region. Others -> -100.
    """
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Build assistant-only text (same formatting as in full chat)
    asst_only_text = tok.apply_chat_template(
        [{"role": "assistant", "content": messages[-1]["content"]}],
        tokenize=False, add_generation_prompt=False
    )

    # Locate assistant substring inside the full chat string
    start_char = full_text.rfind(asst_only_text)
    if start_char == -1:
        # fallback: assume assistant text is at the tail
        start_char = max(0, len(full_text) - len(asst_only_text))

    enc_full = tok(full_text, return_tensors="pt", truncation=False)
    input_ids = enc_full["input_ids"][0]
    attn      = enc_full["attention_mask"][0]

    # Token position for assistant segment via tokenizing prefix
    prefix = full_text[:start_char]
    enc_prefix = tok(prefix, return_tensors="pt", truncation=False)
    start_tok = enc_prefix["input_ids"].shape[1]

    enc_asst  = tok(asst_only_text, return_tensors="pt", truncation=False)
    asst_len  = enc_asst["input_ids"].shape[1]
    end_tok   = start_tok + asst_len

    # Truncate head to fit max_len (keep the tail which usually contains assistant)
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
# TrainingArguments (version-safe construction)
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
            gradient_checkpointing=True,
            bf16=args.bf16,
            fp16=(args.fp16 and not args.bf16),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            remove_unused_columns=False,
            optim=("paged_adamw_8bit" if not args.no_qlora else "adamw_torch"),
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

    # Simple split by id order (or shuffle then split)
    idx = list(range(n))
    random.shuffle(idx)
    k = max(1, int(n * args.val_ratio)) if n > 1 else 1
    val_idx = set(idx[:k])
    train_rows = [all_rows[i] for i in range(n) if i not in val_idx]
    val_rows   = [all_rows[i] for i in range(n) if i in val_idx]

    print(f"[INFO] Loaded samples: {n} (train {len(train_rows)}, val {len(val_rows)})")

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

    # Apply LoRA
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

    # Save LoRA adapter explicitly
    model.save_pretrained(args.out_dir)

    # Merge LoRA → single model
    print("\n[MERGE] Merging LoRA into base...")
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype="auto")
    merged = PeftModel.from_pretrained(base, args.out_dir).merge_and_unload()
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)

    # Quick inference script
    infer_py = f'''# -*- coding: utf-8 -*-
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"{os.path.abspath(args.merged_out)}"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", dtype="auto")
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
