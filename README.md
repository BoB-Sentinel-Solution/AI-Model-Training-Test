# AI-Model-Training
ai 모델에 제작한 데이터셋 학습

## Requirement
```
pip install "transformers>=4.44" "trl>=0.9.6" peft bitsandbytes datasets accelerate

#GPU CUDA12.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

pip install transformers trl peft accelerate datasets bitsandbytes safetensors huggingface_hub tokenizers numpy jsonschema

pip install seqeval wandb einops
```

## How to run
```
# QLoRA(4bit)
python train_and_merge_qwen_custom_mask.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data test_dataset.json \
  --out_dir runs/qwen7b_sft \
  --merged_out runs/qwen7b_sft_merged \
  --epochs 3 --bf16 \
  --samples_per_epoch 2000 --temperature 1.3 \
  --batch 1 --grad_accum 16 --max_len 1024
```

```
# LoRA
python train_and_merge_qwen_custom_mask.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data test_dataset.json \
  --out_dir runs/qwen3b_sft \
  --merged_out runs/qwen3b_sft_merged \
  --epochs 3 --bf16 --no_qlora \
  --batch 4 --grad_accum 8 --max_len 1024
```

<img width="1905" height="307" alt="image" src="https://github.com/user-attachments/assets/0acaf195-3610-4aec-a614-687f8ad15cf2" />
<img width="1899" height="234" alt="image" src="https://github.com/user-attachments/assets/3275fbd6-fc9f-4852-86c5-7d73d6fc456e" />
<img width="1893" height="260" alt="image" src="https://github.com/user-attachments/assets/37f0856f-cd75-407f-b25c-799451327dea" />

