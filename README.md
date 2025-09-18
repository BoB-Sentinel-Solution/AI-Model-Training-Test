# AI-Model-Train
ai 모델에 제작한 데이터셋 학습

## Requirement
```
pip install "transformers>=4.44" "trl>=0.9.6" peft bitsandbytes datasets accelerate
```

## How to run
단일 데이터셋
```
python train_and_merge_qwen_multi.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data test_dataset.json \
  --out_dir runs/qwen3b_sft \
  --merged_out runs/qwen3b_sft_merged \
  --epochs 3 --bf16 --samples_per_epoch 2000 --temperature 1.3
```

여러 데이터셋
```
python train_and_merge_qwen_multi.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data ds1.jsonl --data ds2.json --data ds3.jsonl --data ds4.json \
  --out_dir runs/qwen7b_multi \
  --merged_out runs/qwen7b_multi_merged \
  --epochs 3 --bf16 --samples_per_epoch 8000 --temperature 1.3 \
  --batch 2 --grad_accum 16 --max_len 1024
```
