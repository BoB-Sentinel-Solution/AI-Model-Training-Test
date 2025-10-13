#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 민감정보 탐지: 추론(+예측 생성) + 평가 (v2, 한국어 출력)

변경점
- token-classification 파이프라인 호출에서 `truncation` 인자 제거 (HF 최신 버전과 호환)
- 길이 초과 텍스트 자동 청크 분할(window/overlap) 지원
- --task 선택 추가: "token"(기본) 또는 "generation"
"""

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_jsonl_by_id(path: str) -> Dict[str, dict]:
    data = {}
    for obj in load_jsonl(path):
        sid = str(obj.get("id"))
        data[sid] = obj
    return data

def safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0

def iou_span(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    (a0,a1), (b0,b1) = a, b
    inter = max(0, min(a1,b1) - max(a0,b0))
    if inter == 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def match_entities(preds: List[Tuple[int,int,str]], golds: List[Tuple[int,int,str]],
                   mode: str="exact", iou_th: float=0.5):
    used_g = [False]*len(golds)
    tp = 0
    label_counts = Counter()

    for pb, pe, pl in preds:
        match_idx = -1
        best_score = -1.0
        for gi,(gb,ge,gl) in enumerate(golds):
            if used_g[gi]:
                continue
            if pl != gl:
                continue
            if mode == "exact":
                ok = (pb == gb) and (pe == ge)
                score = 1.0 if ok else -1.0
            elif mode == "overlap":
                score = iou_span((pb,pe), (gb,ge))
                ok = score >= iou_th
            else:
                raise ValueError(f"알 수 없는 매칭 모드: {mode}")
            if ok and score > best_score:
                best_score = score
                match_idx = gi
        if match_idx >= 0:
            used_g[match_idx] = True
            tp += 1
            label_counts[("TP", pl)] += 1
        else:
            label_counts[("FP", pl)] += 1

    fn = 0
    for gi,(gb,ge,gl) in enumerate(golds):
        if not used_g[gi]:
            fn += 1
            label_counts[("FN", gl)] += 1

    fp = sum(v for (t,_),v in label_counts.items() if t=="FP")
    return tp, fp, fn, label_counts

def evaluate(answers_path: str, predictions_path: str, match: str, iou: float):
    gold_by_id = load_jsonl_by_id(answers_path)
    pred_by_id = load_jsonl_by_id(predictions_path)

    common_ids = sorted(set(gold_by_id.keys()) & set(pred_by_id.keys()))
    if not common_ids:
        raise SystemExit("정답과 예측 간에 공통 id가 없습니다. 두 파일을 확인하세요.")

    total_tp = total_fp = total_fn = 0
    per_label = Counter()

    for sid in common_ids:
        gold_ents = [(int(e["begin"]), int(e["end"]), str(e["label"])) for e in gold_by_id[sid].get("entities", [])]
        pred_ents = [(int(e["begin"]), int(e["end"]), str(e["label"])) for e in pred_by_id[sid].get("entities", [])]
        tp, fp, fn, label_counts = match_entities(pred_ents, gold_ents, match, iou)
        total_tp += tp; total_fp += fp; total_fn += fn
        per_label.update(label_counts)

    precision = safe_div(total_tp, total_tp + total_fp)
    recall    = safe_div(total_tp, total_tp + total_fn)
    f1        = safe_div(2*precision*recall, precision+recall) if (precision+recall) else 0.0

    print("==============================================")
    print(" AI 민감정보 탐지 성능 결과 (한국어 출력)")
    print("==============================================\n")
    print("■ 전체 요약 (Micro-averaged)")
    print(f"  - TP(참양성) = {total_tp}")
    print(f"  - FP(거짓양성/오탐) = {total_fp}")
    print(f"  - FN(거짓음성/미탐) = {total_fn}")
    print(f"  - 정밀도(신뢰도, Precision) = {precision:.4f}")
    print(f"  - 재현율(탐지능력, Recall)  = {recall:.4f}")
    print(f"  - F1 점수                  = {f1:.4f}\n")

    labels = sorted({lab for (_,lab) in {(t,l) for (t,l) in per_label.keys()}})
    p_list, r_list = [], []

    print("■ 라벨별 상세 (Per Label)")
    if not labels:
        print("  (라벨 정보 없음)")
    for lab in labels:
        tp_l = per_label.get(("TP", lab), 0)
        fp_l = per_label.get(("FP", lab), 0)
        fn_l = per_label.get(("FN", lab), 0)
        p_l = safe_div(tp_l, tp_l + fp_l)
        r_l = safe_div(tp_l, tp_l + fn_l)
        f1_l = safe_div(2*p_l*r_l, p_l+r_l) if (p_l+r_l) else 0.0
        p_list.append(p_l); r_list.append(r_l)
        print(f"  - [{lab}] TP={tp_l}  FP={fp_l}  FN={fn_l}  |  정밀도={p_l:.4f} 재현율={r_l:.4f} F1={f1_l:.4f}")
    print()

    if labels:
        macro_p = sum(p_list)/len(labels)
        macro_r = sum(r_list)/len(labels)
        macro_f1 = safe_div(2*macro_p*macro_r, macro_p+macro_r) if (macro_p+macro_r) else 0.0
        print("■ 매크로 평균 (라벨별 단순 평균)")
        print(f"  - 정밀도(매크로) = {macro_p:.4f}")
        print(f"  - 재현율(매크로) = {macro_r:.4f}")
        print(f"  - F1(매크로)     = {macro_f1:.4f}")
        print()

def chunk_text(text: str, tokenizer, stride_chars: int = 50) -> List[Tuple[int, str]]:
    if not text:
        return [(0, "")]
    max_chars = 4096
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        chunk = text[i:end]
        chunks.append((i, chunk))
        if end == n:
            break
        i = end - min(stride_chars, end - i)
    return chunks

def infer_token_classification(prompts_path: str, model_id: str, out_path: str,
                               aggregation: str = "simple",
                               device: Optional[str] = None,
                               label_map: Optional[Dict[str,str]] = None,
                               batch_size: int = 8):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    print(f"[INFO] token-classification 모델 로딩: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    pipe_kwargs = {}
    if device:
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    else:
        try:
            import torch
            pipe_kwargs["device"] = 0 if torch.cuda.is_available() else -1
        except Exception:
            pipe_kwargs["device"] = -1

    nlp = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy=aggregation, **pipe_kwargs)

    prompts = load_jsonl(prompts_path)

    def map_label(lab: str) -> str:
        if label_map and lab in label_map:
            return label_map[lab]
        for pref in ("B-","I-","S-","E-","U-","L-"):
            if lab.startswith(pref):
                return lab[len(pref):]
        return lab

    with open(out_path, "w", encoding="utf-8") as w:
        for obj in prompts:
            sid = str(obj.get("id"))
            text = obj.get("text","")
            if not text:
                w.write(json.dumps({"id": sid, "entities": []}, ensure_ascii=False)+"\n")
                continue

            entities = []
            for base_off, chunk in chunk_text(text, tok):
                out = nlp(chunk, batch_size=batch_size)
                for ent in out:
                    b = int(ent.get("start", 0)) + base_off
                    e = int(ent.get("end", 0)) + base_off
                    lab = str(ent.get("entity_group") or ent.get("entity") or "")
                    lab = map_label(lab)
                    entities.append({"begin": b, "end": e, "label": lab})

            w.write(json.dumps({"id": sid, "entities": entities}, ensure_ascii=False)+"\n")

    print(f"[INFO] 예측 저장: {out_path}")

GEN_SYS_PROMPT = (
    "당신은 민감정보 탐지기입니다. 사용자가 제공하는 텍스트에서 민감정보 엔티티를 추출해 "
    '다음 JSON 형식으로만 출력하세요: {"entities":[{"begin":정수,"end":정수,"label":"라벨"} ...]}\n'
    "begin/end는 0-기반 문자 오프셋이며 end는 배타적입니다. 응답에는 JSON만 포함하세요."
)

def extract_first_json(text: str) -> Optional[dict]:
    m = __import__("re").search(r'\{.*\}', text, flags=__import__("re").S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def infer_generation(prompts_path: str, model_id: str, out_path: str,
                     device: Optional[str] = None, max_new_tokens: int = 256,
                     temperature: float = 0.2, top_p: float = 0.9):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"[INFO] generation 모델 로딩: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    if device == "cuda" or (device is None and torch.cuda.is_available()):
        model.to("cuda")
        dev = "cuda"
    else:
        dev = "cpu"

    prompts = load_jsonl(prompts_path)

    with open(out_path, "w", encoding="utf-8") as w:
        for obj in prompts:
            sid = str(obj.get("id"))
            text = obj.get("text","")
            if not text:
                w.write(json.dumps({"id": sid, "entities": []}, ensure_ascii=False)+"\n")
                continue

            user_prompt = f"텍스트:\n{text}\n\n위 텍스트에서 엔티티를 JSON으로만 출력하세요."
            full = GEN_SYS_PROMPT + "\n\n" + user_prompt

            inputs = tok(full, return_tensors="pt")
            inputs = {k: v.to(dev) for k,v in inputs.items()}

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tok.eos_token_id
                )
            out_text = tok.decode(out_ids[0], skip_special_tokens=True)

            parsed = extract_first_json(out_text) or {"entities":[]}
            entities = []
            for e in parsed.get("entities", []):
                try:
                    b = int(e["begin"]); e_ = int(e["end"]); lab = str(e["label"])
                    if 0 <= b < e_ <= len(text):
                        entities.append({"begin": b, "end": e_, "label": lab})
                except Exception:
                    continue

            w.write(json.dumps({"id": sid, "entities": entities}, ensure_ascii=False)+"\n")

    print(f"[INFO] 예측 저장: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["token","generation"], default="token", help="추론 방식 선택")
    ap.add_argument("--prompts", help="프롬프트 JSONL 경로(id+text). 추론 시 필요")
    ap.add_argument("--answers", required=True, help="정답지 JSONL 경로(id+entities)")
    ap.add_argument("--predictions", help="예측 JSONL 경로(있으면 평가만 수행)")
    ap.add_argument("--model", help="허깅페이스 식별자 또는 로컬 경로")
    ap.add_argument("--out", default="predictions.jsonl", help="추론 결과 저장 경로")
    ap.add_argument("--match", choices=["exact","overlap"], default="exact", help="스팬 매칭 방식")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 임계값( --match overlap )")
    ap.add_argument("--aggregation", default="simple", help="[token] aggregation_strategy")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], help="강제 디바이스 선택(미지정시 auto)")
    ap.add_argument("--label-map", help="[token] 라벨 매핑 JSON 경로(모델라벨->평가라벨)")
    ap.add_argument("--batch-size", type=int, default=8, help="[token] 추론 배치 크기]")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="[generation] 생성 길이")
    ap.add_argument("--temperature", type=float, default=0.2, help="[generation] 샘플링 온도")
    ap.add_argument("--top-p", type=float, default=0.9, help="[generation] top-p")
    args = ap.parse_args()

    if args.predictions:
        evaluate(args.answers, args.predictions, args.match, args.iou)
        return

    if not args.model or not args.prompts:
        raise SystemExit("모델 추론을 사용하려면 --model 과 --prompts 를 함께 지정하거나, --predictions 로 예측 파일을 주세요.")

    if args.task == "token":
        label_map = None
        if args.label_map:
            with open(args.label_map, "r", encoding="utf-8") as f:
                label_map = json.load(f)
        infer_token_classification(
            prompts_path=args.prompts,
            model_id=args.model,
            out_path=args.out,
            aggregation=args.aggregation,
            device=args.device,
            label_map=label_map,
            batch_size=args.batch_size
        )
    else:
        infer_generation(
            prompts_path=args.prompts,
            model_id=args.model,
            out_path=args.out,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

    evaluate(args.answers, args.out, args.match, args.iou)

if __name__ == "__main__":
    main()
