
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 민감정보 탐지: 추론(+예측 생성) + 평가 통합 스크립트 (한국어 출력)

두 가지 방식 지원:
1) 이미 만든 예측 JSONL을 평가만 하기
   - --predictions 를 제공하면 모델 로딩 없이 곧바로 평가

2) 직접/허깅페이스 경로의 NER 모델로 추론하여 예측 생성 후 평가
   - --model 에 로컬 경로 또는 hub 식별자 입력(예: "yourname/ner-korean-pii")
   - 토큰 분류(ner) 계열 모델을 가정: transformers pipeline("token-classification")
   - 결과를 JSONL로 저장(--out)하고, 즉시 정답과 비교하여 성능 출력

입력 파일
- prompts(JSONL):   {"id": "...", "text": "원문"} …
- answers(JSONL):   {"id": "...", "entities":[{"begin":int,"end":int,"label":"..."}]}
- predictions(JSONL): 위 answers와 동일 구조(id+entities) — (직접 평가만 할 때 사용)

사용 예
  # A) 허깅페이스 모델로 추론+평가 (IoU 겹침 허용)
  python3 ai_perf_infer_eval_ko.py \
    --prompts "/mnt/data/AI performance test id_prompt.jsonl" \
    --answers "/mnt/data/AI performance test id_answer.jsonl" \
    --model "yourname/ner-korean-pii" \
    --out "/tmp/preds.jsonl" \
    --match overlap --iou 0.5

  # B) 이미 생성된 예측 파일만 평가
  python3 ai_perf_infer_eval_ko.py \
    --answers "/mnt/data/AI performance test id_answer.jsonl" \
    --predictions "/tmp/preds.jsonl" \
    --match exact

필요 패키지
  pip install transformers==4.* torch>=2.*
  (GPU 권장: CUDA 환경에서 자동 사용)
"""
import argparse
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

# ------------------------- 공통: 로드/매칭/지표 -------------------------

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
    print(" AI 중요정보 탐지 성능 결과 (한국어 출력)")
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

# ------------------------- 추론(토큰분류) -------------------------

def infer_with_token_classification(prompts_path: str, model_id: str, out_path: str,
                                    aggregation: str = "simple",
                                    device: Optional[str] = None,
                                    label_map: Optional[Dict[str,str]] = None,
                                    batch_size: int = 8):
    """
    Token-classification pipeline으로 엔티티 예측을 생성해 JSONL로 저장.
    - model_id: 로컬 경로나 hub 식별자
    - aggregation: "simple"|"first"|"max"|None (transformers aggregation_strategy)
    - label_map: 모델 라벨 -> 평가 라벨 매핑(예: {"B-SSN":"SSN","I-SSN":"SSN"})
    - device: "cpu"|"cuda"|"mps"|None(자동)
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    print(f"[INFO] 모델 로딩: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    # device 설정
    pipe_kwargs = {}
    if device:
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    else:
        # auto
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
        # BIO 라벨이면 접두어 제거(B-, I-, S-, E- 등)
        for pref in ("B-","I-","S-","E-","U-","L-"):
            if lab.startswith(pref):
                return lab[len(pref):]
        return lab

    # 추론 및 저장
    with open(out_path, "w", encoding="utf-8") as w:
        for obj in prompts:
            sid = str(obj.get("id"))
            text = obj.get("text","")
            if not text:
                w.write(json.dumps({"id": sid, "entities": []}, ensure_ascii=False)+"\n")
                continue
            out = nlp(text, batch_size=batch_size, truncation=False)
            ents = []
            for ent in out:
                # transformers 토큰분류 파이프라인은 start/end(문자 인덱스) 제공
                b = int(ent.get("start", 0))
                e = int(ent.get("end", 0))
                lab = str(ent.get("entity_group") or ent.get("entity") or "")
                lab = map_label(lab)
                ents.append({"begin": b, "end": e, "label": lab})
            w.write(json.dumps({"id": sid, "entities": ents}, ensure_ascii=False)+"\n")

    print(f"[INFO] 예측 저장: {out_path}")

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="프롬프트 JSONL 경로(id+text). 추론 시 필요")
    ap.add_argument("--answers", required=True, help="정답지 JSONL 경로(id+entities)")
    ap.add_argument("--predictions", help="예측 JSONL 경로(있으면 평가만 수행)")
    ap.add_argument("--model", help="허깅페이스 식별자 또는 로컬 경로(토큰분류 모델)")
    ap.add_argument("--out", default="predictions.jsonl", help="추론 결과 저장 경로")
    ap.add_argument("--match", choices=["exact","overlap"], default="exact", help="스팬 매칭 방식")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 임계값( --match overlap )")
    ap.add_argument("--aggregation", default="simple", help="token-classification aggregation_strategy")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], help="강제 디바이스 선택(미지정시 auto)")
    ap.add_argument("--label-map", help="라벨 매핑 JSON 경로(모델라벨->평가라벨)")
    ap.add_argument("--batch-size", type=int, default=8, help="추론 배치 크기")
    args = ap.parse_args()

    # 1) 예측 파일이 이미 있으면 바로 평가
    if args.predictions:
        evaluate(args.answers, args.predictions, args.match, args.iou)
        return

    # 2) 모델로 추론 → 예측 생성 → 평가
    if not args.model or not args.prompts:
        raise SystemExit("모델 추론을 사용하려면 --model 과 --prompts 를 함께 지정하거나, --predictions 로 예측 파일을 주세요.")

    label_map = None
    if args.label_map:
        with open(args.label_map, "r", encoding="utf-8") as f:
            label_map = json.load(f)  # {"B-FOO":"FOO","I-FOO":"FOO", ...}

    infer_with_token_classification(
        prompts_path=args.prompts,
        model_id=args.model,
        out_path=args.out,
        aggregation=args.aggregation,
        device=args.device,
        label_map=label_map,
        batch_size=args.batch_size
    )

    # 생성한 예측을 즉시 평가
    evaluate(args.answers, args.out, args.match, args.iou)

if __name__ == "__main__":
    main()
