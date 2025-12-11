# object-detection-notes
Personal notes and experiments on modern object detection models (CNN &amp; Transformer based)

## YOLO
- Classic: YOLOv3, YOLOv4, YOLOv5 (anchor-based)
- Modern: YOLOX, YOLOv7, YOLOv8, YOLOv9, RT-DETR, YOLO-World 등
- 목표:
  - 각 구조별 핵심 아이디어/아키텍처 도식화
  - 간단한 재현 실험 & mAP 비교
  - 실전 세팅(TTA, NMS, confidence threshold 등) 메모

## ViT / DETR 계열
- DETR, Deformable DETR, DINO, DN-DETR 등 transformer-based detectors
- 목표:
  - “set prediction”, Hungarian matching, query 개념 정리
  - CNN 백본 vs ViT 백본 성능/속도 비교
  - 작은 데이터셋에서 학습 난이도/수렴 이슈 기록

## Adapter & Alignment (멀티모달 / LVLM 관련)
- Vision encoder + LLM 결합 실험 노트
  - Projector (Linear / MLP)
  - Q-Former / cross-attention 기반 alignment
- 목표:
  - CLIP/ViT + LLM을 이용한 간단 멀티모달 실험 (예: 이미지 → 캡션/QA)
  - Stage 1(캡션 정렬) / Stage 2(VQA·인스트럭션 튜닝) 개념 정리
  - Object hallucination, POPE 같은 평가 기법 메모
