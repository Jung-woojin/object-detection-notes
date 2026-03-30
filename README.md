# 객체검출 모델 노트

개인과 팀에서 실험한 현대적 객체검출 모델 (CNN & Transformer 기반) 에 대한 노트입니다.

**CNN-based: YOLO 시리즈 분석**  
**Transformer-based: DETR 계열 분석**

---

## 📚 목차

### 1. YOLO 시리즈
- [YOLOv8 상세 분석](yolov8-detailed-analysis.md)
- YOLOv9, YOLO-World 개요
- [YOLO 시리즈 Loss 함수](loss/yolo-series-loss.md)
- [YOLO 시리즈 IoU variants](iou/yolo-iou.md)

### 2. DETR 계열
- [DETR, Deformable DETR, DINO, RT-DETR 분석](detr-family-analysis.md)
- [DETR 계열 Loss 함수](loss/detr-loss.md)

### 3. IoU 및 변형 variants
- [IoU 기본](iou/README.md)
  - GIoU, DIoU, CIoU, SIoU, EIoU, WIoU
  - YOLO 버전별 IoU 채택
  - DETR 계열 IoU 사용법
- [IOU 상세 설명](iou/DETAILED.md)

### 4. Loss 함수
- [Loss 함수 기본 개념](loss/README.md)
- [YOLO 시리즈 Loss](loss/yolo-series-loss.md)
- [DETR 계열 Loss](loss/detr-loss.md)
- [모든 객체탐지 모델 Loss](loss/ALL_MODELS.md)

### 5. 비교 및 실전
- 모델 성능 비교
- 실전 세팅 가이드
- 실험 노트

---

## 🎯 객체검출 개요

**객체검출 (Object Detection)** = 위치 + 분류

**주요 접근법:**
1. **Two-stage**: Region Proposals → Classification (Faster R-CNN, Cascade R-CNN)
2. **One-stage**: End-to-end Detection (YOLO, SSD, FCOS, RetinaNet)
3. **Transformer**: Query-based Detection (DETR, DINO, RT-DETR)

---

## 📊 YOLO vs DETR 비교

| 측면 | YOLO 시리즈 | DETR 계열 |
|--|--|--|
| 아키텍처 | One-stage | Transformer |
| 속도 | 빠름 (200-600 FPS) | 느림 (70-150 FPS) |
| 정확도 | 높음 | 매우 높음 |
| small objects | 좋음 (YOLOv9) | 매우 좋음 (RT-DETR) |
| 학습 시간 | 50-100 epochs | 10-100 epochs |
| inference | TensorRT 최적화 | ONNX, TensorRT |
| 사용도 | 실전 추천 | 연구/최적화 필요 |

---

## 🚀 시작 가이드

### YOLOv8 Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')

# Train
model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Predict
results = model.predict(source='image.jpg', conf=0.25)

# Export
model.export(format='onnx')
```

### DETR Training

```python
import torch
from models.detr import DETR
from models.matcher import HungarianMatcher

# Model setup
model = DETR(
    num_classes=80,
    num_queries=100
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4,
    weight_decay=1e-4
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, optimizer, data_loader)
    evaluate(model, val_loader)
```

---

## 📈 성능 벤치마크

### COCO Validation Set

| Model | mAP | Speed | Params | Type |
|--|--|----|----|----|----|
| **YOLOv8n** | 37.3 | 615 FPS | 3.2M | One-stage |
| **YOLOv8s** | 44.9 | 480 FPS | 11.1M | One-stage |
| **YOLOv8m** | 50.2 | 295 FPS | 25.9M | One-stage |
| **YOLOv8l** | 52.9 | 195 FPS | 43.7M | One-stage |
| **YOLOv8x** | 53.9 | 135 FPS | 68.2M | One-stage |
| | | | | |
| **RT-DETRv1** | 53.0 | 135 FPS | 30M | Transformer |
| **DINO** | 56.0 | 90 FPS | 35M | Transformer |
| **DETR** | 42.0 | 70 FPS | 42M | Transformer |
| **Cascade R-CNN** | 51.8 | 20 FPS | 60M | Two-stage |
| **Faster R-CNN** | 39.5 | 30 FPS | 42M | Two-stage |
| **SSD300** | 25.7 | 45 FPS | 15M | One-stage |
| **RetinaNet** | 39.4 | 40 FPS | 40M | One-stage |
| **FCOS** | 40.5 | 35 FPS | 25M | One-stage |
| **CenterNet** | 37.9 | 30 FPS | 10M | One-stage |

---

## 🎯 추천 사용법

### 실시간 인퓨런스
→ **YOLOv8n** or **YOLOv8s**  
→ TensorRT 최적화

### 최고 정확도
→ **YOLOv9** or **DINO**  
→ Multi-scale inference

### 연구 목적
→ **DETR 계열**  
→ Transformer 아키텍처 이해

### Open-vocabulary
→ **Grounding DINO** or **YOLO-World**  
→ Text prompts 기반 detection

---

## 🔬 핵심 개념

### IoU Variants

객체검출의 핵심 지표인 IoU 와 그 변형들:

- **IoU (Intersection over Union)**: 기본 지표
- **GIoU (Generalized IoU)**: CI-2019, 박스 겹치지 않을 때도 gradient 제공
- **DIoU (Distance IoU)**: CVPR-2020, center distance 고려
- **CIoU (Complete IoU)**: YOLOv4~, aspect ratio 포함
- **SIoU (Scylla IoU)**: YOLOv7, angle + distance + shape cost
- **EIoU (Efficient IoU)**: aspect ratio 분리, 더 빠른 수렴

[자세한 IoU 설명](iou/README.md)

### Loss Functions

각 모델별 Loss 함수와 전략:

**YOLO 시리즈**:
- **YOLOv3**: MSE + BCE
- **YOLOv4**: GIoU + Focal Loss
- **YOLOv5-v8**: CIoU + BCE/VFL
- **YOLOv7**: SIoU (Scylla IoU)
- **YOLOv9**: PGI, Reusable Labels

**DETR 계열**:
- **DETR**: L1 + GIoU + Focal
- **Deformable DETR**: Faster convergence (12 epochs)
- **DINO**: GIoU + IoU combined
- **RT-DETR**: DIoU for real-time

[자세한 Loss 함수 설명](loss/README.md)

### Anchor vs Anchor-free

**Anchor-based** (SSD, RetinaNet, Faster R-CNN):
- Default boxes 사용
- Multiple scales와 aspects 학습
- Post-processing (NMS) 필요

**Anchor-free** (FCOS, CenterNet, YOLOv8):
- Point-based detection
- 구현 단순화
- No NMS 가능

[Anchor 비교 및 설명](anchor-free-vs-anchor.md)

---

## 📚 문서 링크

### YOLO 시리즈
- [YOLOv8 상세 분석](yolov8-detailed-analysis.md)
- [YOLOv8 IoU variants](iou/yolo-iou.md)
- [YOLOv8 Loss 함수](loss/yolo-series-loss.md)

### DETR 계열
- [DETR, Deformable DETR, DINO, RT-DETR 분석](detr-family-analysis.md)
- [DETR Loss 함수](loss/detr-loss.md)

### IoU
- [IoU 기본 및 변형](iou/README.md)
- [IoU 상세 설명](iou/DETAILED.md)

### Loss 함수
- [Loss 함수 기본 개념](loss/README.md)
- [YOLO 시리즈 Loss](loss/yolo-series-loss.md)
- [DETR 계열 Loss](loss/detr-loss.md)
- [모든 객체탐지 모델 Loss](loss/ALL_MODELS.md)

### 기타
- [Anchor vs Anchor-free](anchor-free-vs-anchor.md)
- [실전 세팅 가이드](real-world-settings.md)
- [실험 노트](experiments.md)

---

## 🚀 최신 트렌드 (2024-2025)

### 주요 발전 방향
1. **Open-vocabulary Detection**: Grounding DINO, YOLO-World
2. **Real-time Performance**: YOLOv10, RT-DETR
3. **NMS-free Inference**: YOLOv10, YOLO-World
4. **Zero-shot Learning**: Pre-trained models 활용
5. **Efficient Architectures**: Mobile-friendly, edge devices

### Recommended Reading
- [모든 객체탐지 모델 Loss 함수 비교](loss/ALL_MODELS.md)
- [IoU variants 상세 설명](iou/README.md)
- [DETR 계열 완전 분석](detr-family-analysis.md)

---

## 📝 결론

객체검출는 **One-stage** (YOLO), **Two-stage** (R-CNN), **Transformer** (DETR) 의 세 가지 주요 접근법이 있습니다. 각 모델은 고유한 **IoU variant**와 **Loss function**을 사용하며, 응용에 따라 선택해야 합니다.

- **실용적 적용**: YOLOv8/v9 (빠르고 정확함)
- **고성능 연구**: DINO (변환 기반, 최고 정확도)
- **오픈셋**: Grounding DINO (텍스트 기반)

더 자세한 내용은 관련 문서를 참조하세요!

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, 관련 논문, CVPR/ICCV/ECCV proceedings*