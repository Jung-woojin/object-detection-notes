# 객체검출 모델 노트

개인과 팀에서 실험한 현대적 객체검출 모델 (CNN & Transformer 기반) 에 대한 노트입니다.

**CNN-based: YOLO 시리즈 분석**  
**Transformer-based: DETR 계열 분석**

---

## 📚 목차

### 1. YOLO 시리즈
- [YOLOv8 상세 분석](yolov8-detailed-analysis.md)
- YOLOv9, YOLO-World 개요

### 2. DETR 계열
- [DETR, Deformable DETR, DINO, RT-DETR 분석](detr-family-analysis.md)

### 3. 비교 및 실전
- 모델 성능 비교
- 실전 세팅 가이드
- 실험 노트

---

## 🎯 객체검출 개요

**객체검출 (Object Detection)** = 위치 + 분류

**주요 접근법:**
1. **Two-stage**: Region Proposals → Classification (Faster R-CNN)
2. **One-stage**: End-to-end Detection (YOLO, SSD)
3. **Transformer**: Query-based Detection (DETR)

---

## 📊 YOLO vs DETR 비교

| 측면 | YOLO 시리즈 | DETR 계열 |
|------|-------|--------|------|
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

| Model | mAP | Speed | Params |
|-------|-----|-- -----|--------|
| **YOLOv8n** | 37.3 | 615 FPS | 3.2M |
| **YOLOv8s** | 44.9 | 480 FPS | 11.1M |
| **YOLOv8m** | 50.2 | 295 FPS | 25.9M |
| **YOLOv8l** | 52.9 | 195 FPS | 43.7M |
| **YOLOv8x** | 53.9 | 135 FPS | 68.2M |
| | | | |
| **RT-DETRv1** | 53.0 | 135 FPS | 30M |
| **DINO** | 56.0 | 90 FPS | 35M |
| **DETR** | 42.0 | 70 FPS | 42M |

---

## 🎯 추천 사용법

### 실시간 인퓨런스
→ **YOLOv8n** or **YOLOv8s**  
→ TensorRT 최적화

### 최고 정확도
→ **YOLOv9** or **DINO**  
→ High-end GPU

### Small objects
→ **YOLOv8l** or **RT-DETR**  
→ High input resolution

### Limited compute
→ **YOLOv8n** or **YOLOv5n**  
→ CPU inference 가능

---

## 📝 추가 학습 자료

- **YOLO 공식**: https://docs.ultralytics.com/
- **DETR 논문**: "End-to-End Object Detection with Transformers"
- **RT-DETR 논문**: Real-Time DETR

---

## 🎓 학습 경로

1. **기본**: YOLOv8 상세 분석 읽기
2. **심화**: DETR 계열 분석 읽기
3. **비교**: 성능 벤치마크 비교
4. **실전**: 가이드 적용 및 실험

---

*최종 수정일: 2026 년 3 월*
*Created with 💜 for object detection enthusiasts*
