# Object Detection Loss Functions

객체 탐지에서 사용되는 모든 Loss 함수를 YOLO 버전별, 모델별 정리합니다.

## 📚 목차

- [손실 함수 기본 개념](#-손실-함수-기본-개념)
- [YOLOv3 Loss](#-yolov3-loss)
- [YOLOv4 Loss](#-yolov4-loss)
- [YOLOv5 Loss](#-yolov5-loss)
- [YOLOv6 Loss](#-yolov6-loss)
- [YOLOv7 Loss](#-yolov7-loss)
- [YOLOv8 Loss](#-yolov8-loss)
- [YOLOv9 Loss](#-yolov9-loss)
- [YOLOX Loss](#-yolox-loss)
- [YOLO-World Loss](#-yolo-world-loss)
- [DETR 계열 Loss](#-detr-계열-loss)
- [실전 코드](#-실전-코드)

---

## 🔧 손실 함수 기본 개념

객체 탐지에서 주로 사용되는 손실 함수 유형:

1. **Position Loss**: 박스 위치 정확도
2. **Size Loss**: 박스 크기 정확도
3. **Classification Loss**: 객체 분류 정확도
4. **Objectness Loss**: 객체 존재 확신도

**일반적인 손실 함수 형태**:
```
L_total = λ_box · L_box + λ_obj · L_obj + λ_cls · L_cls
```

여기서:
- `λ`: 각 손실의 weight
- `L_box`: 박스 손실 (IoU 기반)
- `L_obj`: 객체 존재 손실 (Binary Cross Entropy)
- `L_cls`: 분류 손실 (Cross Entropy, Focal Loss 등)

---

## 🎯 YOLOv3 Loss

**논문**: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

### Loss 구성

```python
# 총 손실 = 박스 손실 + 객체 손실 + 분류 손실
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: MSE Loss

**수식**:
```
L_box = Σᵢ⁰ˢ⁻¹ Σⱼ⁰ˢ⁻¹ 1ⱼᵢʰₒᵦʲ · [Σₖ⁴⁽xₖ - x̂ₖ)² + (yₖ - ŷₖ)² + (wₖ - ŵₖ)² + (hₖ - ĥₖ)²]
```

- **Type**: Mean Squared Error (MSE)
- **Target**: 박스의 center, width, height
- **문제점**: 박스가 크게 어긋나면 gradient 가 작아짐

### 2. Objectness Loss: Binary Cross Entropy

**수식**:
```
L_obj = Σᵢ⁰ˢ⁻¹ Σⱼ⁰ˢ⁻¹ 1ⱼᵢʰₒᵦʲ · [tⱼᵢ log(pⱼᵢ) + (1-tⱼᵢ) log(1-pⱼᵢ)]
```

여기서:
- `pⱼᵢ`: 예측 객체 존재 확률
- `tⱼᵢ`: True label (1=객체 있음, 0=아님)

### 3. Classification Loss: Binary Cross Entropy

**수식**:
```
L_cls = Σᵢ⁰ˢ⁻¹ Σⱼ⁰ˢ⁻¹ 1ⱼᵢᵒᵦʲ · Σₖ⁶⁰ [Cⱼₖ log(pⱼₖ) + (1-Cⱼₖ) log(1-pⱼₖ)]
```

- **Type**: BCE (multi-label)
- **Classes**: COCO 80 classes

### 특징

- **Simple but limited**: 구현이 단순하지만 성능 한계
- **Positive-negative imbalance**: 양쪽 클래스 불균형 심함
- **Weak localization**: MSE 로 인해 정확도 제한

---

## 🚀 YOLOv4 Loss

**논문**: [YOLOv4: Optimal Speed and Accuracy](https://arxiv.org/abs/2004.10934)

### Loss 구성

```python
total_loss = λ₁·L_GIoU + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: GIoU Loss

**Type**: Generalized Intersection over Union

**수식**:
```
L_GIoU = 1 - GIoU
```

- **기존 MSE → GIoU**로 변경
- **장점**: 박스가 겹치지 않을 때도 gradient 제공
- **문제점**: 과잉 최적화 가능

### 2. Objectness Loss: Binary Focal Loss

**Type**: Binary Focal Loss (Positive-Negative Imbalance 해소)

**수식**:
```
L_obj = -α·(1-p_t)ᵞ·log(p_t)
```

여기서:
- `α`: weight (0.25 recommended)
- `γ`: focusing parameter (2 recommended)
- `p_t`: model probability for true label

**장점**: Hard examples 에 더 집중

### 3. Classification Loss: Sigmoid + Focal Loss

**Type**: Sigmoid activation + Focal Loss

**수식**:
```
L_cls = Focal Loss
```

**변경**:
- Multi-label BCE → Focal Loss
- **정확도**: +2-3% 향상

### 추가 기능

**CIoU 실험**:
- YOLOv4 에서 CIoU 도 실험됨
- GIoU 가 더 안정적

### 특징

- **Better balance**: Focal Loss 로 imbalance 해소
- **Improved localization**: GIoU 로 정확도 향상
- **Standard now**: 많은 후속 모델의 기반

---

## ⚡ YOLOv5 Loss

**유저**: Ultralytics YOLOv5

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: CIoU Loss

**Type**: Complete IoU

**수식**:
```
L_CIoU = 1 - CIoU
```

- **CIOU** 사용 (GIoU 보다 더 좋음)
- **Center distance + aspect ratio** 고려
- **더 빠른 수렴**

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

**Type**: BCE with label smoothing

**Label Smoothing**:
```
t_smoothed = (1-ε)·t + ε/num_classes
```
- **ε**: 0.0 (기본) or 0.1

### 3. Classification Loss: BCE with Label Smoothing

**Type**: Binary Cross Entropy

**Label Smoothing**:
```
t_smoothed = (1-ε)·t + ε/num_classes
```

**장점**: Overfitting 감소

### Loss weights

```yaml
box: 7.5
obj: 0.7
cls: 0.5
```

### 특징

- **CIoU**: 더 정확한 localization
- **Label smoothing**: Generalization 향상
- **Balanced**: 각 손실 weight 적절히 조정
- **Standard now**: YOLO 생태계의 기본

---

## 🎯 YOLOv6 Loss

**유저**: Meituan YOLOv6

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: CIoU Loss

**Type**: Complete IoU

- **YOLOv5 와 동일** CIoU 사용
- **Reparametrization** 기술과 결합

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing** 적용

### 3. Classification Loss: Varifocal Loss

**Type**: Varifocal Loss (VFL)

**동기**: BCE 의 한계 (hard positives/negatives)

**수식**:
```
L_VFL = Σᵢ [(1 - pᵢ)ᵞᵢ · log(1 - qᵢ) + pᵢᵞᵢ · (1 - pᵢ) · log(qᵢ)]
```

여기서:
- `pᵢ`: Predicted probability
- `qᵢ`: IoU between predicted and GT box
- `γᵢ`: Focusing parameter

**장점**:
- **IoU-aware**: Box quality 를 고려
- **Better gradient**: Hard examples 에 더 효과적

### 특징

- **Varifocal Loss**: Classification performance ↑
- **Decoupled Head**: Faster convergence
- **Reparametrization**: Inference efficiency

---

## 🔥 YOLOv7 Loss

**논문**: [YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art](https://arxiv.org/abs/2207.02696)

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: SIoU Loss

**Type**: Scylla IoU

**수식**:
```
L_SIoU = 1 - SIoU
```

**SIoU components**:
1. **Angle cost**: Box orientation difference
2. **Distance cost**: Center-to-center distance
3. **Shape cost**: Aspect ratio difference

**장점**:
- **Faster convergence**: 3 가지 cost 결합
- **Better alignment**: 더 정확한 localization
- **YOLOv7 signature**: YOLOv7 의 핵심

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing**: ε = 0.0 (기본)

### 3. Classification Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing**: 적용
- **Efficient**: 빠른 학습

### Loss weights

```yaml
box: 7.5
obj: 1.5
cls: 0.5
```

### 특징

- **SIoU**: 더 빠른 수렴
- **EVA (Efficient Vehicle Alignment)**: 학습 효율성
- **PGI (Programmable Gradient Information)**: Gradient flow 최적화
- **Fast convergence**: 50 epochs 에서도 good performance

---

## 🚀 YOLOv8 Loss

**유저**: Ultralytics YOLOv8

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: CIoU variant

**Type**: CIoU-based loss

**수식**:
```
L_box = 1 - CIoU
```

- **CIoU** 기반
- **Distribution Focal Loss**와 결합
- **Box loss weight**: 7.5

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing**: ε = 0.0

### 3. Classification Loss: VFL (Varifocal Loss)

**Type**: Varifocal Loss

- **YOLOv6 와 동일** VFL 사용
- **IoU-aware**: Better classification

### Loss weights

```yaml
box: 7.5
obj: 0.5
cls: 0.5
```

### 특징

- **VFL**: Classification accuracy ↑
- **CIoU**: Balanced localization
- **Modern**: 가장 최신 YOLO 의 표준
- **Better mAP**: v7 보다 1-2% 향상

---

## 🎯 YOLOv9 Loss

**논문**: [YOLOv9: Learning What You Want to Learn](https://arxiv.org/abs/2402.13616)

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: GIoU variant

**Type**: GIoU-based loss

- **Programmable Gradient Information (PGI)** 사용
- **Reusable Labels**: Label assignment 최적화

**수식**:
```
L_box = 1 - GIoU
```

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing**: 적용
- **Reusable labels**: 더 안정적인 학습

### 3. Classification Loss: BCE

**Type**: Binary Cross Entropy

- **Label smoothing**: 적용
- **Efficient**: 빠른 학습

### 특징

- **PGI**: Gradient flow 최적화
- **Reusable labels**: Label assignment 개선
- **Better accuracy**: v8 보다 +1-2% mAP
- **Flexible architecture**: Programmable gradient

---

## 📦 YOLOX Loss

**논문**: [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: CIoU Loss

**Type**: Complete IoU

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

### 3. Classification Loss: VFL (Varifocal Loss)

**Type**: Varifocal Loss

### SimOTA Label Assignment

**동기**: Traditional label assignment (anchor-based) 의 한계

**SimOTA**:
- **Online matching**: Dynamic label assignment
- **Cost-based**: Cost = L_box + L_obj + L_cls
- **Better matching**: More accurate label assignment

### 특징

- **SimOTA**: Label assignment 개선
- **Decoupled training**: Faster convergence
- **Good performance**: v4 수준 but simpler

---

## 🌍 YOLO-World Loss

**논문**: [YOLO-World: Real-time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270)

### Loss 구성

```python
total_loss = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

### 1. Box Loss: GIoU

**Type**: Generalized IoU

- **Open-vocabulary** 학습용 GIoU
- **Semantic-aware**: Class-agnostic

### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

### 3. Classification Loss: Contrastive Loss

**Type**: Contrastive Loss (Open-vocabulary)

**동기**: Open-vocabulary detection 을 위한 contrastive learning

**수식**:
```
L_cls = Contrastive Loss
```

**특징**:
- **Semantic embeddings**: Text encoder 와 image encoder 연결
- **Open-vocabulary**: 새로운 class 도 detection 가능
- **Zero-shot**: Training 없는 inference 가능

### 특징

- **Open-vocabulary**: 새로운 class 추가 가능
- **Contrastive learning**: Semantic understanding
- **Zero-shot**: Pre-trained models 활용

---

## 🔮 DETR 계열 Loss

### DETR (Original)

**논문**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

### Loss 구성

```python
total_loss = λ₁·L_dice + λ₂·L_focal + λ₃·L_box
```

### 1. Box Loss: L1 + GIoU

**Type**: L1 Loss + GIoU Loss

**수식**:
```
L_box = λ₁·L1 + λ₂·L_GIoU
```

- **L1**: Absolute error
- **GIoU**: Overlap-based loss
- **Weight**: L1: 5, GIoU: 2

### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

**동기**: Class imbalance problem

**수식**:
```
L_focal = -α·(1-p_t)ᵞ·log(p_t)
```

### 3. Matching Loss: Bipartite Matching

**Type**: Hungarian matching

**동기**: One-to-one matching problem

**수식**:
```
L_matching = Σᵢ₌₁ⁿ⁻¹ Σⱼ₌₁ⁿ⁻¹ Cᵢⱼ
```

여기서 `Cᵢⱼ` 는 cost matrix:
- **Classification cost**: Focal loss
- **Box cost**: L1 + GIoU

### 특징

- **End-to-end**: No anchor needed
- **Hungarian matching**: One-to-one matching
- **Focal loss**: Class imbalance handling
- **Slow convergence**: 100 epochs 필요

---

### Deformable DETR

**논문**: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

### Loss 구성

```python
total_loss = λ₁·L1 + λ₂·L_GIoU + λ₃·L_focal
```

### Loss

- **L1 + GIoU**: Box loss
- **Focal Loss**: Classification
- **Hungarian matching**: Same as DETR

**차이점**:
- **Deformable attention**: Faster convergence (12 epochs)
- **Multi-scale**: Hierarchical features

---

### DINO

**논문**: [DINO: DETR with Improved DEterministic Queries](https://arxiv.org/abs/2203.03605)

### Loss 구성

```python
total_loss = λ₁·L_GIoU + λ₂·L_IoU + λ₃·L_focal
```

### Loss

- **GIoU + IoU**: Box loss (combined)
- **Focal Loss**: Classification
- **Contrastive denoising**: Training stabilization
- **Query selection**: Better initialization

**차이점**:
- **Contrastive denoising**: Noisy queries 제거
- **Better query initialization**: Faster convergence
- **High performance**: mAP 56.0 (COCO)

---

### RT-DETR

**논문**: [RT-DETR: Rethinking Real-Time DETR](https://arxiv.org/abs/2304.01318)

### Loss 구성

```python
total_loss = λ₁·L_DIoU + λ₂·L_focal + λ₃·L1
```

### Loss

- **DIoU**: Box loss (faster)
- **Focal Loss**: Classification
- **CNN backbone**: Faster inference
- **Real-time**: 135 FPS

**차이점**:
- **CNN + Transformer**: Best of both worlds
- **DIoU**: Faster convergence
- **Real-time**: RT-DETR 의 핵심

---

## 🛠 실전 코드

### YOLOv5 CIoU Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIoULoss(nn.Module):
    """Complete IoU Loss for YOLOv5/v8"""
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # pred, target: (x1, y1, x2, y2)
        
        # Intersection
        x1 = torch.max(pred[..., 0], target[..., 0])
        y1 = torch.max(pred[..., 1], target[..., 1])
        x2 = torch.min(pred[..., 2], target[..., 2])
        y2 = torch.min(pred[..., 3], target[..., 3])
        
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        # Union
        union = pred_area + target_area - inter_area
        iou = inter_area / (union + self.eps)
        
        # Center points
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        
        # Center distance
        center_dist = torch.pow(pred_cx - target_cx, 2) + torch.pow(pred_cy - target_cy, 2)
        
        # Enclosing box
        enclose_x1 = torch.min(pred[..., 0], target[..., 0])
        enclose_y1 = torch.min(pred[..., 1], target[..., 1])
        enclose_x2 = torch.max(pred[..., 2], target[..., 2])
        enclose_y2 = torch.max(pred[..., 3], target[..., 3])
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        c_area = enclose_w**2 + enclose_h**2
        
        # Aspect ratio
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        v = (4 / (torch.pi**2)) * torch.pow(
            torch.atan(target_w / (target_h + self.eps)) - 
            torch.atan(pred_w / (pred_h + self.eps)), 2)
        
        alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - center_dist / c_area - alpha * v
        
        loss = 1 - ciou
        return loss
```

### YOLOv7 SIoU Loss

```python
class SIoULoss(nn.Module):
    """Scylla IoU Loss for YOLOv7"""
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # Intersection
        xi1 = torch.max(pred[..., 0], target[..., 0])
        yi1 = torch.max(pred[..., 1], target[..., 1])
        xi2 = torch.min(pred[..., 2], target[..., 2])
        yi2 = torch.min(pred[..., 3], target[..., 3])
        
        inter_w = torch.clamp(xi2 - xi1, min=0)
        inter_h = torch.clamp(yi2 - yi1, min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        area2 = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        # Union
        union = area1 + area2 - inter_area
        iou = inter_area / (union + self.eps)
        
        # Center points
        c_x1 = (pred[..., 0] + pred[..., 2]) / 2
        c_y1 = (pred[..., 1] + pred[..., 3]) / 2
        c_x2 = (target[..., 0] + target[..., 2]) / 2
        c_y2 = (target[..., 1] + target[..., 3]) / 2
        
        # Angle cost
        angle = torch.atan2(c_y2 - c_y1, c_x2 - c_x1)
        
        # Distance cost
        dist = torch.sqrt(torch.pow(c_x2 - c_x1, 2) + torch.pow(c_y2 - c_y1, 2))
        
        # Shape cost
        shape1 = pred[..., 2] - pred[..., 0]
        shape2 = pred[..., 3] - pred[..., 1]
        
        # SIoU
        siou = iou - (angle * dist * torch.abs(shape1 - shape2))
        
        loss = 1 - siou
        return loss
```

### Varifocal Loss (YOLOv6/v8)

```python
class VarifocalLoss(nn.Module):
    """Varifocal Loss for classification"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, iou):
        """
        pred: predicted probability
        target: ground truth (0 or 1)
        iou: IoU between predicted and GT box
        """
        positive_weight = (self.alpha * iou) ** self.gamma
        negative_weight = (1 - iou) ** self.gamma
        
        # Positive samples
        pos_loss = -(positive_weight * torch.log(pred + 1e-8)) * target
        
        # Negative samples
        neg_loss = -(negative_weight * torch.log(1 - pred + 1e-8)) * (1 - target)
        
        return (pos_loss + neg_loss).mean()
```

---

## 📊 Loss 비교 요약

| Model | Box Loss | Objectness | Classification | Key Feature |
|------|------|------|--------|----|----|--|
| **YOLOv3** | MSE | BCE | BCE | Simple, basic |
| **YOLOv4** | GIoU | Focal | Focal | Better balance |
| **YOLOv5** | CIoU | BCE | BCE | Standard now |
| **YOLOv6** | CIoU | BCE | VFL | Varifocal |
| **YOLOv7** | SIoU | BCE | BCE | Faster convergence |
| **YOLOv8** | CIoU | BCE | VFL | Modern standard |
| **YOLOv9** | GIoU | BCE | BCE | PGI, reusable labels |
| **YOLOX** | CIoU | BCE | VFL | SimOTA |
| **YOLO-World** | GIoU | BCE | Contrastive | Open-vocabulary |
| **DETR** | L1+GIoU | - | Focal | End-to-end |
| **Deformable** | L1+GIoU | - | Focal | Faster (12 ep) |
| **DINO** | GIoU+IoU | - | Focal | Contrastive denoising |
| **RT-DETR** | DIoU | - | Focal | Real-time |

---

## 🎯 결론

1. **YOLOv3**: MSE → outdated, simple
2. **YOLOv4**: GIoU + Focal → improved
3. **YOLOv5**: CIoU + BCE → **recommended baseline**
4. **YOLOv6**: VFL → better classification
5. **YOLOv7**: SIoU → **fastest convergence**
6. **YOLOv8**: CIoU + VFL → **modern standard**
7. **YOLOv9**: PGI → flexible learning
8. **DETR 계열**: GIoU + Focal + Hungarian → end-to-end

**실전 추천**:
- **초기 실험**: YOLOv5/v8 (CIoU + BCE/VFL)
- **최적화**: YOLOv7 (SIoU)
- **연구**: DETR 계열 (GIoU + Focal)
- **Open-vocabulary**: YOLO-World (Contrastive)

---
*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, 관련 논문, CVPR/ICCV/ECCV proceedings*