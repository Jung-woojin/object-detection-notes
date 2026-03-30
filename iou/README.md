# IoU (Intersection over Union) 및 변형

객체검출에서 가장 핵심적인 평가 지표이자 Loss function 으로 사용되는 IoU 와 그 변형들을 정리합니다.

## 📚 목차

- [IoU 기본](#-iou-기본)
- [IoU 변형들](#-iou-변형들)
- [YOLO 버전별 IoU 채택](#-yolo-버전별-iou-채택)
- [DETR 계열 IoU](#-detr-계열-iou)
- [실전 코드](#-실전-코드)

---

## 🔢 IoU 기본

### 정의

**IoU (Intersection over Union)** = 예측 박스 와 정답 박스의 겹치는 비율

```
IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)
```

여기서:
- `B_pred`: 예측 bounding box
- `B_gt`: Ground truth bounding box
- `∩`: 교집합 (Intersection)
- `∪`: 합집합 (Union)

### 수학적 표현

두 박스를 (x1, y1, x2, y2) 로 표현하면:

```python
def iou(box1, box2):
    """
    box1, box2: (x1, y1, x2, y2) format
    """
    # Intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union area
    union = area1 + area2 - intersection
    
    # IoU
    return intersection / union
```

### 특징

- **값 범위**: [0, 1] (100% 일치할 때 1)
- **Differentiable 아님**: Area 가 0 이면 gradient 가 0
- **Position insensitive**: 박스가 완전히 겹치지 않으면 gradient 가 작음

---

## 🔄 IoU 변형들

### 1. GIoU (Generalized IoU)

**논문**: [CI-2019 "Generalized Intersection over Union: A Metric and Its Loss for Bounding Box Regression"]

**동기**: IoU 가 박스가 완전히 겹치지 않을 때 gradient 가 0 이 되는 문제

**정의**:
```
GIoU = IoU - (Area(C) - Area(A ∪ B)) / Area(C)
```

여기서 `C` 는 두 박스를 포함하는 가장 작은 축정렬 박스 (minimum enclosing box)

```python
def giou(box1, box2):
    """
    Generalized IoU
    box1, box2: (x1, y1, x2, y2)
    """
    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / union if union > 0 else 0
    
    # Enclosing box
    xc1 = min(box1[0], box2[0])
    yc1 = min(box1[1], box2[1])
    xc2 = max(box1[2], box2[2])
    yc2 = max(box1[3], box2[3])
    
    c_w = xc2 - xc1
    c_h = yc2 - yc1
    c_area = c_w * c_h
    
    # GIoU
    gciou = iou - (c_area - union) / c_area
    
    return gciou
```

**Loss**: `L_GIoU = 1 - GIoU`

**장점**:
- 박스가 겹치지 않을 때도 gradient 제공
- 박스들을 점점 겹치도록 밀어붙임

**단점**:
- 정확도가 높은 경우 과잉 최적화 가능

---

### 2. DIoU (Distance IoU)

**논문**: [CVPR-2020 "Distance-IoU Loss for Faster and More Accurate Object Detection"]

**동기**: GIoU 가 박스의 aspect ratio 를 고려하지 않는 문제

**정의**:
```
DIoU = IoU - ρ²(b, b_gt) / c²
```

여기서:
- `ρ`: 박스 center point 간 거리의 제곱
- `c`: 박스를 포함하는 대각선 길이의 제곱 (최소 encompassing box)

```python
def diou(box1, box2):
    """
    Distance IoU
    box1, box2: (x1, y1, x2, y2)
    """
    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0
    
    # Center points
    c_x = (box1[0] + box1[2] + box2[0] + box2[2]) / 2
    c_y = (box1[1] + box1[3] + box2[1] + box2[3]) / 2
    
    # Diagonal distance
    diag_dist = ((box1[2] - box1[0])**2 + (box1[3] - box1[1])**2 + 
                 (box2[2] - box2[0])**2 + (box2[3] - box2[1])**2) / 2
    
    # DIoU
    rho2 = (c_x - c_x)**2 + (c_y - c_y)**2  # 실제 구현에서는 center point 계산 필요
    diou = iou - rho2 / diag_dist
    
    return diou
```

**Loss**: `L_DIoU = 1 - DIoU + α·v` (v는 aspect ratio term)

**장점**:
- 박스 center 간 거리를 고려
- 더 빠르게 수렴
- aspect ratio 까지 고려 (augmented DIoU)

**단점**:
- 매우 크게 겹치는 경우 GIoU 와 유사한 성능

---

### 3. CIoU (Complete IoU)

**논문**: [arXiv:2011.0828 "CIOU: A Complete IoU Metric for Bounding Box Regression"]

**동기**: DIoU 의 추가 term 과 aspect ratio 고려

**정의**:
```
CIoU = IoU - ρ²(b, b_gt)/c² - α·v
```

여기서:
- `α`: aspect ratio consistency term 의 weight
- `v`: aspect ratio discrepancy, v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²

```python
def ciou(box1, box2):
    """
    Complete IoU
    box1, box2: (x1, y1, x2, y2)
    """
    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0
    
    # Center points
    c_x1, c_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    c_x2, c_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    
    # Center distance squared
    rho2 = (c_x1 - c_x2)**2 + (c_y1 - c_y2)**2
    
    # Diagonal distance of enclosing box
    c_w = max(box1[2], box2[2]) - min(box1[0], box2[0])
    c_h = max(box1[3], box2[3]) - min(box1[1], box2[1])
    c_area = c_w**2 + c_h**2
    
    # Aspect ratio
    ar1 = box1[2] - box1[0]
    ar2 = box1[3] - box1[1]
    ar3 = box2[2] - box2[0]
    ar4 = box2[3] - box2[1]
    
    # Aspect ratio consistency term
    v = (4 / (np.pi**2)) * (np.arctan(ar3/ar4) - np.arctan(ar1/ar2))**2
    
    # Weight for aspect ratio
    alpha = v / (1 - iou + v + epsilon)
    
    # CIoU
    ciou = iou - rho2/c_area - alpha * v
    
    return ciou
```

**Loss**: `L_CIoU = 1 - CIoU`

**장점**:
- IoU, center distance, aspect ratio 를 모두 고려
- YOLOv4 에서 사용되어 성능 크게 향상

**단점**:
- 계산 복잡도 증가

---

### 4. SIoU (Scylla IoU)

**논문**: [arXiv:2205.12740 "YOLOv7: Improved Real-Time Object Detection"]

**동기**: Faster convergence and better convergence speed

**정의**:
```
SIoU = IoU - Σ_i C_i · D_i
```

여기서 `C_i` 와 `D_i` 는 각각 cost function 과 distance function:
- **angle cost**: 박스의 각도 차이
- **distance cost**: center 간 거리
- **shape cost**: aspect ratio 차이

```python
def siou(box1, box2):
    """
    Scylla IoU
    box1, box2: (x1, y1, x2, y2)
    """
    # IoU 계산 (기존과 동일)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0
    
    # Angle cost
    c_x1, c_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    c_x2, c_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    
    # Angle between boxes
    angle = np.arctan2(c_y2 - c_y1, c_x2 - c_x1)
    
    # Distance cost
    dist = np.sqrt((c_x2 - c_x1)**2 + (c_y2 - c_y1)**2)
    
    # Shape cost (aspect ratio)
    shape1 = (box1[2] - box1[0]) / (box1[3] - box1[1])
    shape2 = (box2[2] - box2[0]) / (box2[3] - box2[1])
    
    # SIoU loss
    siou_loss = iou - (angle * dist * abs(shape1 - shape2))
    
    return siou_loss
```

**Loss**: `L_SIoU = 1 - SIoU`

**장점**:
- 3 가지 cost 를 결합하여 더 빠르고 정확하게 수렴
- YOLOv7 에서 사용

**단점**:
- angle 계산이 추가되어 계산량 증가

---

### 5. EIoU (Efficient IoU)

**논문**: [arXiv:2106.06072 "Efficient IoU: A Faster Bounding Box Regression Loss"]

**동기**: CIoU 의 aspect ratio term 을 분리하여 더 효율적으로

**정의**:
```
EIoU = IoU - ρ²(b, b_gt)/c² - ρ²(w, w_gt)/c_w² - ρ²(h, h_gt)/c_h²
```

**Loss**: `L_EIoU = 1 - EIoU`

**장점**:
- aspect ratio 를 width, height 로 분리
- 더 빠른 수렴 속도

---

### 6. WIoU (Wise IoU)

**논문**: [arXiv:2301.12800 "Wise-IoU: Dynamic Non-Monotonic Focusing for Bounding Box Regression"]

**동기**: Overlap-based focusing weight 을 통한 더 효과적인 학습

**정의**:
```
WIou = IoU · ω(v, P)
```

여기서 `ω` 는 dynamic focusing weight 으로, 예측 품질에 따라 가중치를 조절합니다.

**장점**:
- 학습 초기/중기/후기에서 다른 focus 전략 사용
- 더 효과적인 localization

---

## 📊 YOLO 버전별 IoU 채택

### YOLOv3
- **IoU** 기본 사용
- **MSE Loss** 사용
- 단순하지만 성능 제한적

### YOLOv4
- **CIoU Loss** 사용
- **GIoU**도 실험됨
- 성능 크게 향상

### YOLOv5
- **CIoU Loss** 기본
- **WBF (Weighted Box Fusion)** 인코딩
- **Varifocal Loss** with CIoU

### YOLOv6
- **CIoU Loss** 사용
- **Decoupled Head** 구조
- **Reparametrization** 기술

### YOLOv7
- **SIoU Loss** 사용
- **EVA (Efficient Vehicle Alignment)**
- 더 빠른 수렴

### YOLOv8
- **CIoU** 또는 **DIoU** 사용
- **Distribution Focal Loss**
- **Box Loss**: CIoU variant

### YOLOv9
- **GIoU** variant 사용
- **Programmable Gradient Information (PGI)**
- **Reusable Labels** 기술

### YOLO-World
- **Open-vocabulary** 학습
- **GIoU** 기반
- **Semantic-aware** loss

### YOLOX
- **CIoU** 사용
- **SimOTA** label assignment
- **Decoupled training**

---

## 🎯 DETR 계열 IoU

### DETR (Original)
- **Dice Loss** + **Focal Loss**
- **IoU** 기본 사용 (안정화용)
- Hungarian matching

### Deformable DETR
- **GIoU** 사용
- **Deformable attention**으로 성능 향상
- 더 빠른 수렴

### DINO (DETR with Improved DEteR)
- **GIoU** + **IoU** combined loss
- **Contrastive denoising**
- **Query selection**

### RT-DETR (Real-time DETR)
- **DIoU** 사용
- **CNN backbone** + **Transformer**
- 실시간 성능

---

## 🛠 실전 코드

### YOLOv5/Ciou Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIoULoss(nn.Module):
    """Complete IoU Loss for YOLO"""
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
        
        # IoU
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
        enclose_area = enclose_w**2 + enclose_h**2
        
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
        ciou = iou - center_dist / enclose_area - alpha * v
        
        loss = 1 - ciou
        return loss
```

### YOLOv7 SIoU Loss

```python
class SIoULoss(nn.Module):
    """Scylla IoU Loss"""
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # IoU 부분 (同上)
        
        # Angle cost
        c_x1, c_y1 = pred[..., :2].mean(dim=-1), pred[..., 2:].mean(dim=-1)
        c_x2, c_y2 = target[..., :2].mean(dim=-1), target[..., 2:].mean(dim=-1)
        
        angle = torch.atan2(c_y2 - c_y1, c_x2 - c_x1)
        
        # Distance cost
        dist = torch.sqrt(torch.pow(c_x2 - c_x1, 2) + torch.pow(c_y2 - c_y1, 2))
        
        # Shape cost
        shape1 = pred[..., 2] - pred[..., 0]
        shape2 = pred[..., 3] - pred[..., 1]
        
        # SIoU loss
        siou_loss = iou - (angle * dist * torch.abs(shape1 - shape2))
        
        return 1 - siou_loss
```

---

## 📈 성능 비교

| Loss Type | Convergence Speed | Accuracy | Computation |
|-----------|------------------|----------|-------------|
| **IoU** | 느림 | 보통 | 빠름 |
| **GIoU** | 중간 | 좋음 | 보통 |
| **DIoU** | 빠름 | 좋음 | 빠름 |
| **CIoU** | 매우 빠름 | 매우 좋음 | 빠름 |
| **SIoU** | 가장 빠름 | 매우 좋음 | 중간 |
| **EIoU** | 빠름 | 매우 좋음 | 빠름 |

---

## 🎯 결론

1. **YOLOv4~v6**: **CIoU** 최적 (가장 균형잡힘)
2. **YOLOv7**: **SIoU** 사용 (더 빠른 수렴)
3. **YOLOv8~v9**: **CIoU/DIoU** variant (효율성)
4. **DETR 계열**: **GIoU** + **IoU** (안정성)

**실전 추천**:
- **초기 실험**: CIoU (가장 안정적)
- **최적화 필요**: SIoU (더 빠름)
- **연구 목적**: WIoU (동적 focusing)

---
*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, 관련 논문, CVPR/ICCV/ECCV proceedings*