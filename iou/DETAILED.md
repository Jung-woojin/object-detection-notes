# IoU Variants - 상세 설명

각 IoU 변형들의 수학적, 기하학적 세부 사항을 깊이 있게 설명합니다.

---

## 📚 목차

- [IoU 수학적 정의](#-iou-수학적-정의)
- [GIoU: Generalized IoU](#-giou-generalized-iou)
- [DIoU: Distance IoU](#-diou-distance-iou)
- [CIoU: Complete IoU](#-ciou-complete-iou)
- [SIoU: Scylla IoU](#-siou-scylla-iou)
- [EIoU: Efficient IoU](#-eiou-efficient-iou)
- [WIoU: Wise IoU](#-wiou-wise-iou)
- [GIoU vs DIoU vs CIoU](#-giou-vs-diou-vs-ciou)

---

## 🔢 IoU 수학적 정의

### 기본 정의

**IoU (Intersection over Union)**:
```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
```

여기서:
- `A`: 예측된 bounding box
- `B`: Ground truth bounding box
- `∩`: 교집합 (intersection)
- `∪`: 합집합 (union)

### 기하학적 표현

두 박스를 (x₁, y₁, x₂, y₂) 로 표현:

```
A = [x₁A, y₁A, x₂A, y₂A]
B = [x₁B, y₁B, x₂B, y₂B]
```

**교집합 계산**:
```
x_overlap = max(0, min(x₂A, x₂B) - max(x₁A, x₁B))
y_overlap = max(0, min(y₂A, y₂B) - max(y₁A, y₁B))

Area(A ∩ B) = x_overlap × y_overlap
```

**각 박스의 넓이**:
```
Area(A) = (x₂A - x₁A) × (y₂A - y₁A)
Area(B) = (x₂B - x₁B) × (y₂B - y₁B)
```

**합집합 계산**:
```
Area(A ∪ B) = Area(A) + Area(B) - Area(A ∩ B)
```

### 수식 정리

```
IoU = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

### 그래디언트 문제

**문제 상황**: 박스가 완전히 겹치지 않을 때

```
A = [0, 0, 1, 1]   # 1×1 박스
B = [2, 2, 3, 3]   # 1×1 박스, 완전히 떨어져 있음

Area(A ∩ B) = 0
Area(A ∪ B) = 1 + 1 - 0 = 2

IoU = 0 / 2 = 0
```

**그래디언트**:
```
∂IoU/∂x₁A = ?
∂IoU/∂y₁A = ?
...
```

문제: `Area(A ∩ B) = 0` 이므로 그래디언트가 0 입니다. 박스가 겹치지 않을 때 학습이 안 됩니다.

---

## 🔄 GIoU: Generalized IoU

### 동기

**문제**: IoU 가 박스가 완전히 겹치지 않을 때 그래디언트가 0 입니다.

**해결**: 박스를 포함하는 최소 bounding box (enclosing box) 를 고려합니다.

### 정의

**GIoU = IoU - (Area(C) - Area(A ∪ B)) / Area(C)**

여기서:
- `C`: 박스 A 와 B 를 모두 포함하는 최소 축정렬 박스
- `Area(C)`: enclosing box 의 넓이

### 기하학적 이해

```
      ┌─────────────────┐
      │       C         │  ← Enclosing box
  ┌───┴───┐     ┌───┴───┐
  │   A   │     │   B   │  ← Boxes A and B
  └───────┘     └───────┘
```

**C 의 계산**:
```
x₁C = min(x₁A, x₁B)
y₁C = min(y₁A, y₁B)
x₂C = max(x₂A, x₂B)
y₂C = max(y₂A, y₂B)

Area(C) = (x₂C - x₁C) × (y₂C - y₁C)
```

### 수식 전개

```
GIoU = IoU - (Area(C) - Area(A ∪ B)) / Area(C)
     = IoU - (1 - Area(A ∪ B)/Area(C))
     = IoU + Area(A ∪ B)/Area(C) - 1

Loss = 1 - GIoU
     = 1 - IoU - Area(A ∪ B)/Area(C) + 1
     = 2 - IoU - Area(A ∪ B)/Area(C)
```

### 그래디언트 분석

**Case 1: 박스가 겹칠 때**

```
∂GIoU/∂x₁A = ∂IoU/∂x₁A + ∂/∂x₁A[Area(A ∪ B)/Area(C)]
```

**Case 2: 박스가 안 겹칠 때**

```
Area(A ∩ B) = 0
Area(A ∪ B) = Area(A) + Area(B)

GIoU = 0 - (Area(C) - Area(A) - Area(B))/Area(C)
     = (Area(A) + Area(B) - Area(C))/Area(C)

∂GIoU/∂x₁A ≠ 0  ✅ 그래디언트 존재!
```

### 학습 동역학

**초기 (겹치지 않음)**:
```
x₁A = 0, y₁A = 0, x₂A = 1, y₂A = 1
x₁B = 5, y₁B = 5, x₂B = 6, y₂B = 6

C = [0, 0, 6, 6]  → Area(C) = 36

GIoU = (1 + 1 - 36) / 36 = -34/36 = -0.944
```

**학습 진행**:
```
Epoch 1: GIoU = -0.944
Epoch 10: GIoU = -0.5
Epoch 50: GIoU = 0.3
Epoch 100: GIoU = 0.7
```

### 장단점

#### ✅ 장점

1. **Non-overlapping 박스에도 학습 가능**: 그래디언트 존재
2. **Enclosing box 로 수렴 유도**: 박스를 점점 겹치도록
3. **Scale-invariant**: 상대적 거리 고려

#### ❌ 단점

1. **과잉 최적화 가능**: GIoU 가 1 보다 작을 때 박스를 너무 가깝게 밀어붙임
2. **Aspect ratio 고려 안 됨**: 박스의 종횡비 무시
3. **최적 위치 보장 안 됨**: 박스가 겹쳐도 정확한 정렬 보장 안 됨

### 실전 코드

```python
import torch

def giou(box1, box2):
    """
    Calculate Generalized IoU
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
    
    Returns:
        giou: scalar value
    """
    # Intersection
    xi1 = torch.max(box1[..., :2], box2[..., :2])
    yi1 = torch.max(box1[..., 2:4], box2[..., 2:4])
    
    inter_w = torch.clamp(xi1[..., 0] - xi1[..., 1], min=0)
    inter_h = torch.clamp(xi1[..., 2] - xi1[..., 3], min=0)
    
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # Union
    union = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / (union + 1e-8)
    
    # Enclosing box
    xc1 = torch.min(box1[..., :2], box2[..., :2])
    yc1 = torch.min(box1[..., 2:4], box2[..., 2:4])
    
    xc2 = torch.max(box1[..., 2:4], box2[..., 2:4])
    yc2 = torch.max(box1[..., 2:4], box2[..., 2:4])
    
    c_w = xc2 - xc1
    c_h = yc2 - yc1
    c_area = c_w * c_h
    
    # GIoU
    gciou = iou - (c_area - union) / (c_area + 1e-8)
    
    return gciou
```

---

## 📏 DIoU: Distance IoU

### 동기

**문제**: GIoU 는 aspect ratio 를 고려하지 않습니다.

**해결**: 박스 center 간 거리를 고려합니다.

### 정의

**DIoU = IoU - ρ²(b, b_gt) / c²**

여기서:
- `ρ²`: 박스 center point 간 거리 (euclidean distance squared)
- `c²`: 박스를 포함하는 대각선 길이의 제곱

### 기하학적 구성

```
     center(b)  ●────────────────● center(b_gt)
                \                /
                 \              /
                  \            /
                   \          /
                    \        /
                     ●──────●
                     A      B
```

**Center distance**:
```
center(b) = ((x₁ + x₂)/2, (y₁ + y₂)/2)
center(b_gt) = ((x₁_gt + x₂_gt)/2, (y₁_gt + y₂_gt)/2)

ρ²(b, b_gt) = (center_x - center_x_gt)² + (center_y - center_y_gt)²
```

**Diagonal of enclosing box**:
```
c² = (x₂C - x₁C)² + (y₂C - y₁C)²
```

### 완전한 DIoU (augmented DIoU)

**CIoU 와 유사한 버전**:
```
DIoU = IoU - ρ²/c² - α·v

여기서:
- v: aspect ratio discrepancy
- α: weight parameter
```

**Aspect ratio term**:
```
v = (4/π²) × (arctan(w_gt/h_gt) - arctan(w/h))²
```

### 수식 정리

**Basic DIoU Loss**:
```
L_DIoU = 1 - DIoU
       = 1 - IoU + ρ²/c²
```

**Augmented DIoU Loss** (with aspect ratio):
```
L_DIoU = 1 - IoU + ρ²/c² + α·v
```

### 그래디언트 분석

**Center distance gradient**:
```
∂ρ²/∂x₁ = ∂/∂x₁[(x₁/2 + x₂/2 - x₁_gt/2 - x₂_gt/2)² + ...]
        = 2(x₁/2 + x₂/2 - x₁_gt/2 - x₂_gt/2) × (1/2)
        = (x₁ + x₂ - x₁_gt - x₂_gt)
```

**Impact**:
- 박스 center 가 target center 로 수렴
- aspect ratio 도 일치하도록 유도

### 학습 동역학

**초기 (far)**:
```
b = [0, 0, 1, 1]
b_gt = [10, 10, 11, 11]

ρ² = (0.5 - 10.5)² + (0.5 - 10.5)² = 100 + 100 = 200
c² = (11 - 0)² + (11 - 0)² = 121 + 121 = 242

DIoU = 0 - 200/242 = -0.826
```

**학습 진행**:
```
Epoch 1: DIoU = -0.826
Epoch 5: DIoU = -0.3
Epoch 20: DIoU = 0.4
Epoch 50: DIoU = 0.8
```

### 장단점

#### ✅ 장점

1. **Center distance 고려**: 더 빠르게 수렴
2. **Aspect ratio 고려**: augmented DIoU 에서
3. **Rotation invariant**: center 간 거리만 고려

#### ❌ 단점

1. **Aspect ratio term 필요**: 기본 DIoU 는 aspect ratio 무시
2. **Overlapping 필요**: 여전히 IoU term 있음

### 실전 코드

```python
def diou(box1, box2):
    """
    Calculate Distance IoU with aspect ratio consideration
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
    
    Returns:
        diou: scalar value
    """
    # IoU calculation
    # ... (same as GIoU)
    
    # Center points
    c_x1 = (box1[..., 0] + box1[..., 2]) / 2
    c_y1 = (box1[..., 1] + box1[..., 3]) / 2
    
    c_x2 = (box2[..., 0] + box2[..., 2]) / 2
    c_y2 = (box2[..., 1] + box2[..., 3]) / 2
    
    # Center distance
    rho2 = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
    
    # Enclosing box diagonal
    xc1 = torch.min(box1[..., :2], box2[..., :2])
    yc1 = torch.min(box1[..., 2:4], box2[..., 2:4])
    
    xc2 = torch.max(box1[..., :2], box2[..., :2])
    yc2 = torch.max(box1[..., 2:4], box2[..., 2:4])
    
    diag_dist = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2
    
    # Aspect ratio term
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]
    
    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + 1e-8)) - torch.atan(w1 / (h1 + 1e-8))) ** 2
    
    # Weight for aspect ratio
    alpha = v / (1 - iou + v + 1e-8)
    
    # DIoU with aspect ratio
    diou = iou - rho2 / (diag_dist + 1e-8) - alpha * v
    
    return diou
```

---

## 🔷 CIoU: Complete IoU

### 동기

**문제**: DIoU 는 aspect ratio 를 명시적으로 고려하지 않습니다.

**해결**: Center distance 와 aspect ratio 를 모두 고려합니다.

### 정의

**CIoU = IoU - ρ²/c² - α·v**

여기서:
- `ρ²`: Center distance squared
- `c²`: Enclosing box diagonal squared
- `v`: Aspect ratio consistency term
- `α`: Adaptive weight for aspect ratio

### 완전한 수식

```
CIoU(b, b_gt) = IoU(b, b_gt) - ρ²(b, b_gt)/c² - α·v(b, b_gt)
```

**Component terms**:

1. **IoU term**: Overlap ratio
2. **Distance term**: Center distance penalty
3. **Aspect ratio term**: Shape similarity

### Aspect Ratio Term 상세

```
v(b, b_gt) = (4/π²) × (arctan(w_gt/h_gt) - arctan(w/h))²
```

**Characteristics**:
- `v = 0`: Aspect ratio 일치 (arctan 차이 0)
- `v > 0`: Aspect ratio 다름
- **Max value**: Aspect ratio 가 완전히 다를 때

**Adaptive weight**:
```
α(b, b_gt) = v / (1 - IoU + v + ε)
```

- `IoU → 1`: α → 0 (aspect ratio 중요도 감소)
- `IoU → 0`: α → 1 (aspect ratio 중요도 증가)

### 수식 정리

**CIoU Formula**:
```
CIoU = IoU - ρ²/c² - α·v

Loss = 1 - CIoU
     = 1 - IoU + ρ²/c² + α·v
```

**Gradient**:
```
∂CIoU/∂x₁ = ∂IoU/∂x₁ - ∂(ρ²/c²)/∂x₁ - ∂(α·v)/∂x₁
```

### 수렴 속도 분석

**Case 1: 박스가 겹치지만 aspect ratio 다름**

```
Initial:
  b = [0, 0, 1, 1]       # Square
  b_gt = [0, 0, 2, 1]    # Rectangle

CIoU = 0.5 - 0 - 0.3 = 0.2
```

**Case 2: 박스가 겹치고 aspect ratio 일치**

```
After adjustment:
  b = [0, 0, 2, 1]       # Now matches aspect ratio

CIoU = 0.5 - 0 - 0 = 0.5
```

### 장단점

#### ✅ 장점

1. **3 가지 요소 모두 고려**: IoU, distance, aspect ratio
2. **완전한 최적화**: box position과 shape 모두
3. **Adaptive weighting**: 상황에 따라 중요도 조정

#### ❌ 단점

1. **복잡한 계산**: 3 가지 term 계산 필요
2. **Gradient stability**: aspect ratio term 의 gradient 불안정 가능

### 실전 코드

```python
def ciou(box1, box2):
    """
    Calculate Complete IoU
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
    
    Returns:
        ciou: scalar value
    """
    # IoU calculation
    # ... (same as GIoU and DIoU)
    
    # Center distance
    c_x1 = (box1[..., 0] + box1[..., 2]) / 2
    c_y1 = (box1[..., 1] + box1[..., 3]) / 2
    
    c_x2 = (box2[..., 0] + box2[..., 2]) / 2
    c_y2 = (box2[..., 1] + box2[..., 3]) / 2
    
    rho2 = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
    
    # Enclosing box diagonal
    xc1 = torch.min(box1[..., :2], box2[..., :2])
    yc1 = torch.min(box1[..., 2:4], box2[..., 2:4])
    
    xc2 = torch.max(box1[..., :2], box2[..., :2])
    yc2 = torch.max(box1[..., 2:4], box2[..., 2:4])
    
    diag_dist = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2
    
    # Aspect ratio
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]
    
    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + 1e-8)) - torch.atan(w1 / (h1 + 1e-8))) ** 2
    
    # Adaptive weight
    alpha = v / (1 - iou + v + 1e-8)
    
    # CIoU
    ciou = iou - rho2 / (diag_dist + 1e-8) - alpha * v
    
    return ciou
```

---

## ⚡ SIoU: Scylla IoU

### 동기

**문제**: 기존 IoU variants 가 convergence speed 와 accuracy 를 동시에 최적화하지 못합니다.

**해결**: 3 가지 cost 를 결합하여 더 빠른 수렴과 높은 정확도 달성.

### 구성 요소

**SIoU = IoU - Σᵢ Cᵢ · Dᵢ**

여기서 `i ∈ {angle, distance, shape}`:

#### 1. Angle Cost (각도 비용)

```
C_angle = Σ_i (1 - cos(θ_i))

여기서 θ_i 는 박스의 orientation 차이
```

**Orientation 계산**:
```
θ = arctan2(y₂ - y₁, x₂ - x₁)
```

**Cost function**:
```
C_angle = 1 - cos(θ_pred - θ_gt)
```

#### 2. Distance Cost (거리 비용)

```
C_distance = 1 - exp(-ρ²/b²)

여기서:
- ρ²: center distance
- b: bounding box diagonal (normalized)
```

**Normalized distance**:
```
ρ²_norm = ρ² / c²
C_distance = 1 - exp(-ρ²_norm)
```

#### 3. Shape Cost (모양 비용)

```
C_shape = 1 - exp(-v²)

여기서 v 는 aspect ratio discrepancy
```

### 수식 정리

**SIoU Formula**:
```
SIoU = IoU - (C_angle · D_angle + C_distance · D_distance + C_shape · D_shape)

Loss = 1 - SIoU
```

**Components**:
1. **Angle cost**: Box orientation difference
2. **Distance cost**: Center distance penalty
3. **Shape cost**: Aspect ratio difference

### 수렴 속도

**Comparison**:
```
GIoU: 50 epochs to mAP=0.7
DIoU: 35 epochs to mAP=0.7
CIoU: 30 epochs to mAP=0.7
SIoU: 20 epochs to mAP=0.7  ← Fastest!
```

### 장단점

#### ✅ 장점

1. **가장 빠른 수렴**: 3 가지 cost 결합
2. **Higher accuracy**: orientation 도 고려
3. **Better convergence**: gradient 더 효과적

#### ❌ 단점

1. **복잡함**: 3 가지 cost 계산
2. **Calculation overhead**: angle 계산 추가

### 실전 코드

```python
def siou(box1, box2):
    """
    Calculate Scylla IoU
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
    
    Returns:
        siou: scalar value
    """
    # IoU calculation
    # ... (same as previous IoUs)
    
    # Angle cost
    c_x1 = (box1[..., 0] + box1[..., 2]) / 2
    c_y1 = (box1[..., 1] + box1[..., 3]) / 2
    
    c_x2 = (box2[..., 0] + box2[..., 2]) / 2
    c_y2 = (box2[..., 1] + box2[..., 3]) / 2
    
    # Orientation
    angle = torch.atan2(c_y2 - c_y1, c_x2 - c_x1)
    
    # Angle cost
    cos_angle = torch.cos(angle)
    angle_cost = 1 - cos_angle
    
    # Distance cost
    rho2 = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
    diag_dist = torch.max(box1[..., 2], box2[..., 2]) - torch.min(box1[..., 0], box2[..., 0])
    diag_h = torch.max(box1[..., 3], box2[..., 3]) - torch.min(box1[..., 1], box2[..., 1])
    
    C_norm = diag_dist ** 2 + diag_h ** 2
    distance_cost = 1 - torch.exp(-rho2 / (C_norm + 1e-8))
    
    # Shape cost
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]
    
    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + 1e-8)) - torch.atan(w1 / (h1 + 1e-8))) ** 2
    shape_cost = 1 - torch.exp(-v)
    
    # SIoU
    siou = iou - (angle_cost * distance_cost + shape_cost)
    
    return siou
```

---

## 🔧 EIoU: Efficient IoU

### 동기

**문제**: CIoU 에서 aspect ratio term 이 box regression 을 느리게 합니다.

**해결**: aspect ratio 를 width 와 height 로 분리하여 더 효율적으로 최적화합니다.

### 정의

**EIoU = IoU - ρ²/c² - ρ²(w, w_gt)/c_w² - ρ²(h, h_gt)/c_h²**

여기서:
- `ρ²(w, w_gt)`: Width difference squared
- `ρ²(h, h_gt)`: Height difference squared
- `c_w²`: Width enclosing box squared
- `c_h²`: Height enclosing box squared

### 분리된 aspect ratio

**CIoU vs EIoU**:

```
CIoU: v = (arctan(w_gt/h_gt) - arctan(w/h))²

EIoU: separate width and height
      width_cost = (w - w_gt)² / c_w²
      height_cost = (h - h_gt)² / c_h²
```

### 수식 정리

**EIoU Formula**:
```
EIoU = IoU - ρ²/c² - ρ²(w, w_gt)/c_w² - ρ²(h, h_gt)/c_h²

Loss = 1 - EIoU
```

**Components**:
1. **IoU**: Overlap
2. **Center distance**: Position
3. **Width distance**: Width alignment
4. **Height distance**: Height alignment

### 수렴 분석

**CIoU vs EIoU**:
```
CIoU: 30 epochs
EIoU: 25 epochs  ← Faster convergence!
```

**이유**:
- Aspect ratio 를 width, height 로 분리
- Separate optimization 가능
- Faster convergence

### 장단점

#### ✅ 장점

1. **Separate optimization**: width 와 height 독립적 최적화
2. **Faster convergence**: 20% speedup
3. **Simpler gradients**: arctan 불필요

#### ❌ 단점

1. **Similar to CIoU**: aspect ratio 여전히 중요

### 실전 코드

```python
def eiou(box1, box2):
    """
    Calculate Efficient IoU
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
    
    Returns:
        eiou: scalar value
    """
    # IoU calculation
    # ... (same as CIoU)
    
    # Center distance
    c_x1 = (box1[..., 0] + box1[..., 2]) / 2
    c_y1 = (box1[..., 1] + box1[..., 3]) / 2
    
    c_x2 = (box2[..., 0] + box2[..., 2]) / 2
    c_y2 = (box2[..., 1] + box2[..., 3]) / 2
    
    rho2 = (c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2
    
    # Enclosing box diagonal
    xc1 = torch.min(box1[..., :2], box2[..., :2])
    yc1 = torch.min(box1[..., 2:4], box2[..., 2:4])
    
    xc2 = torch.max(box1[..., :2], box2[..., :2])
    yc2 = torch.max(box1[..., 2:4], box2[..., 2:4])
    
    c_w = xc2 - xc1
    c_h = yc2 - yc1
    
    c_area = c_w ** 2 + c_h ** 2
    
    # Width and height distances
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]
    
    width_dist = (w1 - w2) ** 2
    height_dist = (h1 - h2) ** 2
    
    # Separate width and height penalties
    width_penalty = width_dist / (c_w ** 2 + 1e-8)
    height_penalty = height_dist / (c_h ** 2 + 1e-8)
    
    # EIoU
    eiou = iou - rho2 / c_area - width_penalty - height_penalty
    
    return eiou
```

---

## 🎯 WIoU: Wise IoU

### 동기

**문제**: 학습 초기, 중기, 후기에서 다른 focus 전략이 필요합니다.

**해결**: Dynamic focusing weight 을 사용하여 각 training stage 에 최적화된 학습.

### Dynamic Focusing Weight

**WIoU = IoU · ω(v, P)**

여기서:
- `ω`: Dynamic focusing weight
- `v`: IoU-based quality metric
- `P`: Training progress

### Focusing Strategy

**Three phases**:

1. **Early stage**: Focus on hard examples
2. **Mid stage**: Balance hard and easy
3. **Late stage**: Focus on high-quality predictions

**Weight function**:
```
ω = φ(v) × ψ(P)

여기서:
- φ(v): Quality-based weight
- ψ(P): Progress-based weight
```

### 수식 정리

**WIoU Formula**:
```
WIoU = IoU · ω(v, P)

Loss = 1 - WIoU
```

**Components**:
1. **IoU**: Base overlap
2. **Dynamic weight**: Adaptive focusing

### 장단점

#### ✅ 장점

1. **Adaptive training**: 각 stage 에 최적화
2. **Better convergence**: Overfitting 방지
3. **Quality-aware**: 예측 품질 고려

#### ❌ 단점

1. **Complex**: Dynamic weight 계산
2. **Hyperparameter sensitive**: Weight tuning 필요

### 실전 코드

```python
def wiou(box1, box2, epoch, total_epochs):
    """
    Calculate Wise IoU with dynamic focusing
    
    Args:
        box1: (x1, y1, x2, y2) - predicted box
        box2: (x1, y1, x2, y2) - ground truth box
        epoch: Current epoch
        total_epochs: Total training epochs
    
    Returns:
        wiou: scalar value
    """
    # IoU calculation
    iou = calculate_iou(box1, box2)
    
    # Calculate quality metric
    v = calculate_quality(box1, box2)
    
    # Progress metric
    P = epoch / total_epochs
    
    # Dynamic weight based on phase
    if P < 0.3:  # Early stage
        weight = focal_weight(v, gamma=2.0)
    elif P < 0.7:  # Mid stage
        weight = focal_weight(v, gamma=1.0)
    else:  # Late stage
        weight = focal_weight(v, gamma=0.5)
    
    # WIoU
    wiou = iou * weight
    
    return wiou
```

---

## 📊 GIoU vs DIoU vs CIoU

### 비교 테이블

| Metric | GIoU | DIoU | CIoU | SIoU | EIoU |
|--------|------|------|------|------|------|
| **Convergence** | Slow | Fast | Fastest | Fastest | Fast |
| **Accuracy** | Good | Very good | Very good | Best | Very good |
| **Computational** | Low | Low | Medium | Medium | Low |
| **Aspect Ratio** | ❌ | ✅ (augmented) | ✅ | ✅ | ✅ |
| **Orientation** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Separate w/h** | ❌ | ❌ | ❌ | ❌ | ✅ |

### 추천 사용법

**초기 실험**:
- **YOLOv5/v8**: CIoU (가장 안정적)
- **Quick test**: DIoU (빠름)

**최적화**:
- **Speed priority**: SIoU (가장 빠른 수렴)
- **Accuracy priority**: SIoU (높은 정확도)

**연구**:
- **New variants**: EIoU, WIoU
- **Understanding**: GIoU 기본 개념 이해

---

*마지막 업데이트: 2026-03-30*
*참고: GIoU paper, DIoU paper, CIoU paper, SIoU paper, EIoU paper, CVPR/ICCV proceedings*
