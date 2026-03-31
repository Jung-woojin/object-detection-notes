# Detection Head Characteristics & NMS Variations

객체검출에서 **Detection Head**(감지 헤드) 의 특성별로 적합한 **Non-Maximum Suppression (NMS)** 전략을 정리합니다.

## 📚 목차

- [Detection Head 기본 개념](#-detection-head-기본-개념)
- [Head 별 NMS 전략](#-head-별-nms-전략)
- [NMS-Free Head Types](#-nms-free-head-types)
- [Head별 상세 분석](#-head별-상세-분석)
- [실전 코드 및 구현](#-실전-코드-및-구현)
- [결론 및 추천](#-결론-및-추천)

---

## 🔍 Detection Head 기본 개념

### Detection Head 란?

**Detection Head** 는 CNN/Transformer 의 backbone 에서 추출된 feature map 을 기반으로 **bounding box** 와 **class scores** 를 예측하는 네트워크 부분입니다.

**주요 역할**:
1. **Classification Head**: 객체 class 예측
2. **Regression Head**: bounding box coordinates 예측
3. **Objectness Head**: 객체 존재 확신도 예측 (일부 모델)

### Head Output Format

일반적인 detection head 의 출력:
```
Output = {
    'boxes': (N, 4),      # N개 검출 박스 (x1, y1, x2, y2)
    'scores': (N, C),     # 각 class 의 confidence score
    'class_ids': (N,)     # 예측된 class ID
}
```

여기서 `N` 은 예측된 객체 수, `C` 는 class 수입니다.

---

## 🎯 Head 별 NMS 전략

Detection head 의 특성에 따라 NMS 전략이 달라집니다.

### Head Type 별 분류

| Head Type | Representative Models | NMS 필요성 | 추천 NMS |
|------|----------|---|---|
| **Anchor-based** | SSD, Faster R-CNN, YOLOv3-v6 | ⭐⭐⭐ 필요 | Traditional NMS / DIoU-NMS |
| **Anchor-free** | FCOS, CenterNet, YOLOv8 | ⭐⭐ 보통 | Soft-NMS / DIoU-NMS |
| **Corner-based** | CornerNet, CenterNet-v2 | ⭐⭐ 보통 | Pairing / Distance-threshold |
| **Transformer-based** | DETR, Deformable DETR, DINO, RT-DETR | ✗ 불필요 | Set-based matching |
| **One-shot** | YOLOv10, YOLO-World | ✗ 불필요 | NMS-free |

---

## 🚫 NMS-Free Head Types

### 1. **DETR-style (Transformer) Heads**

**Models**: DETR, Deformable DETR, DINO, RT-DETR

**Characteristics**:
- **Set-based prediction**: 고정된 수의 query (예: 100 queries)
- **Bipartite matching**: Hungarian algorithm 으로 one-to-one matching
- **One prediction per class**: 각 class 당 하나만 예측
- **No duplicate removal 필요**: matching 과정이 이미 중복 제거

**Why no NMS?**:
```
1. Query 수 = 최대 예측 객체 수 (예: 100)
2. Hungarian matching: optimal one-to-one assignment
3. 각 query 는 하나의 box + class 예측
4. 동일 클래스의 중복 예측이 발생할 수 없음
```

**Output structure**:
```python
output = {
    'boxes': (100, 4),      # 100 queries 중 active ones
    'scores': (100, C),     # 각 query 의 class confidence
    'class_ids': (100,),    # 각 query 의 predicted class
    'active_mask': (100,)   # 어떤 query 가 active 인지
}
```

**Matching process**:
1. Query embeddings 생성 (각 query = 하나의 예측 후보)
2. Class predictions: 각 query 가 어떤 class 인지 예측
3. Box predictions: 해당 box 좌표 예측
4. Hungarian matching: ground truth 와 optimal matching
5. Matching 되지 않은 queries 는 "no object" prediction

**Loss function**:
```python
L_total = λ₁·L_box + λ₂·L_class + λ₃·L_matching

L_matching = min_π Σᵢ₌₁ⁿ⁻¹ [L_box(yᵢ, π(i)) + L_class(pᵢ, c_π(i))]
```

여기서 `π` 는 permutation (matching) 입니다.

---

### 2. **Anchor-free / Center-based Heads**

**Models**: FCOS, CenterNet, FCOS-TRUNK

**Characteristics**:
- **Point-based detection**: 각 pixel / center point 에서 예측
- **Center heatmaps**: 객체 center 만 예측
- **Single peak per object**: 하나의 객체는 하나의 center point
- **Distance-based filtering**: IoU threshold 대신 center distance 사용

**Why NMS is different?**:
- Traditional overlap-based NMS 대신 **distance threshold** 사용
- Center point 기반 → 자연스럽게 중복 제거
- **Penalty-based**: center point 에서 멀어질수록 score 감소

**FCOS example**:
```python
# FCOS: Fully Convolutional One-Stage detection

# Each point predicts:
# 1. Corner distance: top, bottom, left, right distances
# 2. Classification: class probabilities
# 3. Objectness: whether this point is a center

# NMS variant: Distance-based thresholding
for class_id in classes:
    # Get all points for this class
    points = predictions[class_id]
    
    # Sort by score
    points = sort_by_score(points, descending=True)
    
    # Filter by distance threshold
    for point in points:
        if min_distance(point, selected_points) > threshold:
            selected_points.append(point)
        else:
            # Merge or suppress similar points
            suppress_similar_points(point)
```

**CenterNet approach**:
```python
# CenterNet: Corner pairing + center heatmap

# 1. Predict center heatmap
center_scores = model(center_heatmap)  # shape: (B, 1, H, W)

# 2. NMS-like center suppression
center_scores = soft_nms(center_scores, threshold=0.3)

# 3. Extract top-K centers
centers = extract_top_k(center_scores, k=100)

# 4. Pair corners for each center
#    (top, bottom, left, right offsets)
boxes = pair_corners(centers)
```

---

### 3. **Corner-based Heads**

**Models**: CornerNet, CenterNet-v2

**Characteristics**:
- **Corner detection**: 객체의 top-left, bottom-right corner 예측
- **Pairing logic**: corners 를 pairing 하여 box 형성
- **No overlap suppression**: corner pairing 이 이미 중복 방지

**How it works**:
```python
# CornerNet-style

# 1. Predict top-left corners
tl_heatmap = model(top_left)  # top-left corner heatmap

# 2. Predict bottom-right corners  
br_heatmap = model(bottom_right)  # bottom-right corner heatmap

# 3. Pairing (association)
#    각 top-left 에 가장 유사한 bottom-right 찾기
pairs = association_net(tl_heatmap, br_heatmap)

# 4. Form boxes
boxes = [tl + br_offset for tl, br in pairs]
```

**Key difference from traditional NMS**:
- Traditional: Overlap-based → suppress boxes
- Corner-based: Association-based → pair corners

---

### 4. **One-shot / NMS-free Heads**

**Models**: YOLOv10, YOLO-World (some modes)

**Characteristics**:
- **Explicit NMS elimination**: 설계 단계에서 NMS 불필요하게 최적화
- **Single prediction per object**: 각 객체당 단 한 개의 예측만
- **Efficient inference**: Post-processing 없이 바로 사용

**YOLOv10 approach**:
```python
# YOLOv10: Explicit double-free design

# 1. Training: Learn to suppress duplicates internally
# 2. Inference: No post-processing needed
# 3. Active selection: Network 스스로 중복 예측 억제

# Output: Clean predictions directly
predictions = model(image)  # Already NMS-free
```

**YOLO-World open-vocabulary**:
```python
# YOLO-World: Semantic-based filtering

# 1. Text embeddings 와 image features 연결
# 2. Class-agnostic predictions
# 3. Semantic filtering代替 NMS

predictions = model(image, text_prompt="dog")
# Semantic matching 만 수행 → NMS 불필요
```

---

## 📊 Head 별 상세 분석

### Anchor-based Heads

**Models**: SSD, Faster R-CNN, YOLOv3, YOLOv4, YOLOv5, YOLOv6

**Characteristics**:
- **Anchors**: 미리 정의된 여러 개의 default boxes
- **Multiple predictions per object**: 하나의 객체가 여러 anchor 에서 예측됨
- **Overlap 발생**: 동일 객체에 대한 중복 예측이 필수적

**NMS 필요성**: ⭐⭐⭐ (필수)

**How it works**:
```python
# Anchor-based detection flow

# 1. Generate anchors (predefined boxes)
anchors = generate_anchors(scales=[0.5, 1.0, 2.0],
                          ratios=[0.5, 1.0, 2.0])
# 예: 9 anchors per location

# 2. Predict for each anchor
#    - Class probabilities
#    - Box offsets
predictions = model(feature_map)

# 3. Decode anchors to boxes
boxes = decode_anchors(predictions, anchors)

# 4. NMS: 중복 제거
boxes, scores, classes = nms(boxes, scores, threshold=0.45)
```

**NMS strategies**:

| Strategy | Description |适用场景 |
|------|-----|---|---|
| **Traditional NMS** | IoU threshold 로 필터링 | 기본, 일반적 |
| **DIoU-NMS** | Distance + IoU | 더 정확한 suppression |
| **Soft-NMS** | 점수 점진적 감소 | 밀집 객체 |
| **Multi-level NMS** | Scale별 처리 | multi-scale objects |

---

### Anchor-free Heads

**Models**: FCOS, CenterNet, YOLOv7, YOLOv8

**Characteristics**:
- **No predefined anchors**: 각 point 에서 직접 예측
- **Point sampling**: feature map 의 각 pixel 에서 예측
- **Some duplicates**: 여전히 중복 가능성 있음

**NMS 필요성**: ⭐⭐ (보통)

**YOLOv7/8 approach**:
```python
# YOLOv7/v8: Anchor-free with reduced NMS need

# 1. Point-based prediction
predictions = model(feature_map)  # 각 point 에서 예측

# 2. Objectness filtering
high_conf = predictions.scores > threshold  # 예: 0.5

# 3. NMS (필요시)
boxes, scores, classes = nms(high_conf.boxes, 
                             high_conf.scores, 
                             threshold=0.5)
```

**NMS reduction**:
- Anchor-based 보다 중복이 적음 (9 anchors → 1 point)
- 그래도 NMS 필요 (동일 객체에서 여러 point 가 예측될 수 있음)

---

### Transformer-based Heads

**Models**: DETR, Deformable DETR, DINO, RT-DETR

**Characteristics**:
- **Query-based**: 고정된 수의 query (예: 100)
- **Set prediction**: 각 query 가 하나의 객체 예측
- **One-to-one matching**: Hungarian algorithm 으로 중복 제거

**NMS 필요성**: ✗ (불필요)

**DETR flow**:
```python
# DETR: No NMS needed

# 1. Query initialization
queries = learnable_queries(n_queries=100)  # shape: (100, D)

# 2. Transformer decoder
#    각 query 가 하나의 객체 예측
output = transformer_decoder(queries, features)

# 3. Prediction heads
box_preds = box_head(output)        # (100, 4)
class_preds = class_head(output)    # (100, C)
obj_scores = objectness_head(output)  # (100,)

# 4. One-to-one matching (already done during training)
#    No post-processing NMS needed

# 5. Filtering only
active = obj_scores > threshold
final_boxes = box_preds[active]
final_classes = class_preds[active]
```

---

## 🛠 실전 코드 및 구현

### Traditional NMS (Anchor-based)

```python
import torch
import torch.nn.functional as F

def traditional_nms(boxes, scores, classes, iou_threshold=0.45):
    """
    Traditional NMS for anchor-based detection heads
    """
    # boxes: (N, 4) [x1, y1, x2, y2]
    # scores: (N,) confidence scores
    # classes: (N,) class IDs
    
    keep = []
    
    # Get unique classes
    unique_classes = torch.unique(classes)
    
    for cls in unique_classes:
        # Filter by class
        mask = classes == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        
        # Sort by score
        sorted_idx = torch.argsort(cls_scores, descending=True)
        cls_boxes = cls_boxes[sorted_idx]
        cls_scores = cls_scores[sorted_idx]
        
        # NMS loop
        while len(cls_boxes) > 0:
            # Keep highest score
            idx = 0
            keep.append(sorted_idx[idx])
            
            # Calculate IoU with kept box
            if len(cls_boxes) == 1:
                break
            
            ious = calculate_iou(cls_boxes[0], cls_boxes[1:])
            
            # Filter by IoU threshold
            keep_idx = (ious < iou_threshold).nonzero().squeeze()
            cls_boxes = cls_boxes[keep_idx + 1]
            cls_scores = cls_scores[keep_idx + 1]
    
    return keep
```

### Soft-NMS (밀집 객체용)

```python
def soft_nms(boxes, scores, classes, iou_threshold=0.5, sigma=0.3):
    """
    Soft-NMS: 점수를 점진적으로 감소시키는 NMS
    밀집된 객체 (dense objects) 에 효과적
    """
    keep = []
    
    unique_classes = torch.unique(classes)
    
    for cls in unique_classes:
        mask = classes == cls
        cls_boxes = boxes[mask].clone()
        cls_scores = scores[mask].clone()
        
        while len(cls_boxes) > 0:
            idx = torch.argmax(cls_scores)
            keep.append(idx)
            
            if len(cls_boxes) == 1:
                break
            
            # IoU with current best box
            best_box = cls_boxes[idx]
            ious = calculate_iou(best_box, cls_boxes)
            
            # Weight decay based on IoU
            weights = torch.ones_like(cls_scores)
            high_iou = ious > iou_threshold
            weights[high_iou] = torch.exp(-ious[high_iou]**2 / sigma)
            
            # Apply weights
            cls_scores *= weights
            
            # Remove processed box
            cls_boxes = torch.cat([cls_boxes[:idx], cls_boxes[idx+1:]])
            cls_scores = torch.cat([cls_scores[:idx], cls_scores[idx+1:]])
    
    return keep
```

### DIoU-NMS (Distance-aware)

```python
class DIoUNMS:
    """
    Distance-IoU NMS
    center distance 를 고려한 NMS
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
    
    def __call__(self, boxes, scores, classes):
        keep = []
        unique_classes = torch.unique(classes)
        
        for cls in unique_classes:
            mask = classes == cls
            cls_boxes = boxes[mask].clone()
            cls_scores = scores[mask].clone()
            
            while len(cls_boxes) > 0:
                idx = torch.argmax(cls_scores)
                keep.append(idx.item())
                
                if len(cls_boxes) == 1:
                    break
                
                # DIoU 계산
                best_box = cls_boxes[idx]
                dious = self.calculate_diou(best_box, cls_boxes)
                
                # Filter by DIoU threshold
                keep_idx = (dious > self.iou_threshold).nonzero().squeeze()
                cls_boxes = cls_boxes[keep_idx + 1]
                cls_scores = cls_scores[keep_idx + 1]
        
        return keep
    
    def calculate_diou(self, box1, boxes2):
        """
        Calculate DIoU between box1 and boxes2
        """
        # Intersection
        xi1 = torch.max(box1[:2], boxes2[:, :2])
        yi1 = torch.max(box1[2:4], boxes2[:, 2:4])
        
        inter_w = torch.clamp(xi1 - torch.min(box1[:2], boxes2[:, :2]), min=0)
        inter_h = torch.clamp(yi1 - torch.min(box1[2:4], boxes2[:, 2:4]), min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Union
        union = area1 + area2 - inter_area
        iou = inter_area / (union + 1e-8)
        
        # Center distance
        center_dist = torch.sum((boxes2[:, :2] - box1[:2])**2, dim=1)
        
        # Diagonal distance of enclosing box
        diag_dist = torch.max(box1[2], boxes2[:, 2]) - torch.min(box1[0], boxes2[:, 0])
        diag_h = torch.max(box1[3], boxes2[:, 3]) - torch.min(box1[1], boxes2[:, 1])
        c_area = diag_dist**2 + diag_h**2
        
        # DIoU
        diou = iou - center_dist / (c_area + 1e-8)
        
        return diou
```

---

## 📊 Head 별 NMS 선택 가이드

### Anchor-based Heads (SSD, Faster R-CNN, YOLOv3-v6)

**추천**: Traditional NMS 또는 DIoU-NMS

```python
# YOLOv5/v6 style
if use_diou:
    nms = DIoUNMS(iou_threshold=0.45)
else:
    nms = TraditionalNMS(iou_threshold=0.45)
```

### Anchor-free Heads (FCOS, CenterNet, YOLOv7-v8)

**추천**: Soft-NMS 또는 DIoU-NMS

```python
# YOLOv7/v8 style
nms = SoftNMS(iou_threshold=0.5, sigma=0.3)
```

### Transformer-based Heads (DETR, DINO, RT-DETR)

**추천**: NMS 없음, filtering 만 수행

```python
# DETR style
active_mask = obj_scores > threshold  # 예: 0.3
final_boxes = boxes[active_mask]
final_scores = scores[active_mask]
final_classes = classes[active_mask]
```

---

## 🎯 결론 및 추천

### NMS 필요성 요약

| Head Type | NMS 필요? | 추천 전략 | 이유 |
|------|---|---|---|
| **Anchor-based** | ⭐⭐⭐ 필수 | Traditional / DIoU-NMS | 중복 예측 필수 |
| **Anchor-free** | ⭐⭐ 보통 | Soft-NMS / DIoU-NMS | 일부 중복 가능 |
| **Transformer** | ✗ 불필요 | Filtering only | Matching 이 중복 제거 |
| **Corner-based** | ⭐⭐ 보통 | Pairing | Corner pairing 방식 |
| **One-shot** | ✗ 불필요 | None | Explicit NMS-free 설계 |

### 실전 추천

**초기 실험**:
- Anchor-based: **Traditional NMS** (iou_threshold=0.45-0.5)
- Anchor-free: **Soft-NMS** (sigma=0.3)
- Transformer: **Filtering only** (threshold=0.3)

**최적화**:
- Anchor-based: **DIoU-NMS** (더 정확한 suppression)
- Dense objects: **Soft-NMS** (밀집 객체)
- Real-time: **Traditional NMS** (빠름)

**연구 목적**:
- **Transformer**: NMS-free 의 한계 연구
- **Anchor-free**: distance-based filtering 최적화
- **NMS variant**: 새로운 metric 연구

---

## 📚 참고 자료

1. **FCOS**: "Fully Convolutional One-Stage Object Detection" (ICCV-2019)
2. **CenterNet**: "Objects as Points" (arXiv-2019)
3. **DETR**: "End-to-End Object Detection with Transformers" (ECCV-2020)
4. **DIoU**: "Distance-IoU Loss for Faster and More Accurate Object Detection" (CVPR-2020)
5. **Soft-NMS**: "Improved Object Detection with Soft Non-Maximum Suppression" (arXiv-2017)
6. **DINO**: "DINO: DETR with Improved DEterministic Queries" (NeurIPS-2022)
7. **YOLOv10**: "Explicitly Constructing NMS-Free Object Detector" (arXiv-2024)

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, 관련 논문, CVPR/ICCV/ECCV/NeurIPS proceedings*
