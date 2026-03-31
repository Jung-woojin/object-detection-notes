# Anchor vs Anchor-free Detection Heads

객체검출에서 두 가지 주요 접근법인 **Anchor-based** 와 **Anchor-free** 방식을 상세히 비교 분석합니다.

---

## 📚 목차

- [Anchor-based Detection](#-anchor-based-detection)
- [Anchor-free Detection](#-anchor-free-detection)
- [핵심 차이점](#-핵심-차이점)
- [성능 비교](#-성능-비교)
- [선택 가이드](#-선택-가이드)
- [실전 코드](#-실전-코드)

---

## 🔷 Anchor-based Detection

### 기본 개념

**Anchor-based** 방식은 미리 정의된 **default boxes**(anchors) 를 사용하여 객체를 예측합니다.

#### 특징

1. **Predefined Anchors**: 각 위치에서 여러 개의 anchor box
2. **Multiple Scales**: 다른 크기의 feature map 에서 prediction
3. **Multi-ratios**: 각 위치에서 여러 aspect ratios
4. **Region Proposals**: Anchors 중 "객체가 있는" 영역을 proposals 로 선택

### Anchor 설계

#### Traditional Anchor Design

```python
# Example: YOLOv3 anchors
anchors = [
    [[10, 13],   [16, 30],   [33, 23]],  # P3 - small objects
    [[30, 61],   [62, 45],   [59, 119]], # P4 - medium objects
    [[116, 90],  [156, 198], [373, 326]] # P5 - large objects
]

# Per anchor: [width, height]
# Per level: 3 anchors (3 aspect ratios)
# Total: 3 levels × 3 anchors × 3 ratios = 27 anchors per position
```

#### K-means Clustering for Anchor Design

**Objective**: Dataset 에서 객체들의 bounding box 를 clustering 하여 optimal anchors 학습

```python
def kmeans_anchors(boxes, num_anchors=9):
    """
    Compute anchor boxes using K-means clustering
    
    Args:
        boxes: numpy array of shape (N, 4) [x1, y1, x2, y2]
        num_anchors: number of anchor boxes to compute
    
    Returns:
        anchors: numpy array of shape (num_anchors, 2) [width, height]
    """
    # Convert to width, height
    wh = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(wh)
    
    anchors = kmeans.cluster_centers_
    return anchors
```

#### Anchor Distribution

```
Feature Map: 80×80 (P5)
Anchor per position: 9 (3 scales × 3 ratios)
Total anchors: 80 × 80 × 9 = 57,600

Feature Map: 40×40 (P6)
Anchor per position: 9
Total anchors: 40 × 40 × 9 = 14,400

Feature Map: 20×20 (P7)
Anchor per position: 9
Total anchors: 20 × 20 × 9 = 3,600

Total per image: ~75,600 anchors
```

### Representative Models

| Model | Year | Anchors per Position | Backbone | NMS |
|------|------|---------------------|---------|-----|
| **SSD** | 2016 | 4-6 | VGG16 | Yes |
| **Faster R-CNN** | 2015 | 9 | R-ION | Yes |
| **YOLOv3** | 2018 | 9 | CSPDarknet | Yes |
| **YOLOv4** | 2020 | 9 | CSPDarknet | Yes |
| **YOLOv5** | 2020 | 9 | CSPDarknet | Yes |
| **YOLOv6** | 2022 | 9 | CSPDarknet | Yes |

### Anchor-based Workflow

```python
def anchor_based_detection(model, image, anchors):
    """
    Anchor-based detection workflow
    
    1. Generate feature maps
    2. Predict for each anchor
    3. Decode anchor offsets
    4. Filter by confidence threshold
    5. Apply NMS
    """
    # 1. Feature extraction
    features = backbone(image)
    
    # 2. Predict for each anchor
    predictions = []
    for level, feature in enumerate(features):
        # cls_pred: [N, H, W, num_classes]
        # reg_pred: [N, H, W, 4] (offsets)
        cls_pred, reg_pred = head(feature)
        
        # Decode anchor offsets
        boxes = decode_anchors(reg_pred, anchors[level])
        scores = softmax(cls_pred)
        
        predictions.append({
            'boxes': boxes,
            'scores': scores,
            'level': level
        })
    
    # 3. Combine predictions
    all_boxes = concat(predictions['boxes'])
    all_scores = concat(predictions['scores'])
    all_classes = concat(predictions['classes'])
    
    # 4. Filter by threshold
    mask = all_scores > conf_threshold  # e.g., 0.5
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]
    all_classes = all_classes[mask]
    
    # 5. NMS
    final_boxes, final_scores, final_classes = nms(
        all_boxes, all_scores, all_classes, iou_threshold=0.45
    )
    
    return final_boxes, final_scores, final_classes
```

### 장단점

#### ✅ 장점

1. **Well-studied**: 연구가 많이 됨, 안정적인 성능
2. **Multi-scale**: 여러 scale 에서 detection 가능
3. **Fast inference**: Decoupled heads, optimized
4. **Mature implementations**: Ultralytics, Detectron2 등 구현 풍부

#### ❌ 단점

1. **Hyperparameter sensitive**: Anchor design 에 의존
2. **Fixed priors**: 미리 정해진 anchor 로 제한됨
3. **Complex post-processing**: NMS 필요
4. **Annotation bias**: Dataset bias 에 취약

---

## 🟢 Anchor-free Detection

### 기본 개념

**Anchor-free** 방식은 사전 정의된 anchors 없이, 각 pixel/point 에서 직접 객체를 예측합니다.

#### 특징

1. **Point-based**: 각 pixel 에서 객체 예측
2. **No default boxes**: Anchors 불필요
3. **Simpler design**: 구현이 단순화
4. **Direct prediction**: Center point 에서 직접 예측

### Representative Approaches

#### 1. FCOS (Fully Convolutional One-Stage)

**핵심 아이디어**: 각 pixel 이 bounding box 의 네 변까지 예측

```python
# FCOS prediction per pixel
class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.cls_head = ConvBlock(in_channels, num_classes)
        self.reg_head = ConvBlock(in_channels, 4)  # [l, r, t, b]
        self.centerness_head = ConvBlock(in_channels, 1)
    
    def forward(self, x):
        cls_logits = self.cls_head(x)      # [N, C, H, W]
        bbox_pred = self.reg_head(x)       # [N, 4, H, W]
        centerness = self.centerness_head(x)  # [N, 1, H, W]
        
        return cls_logits, bbox_pred, centerness
```

**Loss**:
- **Classification**: Focal Loss
- **Regression**: L1 loss (l, r, t, b distances)
- **Centerness**: Binary Cross Entropy

#### 2. CenterNet (Objects as Points)

**핵심 아이디어**: 객체를 center point 로 인식

```python
# CenterNet prediction
class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        self.heatmap_head = ConvBlock(in_channels, num_classes)
        self.offset_head = ConvBlock(in_channels, 2)
        self.size_head = ConvBlock(in_channels, 3)  # [w, h, d]
    
    def forward(self, x):
        heatmap = self.heatmap_head(x)      # [N, C, H, W]
        offsets = self.offset_head(x)       # [N, 2, H, W]
        sizes = self.size_head(x)           # [N, 3, H, W]
        
        return heatmap, offsets, sizes
```

#### 3. YOLOv8 (Anchor-free YOLO)

**핵심 아이디어**: Anchor-free YOLOv8 의 decoupled head

```python
# YOLOv8 detection head
class YOLOv8Head(nn.Module):
    def __init__(self, nc, anchors=None):
        self.nc = nc
        self.nl = len(anchors) if anchors else 3
        
        # Shared backbone
        self.cv1 = Conv(256, 256, 1, 1)
        self.cv2 = Conv(256, 256, 1, 1)
        
        # Decoupled branches
        self.cls_branch = nn.Sequential(
            Conv(256, 256, 1, 1),
            Conv(256, nc, 1, 1)
        )
        
        self.reg_branch = nn.Sequential(
            Conv(256, 256, 1, 1),
            Conv(256, 4, 1, 1)  # [x1, y1, x2, y2]
        )
    
    def forward(self, x):
        shared = self.cv1(x)
        
        # Classification
        cls = self.cls_branch(shared)
        
        # Regression
        reg = self.reg_branch(shared)
        
        return torch.cat([reg, cls], dim=1)
```

### Anchor-free Workflow

```python
def anchor_free_detection(model, image):
    """
    Anchor-free detection workflow
    
    1. Extract feature maps
    2. Predict at each point
    3. Filter by centerness/confidence
    4. Distance-based filtering
    5. Optional NMS
    """
    # 1. Feature extraction
    features = backbone(image)
    
    # 2. Point-based predictions
    predictions = model(features)
    
    # 3. Filter by centerness/confidence
    mask = predictions['centerness'] > 0.5
    high_conf_points = predictions[mask]
    
    # 4. Distance-based filtering
    selected_points = []
    for point in high_conf_points:
        if min_distance(point, selected_points) > threshold:
            selected_points.append(point)
    
    # 5. NMS (optional, less needed than anchor-based)
    final_boxes, final_scores, final_classes = nms(
        selected_points['boxes'],
        selected_points['scores'],
        iou_threshold=0.5
    )
    
    return final_boxes, final_scores, final_classes
```

### 장단점

#### ✅ 장점

1. **Simpler**: 구현이 단순, hyperparameter 적음
2. **No anchor design**: Dataset annotation 불필요
3. **Flexible**: 다양한 shape 에 적응 가능
4. **Modern**: 최신 YOLOv8, FCOS 등 채택

#### ❌ 단점

1. **Point sampling**: Dense sampling 필요
2. **Background noise**: 많은 background points
3. **Less mature**: Anchor-based 보다 연구 적음
4. **Still need NMS**: 완전한 NMS-free 는 아님

---

## ⚔️ 핵심 차이점

### Architecture Comparison

| Aspect | Anchor-based | Anchor-free |
|--------|-------------|-------------|
| **Prediction Unit** | Default boxes (anchors) | Each pixel/point |
| **Parameters** | Predefined anchors | None |
| **Complexity** | Higher (anchor design) | Lower |
| **NMS Need** | Required | Reduced/Optional |
| **Training** | More stable | Sensitive to initialization |

### Performance Characteristics

#### Anchor-based

- **Mature**: Well-tested, stable
- **Multi-scale**: Excellent small object detection
- **NMS**: 필수
- **Speed**: Fast (optimized implementations)

#### Anchor-free

- **Modern**: Latest architectures adopt
- **Simple**: Fewer hyperparameters
- **NMS**: Reduced 필요
- **Speed**: Similar, sometimes faster

### Implementation Complexity

```python
# Anchor-based: Complex
class AnchorBasedHead(nn.Module):
    def __init__(self, anchors):
        self.anchors = anchors  # Need to design
        self.cls_layers = [...]
        self.reg_layers = [...]
    
    def forward(self, x):
        # For each anchor, predict offsets
        # Decoding: box = anchor + offset
        pass

# Anchor-free: Simple
class AnchorFreeHead(nn.Module):
    def __init__(self):
        # No anchors needed
        self.cls_head = Conv(...)
        self.reg_head = Conv(...)
    
    def forward(self, x):
        # Direct prediction at each point
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        pass
```

---

## 📊 성능 비교

### COCO Benchmark

| Model | Type | mAP | mAP₅₀ | mAP₇₅ | mAP_S | mAP_M | mAP_L |
|-------|------|-----|-------|-------|-------|-------|-------|
| **SSD300** | Anchor | 25.7 | 42.5 | 26.8 | 11.8 | 30.4 | 35.8 |
| **Faster R-CNN** | Anchor | 36.5 | 56.0 | 39.0 | 18.5 | 39.5 | 46.0 |
| **YOLOv3** | Anchor | 33.6 | 52.9 | 35.6 | 15.9 | 36.5 | 43.1 |
| **YOLOv4** | Anchor | 40.5 | 58.8 | 44.1 | 21.3 | 44.2 | 51.8 |
| **YOLOv5s** | Anchor | 37.4 | 56.8 | 40.9 | 20.3 | 41.0 | 48.3 |
| **FCOS** | Anchor-free | 40.5 | 58.5 | 44.2 | 22.0 | 43.8 | 51.0 |
| **CenterNet** | Anchor-free | 37.9 | 56.1 | 40.5 | 20.5 | 41.2 | 48.5 |
| **YOLOv8s** | Anchor-free | 44.9 | 62.2 | 48.8 | 25.8 | 48.5 | 56.0 |

### Small Object Detection

| Model | mAP_S | Strategy |
|-------|-------|----------|
| **Faster R-CNN** | 18.5 | Multi-scale features |
| **YOLOv3** | 15.9 | P3 feature map |
| **FCOS** | 22.0 | Point-based sampling |
| **YOLOv8s** | 25.8 | Enhanced feature pyramid |

### Inference Speed

| Model | Type | FPS (2080 Ti) |
|-------|------|--------------|
| **YOLOv3** | Anchor | 105 |
| **YOLOv5s** | Anchor | 185 |
| **FCOS** | Anchor-free | 95 |
| **YOLOv8s** | Anchor-free | 220 |

---

## 🎯 선택 가이드

### Anchor-based 을 선택할 때

**추천 시나리오**:
1. **Mature solutions 필요**: 안정성 우선
2. **Multi-scale detection**: 다양한 크기 객체
3. **Small objects**: 작은 객체 검출 중요
4. **Fast deployment**: 검증된 구현 필요

**추천 모델**:
- **YOLOv5/v6**: Real-time, stable
- **Faster R-CNN**: High accuracy
- **SSD**: Lightweight apps

### Anchor-free 를 선택할 때

**추천 시나리오**:
1. **Simple implementation**: 구현 단순화
2. **Modern architecture**: 최신 기술
3. **NMS reduction**: post-processing 최소화
4. **Flexibility**: 다양한 shape 적응

**추천 모델**:
- **YOLOv8**: Latest YOLO, anchor-free
- **FCOS**: Anchor-free pioneer
- **CenterNet**: Point-based approach

### Decision Tree

```
객체검출 모델 선택
├── 안정성/성능 우선?
│   ├── Yes → Anchor-based (YOLOv5/v6, Faster R-CNN)
│   └── No → 최신 기술 우선?
│       ├── Yes → Anchor-free (YOLOv8, FCOS)
│       └── No → Anchor-based
├── Small objects 중요?
│   ├── Yes → Anchor-based (multi-scale)
│   └── No → Either
├── Implementation simplicity?
│   ├── Yes → Anchor-free
│   └── No → Either
└── NMS 불필요?
    ├── Yes → Transformer (DETR, DINO)
    └── No → Either
```

---

## 🛠 실전 코드

### Anchor-based: YOLOv5 Style

```python
class YOLOv5DetectionHead(nn.Module):
    """YOLOv5-style anchor-based detection head"""
    def __init__(self, nc, anchors=None):
        super().__init__()
        self.nc = nc
        self.nl = len(anchors) if anchors else 3
        self.no = nc + 5  # predictions per anchor
        
        # Initialize anchors (example)
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],  # P3
                [[30, 61], [62, 45], [59, 119]], # P4
                [[116, 90], [156, 198], [373, 326]] # P5
            ])
        else:
            self.anchors = anchors
        
        self.stride = torch.zeros(self.nl)
        
        # Detect layers
        self.cv2 = nn.ModuleList([
            Conv(256, self.nc * len(anchors[i]), 1, 1)
            for i in range(self.nl)
        ])
        self.cv3 = nn.ModuleList([
            Conv(256, 4 * len(anchors[i]), 1, 1)
            for i in range(self.nl)
        ])
    
    def forward(self, x):
        predictions = []
        for i in range(self.nl):
            box_pred = self.cv3[i](x[i])
            cls_pred = self.cv2[i](x[i])
            
            # Decode: box = anchor + offset
            box = self.decode_boxes(box_pred, self.anchors[i])
            score = sigmoid(cls_pred)
            
            predictions.append(torch.cat([box, score], dim=1))
        
        return torch.cat(predictions, dim=1)
    
    def decode_boxes(self, reg_pred, anchors):
        """Decode anchor offsets to bounding boxes"""
        # reg_pred: [N, 4*num_anchors, H, W]
        # Decode: [x1, y1, x2, y2]
        pass
```

### Anchor-free: FCOS Style

```python
class FCOSDetectionHead(nn.Module):
    """FCOS-style anchor-free detection head"""
    def __init__(self, nc, in_channels=256):
        super().__init__()
        self.nc = nc
        
        # Shared feature
        self.conv = Conv(in_channels, 256, 1, 1)
        
        # Branches
        self.cls_head = nn.Sequential(
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, nc, 3, 1)
        )
        
        self.reg_head = nn.Sequential(
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 4, 3, 1)  # [l, r, t, b]
        )
        
        self.centerness_head = nn.Sequential(
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 1, 3, 1)  # centerness score
        )
    
    def forward(self, x):
        shared = self.conv(x)
        
        cls_logits = self.cls_head(shared)      # [N, C, H, W]
        bbox_pred = self.reg_head(shared)       # [N, 4, H, W]
        centerness = self.centerness_head(shared)  # [N, 1, H, W]
        
        return cls_logits, bbox_pred, centerness
    
    def decode_boxes(self, bbox_pred, points, stride):
        """Decode [l, r, t, b] to [x1, y1, x2, y2]"""
        # bbox_pred: [l, r, t, b]
        # points: center points of each pixel
        # stride: downsampling stride
        pass
```

### Anchor-free: YOLOv8 Style

```python
class YOLOv8DetectionHead(nn.Module):
    """YOLOv8-style anchor-free detection head with decoupled branches"""
    def __init__(self, nc, in_channels=256):
        super().__init__()
        self.nc = nc
        
        # Shared backbone
        self.shared = Conv(in_channels, 256, 1, 1)
        
        # Decoupled classification branch
        self.cls = nn.Sequential(
            Conv(256, 256, 1, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, nc, 1, 1)
        )
        
        # Decoupled regression branch
        self.reg = nn.Sequential(
            Conv(256, 256, 1, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            Conv(256, 4, 1, 1)  # [x1, y1, x2, y2]
        )
    
    def forward(self, x):
        shared = self.shared(x)
        
        # Classification
        cls = self.cls(shared)
        
        # Regression
        reg = self.reg(shared)
        
        return torch.cat([reg, cls], dim=1)
    
    def decode_boxes(self, reg_pred):
        """Decode reg_pred to [x1, y1, x2, y2]"""
        # reg_pred: normalized [x1, y1, x2, y2] in [0, 1]
        # Decode to pixel coordinates
        pass
```

---

## 📊 비교 요약

### 특징 비교

| Feature | Anchor-based | Anchor-free |
|---------|-------------|-------------|
| **Anchor Design** | Required | None |
| **Implementation** | Complex | Simple |
| **NMS Need** | Required | Reduced |
| **Small Objects** | Excellent | Good |
| **Multi-scale** | Excellent | Good |
| **Training Stability** | High | Medium |
| **Flexibility** | Fixed | Flexible |
| **Speed** | Fast | Fast |

### 성능 비교 (COCO)

| Model | Type | mAP | Speed |
|-------|------|-----|-------|
| **YOLOv5** | Anchor | 37.4 | 185 FPS |
| **YOLOv8** | Anchor-free | 44.9 | 220 FPS |
| **FCOS** | Anchor-free | 40.5 | 95 FPS |
| **Faster R-CNN** | Anchor | 36.5 | 20 FPS |

---

## 🎯 결론

### 선택 가이드

**Anchor-based 추천**:
- ✅ 안정성 우선
- ✅ Multi-scale 필요
- ✅ Small objects 중요
- ✅ 검증된 구현 필요

**Anchor-free 추천**:
- ✅ Simple implementation
- ✅ 최신 기술 우선
- ✅ NMS 최소화
- ✅ Flexibility 필요

### Trend

**2024 현재**:
- **YOLOv8, YOLOv9**: Anchor-free 채택
- **DETR 계열**: Anchor-free
- **Real-time**: Anchor-free 주도

**Future**:
- **Hybrid approaches**: Best of both worlds
- **NMS-free**: Explicit elimination
- **Adaptive anchors**: Learnable anchors

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, FCOS paper, CenterNet paper, CVPR/ICCV/ECCV proceedings*
