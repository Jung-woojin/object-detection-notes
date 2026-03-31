# YOLO 시리즈 IoU Variants

각 YOLO 버전별로 채택된 IoU variant 를 상세히 분석합니다.

---

## 📚 목차

- [YOLO 버전별 IoU 변형](#-yolo-버전별-iou-변형)
- [각 IoU variant详解](#-각-iou-variant 详解)
- [Version별 채택 배경](#-version별-채택-배경)
- [실전 코드](#-실전-코드)
- [선택 가이드](#-선택-가이드)

---

## 🎯 YOLO 버전별 IoU 변형

### Version Evolution Timeline

| Version | Year | IoU Variant | Loss Type | Key Features |
|---------|------|-------------|---------|----|
| **v3** | 2018 | MSE (basic) | MSE + BCE | Simple, basic |
| **v4** | 2020 | GIoU | GIoU + Focal | Better balance |
| **v5** | 2020 | CIoU | CIoU + BCE | Standard now |
| **v6** | 2022 | CIoU | CIoU + VFL | Varifocal |
| **v7** | 2022 | SIoU | SIoU + BCE | Fastest convergence |
| **v8** | 2023 | CIoU | CIoU + VFL | Modern standard |
| **v9** | 2024 | GIoU variant | GIoU + BCE | PGI optimization |
| **v10** | 2024 | CIoU | CIoU + BCE | NMS-free |
| **v11** | 2025 | CIoU | CIoU + VFL | Latest |
| **v12** | 2025 | EIoU | EIoU + VFL | Efficient |

---

## 🔬 각 IoU variant详解

### YOLOv3: MSE Loss

**Paper**: "YOLOv3: An Incremental Improvement" (2018)

#### Loss Function

```
L_total = λ₁·L_box + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: MSE (Mean Squared Error)

```python
def mse_loss(box1, box2):
    """
    MSE loss for bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) - predicted
        box2: (x1, y1, x2, y2) - ground truth
    
    Returns:
        MSE loss value
    """
    loss = torch.nn.MSELoss()
    return loss(box1, box2)
```

**Characteristics**:
- Simple implementation
- Unstable gradients
- Slow convergence

**Why MSE?**:
- Simple and intuitive
- Easy to implement
- Works okay for basic detection

**Limitations**:
- Poor performance for non-overlapping boxes
- Unstable gradients
- Slow convergence

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 33.6 |
| Speed | 105 FPS |
| Convergence | Slow |

### YOLOv4: GIoU

**Paper**: "YOLOv4: Optimal Speed and Accuracy" (2020)

#### Loss Function

```
L_total = λ₁·L_GIoU + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: GIoU (Generalized IoU)

```python
def giou_loss(box1, box2):
    """
    GIoU loss for YOLOv4
    
    Args:
        box1: (x1, y1, x2, y2) - predicted
        box2: (x1, y1, x2, y2) - ground truth
    
    Returns:
        GIoU loss value
    """
    # IoU calculation
    inter = intersection(box1, box2)
    union = union(box1, box2)
    iou = inter / union
    
    # Enclosing box
    enclosing = enclosing_box(box1, box2)
    enclosing_area = area(enclosing)
    
    # GIoU
    gciou = iou - (enclosing_area - union) / enclosing_area
    
    # Loss
    loss = 1 - gciou
    
    return loss
```

**Characteristics**:
- Better gradients for non-overlapping
- More stable than MSE
- Faster convergence

**Why GIoU?**:
- Solves MSE gradient problem
- Better localization
- Industry standard

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 40.5 |
| Speed | 67 FPS |
| Convergence | Medium |

### YOLOv5: CIoU

**Paper**: YOLOv5 (Ultralytics, 2020)

#### Loss Function

```
L_total = λ₁·L_CIoU + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: CIoU (Complete IoU)

```python
def ciou_loss(box1, box2):
    """
    CIoU loss for YOLOv5
    
    Args:
        box1: (x1, y1, x2, y2) - predicted
        box2: (x1, y1, x2, y2) - ground truth
    
    Returns:
        CIoU loss value
    """
    # IoU calculation
    inter = intersection(box1, box2)
    union = union(box1, box2)
    iou = inter / union
    
    # Center distance
    center_dist = center_distance(box1, box2)
    
    # Diagonal distance
    diagonal = enclosing_diagonal(box1, box2)
    
    # Aspect ratio
    aspect_ratio = aspect_ratio_term(box1, box2)
    
    # CIoU
    ciou = iou - center_dist/diagonal - alpha * aspect_ratio
    
    # Loss
    loss = 1 - ciou
    
    return loss
```

**Characteristics**:
- Best balance of speed and accuracy
- Stable training
- Industry standard

**Why CIoU?**:
- Faster convergence than GIoU
- Better aspect ratio handling
- Stable gradients

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 37.4 (s) / 50.2 (m) |
| Speed | 185 FPS (s) |
| Convergence | Fast |

### YOLOv6: CIoU + VFL

**Paper**: "YOLOv6: A Single-stage Object Detection Framework" (2022)

#### Loss Function

```
L_total = λ₁·L_CIoU + λ₂·L_VFL + λ₃·L_obj
```

**Box Loss**: CIoU

**Classification Loss**: VFL (Varifocal Loss)

```python
def vfl_loss(pred, target, iou):
    """
    Varifocal Loss for classification
    
    Args:
        pred: Predicted probabilities
        target: Ground truth labels
        iou: IoU between predicted and GT
    
    Returns:
        VFL value
    """
    # Focal weight based on IoU
    focal_weight = (iou ** 2)
    
    # VFL
    if target == 1:
        loss = -focal_weight * torch.log(pred + 1e-8)
    else:
        loss = -torch.pow(1 - pred, 4) * torch.log(1 - pred + 1e-8)
    
    return loss
```

**Characteristics**:
- VFL for better classification
- CIoU for better localization
- Reparameterized network

**Why CIoU + VFL?**:
- Better classification performance
- Faster convergence
- Improved precision

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 38.9 (s) / 51.1 (m) |
| Speed | 195 FPS (s) |
| Convergence | Fast |

### YOLOv7: SIoU

**Paper**: "YOLOv7: Trainable Bag-of-Freebies" (2022)

#### Loss Function

```
L_total = λ₁·L_SIoU + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: SIoU (Scylla IoU)

```python
def siou_loss(box1, box2):
    """
    SIoU loss for YOLOv7
    
    Args:
        box1: (x1, y1, x2, y2) - predicted
        box2: (x1, y1, x2, y2) - ground truth
    
    Returns:
        SIoU loss value
    """
    # IoU calculation
    inter = intersection(box1, box2)
    union = union(box1, box2)
    iou = inter / union
    
    # Angle cost
    angle_cost = angle_cost(box1, box2)
    
    # Distance cost
    distance_cost = distance_cost(box1, box2)
    
    # Shape cost
    shape_cost = shape_cost(box1, box2)
    
    # SIoU
    siou = iou - (angle_cost * distance_cost + shape_cost)
    
    # Loss
    loss = 1 - siou
    
    return loss
```

**Characteristics**:
- Fastest convergence
- Three cost components
- Best for speed

**Why SIoU?**:
- Fastest training convergence
- Better accuracy
- Angle awareness

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 38.3 (s) / 51.5 (m) |
| Speed | 210 FPS (s) |
| Convergence | **Fastest** |

### YOLOv8: CIoU + VFL

**Paper**: YOLOv8 (Ultralytics, 2023)

#### Loss Function

```
L_total = λ₁·L_CIoU + λ₂·L_VFL + λ₃·L_obj
```

**Box Loss**: CIoU (variant)

**Classification Loss**: VFL (Varifocal Loss)

```python
def yolo_v8_loss(pred, target):
    """
    YOLOv8 loss function
    
    Args:
        pred: Model predictions
        target: Ground truth
    
    Returns:
        Total loss
    """
    # Box loss (CIoU variant)
    box_loss = ciou_loss(pred['boxes'], target['boxes'])
    
    # Classification loss (VFL)
    cls_loss = vfl_loss(pred['scores'], target['classes'], pred['iou'])
    
    # Objectness loss
    obj_loss = bce_loss(pred['objectness'], target['objectness'])
    
    # Combined loss
    total = 7.5 * box_loss + 0.5 * cls_loss + 0.5 * obj_loss
    
    return total
```

**Characteristics**:
- Modern standard
- CIoU + VFL combination
- Stable and accurate

**Why CIoU + VFL?**:
- Best balance
- Industry standard
- Proven performance

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 37.3 (n) / 44.9 (s) / 50.2 (m) |
| Speed | 615 FPS (n) / 480 FPS (s) |
| Convergence | Fast |

### YOLOv9: GIoU variant

**Paper**: "YOLOv9: Learning What You Want to Learn" (2024)

#### Loss Function

```
L_total = λ₁·L_GIoU_variant + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: GIoU variant with PGI

```python
def yolo_v9_loss(pred, target):
    """
    YOLOv9 loss with PGI optimization
    
    Args:
        pred: Model predictions
        target: Ground truth
    
    Returns:
        Total loss
    """
    # Box loss (GIoU variant)
    box_loss = giou_loss_variant(pred['boxes'], target['boxes'])
    
    # Classification loss
    cls_loss = bce_loss(pred['scores'], target['classes'])
    
    # Objectness loss
    obj_loss = bce_loss(pred['objectness'], target['objectness'])
    
    # Combined loss with PGI
    total = 7.5 * box_loss + 0.5 * cls_loss + 0.5 * obj_loss
    
    # PGI enhancement
    total = apply_pgi(total, pred, target)
    
    return total
```

**Characteristics**:
- GIoU with PGI optimization
- Reusable labels
- Better gradient flow

**Why GIoU variant?**:
- PGI optimization
- Efficient training
- Better accuracy

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 39.1 (n) / 46.0 (s) / 51.3 (m) |
| Speed | 310 FPS (c) |
| Convergence | Fast (with PGI) |

### YOLOv10: CIoU

**Paper**: "YOLOv10: Real-Time End-to-End Object Detection" (2024)

#### Loss Function

```
L_total = λ₁·L_CIoU + λ₂·L_obj + λ₃·L_cls
```

**Box Loss**: CIoU

**Special**: NMS-free design

```python
def yolo_v10_loss(pred, target):
    """
    YOLOv10 loss for NMS-free detection
    
    Args:
        pred: Model predictions (double-labeling)
        target: Ground truth
    
    Returns:
        Total loss
    """
    # Box loss (CIoU)
    box_loss = ciou_loss(pred['boxes'], target['boxes'])
    
    # Classification loss
    cls_loss = bce_loss(pred['scores'], target['classes'])
    
    # Objectness loss
    obj_loss = bce_loss(pred['objectness'], target['objectness'])
    
    # Combined loss (same as v8)
    total = 7.5 * box_loss + 0.5 * cls_loss + 0.5 * obj_loss
    
    return total
```

**Characteristics**:
- CIoU (standard)
- NMS-free training
- Direct inference

**Why CIoU?**:
- Stable and proven
- Works with NMS-free
- Simple implementation

#### Performance

| Metric | Value |
|--------|-----|
| mAP | 38.4 (n) / 45.7 (s) / 51.1 (m) |
| Speed | 650 FPS (n) / 320 FPS (m) |
| Convergence | Fast |

---

## 📊 Version별 채택 배경

### Why Different Versions Use Different IoU?

#### YOLOv3: MSE

**Context**: Early YOLO era
- Simple baseline
- Proof of concept
- Not optimized

**Why MSE?**:
- Simple to implement
- Good enough for early version
- Foundation for later improvements

#### YOLOv4: GIoU

**Context**: Need better performance
- MSE gradient problem identified
- Need for better localization

**Why GIoU?**:
- Solves gradient problem
- Better accuracy
- Industry standard

#### YOLOv5: CIoU

**Context**: Standardization
- GIoU improved but still issues
- Need for better aspect ratio handling

**Why CIoU?**:
- Better aspect ratio handling
- Faster convergence
- More stable

#### YOLOv6: CIoU + VFL

**Context**: Optimization
- Classification needs improvement
- Need for better precision

**Why VFL?**:
- Better classification performance
- Handles class imbalance
- More robust

#### YOLOv7: SIoU

**Context**: Speed optimization
- Need faster convergence
- Real-time priority

**Why SIoU?**:
- Fastest convergence
- Three cost components
- Best for speed

#### YOLOv8: CIoU + VFL

**Context**: Modern standard
- Best of both worlds
- Proven performance

**Why CIoU + VFL?**:
- Best balance
- Modern standard
- Industry adoption

#### YOLOv9: GIoU variant

**Context**: PGI optimization
- Need for better gradient flow
- Efficient training

**Why GIoU variant?**:
- PGI optimization
- Better efficiency
- Improved performance

#### YOLOv10: CIoU

**Context**: NMS-free design
- Simple and stable
- Proven to work

**Why CIoU?**:
- Simple and stable
- Works with NMS-free
- Easy to implement

---

## 🛠 실전 코드

### Universal YOLO IoU Loss Factory

```python
class YOLOIoUFactory:
    """
    Factory for creating YOLO IoU loss based on version
    
    Usage:
        loss_factory = YOLOIoUFactory()
        v5_loss = loss_factory.get_loss(version='v5')
        v7_loss = loss_factory.get_loss(version='v7')
    """
    
    VERSION_LOSSES = {
        'v3': 'MSE',
        'v4': 'GIoU',
        'v5': 'CIoU',
        'v6': 'CIoU',
        'v7': 'SIoU',
        'v8': 'CIoU',
        'v9': 'GIoU_variant',
        'v10': 'CIoU',
    }
    
    VERSION_CONFIG = {
        'v3': {
            'box_weight': 1.0,
            'obj_weight': 1.0,
            'cls_weight': 1.0,
            'optimizer': 'SGD',
            'lr': 0.001
        },
        'v4': {
            'box_weight': 7.5,
            'obj_weight': 1.0,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        },
        'v5': {
            'box_weight': 7.5,
            'obj_weight': 0.7,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        },
        'v6': {
            'box_weight': 7.5,
            'obj_weight': 0.5,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        },
        'v7': {
            'box_weight': 7.5,
            'obj_weight': 1.5,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        },
        'v8': {
            'box_weight': 7.5,
            'obj_weight': 0.5,
            'cls_weight': 0.5,
            'optimizer': 'AdamW',
            'lr': 0.01
        },
        'v9': {
            'box_weight': 7.5,
            'obj_weight': 0.5,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        },
        'v10': {
            'box_weight': 7.5,
            'obj_weight': 0.5,
            'cls_weight': 0.5,
            'optimizer': 'SGD',
            'lr': 0.01
        }
    }
    
    def get_loss(self, version):
        """
        Get appropriate IoU loss for version
        
        Args:
            version: YOLO version string (e.g., 'v5', 'v7')
        
        Returns:
            Loss module
        """
        if version not in self.VERSION_LOSSES:
            raise ValueError(f"Unknown version: {version}")
        
        iou_type = self.VERSION_LOSSES[version]
        config = self.VERSION_CONFIG[version]
        
        if iou_type == 'MSE':
            return MSEIoULoss(config)
        elif iou_type == 'GIoU':
            return GIoULoss(config)
        elif iou_type == 'CIoU':
            return CIoULoss(config)
        elif iou_type == 'SIoU':
            return SIoULoss(config)
        elif iou_type == 'GIoU_variant':
            return GIoUVariantLoss(config)
    
    def get_config(self, version):
        """
        Get configuration for version
        
        Args:
            version: YOLO version string
        
        Returns:
            Configuration dictionary
        """
        return self.VERSION_CONFIG[version]
```

### Usage Example

```python
# Create factory
factory = YOLOIoUFactory()

# Get YOLOv5 loss
v5_loss = factory.get_loss('v5')
v5_config = factory.get_config('v5')

# Train with v5
model = YOLOv8()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=v5_config['lr'],
    momentum=0.937,
    weight_decay=0.05
)

for epoch in range(100):
    for batch in dataloader:
        predictions = model(batch['images'])
        loss = v5_loss(predictions, batch['targets'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🎯 선택 가이드

### Which YOLO IoU to Use?

**For new projects**:
1. **YOLOv8** (CIoU + VFL) - Modern standard
2. **YOLOv10** (CIoU) - If NMS-free needed
3. **YOLOv7** (SIoU) - If speed priority

**For research**:
1. **YOLOv9** (GIoU variant) - Latest technology
2. **YOLOv10** (CIoU) - NMS-free exploration

**For legacy**:
1. **YOLOv5** (CIoU) - Stable and proven
2. **YOLOv7** (SIoU) - Good balance

### Performance Summary

| Version | IoU | Speed | Accuracy | Best For |
|---------|-----|-------|----------|----------|
| **v3** | MSE | ⚡️Fast | 🟡Medium | Legacy |
| **v4** | GIoU | 🟡Medium | 🟢Good | Research |
| **v5** | CIoU | 🟢Fast | 🟢Very Good | **Recommended** |
| **v6** | CIoU+VFL | 🟢Fast | 🟢Very Good | Precision |
| **v7** | SIoU | ⚡️Fastest | 🟢Very Good | Speed |
| **v8** | CIoU+VFL | 🟢Fast | 🟢Very Good | **Modern** |
| **v9** | GIoU+PGI | 🟢Fast | 🟢Excellent | Latest |
| **v10** | CIoU | ⚡️Fastest | 🟢Very Good | NMS-free |

---

## 📊 Comparison Chart

```
Speed vs Accuracy Trade-off

High Accuracy:  ● v9 (51.3 mAP)
                ● v7 (51.5 mAP)
                ● v8 (50.2 mAP)
                ● v10 (51.1 mAP)
                ● v5 (50.2 mAP)
                ● v6 (51.1 mAP)
                ● v4 (40.5 mAP)
                ● v3 (33.6 mAP)

High Speed:    ● v3 (105 FPS)
               ● v7 (210 FPS)
               ● v10 (320 FPS)
               ● v5 (185 FPS)
               ● v6 (195 FPS)
               ● v8 (480 FPS)
               ● v9 (310 FPS)
```

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO official papers, Ultralytics documentation, CVPR/ICCV proceedings*
