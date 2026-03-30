# Object Detection Loss Functions

객체 탐지에서 사용되는 모든 Loss 함수를 모델별, YOLO 버전별로 정리합니다.

## 📚 목차

- [손실 함수 기본 개념](#-손실-함수-기본-개념)
- [One-Stage Models](#-one-stage-models)
  - [YOLO Series](#-yolo-series)
  - [SSD](#-ssd-single-shot-multibox-detector)
  - [RetinaNet](#-retinanet)
  - [FCOS](#-fcos-fully-convolutional-one-stage)
  - [CenterNet](#-centernet-object-as-points)
  - [EfficientDet](#-efficientdet)
  - [AnchorDet](#-anchordet)
- [Two-Stage Models](#-two-stage-models)
  - [Fast R-CNN](#-fast-r-cnn)
  - [Faster R-CNN](#-faster-r-cnn)
  - [Cascade R-CNN](#-cascade-r-cnn)
  - [Mask R-CNN](#-mask-r-cnn)
- [Transformer-based Models](#-transformer-based-models)
  - [DETR](#-detr-end-to-end-object-detection-with-transformers)
  - [Deformable DETR](#-deformable-detr)
  - [DINO](#-dino)
  - [RT-DETR](#-rt-detr)
  - [Sparse R-CNN](#-sparse-r-cnn)
- [Recent SOTA Models](#-recent-sota-models)
  - [Grounding DINO](#-grounding-dino)
  - [YOLOv10](#-yolov10)
  - [YOLO-World](#-yolo-world)
- [실전 코드](#-실전-코드)
- [Loss 비교 요약](#-loss-비교-요약)

---

## 🔧 손실 함수 기본 개념

객체 탐지에서 주로 사용되는 손실 함수 유형:

1. **Position Loss**: 박스 위치 정확도 (L1, L2, GIoU, DIoU, CIoU, SIoU)
2. **Size Loss**: 박스 크기 정확도 (L1, L2)
3. **Classification Loss**: 객체 분류 정확도 (Cross Entropy, Focal, BCE, VFL)
4. **Objectness Loss**: 객체 존재 확신도 (BCE, Focal)

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

## 🎯 One-Stage Models

### YOLO Series

| 버전 | Box Loss | Objectness | Classification | 특징 |
|------|------|----|--------|----|----|----|--|
| **v3** | MSE | BCE | BCE | Simple, basic |
| **v4** | GIoU | Focal | Focal | Better balance |
| **v5** | CIoU | BCE | BCE | Standard now |
| **v6** | CIoU | BCE | VFL | Varifocal |
| **v7** | SIoU | BCE | BCE | Fastest convergence |
| **v8** | CIoU | BCE | VFL | Modern standard |
| **v9** | GIoU | BCE | BCE | PGI, reusable labels |
| **v10** | CIoU | BCE | BCE | Latest version |
| **v11** | CIoU | BCE | VFL | Newest |

**YOLO-World**:
- **Box**: GIoU
- **Objectness**: BCE
- **Classification**: Contrastive Loss (Open-vocabulary)

**YOLOX**:
- **Box**: CIoU
- **Objectness**: BCE
- **Classification**: VFL
- **Label Assignment**: SimOTA

### SSD (Single Shot MultiBox Detector)

**논문**: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

**손실 구성**:
```python
L_total = λ_conf · L_conf + λ_loc · L_loc
```

#### 1. Localization Loss: Smooth L1

**Type**: Smooth L1 Loss (Huber Loss)

**수식**:
```
L_loc(x, l) = Σᵢ∈posSmoothL1(xᵢₓ - lᵢₓ) + Σᵢ∈negSmoothL1(xᵢₓ - lᵢₓ)
```

여기서 `Smooth L1`:
```
SmoothL1(x) = 0.5x²          if |x| < 1
             = |x| - 0.5    otherwise
```

**장점**:
- Gradient 가 0 이 아니면 선형 (outliers robust)
- Larger errors 에 대해 더 stable

#### 2. Confidence Loss: Cross Entropy

**Type**: Cross Entropy (Hard Negative Mining)

**수식**:
```
L_conf(x, c, l) = -Σₓᵢₗlog(cₓᵢₚ) - α·Σₓᵢₙlog(cₓᵢ₀)
```

여기서:
- `p`: Positive class
- `n`: Negative class (background)
- `α`: Hard negative mining weight (positive:negative = 1:3)

**특징**:
- **Hard Negative Mining**: Confident negatives 에만 집중
- **Multi-class**: 각 클래스별로 독립적 BCE

#### 특징

- **Anchor-based**: Default boxes 사용
- **Multi-scale**: 여러 resolution 에서 detection
- **Fast**: 45-50 FPS (GTX 1050 Ti)
- **mAP**: 25.7 on VOC 2007

**손실 weight**:
```yaml
loc: 1.0
conf: 1.0
```

---

### RetinaNet

**논문**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

**손실 구성**:
```python
L_total = α·L_norm + β·L_cls
```

#### 1. Localization Loss: Smooth L1

**Type**: Smooth L1 Loss

**수식**:
```
L_loc(x, l) = Σᵢ∈posSmoothL1(xᵢₓ - lᵢₓ) + Σᵢ∈negSmoothL1(xᵢₓ - lᵢₓ)
```

**Anchor-based**:
- 4 parameters: Δx, Δy, Δw, Δh
- **Scale-invariant**: Multi-scale anchors

#### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

**동기**: Class imbalance problem (foreground:background = 1:1000)

**수식**:
```
FL(p_t) = -α_t·(1 - p_t)ᵞ·log(p_t)
```

여기서:
- `p_t`: Model probability for true label
- `α_t`: Balance weight (0.25 for foreground, 0.75 for background)
- `γ`: Focusing parameter (2 recommended)

**장점**:
- **Hard examples**: Easy negatives 를 ignore
- **Better balance**: Dense object detection 에 최적화

#### 특징

- **Focal Loss**: Class imbalance 해결
- **Feature Pyramid**: P3-P5 multi-scale
- **Anchor-based**: Default boxes
- **Fast**: 40-50 FPS

**손실 weight**:
```yaml
loc: 1.0
conf: 1.0
alpha: 0.25
gamma: 2.0
```

---

### FCOS (Fully Convolutional One-Stage)

**논문**: [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

**손실 구성**:
```python
L_total = λ_cls·L_cls + λ_box·L_box + λ_ctr·L_ctr
```

#### 1. Classification Loss: Focal Loss

**Type**: Focal Loss

**수식**:
```
FL(p_t) = -α_t·(1 - p_t)ᵞ·log(p_t)
```

- **Anchor-free**: Point-based detection
- **One-to-one**: 각 pixel 이 하나의 객체 예측

#### 2. Box Loss: L1 Loss

**Type**: L1 Loss (Smooth L1)

**수식**:
```
L_box = Σₖ₌₁⁴ |xₖ - lₖ|
```

- **L, R, T, B**: Left, Right, Top, Bottom distances
- **Center-based**: Object center point 에서 distance 계산

#### 3. Centerness Loss: BCE

**Type**: Binary Cross Entropy

**동기**: Quality of predictions

**수식**:
```
L_ctr = -y·log(y_hat) - (1-y)·log(1-y_hat)
```

여기서:
- `y`: Ground truth centerness
- `y_hat`: Predicted centerness

**장점**:
- **Filter low-quality boxes**: Low centerness 제거
- **No NMS 필요**: Inference 후 post-processing 불필요

#### 특징

- **Anchor-free**: No default boxes
- **Simple**: 구현이 단순
- **Accurate**: mAP 40.5 on COCO
- **No NMS**: Inference 빠름

**손실 weight**:
```yaml
cls: 1.0
box: 5.0
ctr: 1.0
```

---

### CenterNet (Object as Points)

**논문**: [Objects as Points](https://arxiv.org/abs/1904.08616)

**손실 구성**:
```python
L_total = λ_hm·L_hm + λ_off·L_off + λ_dim·L_dim
```

#### 1. Heatmap Loss: Focal Loss

**Type**: Focal Loss

**동기**: Object detection as keypoint detection

**수식**:
```
FL(p_t) = -α_t·(1 - p_t)ᵞ·log(p_t)
```

- **Heatmap**: 각 pixel 이 객체 중심점일 확률
- **Multi-class**: 각 클래스별로 heatmap 생성

#### 2. Offset Loss: L1 Loss

**Type**: L1 Loss (Smooth L1)

**동기**: Sub-pixel precision

**수식**:
```
L_off = |xₖ - x̂ₖ| + |yₖ - ŷₖ|
```

- **Sub-pixel**: Pixel level 에서 정밀 위치
- **8×8 grid**: 각 cell 에서 offset 예측

#### 3. Dimension Loss: L1 Loss

**Type**: L1 Loss

**수식**:
```
L_dim = |wₖ - ŵₖ| + |hₖ - ĥₖ| + |dₖ - d̂ₖ|
```

- **Object size**: Width, Height, Depth
- **3D detection**: Depth estimation 가능

#### 특징

- **Object as Points**: Key point detection
- **Single network**: Simple architecture
- **3D support**: Depth estimation 가능
- **Fast**: Real-time 가능

**손실 weight**:
```yaml
hm: 1.0
off: 0.1
dim: 0.1
```

---

### EfficientDet

**논문**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

**손실 구성**:
```python
L_total = λ_cls·L_cls + λ_box·L_box
```

#### 1. Classification Loss: Focal Loss

**Type**: Focal Loss

**수식**:
```
FL(p_t) = -α_t·(1 - p_t)ᵞ·log(p_t)
```

- **Efficient backbone**: MobileNet, ResNet
- **Efficient PANet**: Bi-directional feature pyramid

#### 2. Box Loss: Smooth L1

**Type**: Smooth L1 Loss

**수식**:
```
L_box = Σᵢ∈posSmoothL1(xᵢₓ - lᵢₓ) + Σᵢ∈negSmoothL1(xᵢₓ - lᵢₓ)
```

**Anchor-based**:
- **AutoAnchor**: AutoML-driven anchor design
- **AutoShape**: Automatic input resolution scaling

#### 특징

- **Efficient backbone**: MobileNetV2, ResNet
- **Efficient PANet**: Bi-directional connections
- **AutoML**: EfficientDet-d0 ~ d7 자동 scaling
- **Fast & Accurate**: 50-60 FPS, mAP 48.8

**손실 weight**:
```yaml
cls: 1.0
box: 5.0
```

---

### AnchorDet

**논문**: [AnchorDet: Anchor-free Object Detection with Centerness-Aware Anchor Points](https://arxiv.org/abs/2003.06667)

**손실 구성**:
```python
L_total = λ_cls·L_cls + λ_box·L_box + λ_ctr·L_ctr
```

#### 1. Classification Loss: Focal Loss

**Type**: Focal Loss

#### 2. Box Loss: CIoU

**Type**: Complete IoU

**동기**: Better convergence with aspect ratio

**수식**:
```
L_CIoU = 1 - CIoU
```

#### 3. Centerness Loss: BCE

**Type**: Binary Cross Entropy

**동기**: Center-aware anchor points

**수식**:
```
L_ctr = -y·log(y_hat) - (1-y)·log(1-y_hat)
```

#### 특징

- **Anchor-free**: No default boxes
- **Centerness-aware**: Better localization
- **Simple**: Single-stage detection

---

## 🔄 Two-Stage Models

### Fast R-CNN

**논문**: [Fast R-CNN](https://arxiv.org/abs/1504.08083)

**손실 구성**:
```python
L_total = L_cls + λ·L_loc
```

#### 1. Classification Loss: Softmax

**Type**: Multiclass Softmax

**수식**:
```
L_cls = -log(p_c)
```

여기서:
- `p_c`: Predicted probability for true class
- **Multi-class**: 80 COCO classes + background

#### 2. Localization Loss: Smooth L1

**Type**: Smooth L1 Loss

**수식**:
```
L_loc(x, l) = Σᵢ∈posSmoothL1(xᵢₓ - lᵢₓ) + Σᵢ∈negSmoothL1(xᵢₓ - lᵢₓ)
```

- **R-ION**: Region of Interest pooling
- **Bounding box regression**: Fine-tuning

#### 특징

- **Two-stage**: RPN (Region Proposal Network) + R-ION
- **Fast**: 200 FPS (shared convolution)
- **Accurate**: mAP 58.4 on VOC 2007

**손실 weight**:
```yaml
cls: 1.0
box: 1.0
```

---

### Faster R-CNN

**논문**: [Faster R-CNN: Towards Real-time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

**손실 구성**:
```python
L_total = L_cls + λ·L_loc
```

#### 1. Classification Loss: Softmax

**Type**: Multiclass Softmax

#### 2. Localization Loss: Smooth L1

**Type**: Smooth L1 Loss

#### RPN (Region Proposal Network)

**Loss**:
```python
L_rpn = L_cls + λ·L_loc
```

- **Objectness**: Foreground/background classification
- **Box regression**: Proposal refinement

#### 특징

- **Two-stage**: RPN + Fast R-CNN
- **Near real-time**: 7 FPS (VGG16)
- **State-of-the-art**: mAP 56.3 on VOC 2007

**손실 weight**:
```yaml
cls: 1.0
box: 1.0
λ_rpn: 1.0
```

---

### Cascade R-CNN

**논문**: [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1906.06575)

**손실 구성**:
```python
L_total = Σᵢ₌₁ⁿ(L_cls + λᵢ·L_boxᵢ)
```

#### Architecture: Cascade of Detectors

**Stage 1**:
- **IoU threshold**: 0.5
- **Box Loss**: Smooth L1

**Stage 2**:
- **IoU threshold**: 0.6
- **Box Loss**: IoU Loss

**Stage 3**:
- **IoU threshold**: 0.7
- **Box Loss**: GIoU Loss

#### 특징

- **Cascade**: Progressive refinement
- **Multi-stage**: Each stage uses higher IoU threshold
- **High quality**: mAP 47.1 on COCO

---

### Mask R-CNN

**논문**: [Mask R-CNN](https://arxiv.org/abs/1703.06870)

**손실 구성**:
```python
L_total = L_cls + L_box + L_mask + L_rpn
```

#### 1. Classification Loss: Softmax

**Type**: Multiclass Softmax

#### 2. Box Loss: Smooth L1

**Type**: Smooth L1 Loss

#### 3. Mask Loss: Binary Cross Entropy

**Type**: Binary Cross Entropy

**동기**: Instance segmentation

**수식**:
```
L_mask = -Σₖ[y·log(p) + (1-y)·log(1-p)]
```

- **Per-pixel**: 각 pixel 이 mask 인 확률
- **Per-instance**: 각 object 인스턴스별로 mask

#### 특징

- **Two-stage**: Faster R-CNN + Mask head
- **Instance segmentation**: Pixel-level mask
- **Parallel heads**: Classification, Box, Mask

---

## 🤖 Transformer-based Models

### DETR (End-to-End Object Detection with Transformers)

**논문**: [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

**손실 구성**:
```python
L_total = λ_box·(λ₁·L1 + λ₂·L_GIoU) + λ_cls·L_focal + L_matching
```

#### 1. Box Loss: L1 + GIoU

**Type**: L1 Loss + GIoU Loss

**수식**:
```
L_box = λ₁·L1 + λ₂·L_GIoU
```

- **L1**: Absolute error
- **GIoU**: Overlap-based loss
- **Weight**: L1: 5, GIoU: 2

#### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

**동기**: Class imbalance problem

**수식**:
```
FL(p_t) = -α·(1 - p_t)ᵞ·log(p_t)
```

#### 3. Matching Loss: Hungarian Matching

**Type**: Bipartite Matching

**동기**: One-to-one matching problem

**수식**:
```
L_matching = min_π Σᵢ₌₁ⁿ⁻¹ Cᵢπ(ᵢ)
```

여기서 `Cᵢⱼ` 는 cost matrix:
- **Classification cost**: Focal loss
- **Box cost**: L1 + GIoU

#### 특징

- **End-to-end**: No anchor needed
- **Transformer**: Self-attention mechanism
- **Bipartite matching**: One-to-one matching
- **Slow convergence**: 100 epochs 필요

**손실 weight**:
```yaml
box: 5.0
conf: 2.0
cls: 1.0
λ1: 5.0
λ2: 2.0
```

---

### Deformable DETR

**논문**: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

**손실 구성**: DETR 와 동일

```python
L_total = λ_box·(λ₁·L1 + λ₂·L_GIoU) + λ_cls·L_focal + L_matching
```

#### 특징

- **Deformable attention**: Faster convergence
- **Multi-scale**: Hierarchical features
- **Fast convergence**: 12 epochs
- **High resolution**: Better small object detection

---

### DINO (DETR with Improved DEterministic Queries)

**논문**: [DINO: DETR with Improved DEterministic Queries](https://arxiv.org/abs/2203.03605)

**손실 구성**:
```python
L_total = λ_box·(λ₁·L_GIoU + λ₂·L_IoU) + λ_cls·L_focal + L_matching + L_denoising
```

#### 1. Box Loss: GIoU + IoU Combined

**Type**: GIoU + IoU Loss

**수식**:
```
L_box = λ₁·L_GIoU + λ₂·L_IoU
```

- **GIoU**: Convergence
- **IoU**: Precision
- **Combined**: Better performance

#### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

#### 3. Contrastive Denoising Loss

**Type**: Contrastive Loss

**동기**: Training stabilization

**동기**:
- **Denoising queries**: Noisy training data
- **Contrastive learning**: Query separation

#### 특징

- **Contrastive denoising**: Training stabilization
- **Better query initialization**: Faster convergence
- **High performance**: mAP 56.0 (COCO)
- **Fast**: 90 FPS

---

### RT-DETR (Real-time DETR)

**논문**: [RT-DETR: Rethinking Real-Time DETR](https://arxiv.org/abs/2304.01318)

**손실 구성**:
```python
L_total = λ_box·(λ₁·L_DIoU + λ₂·L1) + λ_cls·L_focal + L_matching
```

#### 1. Box Loss: DIoU + L1

**Type**: DIoU + L1 Loss

**수식**:
```
L_box = λ₁·L_DIoU + λ₂·L1
```

- **DIoU**: Distance-aware
- **L1**: Absolute error

#### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

#### 특징

- **CNN backbone**: Faster inference
- **DIoU**: Faster convergence
- **Real-time**: 135 FPS
- **High accuracy**: mAP 53.0

---

### Sparse R-CNN

**논문**: [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12455)

**손실 구성**:
```python
L_total = L_cls + λ_box·L_box
```

#### 1. Box Loss: L1 + GIoU

**Type**: L1 + GIoU

#### 2. Classification Loss: Focal Loss

**Type**: Focal Loss

#### 특징

- **Learnable proposals**: Instead of RPN
- **Sparse**: Fewer queries (100 instead of 300)
- **Faster**: 30 FPS
- **Accurate**: mAP 46.0

---

## 🚀 Recent SOTA Models

### Grounding DINO

**논문**: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2305.14312)

**손실 구성**:
```python
L_total = L_detection + L_grounding
```

#### 1. Detection Loss: DETR-style

**Type**: L1 + GIoU + Focal Loss

#### 2. Grounding Loss: Contrastive Loss

**Type**: Contrastive Loss (Vision-Language)

**동기**: Open-set detection

**수식**:
```
L_grounding = Contrastive Loss(image, text)
```

- **Vision-language**: Text encoder 와 연결
- **Open-set**: 새로운 class 도 detection 가능

#### 특징

- **Open-set**: Text prompts 기반 detection
- **Zero-shot**: Training 없는 inference
- **Grounding DINO**: DINO + CLIP
- **Versatile**: 다양한 text descriptions

---

### YOLOv10

**논문**: [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)

**손실 구성**:
```python
L_total = λ_box·L_box + λ_obj·L_obj + λ_cls·L_cls
```

#### 1. Box Loss: CIoU

**Type**: CIoU

#### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

#### 3. Classification Loss: BCE

**Type**: Binary Cross Entropy

#### 특징

- **NMS-free**: Label assignment 없음
- **Dual-labeling**: Simpler inference
- **Efficient**: No post-processing
- **State-of-the-art**: Faster than YOLOv8/v9

---

### YOLO-World

**논문**: [YOLO-World: Real-time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270)

**손실 구성**:
```python
L_total = L_box + L_obj + L_cls
```

#### 1. Box Loss: GIoU

**Type**: GIoU

#### 2. Objectness Loss: BCE

**Type**: Binary Cross Entropy

#### 3. Classification Loss: Contrastive Loss

**Type**: Contrastive Loss (Open-vocabulary)

**동기**: Open-vocabulary detection

**동기**:
- **Semantic embeddings**: Text encoder 와 image encoder 연결
- **Open-vocabulary**: 새로운 class 도 detection 가능
- **Zero-shot**: Training 없는 inference 가능

#### 특징

- **Open-vocabulary**: 새로운 class 추가 가능
- **Contrastive learning**: Semantic understanding
- **Zero-shot**: Pre-trained models 활용
- **Real-time**: Fast inference

---

## 🛠 실전 코드

### Faster R-CNN Style Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastRCNNLoss(nn.Module):
    """Faster R-CNN Loss: Classification + Box Regression"""
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.loc_criterion = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, cls_logits, box_preds, gt_labels, gt_boxes, pos_indices):
        """
        cls_logits: (N, num_classes)
        box_preds: (N, 4)
        gt_labels: (N,)
        gt_boxes: (N, 4)
        pos_indices: list of positive indices
        """
        # Classification loss
        cls_loss = self.cls_criterion(cls_logits, gt_labels)
        
        # Box regression loss (positive samples only)
        box_loss = 0
        for idx in pos_indices:
            pred = box_preds[idx]
            gt = gt_boxes[idx]
            loss = self.loc_criterion(pred, gt)
            box_loss += loss.sum()
        
        box_loss = box_loss / max(len(pos_indices), 1)
        
        return cls_loss + box_loss
```

### FCOS Loss

```python
class FCOSLoss(nn.Module):
    """FCOS Loss: Classification + Box + Centerness"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, cls_preds, box_preds, ctr_preds, gt_labels, gt_boxes):
        """
        cls_preds: (N, C) classification
        box_preds: (N, 4) box (L, R, T, B)
        ctr_preds: (N,) centerness
        gt_labels: (N,)
        gt_boxes: (N, 4)
        """
        # Focal Loss for classification
        pos_mask = gt_labels >= 0
        pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]
        
        # Classification loss (Focal)
        cls_loss = self._focal_loss(cls_preds, gt_labels)
        
        # Box loss (L1) for positive samples
        box_loss = 0
        for idx in pos_indices:
            pred = box_preds[idx]
            gt = gt_boxes[idx]
            box_loss += F.l1_loss(pred, gt)
        box_loss /= max(len(pos_indices), 1)
        
        # Centerness loss
        ctr_loss = F.binary_cross_entropy(ctr_preds[pos_mask], ctr_gt)
        
        return cls_loss + box_loss + ctr_loss
    
    def _focal_loss(self, preds, labels):
        pos_mask = labels >= 0
        pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]
        
        focal_weight = (1 - F.softmax(preds[pos_indices], dim=1)) ** self.focal_gamma
        cls_loss = -focal_weight * F.log_softmax(preds[pos_indices], dim=1) * (labels[pos_indices] >= 0).float()
        
        return cls_loss.sum() / max(len(pos_indices), 1)
```

### CenterNet Loss

```python
class CenterNetLoss(nn.Module):
    """CenterNet Loss: Heatmap + Offset + Size"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, hm_preds, off_preds, dim_preds, gt_hm, gt_off, gt_dim):
        """
        hm_preds: (N, C, H, W) heatmap
        off_preds: (N, 2, H, W) offsets
        dim_preds: (N, 3, H, W) dimensions
        gt_hm: (N, C, H, W) ground truth heatmap
        gt_off: (N, 2, H, W) ground truth offsets
        gt_dim: (N, 3, H, W) ground truth dimensions
        """
        # Focal Loss for heatmap
        hm_loss = self._focal_loss(hm_preds, gt_hm)
        
        # L1 Loss for offsets and dimensions
        off_loss = F.l1_loss(off_preds, gt_off)
        dim_loss = F.l1_loss(dim_preds, gt_dim)
        
        return hm_loss + 0.1 * off_loss + 0.1 * dim_loss
    
    def _focal_loss(self, preds, targets):
        pos_mask = targets > 0
        focal_weight = (1 - F.softmax(preds, dim=1)) ** self.focal_gamma
        
        pred_pos = preds[pos_mask]
        target_pos = targets[pos_mask]
        
        pos_loss = -focal_weight * torch.log(pred_pos + 1e-8) * target_pos
        neg_loss = -(1 - focal_weight) * torch.log(1 - pred_pos + 1e-8) * (1 - target_pos)
        
        return (pos_loss + neg_loss).sum() / (pos_mask.sum() + 1)
```

### Sparse R-CNN Loss

```python
class SparseRCNNLoss(nn.Module):
    """Sparse R-CNN Loss: Classification + Box Regression"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, cls_preds, box_preds, gt_labels, gt_boxes):
        """
        cls_preds: (N, C) classification
        box_preds: (N, 4) boxes
        gt_labels: (N,)
        gt_boxes: (N, 4)
        """
        # Hungarian matching
        cost_matrix = self._compute_cost(cls_preds, box_preds, gt_labels, gt_boxes)
        indices = torch.linear_assignment(cost_matrix)
        
        # Classification loss (Focal) for matched pairs
        matched_indices = indices[0]
        matched_labels = gt_labels[indices[1]]
        matched_preds = cls_preds[matched_indices]
        
        focal_weight = (1 - F.softmax(matched_preds, dim=1)) ** self.focal_gamma
        cls_loss = -focal_weight * torch.log(matched_preds + 1e-8) * matched_labels
        
        # Box loss (L1 + GIoU) for matched pairs
        box_loss = F.l1_loss(box_preds[matched_indices], gt_boxes[indices[1]])
        
        return cls_loss.sum() + box_loss
    
    def _compute_cost(self, cls_preds, box_preds, gt_labels, gt_boxes):
        # Focal cost
        focal_cost = -torch.log(cls_preds + 1e-8) * gt_labels
        # Box cost (L1)
        box_cost = F.l1_loss(box_preds, gt_boxes)
        
        return focal_cost + 5.0 * box_cost
```

---

## 📊 Loss 비교 요약

### One-Stage Models

| Model | Box Loss | Classification | Speed | Accuracy | Type |
|------|------|----|----|----|--|
| **YOLOv3** | MSE | BCE | ⚡️빠름 | 🟡보통 | Anchor-based |
| **YOLOv5** | CIoU | BCE | 🟢빠름 | 🟢좋음 | Anchor-based |
| **YOLOv7** | SIoU | BCE | ⚡️가장 빠름 | 🟢매우 좋음 | Anchor-based |
| **YOLOv8** | CIoU | VFL | 🟢빠름 | 🟢매우 좋음 | Anchor-based |
| **YOLOv9** | GIoU | BCE | 🟢빠름 | 🟢매우 좋음 | Anchor-based |
| **YOLOX** | CIoU | VFL | 🟢빠름 | 🟢좋음 | Anchor-based |
| **YOLO-World** | GIoU | Contrastive | 🟡보통 | 🟢매우 좋음 | Anchor-based |
| **SSD** | Smooth L1 | CE | ⚡️빠름 | 🟡보통 | Anchor-based |
| **RetinaNet** | Smooth L1 | Focal | 🟢빠름 | 🟢좋음 | Anchor-based |
| **FCOS** | L1 | Focal | 🟢빠름 | 🟢매우 좋음 | Anchor-free |
| **CenterNet** | L1 | Focal | 🟢빠름 | 🟢매우 좋음 | Anchor-free |
| **EfficientDet** | Smooth L1 | Focal | 🟢빠름 | 🟢매우 좋음 | Anchor-based |

### Two-Stage Models

| Model | Box Loss | Classification | Speed | Accuracy | Type |
|------|------|----|----|----|--|
| **Fast R-CNN** | Smooth L1 | Softmax | 🟡보통 | 🟢좋음 | Two-stage |
| **Faster R-CNN** | Smooth L1 | Softmax | 🟡보통 | 🟢매우 좋음 | Two-stage |
| **Cascade R-CNN** | GIoU | Softmax | 🐢느림 | 🟢매우 좋음 | Two-stage |
| **Mask R-CNN** | Smooth L1 | Softmax | 🐢느림 | 🟢매우 좋음 | Two-stage |

### Transformer-based Models

| Model | Box Loss | Classification | Speed | Accuracy | Type |
|------|------|----|----|----|--|
| **DETR** | L1+GIoU | Focal | 🐢느림 | 🟢매우 좋음 | Transformer |
| **Deformable DETR** | L1+GIoU | Focal | 🟡보통 | 🟢매우 좋음 | Transformer |
| **DINO** | GIoU+IoU | Focal | 🟡보통 | 🟢매우 좋음 | Transformer |
| **RT-DETR** | DIoU+L1 | Focal | ⚡️빠름 | 🟢매우 좋음 | Transformer |
| **Sparse R-CNN** | L1+GIoU | Focal | 🟡보통 | 🟢매우 좋음 | Transformer |

### SOTA Models

| Model | Box Loss | Classification | Speed | Accuracy | Features |
|------|------|----|----|----|----|
| **Grounding DINO** | L1+GIoU | Contrastive | 🟡보통 | 🟢매우 좋음 | Open-set |
| **YOLOv10** | CIoU | BCE | ⚡️빠름 | 🟢매우 좋음 | NMS-free |
| **YOLOv11** | CIoU | VFL | ⚡️빠름 | 🟢매우 좋음 | Newest |

---

## 🎯 결론

### Model Type 별 추천

**1. Real-time Applications**:
- ✅ **YOLOv7/v8/v9** (Fastest convergence)
- ✅ **RT-DETR** (Transformer + Real-time)
- ✅ **YOLOv10** (NMS-free)

**2. High Accuracy**:
- ✅ **Cascade R-CNN** (Two-stage)
- ✅ **DINO** (Transformer, best performance)
- ✅ **Grounding DINO** (Open-set)

**3. Simple Implementation**:
- ✅ **FCOS** (Anchor-free, simple)
- ✅ **CenterNet** (Object as points)
- ✅ **YOLOv5** (Standard baseline)

**4. Research/Open-vocabulary**:
- ✅ **Grounding DINO** (Text prompts)
- ✅ **YOLO-World** (Open-vocabulary)
- ✅ **Sparse R-CNN** (Learnable proposals)

### Loss Type 별 비교

| Loss Type | Convergence | Accuracy | Computation | Best for |
|------|----|----|----|----|
| **MSE** | ⚡️빠름 | 🟡보통 | ⚡️빠름 | YOLOv3 (deprecated) |
| **Smooth L1** | 🟢빠름 | 🟢좋음 | ⚡️빠름 | SSD, Faster R-CNN |
| **L1** | 🟢빠름 | 🟢매우 좋음 | ⚡️빠름 | FCOS, CenterNet |
| **GIoU** | 🟡보통 | 🟢매우 좋음 | 🟡보통 | YOLOv4, YOLOv9 |
| **DIoU** | 🟢빠름 | 🟢매우 좋음 | ⚡️빠름 | RT-DETR |
| **CIoU** | 🟢빠름 | 🟢매우 좋음 | ⚡️빠름 | YOLOv5/v8/v10 |
| **SIoU** | ⚡️가장 빠름 | 🟢매우 좋음 | 🟡보통 | YOLOv7 |

---
*마지막 업데이트: 2026-03-30*
*참고: YOLO 공식 GitHub, 관련 논문, CVPR/ICCV/ECCV proceedings*