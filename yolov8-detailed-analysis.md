# YOLOv8 상세 분석

YOLOv8 (2023) 은 Ultralytics 에서 발표한 최신 YOLO 버전으로, Anchor-free 방식과 Decoupled Head 를 도입했습니다.

---

## 🏗️ 아키텍처

### 기본 구조

```
Input (3×640×640)
  ↓
Backbone (CSPDarknet)
  ├── P3 (160, 160, 160)
  ├── P4 (320, 80, 320)
  └── P5 (640, 40, 640)
  ↓
SPPF (Spatial Pyramid Pooling Fast)
  ↓
Neck (PANet - Top-down + Bottom-up)
  ├── P3_out (160)
  ├── P4_out (320)
  └── P5_out (640)
  ↓
Head (Decoupled: Classification + Regression)
  ├── Class Branch
  └── Reg Branch
  ↓
Output (84×84×85)
```

### 주요 구성요소

#### 1. Backbone: CSPDarknet

```python
# CSPBlock 구현 (CSP = Cross Stage Partial)
class CSPBlock(nn.Module):
    def __init__(self, c1, c2):
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = Bottlenecks(c1, c2)  # Bottleneck 블록
        self.cv3 = Conv(2*c2, c2, 1, 1)
    
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))
```

#### 2. SPPF (Spatial Pyramid Pooling Fast)

```python
# 기존 SPP의 느린 maxpool2d 대체
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        self.cv1 = Conv(c1, c2//2, 1, 1)
        self.cv2 = Conv(c2//2, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.m(x), self.m(self.m(x))], dim=1))
```

#### 3. Decoupled Head

```python
# Classification + Regression 분리
class DetectionHead(nn.Module):
    def __init__(self, nc, anchors=84):
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.no = nc + 5  # predictions per anchor
        self.stride = torch.zeros(self.nl)
        
        # Classification
        self.cv1 = Conv(256, 256, 1, 1)
        self.cv2 = Conv(256, nc, 1, 1)  # class prediction
        
        # Regression
        self.cv3 = Conv(256, 256, 1, 1)
        self.cv4 = Conv(256, 4, 1, 1)   # bbox prediction
    
    def forward(self, x):
        c = self.cv1(x)
        box = self.cv4(c)
        cls = self.cv2(c)
        return torch.cat([box, cls], dim=1)
```

---

## 🎯 Loss 함수

### Composite Loss

```
L_total = λ_box·L_box + λ_cls·L_cls + λ_obj·L_obj
```

#### 1. Box Loss: CIoU Loss

```
L_box = 1 - CIoU + ρ²(b,b_gt)/c² + α·v
CIoU = IoU - ρ²(b,b_gt)/c² - α·v
v = (4/π²) · (arctan(w_gt/h_gt) - arctan(w/h))²
```

**구성 요소:**
- **IoU**: Overlap ratio
- **ρ²**: Center distance penalty
- **c²**: Diagonal distance
- **v**: Aspect ratio consistency
- **α**: Weight parameter

#### 2. Classification Loss: DFL + BCE

**Distribution Focal Loss (DFL):**
```python
def DFL(c1, c2):
    # bounding box distribution
    return F.cross_entropy(c1.view(-1, 4), c2)
```

**Binary Cross Entropy (BCE):**
```python
def BCE(pred, target):
    return F.binary_cross_entropy(pred, target, reduction='none')
```

#### 3. Objectness Loss: VFL (Variable Focal Loss)

```
L_obj = -α·(1-p)ᵞ·log(p) - (1-α)·pᵞ·log(1-p)
```

**Dynamic Label Assignment:**
- Task-Aware Classification
- Adaptive positive/negative samples

---

## 🔧 Hyperparameters

### Recommended Settings

**COCO Training:**
```python
# YOLOv8m configuration
model = YOLO('yolov8m.yaml')

train_args = {
    'data': 'coco128.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'lr0': 0.01,
    'lrf': 0.01,  # final LR = lr0 * lrf
    'optimizer': 'AdamW',
    'weight_decay': 0.05,
    'warmup_epochs': 3.0,
    'patience': 50,  # early stopping
    'preserve_checkpoint': True
}

results = model.train(**train_args)
```

**학습률 스케줄:**
```
LR = lr0 * (1 - epoch/(max_epochs-1))^2.0
Initial: 0.01
Final: 0.0001 (lr0 * lrf)
```

---

## ⚡ Inference Optimization

### 1. TensorRT Export

```bash
# YOLOv8 TensorRT export
yolo export model=yolov8m.pt format=engine device=0 half=True

# Inference
results = model.predict(source='image.jpg', engine=yolov8m.engine)
```

**속도 향상:**
- FP32: 1.5~2x
- FP16: 2~3x

### 2. ONNX Export

```bash
# ONNX export
yolo export model=yolov8m.pt format=onnx opset=12

# Python inference
from onnxruntime import InferenceSession
session = InferenceSession('yolov8m.onnx')
```

### 3. Batch Inference

```python
# Multiple images
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.predict(source=images, batch_size=4)
```

**Throughput:**
- Single: ~5 FPS
- Batch=4: ~15 FPS

---

## 📊 Benchmark

### GPU Performance (TensorRT FP16)

| Model | mAP | FPS (2080 Ti) | FPS (A100) |
|-------|-----|-------------|----------|
| YOLOv8n | 37.3 | 580 | 1200 |
| YOLOv8s | 44.9 | 420 | 900 |
| YOLOv8m | 50.2 | 260 | 550 |
| YOLOv8l | 52.9 | 160 | 340 |
| YOLOv8x | 53.9 | 105 | 220 |

### CPU Performance

| Model | FPS (8-core) |
|-------|-------------|
| YOLOv8n | 95 |
| YOLOv8s | 75 |
| YOLOv8m | 45 |
| YOLOv8l | 30 |
| YOLOv8x | 20 |

---

## 🛠️ Fine-tuning Guide

### 1. Custom Dataset Preparation

**YOLO Format:**
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

**data.yaml:**
```yaml
path: ../datasets/custom
train: images/train
val: images/val

nc: 3  # number of classes
names: ['class0', 'class1', 'class2']
```

**Label Format:**
```
<class_id> <x_center> <y_center> <width> <height>
0 0.5 0.5 0.3 0.4
```

### 2. Transfer Learning

```python
# Load pre-trained model
model = YOLO('yolov8m.pt')

# Freeze backbone
for i, param in enumerate(model.model[:10]):
    param.requires_grad = False

# Train
results = model.train(
    data='custom.yaml',
    epochs=50,
    pretrained=True
)
```

### 3. Data Augmentation

```python
# YOLOv8 augmentations
augmentations = {
    'mixup': 0.15,      # blend two images
    'copy_paste': 0.15, # paste object
    'mosaic': 1.0,      # 4-image mosaic (disabled in last 10 epochs)
    'hsv_h': 0.015,     # hue
    'hsv_s': 0.7,       # saturation
    'hsv_v': 0.4,       # value
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.0,
    'fliplr': 0.5,
}
```

---

## 🎯 실전 팁

### Tips 1: Small Object Detection

**Strategy:**
```python
# Increase input size
results = model.predict(source='image.jpg', imgsz=1280)

# Use smaller anchor boxes
model.overrides['anchors'] = [
    [10,13, 16,30, 33,23],  # P3 - small
    [30,61, 62,45, 59,119], # P4 - medium
    [116,90, 156,198, 373,326] # P5 - large
]
```

### Tips 2: Class Imbalance

```python
# Use class weights
class_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
results = model.train(
    data='custom.yaml',
    weights=class_weights
)
```

### Tips 3: Overfitting Prevention

```python
# Early stopping
results = model.train(
    patience=30,  # stop if no improvement for 30 epochs
    save=True,
    plot=True
)
```

---

## 📈 Training Analysis

### Loss Trends

**Typical training curve:**
```
Epoch | Box Loss | Cls Loss | Obj Loss | Total
------|----------|----------|----------|-------
  1   |  4.5     |  3.2     |  2.1     |  9.8
 10   |  1.2     |  0.8     |  0.6     |  2.6
 20   |  0.8     |  0.5     |  0.4     |  1.7
 50   |  0.6     |  0.4     |  0.3     |  1.3
100   |  0.5     |  0.3     |  0.3     |  1.1
```

**Expected convergence:**
- Box loss: stabilizes by epoch 20-30
- Class loss: stabilizes by epoch 30-50
- Total loss: stable after epoch 50

---

## 🚀 Common Issues & Solutions

### Issue 1: Overfitting

**Symptoms:**
- Training mAP > Validation mAP
- Loss curves diverge

**Solutions:**
1. Increase data augmentation
2. Use early stopping
3. Reduce model size
4. Add dropout

### Issue 2: Poor Small Object Detection

**Symptoms:**
- Small objects missed
- Low AP for small objects

**Solutions:**
1. Increase imgsz to 1280
2. Add P2 detection head
3. Use larger batch size
4. Increase training epochs

### Issue 3: Slow Training

**Solutions:**
1. Use larger batch size (if GPU memory allows)
2. Enable AMP (Automatic Mixed Precision)
3. Use gradient accumulation
4. Optimize with TensorRT

---

*최종 수정일: 2026 년 3 월*
*Created for deep understanding of YOLOv8*
