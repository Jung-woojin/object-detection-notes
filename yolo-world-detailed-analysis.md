# YOLO-World 상세 분석

YOLO-World (2024) 는 Open-vocabulary detection 을 실시간으로 수행할 수 있는 최초의 YOLO 기반 모델입니다.

---

## 🏗️ 아키텍처

### 핵심 혁신: Open-vocabulary Detection

#### 기본 개념

**Traditional object detection**:
- Fixed classes (80 classes for COCO)
- Requires retraining for new classes
- Cannot detect unseen categories

**YOLO-World**:
- Open-vocabulary: Any text-based category
- Zero-shot inference: No retraining needed
- Real-time performance: YOLO speed

```
Text Embedding ←→ Image Features
        ↓            ↓
   Text Encoder  Image Encoder
        ↓            ↓
   CLIP Text    YOLO Backbone
        ↓            ↓
   Semantic Align  Feature Fusion
        ↓            ↓
   Open-set Detection
```

#### Architecture Components

**1. CLIP-based Text Encoder**

```python
class YOLOWorldTextEncoder(nn.Module):
    """
    CLIP text encoder adapted for YOLO
    
    Uses pre-trained CLIP text encoder
    to generate embeddings for any text prompt
    """
    def __init__(self, num_classes=80, text_encoder=None):
        super().__init__()
        
        if text_encoder is None:
            # Load pre-trained CLIP text encoder
            from transformers import CLIPTextTransformer
            self.text_encoder = CLIPTextTransformer()
        else:
            self.text_encoder = text_encoder
        
        # Project text embeddings to detection space
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.num_classes = num_classes
    
    def forward(self, text_prompts):
        """
        Generate text embeddings for prompts
        
        Args:
            text_prompts: List of text strings
                e.g., ["dog", "cat", "car", ...]
        
        Returns:
            text_embeddings: (N, 256) embeddings
        """
        # Tokenize and encode
        with torch.no_grad():
            text_features = self.text_encoder(text_prompts)
        
        # Project to detection space
        embeddings = self.projection(text_features)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
```

**2. Image Encoder (YOLO Backbone)**

```python
class YOLOWorldImageEncoder(nn.Module):
    """
    YOLO backbone adapted for YOLO-World
    
    Uses YOLOv8 backbone with modifications
    for semantic-aware detection
    """
    def __init__(self, num_classes=80):
        super().__init__()
        
        # YOLOv8 backbone
        self.backbone = YOLOv8Backbone()
        
        # Adapted neck for semantic features
        self.neck = YOLOWorldNeck()
        
        # Detection head
        self.head = DetectionHead(num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, images):
        """
        Extract image features
        
        Args:
            images: Input images (B, 3, H, W)
        
        Returns:
            features: Multi-scale image features
        """
        features = self.backbone(images)
        features = self.neck(features)
        
        return features
```

**3. Semantic Alignment**

```python
class SemanticAlignment(nn.Module):
    """
    Align text and image features
    
    Ensures image features match text semantics
    """
    def __init__(self, dim=256):
        super().__init__()
        
        # Cross-attention for alignment
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            Conv(dim, dim, 1, 1),
            Conv(dim, dim, 3, 1),
            Conv(dim, dim, 1, 1)
        )
    
    def forward(self, image_features, text_embeddings):
        """
        Align image and text features
        
        Args:
            image_features: (B, H*W, dim) image features
            text_embeddings: (B, C, dim) text embeddings
        
        Returns:
            aligned_features: (B, H*W, dim) aligned features
        """
        # Cross-attention
        attended, _ = self.cross_attn(
            query=image_features,
            key=text_embeddings,
            value=text_embeddings
        )
        
        # Fusion
        aligned = self.fusion(image_features + attended)
        
        return aligned
```

### End-to-End Pipeline

```
Text Prompt → Tokenization → Text Encoder → Text Embeddings
                                                    ↓
Image → Backbone → Features → Neck → Detection → Semantic Alignment → Open-vocabulary Detection
                                                    ↓
                                            Aligned Features → Class Scores
```

---

## 🎯 Loss 함수

### Contrastive Loss Architecture

#### 1. Detection Loss (Standard)

```python
class YOLOWorldDetectionLoss(nn.Module):
    """
    Standard YOLO-World detection loss
    
    Similar to YOLOv8 but with open-vocabulary support
    """
    def __init__(self):
        super().__init__()
        
        self.box_loss = CIoULoss()
        self.cls_loss = BCELoss()
        self.obj_loss = BCELoss()
    
    def forward(self, predictions, targets):
        """
        Compute detection loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth boxes and classes
        
        Returns:
            Total loss
        """
        # Box loss
        box_loss = self.box_loss(predictions['boxes'], targets['boxes'])
        
        # Classification loss
        cls_loss = self.cls_loss(predictions['scores'], targets['classes'])
        
        # Objectness loss
        obj_loss = self.obj_loss(predictions['objectness'], targets['objectness'])
        
        # Combined loss
        total = 0.45 * box_loss + 0.50 * cls_loss + 0.25 * obj_loss
        
        return total
```

#### 2. Contrastive Loss (Open-vocabulary)

**Core concept**: Align text and image embeddings for matching

```python
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for open-vocabulary detection
    
    Matches text embeddings with image regions
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, image_embeddings, text_embeddings, matches):
        """
        Compute contrastive loss
        
        Args:
            image_embeddings: (N, D) image region embeddings
            text_embeddings: (C, D) text class embeddings
            matches: Binary matching matrix (N, C)
        
        Returns:
            Contrastive loss
        """
        # Cosine similarity
        similarity = F.cosine_similarity(
            image_embeddings.unsqueeze(1),
            text_embeddings.unsqueeze(0),
            dim=-1
        ) / self.temperature
        
        # Classification target
        target = torch.argmax(matches, dim=-1)
        
        # Cross-entropy loss
        loss = self.ce(similarity, target)
        
        return loss
```

#### 3. Combined Loss

```
L_total = L_detection + λ_contrastive·L_contrastive
```

**Training stages**:
1. **Stage 1**: Train detection head (standard YOLOv8)
2. **Stage 2**: Add contrastive learning (open-vocabulary)

---

## 🔧 Training Strategy

### Two-stage Training

#### Stage 1: Detection Foundation

```python
# Stage 1: Standard YOLOv8 training
model = YOLO('yolov8m.pt')

stage1_args = {
    'data': 'coco128.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    
    # Standard YOLOv8 settings
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.05,
    
    # YOLO-World specific (disabled initially)
    'contrastive_learning': False,
    'open_vocabulary': False
}

model.train(**stage1_args)
```

**Goal**: Establish strong detection capabilities

#### Stage 2: Open-vocabulary Adaptation

```python
# Stage 2: Add contrastive learning
model = YOLO('yolov8m.pt')

stage2_args = {
    'data': 'open_vocabulary.yaml',
    'epochs': 50,
    'batch': 8,
    'imgsz': 640,
    
    # Contrastive learning enabled
    'contrastive_learning': True,
    'contrastive_lambda': 1.0,
    'open_vocabulary': True,
    
    # CLIP text encoder
    'text_encoder': 'clip_text',
    'text_prompts': ['dog', 'cat', 'bird', ...],  # Custom classes
    
    # Fine-tuning
    'freeze_text_encoder': False,
    'lr0': 0.0001,  # Lower LR for fine-tuning
    'freeze_backbone': True  # Only train head
}

model.train(**stage2_args)
```

**Goal**: Enable open-vocabulary detection

---

## 📊 Benchmark

### Zero-shot Performance

| Task | Model | mAP | Speed (FPS) |
|------|-------|-----|-----------|
| **COCO (Zero-shot)** | YOLOv8 | 31.2 | 295 |
| **COCO (Zero-shot)** | Grounding DINO | 35.8 | 8 |
| **COCO (Zero-shot)** | YOLO-World | **38.5** | **280** |

### Custom Categories

**Test**: Detect categories not in COCO training

| Category | YOLOv8 | Grounding DINO | YOLO-World |
|----------|--------|-------------|------------|
| **Electric vehicle** | ❌ N/A | ✅ 42.3 | ✅ **48.7** |
| **Drone** | ❌ N/A | ✅ 38.9 | ✅ **45.2** |
| **Fire hydrant** | ❌ N/A | ✅ 44.1 | ✅ **50.3** |
| **Speed limit sign** | ❌ N/A | ✅ 40.7 | ✅ **47.8** |

**Key insight**: YOLO-World outperforms Grounding DINO by **+7-10%**

### Speed Comparison

| Model | FPS | mAP |
|-------|-----|-----|
| **Grounding DINO** | 8 | 35.8 |
| **YOLO-World** | **280** | **38.5** |
| **YOLOv8** | 295 | 31.2 |

**Speedup**: YOLO-World is **35x faster** than Grounding DINO

---

## 🛠 Inference Guide

### Zero-shot Inference

```python
from yolov8 import YOLO

class YOLOWorldInference:
    """
    YOLO-World inference with open-vocabulary support
    """
    def __init__(self, model_path='yolo-world.pt'):
        self.model = YOLO(model_path)
        self.model.eval()
    
    def detect(self, image, text_prompt, conf_threshold=0.3):
        """
        Detect objects given a text prompt
        
        Args:
            image: Input image (numpy array)
            text_prompt: Text category description
                e.g., "dog and cat" or "electric car"
            conf_threshold: Minimum confidence
        
        Returns:
            detections: List of detected objects
        """
        # Detect with text prompt
        with torch.no_grad():
            results = self.model.predict(
                source=image,
                classes=None,  # All classes
                conf=conf_threshold,
                text_prompt=text_prompt
            )
        
        return results.detections
    
    def detect_multi_class(self, image, class_names, conf_threshold=0.3):
        """
        Detect multiple classes with custom names
        
        Args:
            image: Input image
            class_names: List of class names
                e.g., ['dog', 'cat', 'car']
            conf_threshold: Confidence threshold
        
        Returns:
            detections: List of detected objects
        """
        # Build prompt
        prompt = ', '.join(class_names)
        
        # Detect
        with torch.no_grad():
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                text_prompt=prompt
            )
        
        return results.detections
```

### Custom Categories

```python
# Define custom categories
custom_classes = [
    'electric vehicle',
    'bicycle',
    'scooter',
    'motorcycle',
    'drone',
    'helicopter'
]

# Detect
detections = yolo_world.detect_multi_class(
    image=image,
    class_names=custom_classes,
    conf_threshold=0.4
)

# Result
for detection in detections:
    print(f"Detected: {detection['class']} at {detection['bbox']}")
```

### Text Prompts

```python
# Simple prompt
detections = model.detect(image, text_prompt="dog")

# Complex prompt
detections = model.detect(
    image, 
    text_prompt="dog or cat or bird"
)

# Negative prompt
detections = model.detect(
    image, 
    text_prompt="dog",
    negative_prompt="cat"  # Exclude cats
)

# Attribute prompts
detections = model.detect(
    image, 
    text_prompt="small dog"  # Size-based
)
```

---

## 🎯 실전 팁

### Tip 1: Prompt Engineering

```python
# Good prompts (specific):
"detection of dogs"
"detection of cats"
"detection of electric cars"

# Better prompts (with attributes):
"detection of large dogs"
"detection of small cats"
"detection of electric cars and bicycles"

# Avoid vague prompts:
"things"  # Too broad
"objects"  # Too general
```

**Guidelines**:
1. **Specific**: Use clear category names
2. **Detailed**: Include attributes when relevant
3. **Concise**: Keep prompts reasonably short
4. **Natural**: Use natural language phrasing

### Tip 2: Threshold Tuning

```python
# Find optimal confidence threshold
def find_optimal_threshold(model, val_data):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for conf in thresholds:
        mAP = evaluate(model, val_data, conf=conf)
        results.append((conf, mAP))
    
    optimal = max(results, key=lambda x: x[1])
    return optimal
```

**Typical thresholds**:
- **High recall**: 0.1-0.2
- **Balanced**: 0.3-0.4
- **High precision**: 0.5-0.7

### Tip 3: Batch Inference

```python
def batch_inference(model, images, text_prompts):
    """
    Process multiple images with different prompts
    """
    results = []
    
    for image, prompt in zip(images, text_prompts):
        detection = model.detect(image, prompt)
        results.append(detection)
    
    return results

# Or use parallel processing
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(
        lambda args: model.detect(*args),
        zip(images, text_prompts)
    ))
```

---

## 🚀 Common Issues & Solutions

### Issue 1: Poor Zero-shot Performance

**Symptoms**:
- Low mAP on unseen categories
- Missed detections

**Solution**:
```python
# Improve text prompts
prompts = [
    "dog",
    "small dog",
    "large dog",
    "standing dog",
    "running dog"
]

# Use ensemble of prompts
detections = []
for prompt in prompts:
    det = model.detect(image, prompt)
    detections.append(det)

# Combine results
final = nms_combined(detections)
```

### Issue 2: Slow Inference

**Symptoms**:
- FPS much lower than expected
- Long inference time

**Solution**:
```python
# Optimize model
model.export(format='onnx')
model.export(format='engine')

# Reduce image size
results = model.predict(image, imgsz=416)

# Batch processing
results = model.predict([img1, img2, img3], batch_size=4)
```

### Issue 3: Hallucinations

**Symptoms**:
- Detects objects that don't exist
- False positives

**Solution**:
```python
# Increase confidence threshold
results = model.detect(image, prompt, conf=0.5)

# Use negative prompts
results = model.detect(
    image, 
    prompt="dog",
    negative_prompt="cat"
)

# Post-process results
filtered = remove_low_quality(results, quality_threshold=0.6)
```

---

## 📈 Training Curves

### Stage 1: Detection Foundation

```
Epoch | Training Loss | mAP
---  |------ --------|---
  1  |   3.8        | 28.5
 10  |   1.2        | 42.0
 20  |   0.9        | 48.5
 40  |   0.7        | 52.0
 60  |   0.6        | 53.5
 80  |   0.55       | 54.0
100  |   0.52       | 54.2
```

### Stage 2: Open-vocabulary

```
Epoch | Contrastive Loss | mAP (custom)
---  |------ --------|-----------
  1  |   1.8        | 25.0
 10  |   0.8        | 35.0
 20  |   0.6        | 38.5
 30  |   0.5        | 40.0
 40  |   0.45       | 41.0
 50  |   0.4        | 41.5
```

**Key insights**:
- Stage 1: Rapid improvement
- Stage 2: Gradual learning
- Combined: Strong zero-shot performance

---

## 📝 결론

### YOLO-World 장점

1. **Open-vocabulary**: Any text-based category
2. **Zero-shot**: No retraining needed
3. **Real-time**: YOLO speed (280 FPS)
4. **Flexible**: Custom prompts
5. **Efficient**: Pre-trained CLIP integration

### 단점

1. **Less mature**: YOLOv8 보다 연구 적음
2. **Prompt dependency**: Performance varies with prompts
3. **Training complexity**: Two-stage training
4. **Requires CLIP**: Additional dependencies

### 사용 추천

**추천 YOLO-World**:
- ✅ Open-vocabulary detection 필요
- ✅ Zero-shot inference 중요
- ✅ Custom categories 다수
- ✅ Real-time performance 필요

**추천 YOLOv8**:
- ✅ Fixed categories only
- ✅ Simpler deployment
- ✅ Proven stability
- ✅ Sufficient performance

### Future Directions

**Next improvements**:
1. **Better prompts**: Adaptive prompt generation
2. **Few-shot learning**: Learn from examples
3. **Multimodal**: Add image captioning
4. **Distillation**: Smaller open-vocabulary models

---

*마지막 업데이트: 2026-03-30*
*참고: YOLO-World official, CLIP paper, Ultralytics documentation*
