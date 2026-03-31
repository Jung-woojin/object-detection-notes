# 실전 세팅 가이드

객체검출 모델을 실제 환경에 배포하기 위한 최적화, export, 그리고 배포 전략을 다룹니다.

---

## 📚 목차

- [Inference 최적화](#-inference-최적화)
- [Model Export](#-model-export)
- [Deployment Strategies](#-deployment-strategies)
- [Hardware 최적화](#-hardware-최적화)
- [Memory 최적화](#-memory-최적화)
- [실전 팁](#-실전-팁)

---

## ⚡ Inference 최적화

### 1. TensorRT 최적화

#### YOLOv8 TensorRT Export

```bash
# FP16 Export (recommended)
yolo export model=yolov8m.pt format=engine device=0 half=True

# FP32 Export (maximum compatibility)
yolo export model=yolov8m.pt format=engine device=0

# INT8 Quantization (requires calibration)
yolo export model=yolov8m.pt format=engine device=0 int8 calib=data
```

#### Python TensorRT Inference

```python
import tensorrt as trt
import cv2
import numpy as np

class TRTInference:
    def __init__(self, engine_path, device_id=0):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Get input/output bindings
        self.inputs, self.outputs, self.bindings, self.stream = self._setup_bindings()
    
    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        return engine
    
    def _setup_bindings(self):
        inputs = []
        outputs = []
        bindings = []
        stream = trt.Stream()
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # Allocate memory
            if self.engine.binding_is_input(i):
                input_tensor = trt.tensor_dtype_to_np_dtype(self.engine.get_tensor_dtype(name))
                device_ptr = cuda.cudaMalloc(self.engine.get_tensor_shape(name)[0] * np.prod(shape[1:]) * np.dtype(input_tensor).itemsize)
                bindings.append(device_ptr)
                inputs.append({'name': name, 'shape': shape})
            else:
                output_tensor = trt.tensor_dtype_to_np_dtype(self.engine.get_tensor_dtype(name))
                device_ptr = cuda.cudaMalloc(self.engine.get_tensor_shape(name)[0] * np.prod(shape[1:]) * np.dtype(output_tensor).itemsize)
                bindings.append(device_ptr)
                outputs.append({'name': name, 'shape': shape})
        
        return inputs, outputs, bindings, stream
    
    def __call__(self, image):
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Copy input to device
        cuda.cudaMemcpyAsync(self.bindings[0], input_tensor, trt.volume(input_tensor.shape), 
                           cudaMemcpyHostToDevice, self.stream.value)
        
        # Inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.value)
        
        # Copy output from device
        output_buffer = np.empty_like(self.outputs[0]['shape'])
        cuda.cudaMemcpyAsync(output_buffer, self.bindings[-1], trt.volume(self.outputs[-1]['shape']),
                           cudaMemcpyDeviceToHost, self.stream.value)
        cuda.streamSynchronize(self.stream.value)
        
        # Postprocess
        return self._postprocess(output_buffer)
    
    def _preprocess(self, image):
        # Resize and normalize
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        
        return img
    
    def _postprocess(self, output):
        # Decode output to boxes, scores, classes
        pass
```

**속도 향상**:
- **FP32**: 1.5~2x faster
- **FP16**: 2~3x faster
- **INT8**: 3~4x faster (with calibration)

#### TensorRT 성능 비교

| Platform | FP32 | FP16 | INT8 |
|----------|------|------|------|
| **RTX 2080 Ti** | 1.8x | 2.5x | 3.2x |
| **RTX 3090** | 1.7x | 2.3x | 3.0x |
| **Jetson AGX** | 1.5x | 1.9x | 2.4x |

### 2. ONNX Export

#### Export YOLOv8 to ONNX

```bash
# Export with specific opset
yolo export model=yolov8m.pt format=onnx opset=12

# Export with dynamic axes (for variable input size)
yolo export model=yolov8m.pt format=onnx dynamic=True

# Export with simplified graph
yolo export model=yolov8m.pt format=onnx simplify=True
```

#### ONNX Runtime Inference

```python
from onnxruntime import InferenceSession
import numpy as np

class ONNXInference:
    def __init__(self, onnx_path, providers=None):
        providers = providers or ['CPUExecutionProvider']
        self.session = InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def __call__(self, image):
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Postprocess
        return self._postprocess(outputs[0])
    
    def _preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        return img
    
    def _postprocess(self, output):
        # Decode output to boxes, scores, classes
        # Similar to PyTorch postprocessing
        pass
```

**장점**:
- ✅ Cross-platform inference
- ✅ GPU acceleration (CUDA, TensorRT, DirectML)
- ✅ No framework dependency

### 3. Batch Inference

#### PyTorch Batch Processing

```python
class BatchInference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_batch(self, images, batch_size=4):
        """
        Process multiple images in batches
        
        Args:
            images: List of PIL.Image or numpy arrays
            batch_size: Number of images to process together
        
        Returns:
            List of detection results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Preprocess
            processed = [self._preprocess(img) for img in batch]
            batch_tensor = torch.stack(processed).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.model(batch_tensor)
            
            # Postprocess
            batch_results = self._postprocess_batch(pred, len(batch))
            results.extend(batch_results)
        
        return results
    
    def _preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        
        return torch.tensor(img)
    
    def _postprocess_batch(self, predictions, batch_size):
        # Decode predictions for each image in batch
        results = []
        for i in range(batch_size):
            pred = predictions[i]
            boxes, scores, classes = decode_predictions(pred)
            results.append({
                'boxes': boxes,
                'scores': scores,
                'classes': classes
            })
        
        return results
```

**Throughput 향상**:
- **Single**: ~5 FPS
- **Batch=4**: ~15 FPS (3x improvement)
- **Batch=16**: ~20 FPS (4x improvement)

---

## 📦 Model Export

### Export Formats

#### PyTorch (.pt)

```python
# Save model
import torch
torch.save(model.state_dict(), 'model.pt')

# Load model
model = YOLO('yolov8m.pt')
```

**장점**:
- ✅ PyTorch native
- ✅ Easy to debug
- ✅ Full training capability

**단점**:
- ❌ PyTorch dependency
- ❌ Large file size
- ❌ No optimization

#### ONNX (.onnx)

```bash
# Export
yolo export model=yolov8m.pt format=onnx

# Use with ONNX Runtime
from onnxruntime import InferenceSession
session = InferenceSession('yolov8m.onnx')
```

**장점**:
- ✅ Platform independent
- ✅ GPU support (CUDA, TensorRT, etc.)
- ✅ Optimized inference

**단점**:
- ❌ Limited debugging
- ❌ Version compatibility issues

#### TensorRT (.engine)

```bash
# Export
yolo export model=yolov8m.pt format=engine device=0 half=True
```

**장점**:
- ✅ Maximum performance
- ✅ GPU optimization
- ✅ Reduced latency

**단점**:
- ❌ Platform specific (NVIDIA only)
- ❌ Complex export
- ❌ Not easily debuggable

### Export Comparison

| Format | Size | Speed | Platform | Optimization |
|--------|------|-------|----------|--------------|
| **.pt** | 25MB | 1x | PyTorch | Limited |
| **.onnx** | 23MB | 1.5x | Cross-platform | Moderate |
| **.engine** | 20MB | 2.5x | NVIDIA | Maximum |

### Best Practices

#### Export Checklist

- [ ] Test with validation data before export
- [ ] Compare inference results (PT vs ONNX vs TRT)
- [ ] Use appropriate precision (FP16/INT8)
- [ ] Include calibration data for INT8
- [ ] Optimize input size for target hardware

#### Export Validation

```python
def validate_export(model_pt, model_onnx, model_trt, test_images):
    """Validate exported models match original"""
    pt_results = []
    onnx_results = []
    trt_results = []
    
    for img in test_images:
        # PT inference
        pt_pred = model_pt(img)
        pt_results.append(pt_pred)
        
        # ONNX inference
        onnx_pred = model_onnx(img)
        onnx_results.append(onnx_pred)
        
        # TRT inference
        trt_pred = model_trt(img)
        trt_results.append(trt_pred)
    
    # Compare results
    pt_onnx_diff = compare_predictions(pt_results, onnx_results)
    pt_trt_diff = compare_predictions(pt_results, trt_results)
    
    return {
        'pt_onnx_mAP_diff': pt_onnx_diff,
        'pt_trt_mAP_diff': pt_trt_diff
    }
```

---

## 🚀 Deployment Strategies

### 1. Edge Deployment (Mobile/Embedded)

#### Jetson Nano/AGX

```bash
# Optimize for Jetson
yolo export model=yolov8n.pt format=engine device=0 half=True

# Install TensorRT on Jetson
sudo apt-get install python3-libnvinfer
```

**Constraints**:
- Limited compute
- Memory constrained
- Power constrained

**Optimizations**:
- INT8 quantization
- Model pruning (30-50% reduction)
- Reduced input size (416x416)
- Batch size = 1

#### Mobile (iOS/Android)

```python
# TensorFlow Lite export
yolo export model=yolov8n.pt format=tflite

# Convert to Core ML (iOS)
yolo export model=yolov8n.pt format=coreml
```

**Optimizations**:
- INT8 quantization
- Dynamic shape support
- Neural engine optimization

### 2. Server Deployment

#### Production Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  inference:
    image: yolov8-server:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/yolov8m.engine
      - DEVICE=cuda:0
      - BATCH_SIZE=4
```

**Microservices Architecture**:
```
Client → Load Balancer → Inference Service → Result Cache
                              ↓
                        Object Detection
                              ↓
                        Database
```

**Scaling**:
- Horizontal scaling: Multiple inference instances
- Auto-scaling based on queue length
- GPU shared across instances

### 3. Cloud Deployment

#### AWS Lambda (Serverless)

```python
# Lambda function for inference
def lambda_handler(event, context):
    image = base64.b64decode(event['body']['image'])
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Inference
    results = model.predict(image)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

**Optimizations**:
- Model size < 250MB (lambda limit)
- Warm-start strategies
- EFS for model storage

#### Azure Container Instances

```bash
# Build container
docker build -t yolov8-inference:latest .

# Deploy to ACR
docker tag yolov8-inference:latest myregistry.azurecr.io/yolov8-inference:latest
docker push myregistry.azurecr.io/yolov8-inference:latest

# Deploy to ACI
az container create --resource-group mygroup \
    --name yolov8-acic \
    --image myregistry.azurecr.io/yolov8-inference:latest \
    --cpu 2 --memory 4 \
    --ports 8000 \
    --ip-address public
```

---

## 💾 Hardware 최적화

### GPU Memory Optimization

#### Dynamic Batch Sizing

```python
class AdaptiveBatchSize:
    def __init__(self, model, device, max_memory=10GB):
        self.model = model
        self.device = device
        self.max_memory = max_memory
    
    def compute_optimal_batch_size(self, input_size=(640, 640)):
        """Compute optimal batch size given memory constraints"""
        # Estimate memory usage
        mem_per_batch = self.estimate_memory_usage(input_size)
        
        # Compute max batch size
        available_mem = torch.cuda.get_device_properties(self.device).total_memory
        optimal_batch = int((available_mem * 0.7) / mem_per_batch)
        
        return max(1, optimal_batch)
    
    def estimate_memory_usage(self, input_size):
        """Estimate memory usage per batch"""
        # Forward pass to measure memory
        batch = torch.randn(1, *input_size).to(self.device)
        
        with torch.cuda.memory_stats(self.device) as stats:
            self.model(batch)
        
        mem_used = torch.cuda.max_memory_allocated(self.device)
        return mem_used
```

#### Gradient Checkpointing

```python
# Reduce memory usage during training
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to reduce memory"""
    def custom_forward(*inputs):
        features, targets = inputs
        return model(features, targets)
    
    model.backbone.apply(partial(gradient_checkpointing, custom_forward))
```

**Memory Reduction**:
- **Without**: 16GB GPU, batch=8
- **With**: 16GB GPU, batch=16

### CPU Optimization

#### OpenMP / MKL

```bash
# Install optimized BLAS
pip install mkl
```

#### Multi-threading

```python
import torch

# Optimize for CPU
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

# Enable MKL-DNN
torch.backends.mkldnn.enabled = True
```

### Quantization

#### Post-training Quantization (PTQ)

```python
import torch.quantization as quantization

# PTQ for YOLOv8
model = YOLO('yolov8m.pt')

# Convert to quantized model
model.quantization_enabled = True
model.eval()

# Quantize
quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    quantization_mappings
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'yolov8m_quantized.pt')
```

**Performance**:
- **Size Reduction**: 4x smaller
- **Speed**: 2x faster
- **Accuracy**: <1% drop

---

## 🧠 Memory 최적화

### Model Pruning

#### Iterative Pruning

```python
class PruningManager:
    def __init__(self, model, target_sparsity=0.5):
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruner = torch.nn.utils.prune.L1Pruner()
    
    def prune_layers(self, layers_to_prune=['cv1', 'cv2', 'cv3']):
        """Prune specified layers"""
        for name in layers_to_prune:
            module = getattr(self.model, name)
            self.pruner.prune_from(module, 'weight', amount=0.5)
    
    def compute_sparsity(self):
        """Compute current sparsity"""
        total_params = 0
        sparse_params = 0
        
        for name, param in self.model.named_parameters():
            if 'mask' in name:
                sparse_params += (param.data == 0).sum().item()
            total_params += param.numel()
        
        return sparse_params / total_params
```

**Results**:
- **Original**: 25.9M params
- **Pruned**: 12.9M params (50% reduction)
- **Accuracy**: -2.3% mAP

### Knowledge Distillation

```python
class KnowledgeDistillation:
    def __init__(self, teacher, student, temperature=4.0):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
    
    def forward(self, x, targets):
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Student predictions
        student_logits = self.student(x)
        
        # Distillation loss
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = 0.5 * kl_loss + 0.5 * ce_loss
        
        return total_loss, kl_loss, ce_loss
```

**Benefits**:
- **Teacher**: Large, accurate model
- **Student**: Smaller, faster model
- **Performance**: 95% of teacher, 2x faster

---

## 💡 실전 팁

### Tip 1: Multi-scale Inference

```python
def multiscale_inference(model, image, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """Multi-scale inference for small object detection"""
    results = []
    scores = []
    boxes = []
    
    for scale in scales:
        img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        pred = model.predict(img)
        
        # Scale back to original
        pred.boxes *= scale
        
        results.append(pred)
    
    # Combine results (non-maximum suppression across scales)
    final_boxes = torch.cat([r.boxes for r in results])
    final_scores = torch.cat([r.scores for r in results])
    final_classes = torch.cat([r.classes for r in results])
    
    final_boxes, final_scores, final_classes = nms(final_boxes, final_scores, final_classes, iou_threshold=0.45)
    
    return final_boxes, final_scores, final_classes
```

**Impact**:
- **Small objects**: +5-10% mAP
- **Speed**: -30% (5x inference)
- **Recommendation**: Only for small objects important

### Tip 2: Confidence Threshold Tuning

```python
def find_optimal_confidence_threshold(model, val_loader, iou_threshold=0.45):
    """Find optimal confidence threshold"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for conf_thresh in thresholds:
        mAP = 0
        
        for images, targets in val_loader:
            preds = model.predict(images, conf=conf_thresh)
            mAP += compute_mAP(preds, targets, iou_threshold)
        
        mAP /= len(val_loader)
        results.append((conf_thresh, mAP))
    
    # Find optimal threshold
    optimal = max(results, key=lambda x: x[1])
    
    return optimal[0], optimal[1]
```

**Optimal Thresholds**:
- **YOLOv8**: 0.25-0.35
- **DETR**: 0.3-0.4
- **FCOS**: 0.3-0.5

### Tip 3: NMS Parameters

```python
def tune_nms_parameters(model, val_loader):
    """Tune NMS parameters"""
    iou_thresholds = [0.4, 0.45, 0.5, 0.55]
    nms_types = ['traditional', 'soft', 'diou']
    
    best_config = None
    best_mAP = 0
    
    for iou_thresh in iou_thresholds:
        for nms_type in nms_types:
            mAP = evaluate_with_nms(model, val_loader, iou_thresh, nms_type)
            
            if mAP > best_mAP:
                best_mAP = mAP
                best_config = {
                    'iou_threshold': iou_thresh,
                    'nms_type': nms_type
                }
    
    return best_config
```

**Best Practices**:
- **General**: IoU=0.45, Traditional NMS
- **Dense objects**: Soft NMS
- **High precision**: IoU=0.5, DIoU-NMS

---

## 📊 Performance Summary

### Deployment Recommendations

| Scenario | Recommended Approach | Expected Speed |
|----------|-------------------|----------------|
| **Mobile/Edge** | INT8 quantization + pruning | 30-50 FPS |
| **Real-time** | FP16 TensorRT | 100-200 FPS |
| **Production** | Batch inference + caching | 50-100 FPS |
| **Research** | FP32 PyTorch | 20-50 FPS |

### Optimization Checklist

- [ ] Quantization (FP16/INT8)
- [ ] Model pruning
- [ ] Knowledge distillation
- [ ] TensorRT export
- [ ] ONNX export
- [ ] Batch inference optimization
- [ ] Multi-scale inference (if needed)
- [ ] Confidence threshold tuning
- [ ] NMS parameter tuning

---

*마지막 업데이트: 2026-03-30*
*참고: TensorRT docs, ONNX Runtime docs, Ultralytics documentation*
