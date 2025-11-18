# CLAUDE.md - TransAxx Repository Guide

## Project Overview

**TransAxx** is a PyTorch-based framework for fast emulation of approximate Vision Transformer (ViT) models using approximate computing techniques. It extends PyTorch to support approximate inference and approximation-aware retraining with GPU acceleration.

### Key Capabilities
- Approximate arithmetic for DNN inference
- GPU-accelerated custom CUDA kernels for approximate multipliers
- Approximation-aware retraining workflows
- Support for Vision Transformer models and CNNs
- 8-bit quantization with custom approximate multipliers
- Monte Carlo Tree Search (MCTS) for optimization

### Academic Context
- **Paper**: "TransAxx: Efficient Transformers with Approximate Computing" (arXiv:2402.07545, 2024)
- **Authors**: Dimitrios Danopoulos, Georgios Zervakis, Dimitrios Soudris, Jörg Henkel
- **License**: MIT
- **Based on**: AdaPT framework (https://github.com/dimdano/adapt)

---

## Repository Structure

```
transaxx-plus/
├── classification/          # Training and evaluation scripts
│   ├── train.py            # Main training script with pruning support
│   ├── train_quantization.py  # Quantization-aware training
│   ├── utils.py            # Utility functions for training/eval
│   ├── presets.py          # Training presets and configurations
│   └── ptflops/            # FLOPs counting utilities
├── layers/                 # Custom approximate layers (Python bindings)
│   ├── adapt_linear_layer.py        # Approximate linear/matmul layer
│   ├── adapt_convolution_layer.py   # Approximate Conv2D layer
│   ├── adapt_matmul_layer.py        # Matrix multiplication layer
│   ├── transposed_adapt_convolution_layer.py  # Transposed conv
│   └── layer_utils.py      # Shared layer utilities
├── ext_modules/            # CUDA/C++ extensions (JIT compiled)
│   ├── include/
│   │   ├── nn/cuda/axx_mults/  # Approximate multiplier LUTs (*.h files)
│   │   ├── nn/layers/          # Layer header files
│   │   └── core/               # Core utilities
│   └── src/
│       └── nn/
│           ├── cpp/            # CPU implementations
│           └── cuda/           # CUDA kernel implementations
├── mcts/                   # Monte Carlo Tree Search for optimization
│   └── mcts.py            # MCTS implementation with UCT policy
├── examples/              # Jupyter notebooks for workflows
│   ├── cifar10_eval.ipynb    # CIFAR-10 evaluation example
│   ├── imagenet_eval.ipynb   # ImageNet evaluation example
│   ├── models/               # Model definitions
│   └── run_jupyter.sh        # Start Jupyter server
├── tools/                 # Utility tools
│   └── LUT_convert.ipynb    # Convert custom multipliers to C headers
├── pytorch-quantization/  # Quantization library (submodule/included)
├── datasets/              # Dataset handling (README only)
├── docker/                # Docker configuration
│   ├── run_docker.sh         # Launch Docker container
│   ├── build_docker.sh       # Build custom Docker image
│   └── banner.sh             # Docker banner
├── docs/                  # Documentation and assets
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md             # Project README
```

---

## Technology Stack

### Core Dependencies
- **Python**: 3.x
- **PyTorch**: 2.0.1 with CUDA 11.7
- **CUDA**: Compute capability 7.0+ (Volta architecture or newer)
- **Ninja**: 1.11.1.1 (for JIT compilation)
- **Cython**: 3.0.7

### Key Libraries
- **timm**: 0.9.7 (PyTorch Image Models)
- **numpy**: 1.24.3
- **matplotlib**: 3.8.0
- **pygad**: 3.2.0 (Genetic algorithm optimization, optional)
- **jupyter**: For interactive notebooks

### Development Environment
- **Docker**: Primary development environment
- **Base Image**: `dimdano/transaxx:1.0` (includes PyTorch 2.0.1 + CUDA 11.7)
- **GPU**: NVIDIA GPU required for CUDA operations

---

## Core Concepts

### 1. Approximate Computing
TransAxx replaces exact arithmetic operations with approximate versions to reduce power consumption while maintaining acceptable accuracy. This is achieved through:
- **Approximate multipliers**: Lookup tables (LUTs) for 8x8 signed multiplication
- **Quantization**: 8-bit integer quantization of weights and activations
- **Custom CUDA kernels**: Hardware-accelerated approximate operations

### 2. Layer Architecture

#### Custom Layers
All custom layers follow the pattern:
```python
class AdaPT_LayerType(torch.nn.Module):
    def __init__(self, ..., axx_mult='mul8s_acc', quant_bits=8, fake_quant=False):
        # axx_mult: Multiplier type (e.g., 'mul8s_acc', 'mul8s_1L2H')
        # quant_bits: Quantization bitwidth (currently only 8 supported)
        # fake_quant: If True, use float simulation; if False, use actual int8
```

#### Available Layers
- **AdaPT_Linear**: Approximate fully-connected/linear layers (supports ViT multi-head attention)
- **AdaptConv2D**: Approximate 2D convolution
- **AdaptMMConvolution**: Matrix multiplication-based convolution
- **TransposedAdaptConvolution**: Approximate transposed convolution

#### JIT Compilation
Layers are compiled on-demand using PyTorch's JIT extension loader:
```python
from torch.utils.cpp_extension import load
# Compiles C++/CUDA sources at runtime
```

### 3. Approximate Multipliers

#### Built-in Multipliers
- **mul8s_acc**: 8-bit accurate multiplier (baseline)
- **mul8s_1L2H**: 8-bit approximate multiplier with specific power profile

#### Custom Multipliers
1. Create 256x256 LUT (8-bit signed multiplier)
2. Use `tools/LUT_convert.ipynb` to convert to C header
3. Place in `ext_modules/include/nn/cuda/axx_mults/`
4. Reference by filename (without extension) in layer initialization

### 4. Quantization Workflow

```python
# 1. Collect statistics on calibration data
stats = collect_stats(model, calib_data, num_batches=2)

# 2. Compute activation max values (amax)
amax = compute_amax(model, method="percentile", percentile=99.99)
# Alternatives: method="mse", method="entropy"

# 3. Quantizers use amax to scale activations to int8 range
```

### 5. Model Conversion Workflow

```python
# 1. Load pretrained model
model = torch.hub.load(...).to(device)

# 2. Identify layers to approximate
conv_layers = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Conv2d)]

# 3. Define approximation config for each layer
axx_list = [{
    'axx_mult': 'mul8s_acc',     # Multiplier type
    'axx_power': 1.0,             # Relative power consumption
    'quant_bits': 8,              # Quantization bits
    'fake_quant': False           # Use int8 operations
}] * len(conv_layers)

# 4. Replace layers (initialize=True for first time to compile kernels)
replace_conv_layers(model, AdaptConv2D, axx_list, ..., initialize=True)

# 5. Calibrate quantization
collect_stats(model, calib_data)
compute_amax(model, method="percentile", percentile=99.99)

# 6. Evaluate
top1_accuracy = evaluate(model, val_data)

# 7. Optional: Approximation-aware retraining
train_one_epoch(model, criterion, optimizer, train_data, device, epoch, print_freq)
```

---

## Development Workflows

### Setup and Environment

#### Using Docker (Recommended)
```bash
# 1. Run Docker container (requires NVIDIA GPU)
./docker/run_docker.sh

# 2. Inside container, launch Jupyter
./examples/run_jupyter.sh

# 3. Access Jupyter at http://localhost:8888
```

#### Custom Docker Build
```bash
./docker/build_docker.sh  # Modify dependencies as needed
```

#### Manual Setup (without Docker)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set PYTHONPATH (required for layer JIT compilation)
export PYTHONPATH=/path/to/transaxx-plus:$PYTHONPATH

# 3. Ensure CUDA toolkit is available (nvcc)
```

### Running Examples

#### CIFAR-10 Evaluation
```bash
# Open examples/cifar10_eval.ipynb
# Follow notebook cells to:
# - Load CIFAR-10 dataset
# - Load pretrained model
# - Convert to approximate layers
# - Calibrate quantization
# - Evaluate accuracy
# - Optionally retrain
```

#### ImageNet Evaluation
```bash
# Open examples/imagenet_eval.ipynb
# Similar workflow for larger-scale models
```

### Creating Custom Multipliers

1. **Generate LUT**: Create 256x256 array of int8 values
2. **Convert**: Use `tools/LUT_convert.ipynb` to generate C header
3. **Deploy**: Place `mul8s_<name>.h` in `ext_modules/include/nn/cuda/axx_mults/`
4. **Use**: Reference as `axx_mult='mul8s_<name>'`

**Important**: Only 8-bit signed multipliers are supported (256x256 LUT)

### Training Workflows

#### Standard Training
```python
from classification.train import train_one_epoch

# Configure optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train
for epoch in range(num_epochs):
    train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, print_freq=100)
    lr_scheduler.step()
```

#### Quantization-Aware Training
```python
from classification.train_quantization import train_one_epoch
# Similar API but with quantization support
```

### MCTS Optimization

The `mcts/mcts.py` module provides Monte Carlo Tree Search for finding optimal approximation configurations:
- **UCT Policy**: Upper Confidence Bound for Trees
- **Random policies**: Weighted or uniform action selection
- Use for automated design space exploration

---

## Code Conventions

### Python Style
- **Naming**:
  - Classes: `AdaPT_Linear`, `AdaptConv2D`
  - Functions: `snake_case` (e.g., `collect_stats`, `compute_amax`)
  - Constants: `UPPER_CASE`
- **Imports**: Standard library → Third-party → Local modules
- **Type hints**: Not consistently used (legacy code)

### Layer Development

#### When Creating New Approximate Layers:
1. **Inherit from `torch.nn.Module`**
2. **Define custom autograd function** (`torch.autograd.Function`)
   - Forward: Quantize → Approximate operation → Dequantize
   - Backward: Standard gradient computation (no approximation)
3. **JIT compile C++/CUDA extensions**
   ```python
   from torch.utils.cpp_extension import load
   module = load(name='...', sources=[...], extra_cuda_cflags=[...])
   ```
4. **Support both fake_quant and int8 modes**
5. **Store calibration values** (amax, amax_w) as buffers

#### CUDA Kernel Conventions:
- Place headers in `ext_modules/include/nn/`
- Place implementations in `ext_modules/src/nn/cuda/`
- Use `compute_70` or higher for architecture
- Include approximate multiplier LUTs via `#include "nn/cuda/axx_mults/<name>.h"`

### Model Evaluation

#### Standard Evaluation Pattern:
```python
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        # Compute metrics
```

#### With Quantization Calibration:
```python
# First collect stats (once per model configuration)
with torch.no_grad():
    stats = collect_stats(model, calib_loader, num_batches=N)
    amax = compute_amax(model, method="percentile", percentile=99.99)

# Then evaluate
accuracy = evaluate(model, val_loader)
```

### File Organization
- **Training scripts**: Place in `classification/`
- **Layer definitions**: Place in `layers/`
- **CUDA kernels**: Place in `ext_modules/src/nn/cuda/`
- **Headers**: Place in `ext_modules/include/`
- **Examples**: Place in `examples/` as Jupyter notebooks
- **Tools**: Place in `tools/`

---

## Common Tasks for AI Assistants

### 1. Adding a New Approximate Multiplier

```bash
# 1. Create LUT as numpy array (256x256, int8)
# 2. Use tools/LUT_convert.ipynb to generate header
# 3. Save as ext_modules/include/nn/cuda/axx_mults/mul8s_<name>.h
# 4. Test with existing layers by setting axx_mult='mul8s_<name>'
```

**Files to modify**: None (just add new header)

### 2. Implementing a New Layer Type

**Files to create**:
- `layers/adapt_<layer_type>_layer.py` (Python binding)
- `ext_modules/include/nn/layers/adapt_<layer_type>_layer.h` (C++ header)
- `ext_modules/src/nn/cpp/adapt_<layer_type>_layer.cpp` (CPU impl)
- `ext_modules/src/nn/cuda/adapt_<layer_type>_layer.cu` (CUDA impl)

**Files to modify**:
- `layers/__init__.py` (add import)

**Pattern to follow**: Use `adapt_linear_layer.py` as template

### 3. Adding Support for New Model Architecture

**Files to create/modify**:
- `examples/models/<model_name>.py` (if custom model)
- `examples/<model_name>_eval.ipynb` (evaluation notebook)

**Key considerations**:
- Identify which layers to approximate (Conv2D, Linear, etc.)
- Determine optimal quantization calibration method
- Test with accurate multiplier (`mul8s_acc`) first
- Gradually introduce approximation

### 4. Implementing New Calibration Method

**File to modify**: `classification/utils.py` or `pytorch-quantization/` modules

**Current methods**:
- Percentile (default: 99.99)
- MSE (Mean Squared Error)
- Entropy

### 5. Extending to New Quantization Bitwidths

**Current limitation**: Only 8-bit supported

**To extend**:
1. Modify LUT sizes in `ext_modules/include/nn/cuda/axx_mults/`
2. Update CUDA kernels to handle different bitwidths
3. Modify quantizer logic in layer implementations
4. Update `tools/LUT_convert.ipynb`

**Complexity**: High (requires extensive CUDA modifications)

### 6. Debugging JIT Compilation Issues

**Common issues**:
1. **PYTHONPATH not set**: Ensure `export PYTHONPATH=/path/to/repo:$PYTHONPATH`
2. **CUDA architecture mismatch**: Check `compute_arch` in layer files
3. **Missing CUDA toolkit**: Ensure `nvcc` is in PATH
4. **Include path errors**: Verify `include_dir` points to `ext_modules/include`

**Debugging tips**:
```python
# Add verbose flag to load()
module = load(name='...', sources=[...], verbose=True)

# Check compiled module location
print(module.__file__)
```

### 7. Performance Profiling

```python
# Use ptflops for FLOP counting
from classification.ptflops import get_model_complexity_info

with torch.cuda.device(0):
    macs, params, layer_specs = get_model_complexity_info(
        model, (3, 224, 224),
        custom_modules_hooks={AdaptConv2D: conv_flops_counter_hook}
    )
```

### 8. Running Tests

**Location**: `pytorch-quantization/tests/`

```bash
# Run quantization tests
cd pytorch-quantization
pytest tests/
```

### 9. Model Conversion Checklist

- [ ] Load pretrained model
- [ ] Identify layers to approximate (Conv2D, Linear)
- [ ] Create `axx_list` configuration
- [ ] Initialize layers with `initialize=True` (first time only)
- [ ] Collect calibration statistics
- [ ] Compute amax values
- [ ] Evaluate baseline accuracy
- [ ] Apply approximation progressively
- [ ] Monitor accuracy degradation
- [ ] Apply approximation-aware retraining if needed

### 10. Git Workflow

**Important**: This repo uses a specific branch naming convention:
- Development branches: `claude/claude-md-<session-id>`
- Always push with: `git push -u origin <branch-name>`
- Branch must start with `claude/` and match session ID

**Typical workflow**:
```bash
# Make changes
git add <files>
git commit -m "descriptive message"

# Push to designated branch
git push -u origin claude/claude-md-mi41nfrgdxbxfcrm-014nmJjsn9bjmP1zYPQGyWb4
```

---

## Testing and Validation

### Unit Tests
- Location: `pytorch-quantization/tests/`
- Framework: pytest
- Coverage: Quantization modules, tensor quantizers, calibration

### Integration Tests
- Examples serve as integration tests
- `examples/cifar10_eval.ipynb`: End-to-end workflow validation
- `examples/imagenet_eval.ipynb`: Large-scale model testing

### Validation Workflow
1. **Baseline**: Test with `fake_quant=True` (should match FP32)
2. **Quantization**: Test with `mul8s_acc` (accurate int8)
3. **Approximation**: Test with approximate multipliers
4. **Retraining**: Validate accuracy recovery

---

## Troubleshooting

### Docker Issues
**Problem**: Container fails to start
**Solution**: Ensure NVIDIA Docker runtime installed (`nvidia-docker2`)

**Problem**: GPU not accessible
**Solution**: Check `docker run --gpus` flag and CUDA driver compatibility

### Compilation Issues
**Problem**: JIT compilation fails
**Solution**:
- Verify CUDA toolkit installation (`nvcc --version`)
- Check `compute_arch` matches your GPU
- Ensure Ninja build system installed

**Problem**: Include file not found
**Solution**: Verify PYTHONPATH includes repository root

### Runtime Issues
**Problem**: "CUDA out of memory"
**Solution**: Reduce batch size or use gradient checkpointing

**Problem**: Accuracy significantly degraded
**Solution**:
- Check calibration data quality
- Verify amax computation
- Try different calibration methods
- Increase calibration data size
- Apply retraining

**Problem**: Slow first run
**Solution**: Expected - JIT compilation occurs on first layer initialization

---

## Performance Considerations

### Memory Usage
- **JIT compilation**: Temporary increase during first load
- **Quantization**: ~4x memory reduction (FP32 → INT8)
- **Calibration**: Requires subset of training data in memory

### Compute Performance
- **Speedup**: Depends on GPU architecture and multiplier complexity
- **Power savings**: Main benefit (not necessarily speed)
- **Retraining**: Similar to standard training (approximation only in forward pass)

### Best Practices
1. **Calibration**: Use 5-10% of training data, at least 2 batches
2. **Batch size**: Maximize GPU utilization without OOM
3. **Mixed approximation**: Apply approximation selectively to sensitive layers
4. **Progressive approximation**: Start with later layers, move to earlier layers
5. **Monitor**: Track both accuracy and estimated power consumption

---

## Key Files Reference

| File | Purpose | Modify When |
|------|---------|-------------|
| `layers/adapt_linear_layer.py` | Linear layer implementation | Adding features to linear layers |
| `layers/adapt_convolution_layer.py` | Conv2D layer implementation | Adding features to conv layers |
| `layers/layer_utils.py` | Shared layer utilities | Adding common functionality |
| `classification/train.py` | Training script | Modifying training loop |
| `classification/utils.py` | Evaluation utilities | Adding metrics or data loaders |
| `examples/cifar10_eval.ipynb` | CIFAR-10 workflow | Updating example workflows |
| `ext_modules/include/nn/cuda/axx_mults/*.h` | Multiplier LUTs | Adding new multipliers |
| `ext_modules/src/nn/cuda/*.cu` | CUDA kernels | Optimizing kernels |
| `tools/LUT_convert.ipynb` | LUT converter | Changing LUT format |
| `requirements.txt` | Python dependencies | Adding libraries |
| `docker/run_docker.sh` | Docker launcher | Changing container config |

---

## Additional Resources

### Documentation
- Main README: `/README.md`
- PyTorch Quantization: `/pytorch-quantization/README.md`
- Dataset Info: `/datasets/README.md`

### External Links
- **Paper**: https://arxiv.org/abs/2402.07545
- **AdaPT Framework**: https://github.com/dimdano/adapt
- **CONVOLVE Project**: https://convolve.eu

### Contact
- **Maintainer**: Dimitrios Danopoulos (dimdano@microlab.ntua.gr)

---

## Version History

- **Current Version**: 1.0
- **Docker Image**: `dimdano/transaxx:1.0`
- **PyTorch**: 2.0.1
- **CUDA**: 11.7

---

## Notes for AI Assistants

### When Analyzing Code
1. Check if operations use approximate or accurate multipliers
2. Verify quantization is properly calibrated (amax values set)
3. Look for `fake_quant` flag to understand if simulation or actual int8
4. Check layer dimension handling (2D vs 3D tensors for ViT models)

### When Modifying Code
1. Always test with `mul8s_acc` (accurate) first
2. Maintain backward compatibility with existing notebooks
3. Document power consumption implications
4. Consider both Conv2D and Linear layer impacts
5. Update relevant example notebooks
6. Test JIT compilation on first run

### When Debugging
1. Check PYTHONPATH environment variable
2. Verify CUDA availability (`torch.cuda.is_available()`)
3. Look at JIT compilation logs (verbose=True)
4. Compare fake_quant vs int8 results
5. Validate calibration data distribution

### Best Practices
- Prefer modifying existing examples over creating new files
- Keep approximate multiplier LUTs in designated directory
- Use existing utility functions from `classification/utils.py`
- Follow existing naming conventions (AdaPT, Adapt prefix)
- Test with small datasets (CIFAR-10) before large ones (ImageNet)
- Document power consumption estimates alongside accuracy

---

*This CLAUDE.md file is current as of 2025-11-18 and reflects the state of the TransAxx repository.*
