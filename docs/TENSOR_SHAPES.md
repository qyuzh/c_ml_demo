# Tensor Shape Guide

This document explains the shapes of tensors and matrices throughout the training process in the C ML Demo project.

## Table of Contents

- [Network Architecture](#network-architecture)
- [Forward Pass Shapes](#forward-pass-shapes)
- [Backward Pass Shapes](#backward-pass-shapes)
- [Parameter Shapes](#parameter-shapes)
- [Data Flow Visualization](#data-flow-visualization)

---

## Network Architecture

The default MLP architecture for MNIST:

```
Input (784) → FC1 (128) → ReLU → FC2 (64) → ReLU → FC3 (10) → Softmax → Output
```

Where:

- **784**: Flattened 28×28 MNIST image
- **128**: First hidden layer neurons
- **64**: Second hidden layer neurons
- **10**: Output classes (digits 0-9)

---

## Forward Pass Shapes

### Batch Data

```
batch_images: [batch_size, 784]
batch_labels: [batch_size, 10]  (one-hot encoded)
```

**Example with batch_size=32:**

```
batch_images: [32, 784]  - 32 flattened images
batch_labels: [32, 10]   - 32 one-hot labels
```

### Layer 1: FC1 + ReLU

```
Input:   [32, 784]
         ×
Weights: [784, 128]  (fc1->weights)
         +
Bias:    [1, 128]    (fc1->bias, broadcasted)
         ↓
h1:      [32, 128]   (linear output)
         ↓ (ReLU element-wise)
a1:      [32, 128]   (activation output)
```

**Shape transformation:**

- Matrix multiplication: `[32, 784] @ [784, 128] = [32, 128]`
- Broadcast addition: `[32, 128] + [1, 128] = [32, 128]`
- ReLU (element-wise): `[32, 128] → [32, 128]`

### Layer 2: FC2 + ReLU

```
Input:   a1 [32, 128]
         ×
Weights: [128, 64]   (fc2->weights)
         +
Bias:    [1, 64]     (fc2->bias, broadcasted)
         ↓
h2:      [32, 64]    (linear output)
         ↓ (ReLU element-wise)
a2:      [32, 64]    (activation output)
```

### Layer 3: FC3 + Softmax

```
Input:   a2 [32, 64]
         ×
Weights: [64, 10]    (fc3->weights)
         +
Bias:    [1, 10]     (fc3->bias, broadcasted)
         ↓
h3:      [32, 10]    (linear output, logits)
         ↓ (Softmax row-wise)
output:  [32, 10]    (probability distributions)
```

**Output interpretation:**
Each row is a probability distribution over 10 classes:

```
output[0] = [0.01, 0.02, 0.85, 0.03, ...]  ← Predicts class 2
output[1] = [0.70, 0.05, 0.10, 0.02, ...]  ← Predicts class 0
...
```

---

## Backward Pass Shapes

### Loss Gradient (Seed Gradient)

```
Loss value: scalar (averaged over batch)
Loss gradient: [32, 10]  (∂L/∂logits = predictions - labels)
```

**Why this shape?**
The gradient has the same shape as the output because we need to know how much each prediction contributed to the loss.

### Gradient Flow Through Network

#### Layer 3 Backward

```
∂L/∂h3:           [32, 10]   (gradient at fc3 output)
                  ↓
∂L/∂fc3.weights:  [64, 10]   (accumulated from all batch samples)
∂L/∂fc3.bias:     [1, 10]    (summed across batch dimension)
                  ↓
∂L/∂a2:           [32, 64]   (gradient flows to previous layer)
```

**Gradient computation:**

- Weight gradient: `a2ᵀ @ ∂L/∂h3 = [64, 32] @ [32, 10] = [64, 10]`
- Bias gradient: `sum(∂L/∂h3, axis=0) = [32, 10] → [1, 10]`
- Input gradient: `∂L/∂h3 @ fc3.weightsᵀ = [32, 10] @ [10, 64] = [32, 64]`

#### Layer 2 Backward (through ReLU)

```
∂L/∂a2:           [32, 64]   (from layer 3)
                  ↓ (ReLU gates gradient)
∂L/∂h2:           [32, 64]   (zero where h2 was negative)
                  ↓
∂L/∂fc2.weights:  [128, 64]
∂L/∂fc2.bias:     [1, 64]
                  ↓
∂L/∂a1:           [32, 128]
```

#### Layer 1 Backward (through ReLU)

```
∂L/∂a1:           [32, 128]  (from layer 2)
                  ↓ (ReLU gates gradient)
∂L/∂h1:           [32, 128]
                  ↓
∂L/∂fc1.weights:  [784, 128]
∂L/∂fc1.bias:     [1, 128]
```

---

## Parameter Shapes

### All Model Parameters

```c
// Layer 1: Input → Hidden1
fc1->weights: [784, 128]    75,264 parameters
fc1->bias:    [1, 128]         128 parameters

// Layer 2: Hidden1 → Hidden2  
fc2->weights: [128, 64]      8,192 parameters
fc2->bias:    [1, 64]           64 parameters

// Layer 3: Hidden2 → Output
fc3->weights: [64, 10]         640 parameters
fc3->bias:    [1, 10]           10 parameters

// Total: 84,298 trainable parameters
```

### Gradient Shapes Match Parameter Shapes

After `tensor_backward()`, each parameter has a gradient of the same shape:

```c
fc1->weights->grad: [784, 128]  ← Same as fc1->weights
fc1->bias->grad:    [1, 128]    ← Same as fc1->bias
fc2->weights->grad: [128, 64]   ← Same as fc2->weights
fc2->bias->grad:    [1, 64]     ← Same as fc2->bias
fc3->weights->grad: [64, 10]    ← Same as fc3->weights
fc3->bias->grad:    [1, 10]     ← Same as fc3->bias
```

---

## Data Flow Visualization

### Complete Training Step

```
INPUT DATA
┌─────────────────────────────────────┐
│ batch_images: [32, 784]             │
│ batch_labels: [32, 10]              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ FORWARD PASS                        │
├─────────────────────────────────────┤
│ [32, 784] → FC1 → [32, 128]        │
│             ↓ ReLU                  │
│           [32, 128] → FC2 → [32, 64]│
│                       ↓ ReLU        │
│                     [32, 64] → FC3  │
│                              → [32, 10]│
│                                ↓ Softmax│
│                           output [32, 10]│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ LOSS COMPUTATION                    │
├─────────────────────────────────────┤
│ predictions: [32, 10]               │
│ labels:      [32, 10]               │
│ loss:        scalar                 │
│ gradient:    [32, 10]               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ BACKWARD PASS                       │
├─────────────────────────────────────┤
│ gradient [32, 10] flows back        │
│   ↓                                 │
│ FC3: accumulates to [64,10] + [1,10]│
│   ↓                                 │
│ ReLU: gates gradient [32, 64]      │
│   ↓                                 │
│ FC2: accumulates to [128,64] + [1,64]│
│   ↓                                 │
│ ReLU: gates gradient [32, 128]     │
│   ↓                                 │
│ FC1: accumulates to [784,128] + [1,128]│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ OPTIMIZER UPDATE                    │
├─────────────────────────────────────┤
│ For each parameter θ:               │
│   θ_new = θ_old - lr × ∇θ           │
│                                     │
│ Updates all 84,298 parameters       │
└─────────────────────────────────────┘
```

### Broadcast Addition Detail

The bias broadcast operation is key to understanding bias gradients:

```
Forward:
  matmul_out: [32, 128]    bias: [1, 128]
  ┌─────────────────┐      ┌──────────┐
  │ row 0: 128 vals │  +   │ 128 vals │  → broadcasted to all 32 rows
  │ row 1: 128 vals │  +   │ 128 vals │
  │     ...         │  +   │    ...   │
  │ row 31: 128 vals│  +   │ 128 vals │
  └─────────────────┘      └──────────┘
  
Backward:
  gradient: [32, 128]      bias_grad: [1, 128]
  ┌─────────────────┐      ┌──────────┐
  │ row 0: 128 vals │      │          │
  │ row 1: 128 vals │  →   │ sum all  │ → [1, 128]
  │     ...         │      │  rows    │
  │ row 31: 128 vals│      │          │
  └─────────────────┘      └──────────┘
```

This is why bias gradients are computed by summing across the batch dimension!

---

## Quick Reference

### Common Shape Patterns

| Operation | Input A | Input B | Output | Notes |
|-----------|---------|---------|--------|-------|
| Matrix Multiply | [M, K] | [K, N] | [M, N] | Standard matmul |
| Broadcast Add | [M, N] | [1, N] | [M, N] | Bias addition |
| Element-wise | [M, N] | - | [M, N] | ReLU, Sigmoid |
| Row-wise Softmax | [M, N] | - | [M, N] | Normalize each row |
| Batch Sum | [M, N] | - | [1, N] | For bias gradients |

### Debugging Tips

1. **Check batch dimension**: Most operations preserve the first dimension (batch_size)
2. **Weight shape**: `[input_features, output_features]`
3. **Bias shape**: Always `[1, output_features]`
4. **Gradient shape**: Always matches the parameter shape
5. **Broadcasting**: Smaller tensor is "copied" to match larger tensor's shape

### Shape Errors

Common mistakes and their fixes:

```c
// ❌ Wrong: bias as [output_size] 
Matrix* bias = matrix_create(output_size, 1);  // [output, 1]

// ✅ Correct: bias as [1, output_size]
Matrix* bias = matrix_create(1, output_size);  // [1, output]

// ❌ Wrong: weights transposed
Matrix* weights = matrix_create(output_size, input_size);  // [out, in]

// ✅ Correct: weights for matmul
Matrix* weights = matrix_create(input_size, output_size);  // [in, out]
```

---

## Summary

**Key takeaways:**

1. **Batch first**: All tensors have batch_size as the first dimension
2. **Weights**: Always `[input_size, output_size]` for proper matrix multiplication
3. **Bias**: Always `[1, output_size]` for broadcasting across batch
4. **Gradients**: Always match their parameter's shape
5. **Broadcast backward**: Automatically sums across batch dimension

Understanding these shapes is crucial for:

- Debugging dimension mismatches
- Understanding gradient flow
- Implementing new layer types
- Optimizing memory usage
