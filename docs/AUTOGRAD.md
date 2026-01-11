# Automatic Differentiation (Autograd)

A comprehensive guide to understanding and implementing automatic differentiation for neural networks.

## Table of Contents

1. [What is Autograd?](#what-is-autograd)
2. [Core Principles](#core-principles)
3. [Implementation Overview](#implementation-overview)
4. [Practical Examples](#practical-examples)
5. [How It Works](#how-it-works)
6. [Mathematical Foundation](#mathematical-foundation)

---

## What is Autograd?

**Automatic differentiation** (autograd) automatically computes derivatives of functions represented as computer programs. Our implementation uses **reverse-mode automatic differentiation** (backpropagation), which efficiently computes gradients for training neural networks.

### Why Autograd?

**Without autograd:**
```c
// Manual gradient computation - error-prone and tedious
void backward_linear(Linear* layer, Matrix* grad_output) {
    // Manually derive: ∂Loss/∂weights = input^T @ grad_output
    Matrix* input_T = matrix_transpose(layer->input_cache);
    layer->grad_weights = matrix_matmul(input_T, grad_output);
    // ... manually compute each gradient
}
```

**With autograd:**
```c
// Automatic gradient computation - one line!
ForwardResult* forward = mlp_forward(model, input);
forward->output->grad = predictions - labels;
tensor_backward(forward->output);  // ALL gradients computed automatically!
```

---

## Core Principles

### 1. Computational Graph

Every computation builds a **directed acyclic graph (DAG)**:

- **Nodes** = Tensors (data + gradient)
- **Edges** = Operations (add, multiply, matmul, relu, etc.)
- **Built dynamically** during forward pass

**Example:** `z = (x + y) * w`

```
    x [2.0]   y [3.0]     w [4.0]
        \      /            |
         \    /             |
          ADD               |
           |                |
         [5.0]              |
           \                /
            \              /
             \            /
              \          /
               \        /
                \      /
                  MUL
                   |
                 [20.0] = z
```

### 2. The Tensor Structure

```c
typedef struct Tensor {
    Matrix* data;           // Forward pass values
    Matrix* grad;           // Backward pass gradients (∂Loss/∂data)
    
    // Graph structure
    struct Tensor* parent1; // First input tensor
    struct Tensor* parent2; // Second input tensor (for binary ops)
    OpType op;              // Operation type that created this tensor
    
    int requires_grad;      // Track gradients?
} Tensor;
```

**Key insight:** Each tensor remembers:
- Its value (`data`)
- Its gradient (`grad`)
- How it was created (`op`, `parent1`, `parent2`)

### 3. Chain Rule

The foundation of backpropagation:

```
If z = f(y) and y = g(x), then:

∂z/∂x = (∂z/∂y) × (∂y/∂x)
         ↑          ↑
    upstream    local
    gradient    gradient
```

For multiple paths (when a variable is used multiple times):

```
∂Loss/∂x = Σ (∂Loss/∂yᵢ) × (∂yᵢ/∂x)
           i
```

---

## Implementation Overview

### Forward Pass: Building the Graph

```c
Tensor* tensor_add(Tensor* a, Tensor* b) {
    // 1. Compute forward result
    Matrix* result_data = matrix_add(a->data, b->data);
    
    // 2. Create new tensor
    Tensor* result = tensor_create(result_data, a->requires_grad || b->requires_grad);
    
    // 3. Record the operation and parents
    result->op = OP_ADD;
    result->parent1 = a;
    result->parent2 = b;
    
    return result;  // Graph is built!
}
```

### Backward Pass: Computing Gradients

```c
void tensor_backward(Tensor* t) {
    if (!t->requires_grad) return;
    
    // Ensure gradient is initialized
    if (t->grad == NULL) {
        t->grad = matrix_create(t->data->rows, t->data->cols);
        matrix_ones(t->grad);  // Gradient at output = 1
    }
    
    // Propagate gradient based on operation
    switch (t->op) {
        case OP_ADD:
            // ∂(a+b)/∂a = 1, gradient flows unchanged
            if (t->parent1->requires_grad) {
                accumulate_grad(t->parent1, t->grad);
                tensor_backward(t->parent1);
            }
            if (t->parent2->requires_grad) {
                accumulate_grad(t->parent2, t->grad);
                tensor_backward(t->parent2);
            }
            break;
            
        case OP_MUL:
            // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            if (t->parent1->requires_grad) {
                Matrix* grad_a = matrix_mul(t->grad, t->parent2->data);
                accumulate_grad(t->parent1, grad_a);
                tensor_backward(t->parent1);
                matrix_free(grad_a);
            }
            if (t->parent2->requires_grad) {
                Matrix* grad_b = matrix_mul(t->grad, t->parent1->data);
                accumulate_grad(t->parent2, grad_b);
                tensor_backward(t->parent2);
                matrix_free(grad_b);
            }
            break;
            
        // ... other operations
    }
}
```

### Gradient Accumulation

When a tensor appears multiple times, gradients must be **summed**:

```c
void accumulate_grad(Tensor* t, Matrix* new_grad) {
    if (t->grad == NULL) {
        t->grad = matrix_copy(new_grad);  // First gradient
    } else {
        // Add to existing gradient
        Matrix* sum = matrix_add(t->grad, new_grad);
        matrix_free(t->grad);
        t->grad = sum;
    }
}
```

---

## Practical Examples

### Example 1: Simple Computation

**Forward:**
```c
// Create inputs
Tensor* x = tensor_from_array((float[]){2.0, 3.0}, 1, 2, 1);
Tensor* w = tensor_from_array((float[]){0.5, 0.5}, 1, 2, 1);

// Forward computation
Tensor* y = tensor_mul(x, w);     // y = x * w = [1.0, 1.5]
Tensor* z = tensor_sum(y);        // z = sum(y) = 2.5
```

**Computational graph:**
```
x [2.0, 3.0]    w [0.5, 0.5]
    \              /
     \            /
      \          /
        MUL (element-wise)
         |
    y [1.0, 1.5]
         |
        SUM
         |
      z [2.5]
```

**Backward:**
```c
// Set output gradient
matrix_fill(z->grad, 1.0);  // ∂Loss/∂z = 1

// Compute all gradients automatically
tensor_backward(z);

// Results:
// z->grad = [1.0]
// y->grad = [1.0, 1.0]        (from sum backward)
// x->grad = [0.5, 0.5]        (= y->grad * w)
// w->grad = [2.0, 3.0]        (= y->grad * x)
```

### Example 2: Neural Network Layer

**Forward:**
```c
// Linear layer: output = input @ weights + bias
Tensor* input = tensor_create(batch_data, 1);      // [batch, in_dim]
Tensor* h = linear_forward(layer, input);           // [batch, out_dim]
Tensor* a = tensor_relu(h);                         // [batch, out_dim]
```

**Computational graph:**
```
input [N, Din]    weights [Din, Dout]    bias [1, Dout]
     \               /                      /
      \             /                      /
       \           /                      /
         MATMUL                          /
            \                           /
             \                         /
              \                       /
                 ADD (broadcast)
                      |
                  h [N, Dout]
                      |
                    RELU
                      |
                  a [N, Dout]
```

**Backward:**
```c
// Gradient from next layer
matrix_copy(a->grad, grad_from_next_layer);

// One call computes ALL gradients
tensor_backward(a);

// Results automatically computed:
// - layer->weights->grad  (gradient for weights)
// - layer->bias->grad     (gradient for bias)
// - input->grad           (gradient to pass to previous layer)
```

### Example 3: Complete Training Loop

```c
// Training iteration
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Zero gradients from previous iteration
    optimizer_zero_grad(optimizer);
    
    // === FORWARD PASS ===
    ForwardResult* forward = mlp_forward(model, train_x);
    
    // Compute loss
    float loss = cross_entropy_loss(forward->output->data, train_y);
    
    // === BACKWARD PASS ===
    // Set gradient at output: ∂Loss/∂output = predictions - labels
    Matrix* loss_grad = matrix_sub(forward->output->data, train_y);
    matrix_copy(forward->output->grad, loss_grad);
    
    // Automatic gradient computation
    tensor_backward(forward->output);
    
    // Manually handle bias gradients (due to broadcasting)
    for (int i = 0; i < model->fc1->bias->data->cols; i++) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += model->fc1->weight_grad->data[b * cols + i];
        }
        model->fc1->bias->grad->data[i] = sum;
    }
    // (Repeat for fc2, fc3...)
    
    // === UPDATE WEIGHTS ===
    optimizer_step(optimizer, model);
    
    // Cleanup
    forward_result_free(forward);
    matrix_free(loss_grad);
}
```

---

## How It Works

### Gradient Computation Rules

#### Addition: z = x + y
```
∂z/∂x = 1
∂z/∂y = 1
```
Gradient flows unchanged to both inputs.

#### Element-wise Multiplication: z = x ⊙ y
```
∂z/∂x = y
∂z/∂y = x
```
Gradient is scaled by the other input.

#### Matrix Multiplication: Z = X @ Y
```
∂Z/∂X = grad_output @ Y^T
∂Z/∂Y = X^T @ grad_output
```
Uses transpose for dimension matching.

#### ReLU: y = max(0, x)
```
∂y/∂x = { 1  if x > 0
        { 0  if x ≤ 0
```
Gradient passes through for positive values only.

#### Softmax: yᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```
∂yᵢ/∂xⱼ = yᵢ(δᵢⱼ - yⱼ)
```
Where δᵢⱼ = 1 if i=j, else 0.

### Memory Management

**Gradient Zeroing:**
```c
void tensor_zero_grad(Tensor* t) {
    if (t->grad != NULL) {
        matrix_fill(t->grad, 0.0f);
    }
}
```
Must be called before each training iteration!

**Intermediate Tensors:**
```c
// Keep track of all intermediate results
typedef struct {
    Tensor* h1;
    Tensor* a1;
    Tensor* h2;
    Tensor* output;
} ForwardResult;

// Free after backward pass
void forward_result_free(ForwardResult* result) {
    tensor_free(result->h1);
    tensor_free(result->a1);
    tensor_free(result->h2);
    tensor_free(result->output);
    free(result);
}
```

---

## Mathematical Foundation

### Reverse-Mode Automatic Differentiation

Given function f: ℝⁿ → ℝ (many inputs, one output - typical in ML):

1. **Forward pass:** Compute f(x) = vₙ
   - Store all intermediate values v₁, v₂, ..., vₙ

2. **Initialize:** Set v̄ₙ = 1 (gradient at output)

3. **Backward pass:** For each vᵢ in reverse order:
   ```
   v̄ᵢ = Σⱼ (∂vⱼ/∂vᵢ) × v̄ⱼ
   ```
   where j ranges over children of i

4. **Result:** All partial derivatives ∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ

**Complexity:** O(forward_time) - extremely efficient!

### Why Reverse Mode?

**Forward-mode AD:**
- Computes Jv (Jacobian-vector product)
- Efficient for: few inputs, many outputs
- Example: f: ℝ → ℝⁿ

**Reverse-mode AD (our implementation):**
- Computes J^T v (vector-Jacobian product)
- Efficient for: many inputs, few outputs
- Example: f: ℝⁿ → ℝ (neural network loss)

For neural networks with millions of parameters and one loss value, reverse-mode is **vastly more efficient**.

### Comparison with Alternatives

| Method | Accuracy | Speed | Implementation |
|--------|----------|-------|----------------|
| **Numerical** | Approximate | Slow (2n evals) | Easy |
| **Symbolic** | Exact | Can be slow | Complex |
| **Forward AD** | Exact | O(n × forward) | Medium |
| **Reverse AD** | Exact | O(forward) | Medium |

Our reverse-mode AD is **exact** and **fast** for ML.

---

## Advanced Topics

### Gradient Checkpointing

Save memory by recomputing forward pass during backward:
```c
// Trade computation for memory
// Useful for very deep networks
```

### Higher-Order Derivatives

Computing gradients of gradients:
```c
// Would require tracking gradients as tensors
// Not implemented in current version
```

### In-Place Operations

Modify tensors without creating new ones:
```c
// Must be careful with autograd
// Can break gradient computation if not handled properly
```

---

## Usage in Training

### Complete MNIST Example

```c
// 1. Create model
MLPAutograd* model = mlp_create(784, 128, 64, 10);

// 2. Create optimizer
SGD* optimizer = sgd_create(0.01, 0.9, 0.0001);

// 3. Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // Get batch
        Matrix* batch_x = get_batch_x(train_data, batch);
        Matrix* batch_y = get_batch_y(train_data, batch);
        
        // Zero gradients
        optimizer_zero_grad(optimizer);
        
        // Forward pass
        Tensor* input = tensor_from_matrix(batch_x, 1);
        ForwardResult* forward = mlp_forward(model, input);
        
        // Backward pass
        Matrix* grad = matrix_sub(forward->output->data, batch_y);
        forward->output->grad = matrix_copy(grad);
        tensor_backward(forward->output);
        
        // Update weights
        optimizer_step(optimizer, model);
        
        // Cleanup
        tensor_free(input);
        forward_result_free(forward);
        matrix_free(grad);
        matrix_free(batch_x);
        matrix_free(batch_y);
    }
}
```

### Key Takeaways

1. **One backward call** computes all gradients
2. **Graph is dynamic** - rebuilt each iteration
3. **Memory efficient** - only stores necessary intermediates
4. **Exact gradients** - no numerical approximation
5. **Easy to extend** - add new operations with their backward rules

---

## Further Reading

- **Implementation:** See `src/autograd.c` for complete code
- **Usage:** See `src/train.c` for training example
- **Papers:** "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al., 2018)

---

## Summary

Automatic differentiation transforms neural network training from **manual gradient derivation** to **automatic computation**:

- **Computational graph** tracks operations
- **Reverse-mode traversal** applies chain rule
- **One backward call** computes all gradients
- **Exact and efficient** for machine learning

This is the same approach used by PyTorch, TensorFlow, and JAX - implemented from scratch in pure C!