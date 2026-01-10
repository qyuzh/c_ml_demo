# Automatic Differentiation (Autograd) - Under the Hood

## Overview

Automatic differentiation (autograd) is a technique for automatically computing derivatives of functions represented as computer programs. Our implementation uses **reverse-mode automatic differentiation** (also called backpropagation), which efficiently computes gradients needed for training neural networks.

## Core Concepts

### 1. Computational Graph

Every computation is represented as a directed acyclic graph (DAG) where:

- **Nodes** represent tensors (our `Tensor` struct)
- **Edges** represent operations that transform tensors
- The graph is built **dynamically** during forward pass

```
Example: z = (x + y) * w

    x    y        w
     \  /         |
      +  ------   |
       \       \ /
        \       *
         \     /
           z
```

### 2. The Tensor Structure

```c
typedef struct Tensor {
    Matrix* data;           // The actual values
    Matrix* grad;           // Gradient (∂Loss/∂data)
    struct Tensor* parent1; // First parent in computation graph
    struct Tensor* parent2; // Second parent (if binary operation)
    OpType op;              // Operation that created this tensor
    int requires_grad;      // Whether to track gradients
} Tensor;
```

**Key insight**: Each tensor "remembers" how it was created by storing:

- Its parent tensors (inputs to the operation)
- The operation type that combined them

## How It Works

### Phase 1: Forward Pass (Building the Graph)

When you perform operations, the graph is built automatically:

```c
Tensor* a = tensor_create(matrix_a, 1);
Tensor* b = tensor_create(matrix_b, 1);
Tensor* c = tensor_add(a, b);        // c remembers: parents=(a,b), op=ADD
Tensor* d = tensor_mul(c, some_weight);  // d remembers: parents=(c,weight), op=MUL
```

Each operation:

1. Computes the forward result
2. Creates a new tensor
3. Records the operation and parent tensors

### Phase 2: Backward Pass (Computing Gradients)

Starting from the output, we traverse the graph **backwards** using the **chain rule**:

```
∂Loss/∂x = (∂Loss/∂z) * (∂z/∂x)
           ↑            ↑
      gradient at z   local gradient
```

The `tensor_backward()` function implements this recursively:

```c
void tensor_backward(Tensor* t, Matrix* grad) {
    if (!t->requires_grad) return;
    
    // Accumulate gradient (for nodes with multiple children)
    if (t->grad == NULL) {
        t->grad = matrix_copy(grad);
    } else {
        Matrix* sum = matrix_add(t->grad, grad);
        matrix_free(t->grad);
        t->grad = sum;
    }
    
    // Propagate gradient to parents based on operation
    switch (t->op) {
        case OP_ADD:
            // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            tensor_backward(t->parent1, grad);
            tensor_backward(t->parent2, grad);
            break;
            
        case OP_MUL:
            // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            Matrix* grad_a = matrix_mul(grad, t->parent2->data);
            Matrix* grad_b = matrix_mul(grad, t->parent1->data);
            tensor_backward(t->parent1, grad_a);
            tensor_backward(t->parent2, grad_b);
            matrix_free(grad_a);
            matrix_free(grad_b);
            break;
            
        // ... other operations
    }
}
```

## Gradient Computation Rules

### Addition: z = x + y

```
∂z/∂x = 1
∂z/∂y = 1
```

Gradient flows unchanged to both parents.

### Element-wise Multiplication: z = x ⊙ y

```
∂z/∂x = y
∂z/∂y = x
```

Gradient is multiplied by the other input.

### Matrix Multiplication: Z = X @ Y

```
∂Z/∂X = grad_output @ Y^T
∂Z/∂Y = X^T @ grad_output
```

Uses transposed matrices for proper dimension matching.

### ReLU: y = max(0, x)

```
∂y/∂x = 1 if x > 0, else 0
```

Gradient passes through for positive inputs, blocked for negative.

### Sigmoid: y = 1 / (1 + e^(-x))

```
∂y/∂x = y * (1 - y)
```

Uses the output value itself for efficient computation.

### Softmax: y_i = e^(x_i) / Σ(e^(x_j))

```
∂y_i/∂x_j = y_i * (δ_ij - y_j)
```

Where δ_ij is the Kronecker delta (1 if i=j, else 0).

## Practical Example

Let's trace through a simple computation:

```c
// Create inputs
Matrix* x_data = matrix_create(1, 3);
matrix_fill(x_data, 2.0);  // [2, 2, 2]

Matrix* w_data = matrix_create(1, 3);
matrix_fill(w_data, 3.0);  // [3, 3, 3]

Tensor* x = tensor_create(x_data, 1);
Tensor* w = tensor_create(w_data, 1);

// Forward pass
Tensor* y = tensor_mul(x, w);  // y = x * w = [6, 6, 6]

// Backward pass
Matrix* grad_output = matrix_create(1, 3);
matrix_fill(grad_output, 1.0);  // Gradient from loss = [1, 1, 1]

tensor_backward(y, grad_output);

// Results:
// x->grad = w->data * grad_output = [3, 3, 3] (gradient for x)
// w->grad = x->data * grad_output = [2, 2, 2] (gradient for w)
```

**Computation graph:**

```
x [2,2,2]     w [3,3,3]
    \           /
     \         /
      \       /
       \     /
        *   
         \ /
          y [6,6,6]
          
Backward:
          y 
         /|\
        / | \
grad=[1,1,1]

x.grad = [3,3,3]  (gradient flows back)
w.grad = [2,2,2]
```

## Memory Management

### Gradient Accumulation

When a tensor is used multiple times, gradients must be **accumulated** (summed):

```c
// z = x + x
Tensor* z = tensor_add(x, x);

// x receives gradient from both parent1 and parent2
// x.grad = grad_from_parent1 + grad_from_parent2
```

This is why we check and accumulate in `tensor_backward()`:

```c
if (t->grad == NULL) {
    t->grad = matrix_copy(grad);  // First gradient
} else {
    Matrix* sum = matrix_add(t->grad, grad);  // Accumulate
    matrix_free(t->grad);
    t->grad = sum;
}
```

### Clearing Gradients

Before each training iteration, gradients must be zeroed:

```c
void tensor_zero_grad(Tensor* t) {
    if (t->grad != NULL) {
        matrix_fill(t->grad, 0.0f);
    }
}
```

This prevents gradients from accumulating across multiple backward passes.

## Integration with Neural Networks

### Linear Layer Example

```c
// Forward pass
Matrix* output = linear_forward(layer, input);

// Layer internally caches input for backward pass
layer->input_cache = matrix_copy(input);

// Backward pass
void linear_backward(Linear* layer, Matrix* grad_output, Matrix* grad_input) {
    // Gradient w.r.t. weights: input^T @ grad_output
    Matrix* input_T = matrix_transpose(layer->input_cache);
    Matrix* grad_w = matrix_matmul(input_T, grad_output);
    
    // Accumulate gradient
    Matrix* new_grad = matrix_add(layer->grad_weights, grad_w);
    matrix_free(layer->grad_weights);
    layer->grad_weights = new_grad;
    
    // Gradient w.r.t. input: grad_output @ weights^T
    Matrix* w_T = matrix_transpose(layer->weights);
    Matrix* grad_in = matrix_matmul(grad_output, w_T);
    
    // Copy to output parameter
    memcpy(grad_input->data, grad_in->data, 
           grad_in->rows * grad_in->cols * sizeof(float));
    
    matrix_free(input_T);
    matrix_free(grad_w);
    matrix_free(w_T);
    matrix_free(grad_in);
}
```

## Advantages of Our Implementation

### 1. **Simplicity**

- Direct C implementation, no external dependencies
- Easy to understand and debug
- Explicit memory management

### 2. **Efficiency**

- Reverse-mode AD computes all gradients in one backward pass
- Time complexity: O(forward_time) for backward pass
- Memory: stores only necessary intermediate values

### 3. **Flexibility**

- Can handle arbitrary computation graphs
- Supports branching and merging
- Easy to add new operations

## Limitations and Tradeoffs

### 1. **Memory Usage**

- Must store all intermediate tensors for backward pass
- Trade memory for computation speed

### 2. **Dynamic Graphs Only**

- Graph is rebuilt on each forward pass
- No ahead-of-time optimization (unlike static graphs)

### 3. **No Higher-Order Derivatives**

- Current implementation computes only first derivatives
- Would need gradients of gradients for second-order methods

## Mathematical Foundation

### The Chain Rule

The entire autograd system is based on the chain rule:

```
If y = f(u) and u = g(x), then:
dy/dx = (dy/du) * (du/dx)
```

For multiple paths:

```
If z depends on y through multiple paths:
dz/dx = Σ (dz/dy_i) * (dy_i/dx)
```

### Reverse-Mode AD Algorithm

Given function f: ℝⁿ → ℝ (scalar output):

1. **Forward pass**: Compute f(x) and store all intermediate values
2. **Initialize**: Set v̄ₙ = 1 (gradient at output)
3. **Backward pass**: For each operation i in reverse order:

   ```
   v̄ᵢ = Σⱼ (∂vⱼ/∂vᵢ) * v̄ⱼ
   ```

   where j are children of i

This gives us all partial derivatives ∂f/∂xᵢ efficiently.

## Comparison with Other AD Methods

### Forward-Mode AD

```
Computes: J·v (Jacobian-vector product)
Efficient for: few inputs, many outputs
Example: f: ℝ → ℝⁿ
```

### Reverse-Mode AD (Our Implementation)

```
Computes: J^T·v (vector-Jacobian product)
Efficient for: many inputs, few outputs
Example: f: ℝⁿ → ℝ (typical in ML)
```

### Numerical Differentiation

```
Approximates: (f(x+h) - f(x)) / h
Problems: numerical instability, slow (2n evaluations)
```

Our reverse-mode AD is **exact** and **fast** for machine learning.

## Further Reading

- **Papers**:
  - "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al., 2018)
  - "Efficient Backprop" (LeCun et al., 1998)

- **Books**:
  - "Automatic Differentiation: Applications, Theory, and Implementations"
  - "Deep Learning" by Goodfellow et al., Chapter 6

- **Code Examples**:
  - See `src/autograd.c` for full implementation
  - See `src/train.c` for usage in training loop
  - See `src/test.c` for unit tests

## Summary

Our autograd system provides automatic gradient computation through:

1. **Dynamic computational graph** built during forward pass
2. **Reverse traversal** using chain rule during backward pass
3. **Gradient accumulation** for shared tensors
4. **Efficient implementation** with minimal memory overhead

This enables training neural networks without manually deriving or coding gradients - the system handles it automatically!
