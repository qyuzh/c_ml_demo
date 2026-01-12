#include "autograd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tensor creation and destruction

/**
 * Creates a tensor for automatic differentiation
 * This is a leaf node in the computational graph (no parents)
 */
Tensor* tensor_create(Matrix* data, int requires_grad) {
  Tensor* t = (Tensor*)malloc(sizeof(Tensor));
  t->data = data;  // Store the actual values
  t->grad = NULL;  // Gradient starts as NULL

  // If gradient tracking is enabled, allocate gradient matrix
  // Initialize to zeros - gradients accumulate during backprop
  if (requires_grad) {
    t->grad = matrix_create(data->rows, data->cols);
    matrix_zeros(t->grad);
  }

  // Leaf nodes have no operation or parents
  t->op = OP_NONE;
  t->parent1 = NULL;
  t->parent2 = NULL;
  t->requires_grad = requires_grad;
  return t;
}

Tensor* tensor_create_from_shape(size_t rows, size_t cols, int requires_grad) {
  Matrix* data = matrix_create(rows, cols);
  return tensor_create(data, requires_grad);
}

void tensor_free(Tensor* t) {
  if (t) {
    if (t->data) {
      matrix_free(t->data);
    }
    if (t->grad) {
      matrix_free(t->grad);
    }
    free(t);
  }
}

// Forward operations

/**
 * Addition: z = a + b
 * Builds computational graph by recording operation and parents
 * During backward: ∂z/∂a = 1, ∂z/∂b = 1 (gradient flows unchanged)
 */
Tensor* tensor_add(Tensor* a, Tensor* b) {
  // Forward computation: compute the result
  Matrix* result_data = matrix_add(a->data, b->data);

  // Enable gradient tracking if either parent requires it
  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  // Build the computational graph: record how this tensor was created
  if (requires_grad) {
    result->op = OP_ADD;  // Remember this came from addition
    result->parent1 = a;  // Link to parent tensors
    result->parent2 = b;  // This allows backward pass to propagate gradients
  }

  return result;
}

/**
 * Subtraction: z = a - b
 * During backward: ∂z/∂a = 1, ∂z/∂b = -1 (gradient negated for subtrahend)
 */
Tensor* tensor_sub(Tensor* a, Tensor* b) {
  Matrix* result_data = matrix_sub(a->data, b->data);
  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  if (requires_grad) {
    result->op = OP_SUB;
    result->parent1 = a;
    result->parent2 = b;
  }

  return result;
}

/**
 * Element-wise multiplication: z = a ⊙ b
 * During backward: ∂z/∂a = b, ∂z/∂b = a (gradient scaled by other input)
 */
Tensor* tensor_mul(Tensor* a, Tensor* b) {
  Matrix* result_data = matrix_mul(a->data, b->data);
  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  if (requires_grad) {
    result->op = OP_MUL;
    result->parent1 = a;
    result->parent2 = b;
  }

  return result;
}

/**
 * Matrix multiplication: Z = A @ B
 * During backward: ∂Z/∂A = grad_output @ B^T, ∂Z/∂B = A^T @ grad_output
 * This is the core operation for neural network layers
 */
Tensor* tensor_matmul(Tensor* a, Tensor* b) {
  Matrix* result_data = matrix_matmul(a->data, b->data);
  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  if (requires_grad) {
    result->op = OP_MATMUL;
    result->parent1 = a;
    result->parent2 = b;
  }

  return result;
}

/**
 * ReLU activation: y = max(0, x)
 * During backward: ∂y/∂x = 1 if x > 0, else 0 (gate gradient)
 * Unary operation - only one parent
 */
Tensor* tensor_relu(Tensor* a) {
  Matrix* result_data = matrix_relu(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_RELU;
    result->parent1 = a;  // Only one parent for unary operations
  }

  return result;
}

/**
 * Sigmoid activation: y = 1 / (1 + e^(-x))
 * During backward: ∂y/∂x = y(1-y) (uses output value for efficiency)
 */
Tensor* tensor_sigmoid(Tensor* a) {
  Matrix* result_data = matrix_sigmoid(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_SIGMOID;
    result->parent1 = a;
  }

  return result;
}

/**
 * Softmax activation: y_i = e^(x_i) / Σ(e^(x_j))
 * Typically used with cross-entropy loss
 * During backward: gradient depends on Jacobian matrix of softmax
 */
Tensor* tensor_softmax(Tensor* a) {
  Matrix* result_data = matrix_softmax(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_SOFTMAX;
    result->parent1 = a;
  }

  return result;
}

/**
 * Broadcast addition: z = a + b (broadcast b across batch dimension)
 *
 * SHAPES:
 *   Input a: [batch_size, features]  - Main tensor (e.g., layer output)
 *   Input b: [1, features]            - Bias vector to broadcast
 *   Output:  [batch_size, features]  - Result after broadcasting
 *
 * FORWARD PASS:
 *   For each sample i in batch: output[i, j] = a[i, j] + b[0, j]
 *   Bias b[0, j] is added to corresponding feature j in all batch samples
 *
 * BACKWARD PASS (implemented in tensor_backward):
 *   Gradient for a: flows unchanged (shape preserved)
 *     ∂L/∂a[i,j] = ∂L/∂output[i,j]
 *
 *   Gradient for b: sum across batch dimension (reduction)
 *     ∂L/∂b[0,j] = Σᵢ ∂L/∂output[i,j]
 *     This is why bias gradient is computed by summing over batch!
 *
 * EXAMPLE:
 *   a = [[1, 2, 3],    b = [[10, 20, 30]]
 *        [4, 5, 6]]     (broadcasted to each row)
 *
 *   result = [[11, 22, 33],
 *             [14, 25, 36]]
 *
 * Typical use: adding bias to matrix multiplication output in linear layers
 */
Tensor* tensor_broadcast_add(Tensor* a, Tensor* b) {
  // Forward: broadcast b (bias) across batch dimension of a
  // a: [batch_size, features], b: [1, features] → result: [batch_size,
  // features]
  Matrix* result_data = matrix_create(a->data->rows, a->data->cols);

  for (size_t i = 0; i < a->data->rows; i++) {
    for (size_t j = 0; j < a->data->cols; j++) {
      result_data->data[i * a->data->cols + j] =
          a->data->data[i * a->data->cols + j] + b->data->data[j];
    }
  }

  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  if (requires_grad) {
    result->op = OP_BROADCAST_ADD;
    result->parent1 = a;
    result->parent2 = b;
  }

  return result;
}

// Backward operations

/**
 * Backward pass: Computes gradients using reverse-mode automatic
 * differentiation Traverses computational graph backwards, applying chain rule
 * at each node This is the heart of backpropagation!
 */
void tensor_backward(Tensor* t) {
  if (!t->requires_grad) {
    return;  // Skip nodes that don't need gradients
  }

  // Initialize gradient at output (loss) node
  // By convention, ∂Loss/∂Loss = 1 (seed gradient for chain rule)
  if (t->grad->rows == 1 && t->grad->cols == 1 && t->grad->data[0] == 0.0f) {
    t->grad->data[0] = 1.0f;
  }

  // Apply chain rule based on operation type
  // For each case: local_gradient = ∂output/∂input
  // Then: input.grad += local_gradient * t.grad (chain rule)
  switch (t->op) {
    case OP_ADD:
      // Addition: z = a + b
      // ∂z/∂a = 1, ∂z/∂b = 1
      // Gradient flows unchanged to both parents
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Accumulate gradient (important for tensors used multiple times)
        matrix_add_inplace(t->parent1->grad, t->grad);
      }
      if (t->parent2 && t->parent2->requires_grad) {
        if (!t->parent2->grad) {
          t->parent2->grad =
              matrix_create(t->parent2->data->rows, t->parent2->data->cols);
          matrix_zeros(t->parent2->grad);
        }
        matrix_add_inplace(t->parent2->grad, t->grad);
      }
      break;

    case OP_SUB:
      // Subtraction: z = a - b
      // ∂z/∂a = 1, ∂z/∂b = -1
      // Gradient flows to first parent unchanged, negated to second parent
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        matrix_add_inplace(t->parent1->grad, t->grad);
      }
      if (t->parent2 && t->parent2->requires_grad) {
        if (!t->parent2->grad) {
          t->parent2->grad =
              matrix_create(t->parent2->data->rows, t->parent2->data->cols);
          matrix_zeros(t->parent2->grad);
        }
        // Negate gradient for subtraction
        Matrix* neg_grad = matrix_scale(t->grad, -1.0f);
        matrix_add_inplace(t->parent2->grad, neg_grad);
        matrix_free(neg_grad);
      }
      break;

    case OP_MUL:
      // Element-wise multiplication: z = a ⊙ b
      // ∂z/∂a = b, ∂z/∂b = a
      // Gradient is scaled by the other input (product rule)
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Chain rule: ∂Loss/∂a = ∂Loss/∂z * ∂z/∂a = grad * b
        Matrix* grad1 = matrix_mul(t->grad, t->parent2->data);
        matrix_add_inplace(t->parent1->grad, grad1);
        matrix_free(grad1);
      }
      if (t->parent2 && t->parent2->requires_grad) {
        if (!t->parent2->grad) {
          t->parent2->grad =
              matrix_create(t->parent2->data->rows, t->parent2->data->cols);
          matrix_zeros(t->parent2->grad);
        }
        // Chain rule: ∂Loss/∂b = ∂Loss/∂z * ∂z/∂b = grad * a
        Matrix* grad2 = matrix_mul(t->grad, t->parent1->data);
        matrix_add_inplace(t->parent2->grad, grad2);
        matrix_free(grad2);
      }
      break;

    case OP_MATMUL:
      // Matrix multiplication: Z = A @ B
      // ∂Z/∂A = grad_output @ B^T
      // ∂Z/∂B = A^T @ grad_output
      // This is crucial for neural network weight updates!
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Gradient for left matrix: multiply grad by transpose of right matrix
        // This ensures dimensions match: (m×n) @ (n×p) -> (m×p)
        // grad_a shape must match a shape: (m×n) = (m×p) @ (p×n)
        Matrix* b_t = matrix_transpose(t->parent2->data);
        Matrix* grad1 = matrix_matmul(t->grad, b_t);
        matrix_add_inplace(t->parent1->grad, grad1);
        matrix_free(b_t);
        matrix_free(grad1);
      }
      if (t->parent2 && t->parent2->requires_grad) {
        if (!t->parent2->grad) {
          t->parent2->grad =
              matrix_create(t->parent2->data->rows, t->parent2->data->cols);
          matrix_zeros(t->parent2->grad);
        }
        // Gradient for right matrix: multiply transpose of left matrix by grad
        // grad_b shape must match b shape: (n×p) = (n×m) @ (m×p)
        Matrix* a_t = matrix_transpose(t->parent1->data);
        Matrix* grad2 = matrix_matmul(a_t, t->grad);
        matrix_add_inplace(t->parent2->grad, grad2);
        matrix_free(a_t);
        matrix_free(grad2);
      }
      break;

    case OP_RELU:
      // ReLU: y = max(0, x)
      // ∂y/∂x = 1 if x > 0, else 0
      // Acts as a gradient gate - blocks gradient for negative inputs
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Compute local gradient (1 for positive, 0 for negative)
        Matrix* relu_grad = matrix_relu_derivative(t->parent1->data);
        // Chain rule: multiply incoming gradient by local gradient
        Matrix* grad = matrix_mul(t->grad, relu_grad);
        matrix_add_inplace(t->parent1->grad, grad);
        matrix_free(relu_grad);
        matrix_free(grad);
      }
      break;

    case OP_SIGMOID:
      // Sigmoid: y = 1 / (1 + e^(-x))
      // ∂y/∂x = y(1-y) = sigmoid(x)(1-sigmoid(x))
      // Efficient: uses output value instead of recomputing sigmoid
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Local gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        Matrix* sig_grad = matrix_sigmoid_derivative(t->parent1->data);
        Matrix* grad = matrix_mul(t->grad, sig_grad);
        matrix_add_inplace(t->parent1->grad, grad);
        matrix_free(sig_grad);
        matrix_free(grad);
      }
      break;

    case OP_SOFTMAX:
      // Softmax: y_i = e^(x_i) / Σ(e^(x_j))
      // Full gradient: ∂y_i/∂x_j = y_i(δ_ij - y_j) [Jacobian matrix]
      // When combined with cross-entropy loss, simplifies to (y - target)
      // Here we implement simplified version - in practice, combined with loss
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // Pass gradient through unchanged (simplified for
        // softmax+cross-entropy)
        matrix_add_inplace(t->parent1->grad, t->grad);
      }
      break;

    case OP_BROADCAST_ADD:
      // Broadcast addition: z = a + b (b broadcast across batch)
      // ∂z/∂a = 1 (gradient flows unchanged)
      // ∂z/∂b = sum across batch dimension (reduction)
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        matrix_add_inplace(t->parent1->grad, t->grad);
      }

      // Gradient for bias: sum across batch dimension (rows)
      if (t->parent2 && t->parent2->requires_grad) {
        if (!t->parent2->grad) {
          t->parent2->grad =
              matrix_create(t->parent2->data->rows, t->parent2->data->cols);
          matrix_zeros(t->parent2->grad);
        }

        // Sum gradients across rows (batch dimension)
        for (size_t j = 0; j < t->grad->cols; j++) {
          float sum = 0.0f;
          for (size_t i = 0; i < t->grad->rows; i++) {
            sum += t->grad->data[i * t->grad->cols + j];
          }
          t->parent2->grad->data[j] += sum;
        }
      }
      break;

    case OP_NONE:
      // Leaf node (input/parameter), no backward pass needed
      // Gradients stop here - these are the values we update during training
      break;
  }

  // Recursively backward through parents (reverse topological order)
  // This implements reverse-mode automatic differentiation
  // Ensures gradients propagate from outputs back to all inputs
  if (t->parent1) {
    tensor_backward(t->parent1);
  }
  if (t->parent2) {
    tensor_backward(t->parent2);
  }
}

/**
 * Reset gradients to zero
 * CRITICAL: Must be called before each training iteration
 * Otherwise gradients accumulate across batches (usually unwanted)
 * Exception: gradient accumulation for large batches
 */
void tensor_zero_grad(Tensor* t) {
  if (t && t->grad) {
    matrix_zeros(t->grad);
  }
}

// Utility functions
void tensor_print(const Tensor* t) {
  printf("Tensor Data:\n");
  matrix_print(t->data);
  if (t->grad) {
    printf("Tensor Gradient:\n");
    matrix_print(t->grad);
  }
}
