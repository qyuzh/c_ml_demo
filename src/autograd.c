#include "autograd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tensor creation and destruction
Tensor* tensor_create(Matrix* data, int requires_grad) {
  Tensor* t = (Tensor*)malloc(sizeof(Tensor));
  t->data = data;
  t->grad = NULL;
  if (requires_grad) {
    t->grad = matrix_create(data->rows, data->cols);
    matrix_zeros(t->grad);
  }
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
Tensor* tensor_add(Tensor* a, Tensor* b) {
  Matrix* result_data = matrix_add(a->data, b->data);
  int requires_grad = a->requires_grad || b->requires_grad;
  Tensor* result = tensor_create(result_data, requires_grad);

  if (requires_grad) {
    result->op = OP_ADD;
    result->parent1 = a;
    result->parent2 = b;
  }

  return result;
}

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

Tensor* tensor_relu(Tensor* a) {
  Matrix* result_data = matrix_relu(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_RELU;
    result->parent1 = a;
  }

  return result;
}

Tensor* tensor_sigmoid(Tensor* a) {
  Matrix* result_data = matrix_sigmoid(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_SIGMOID;
    result->parent1 = a;
  }

  return result;
}

Tensor* tensor_softmax(Tensor* a) {
  Matrix* result_data = matrix_softmax(a->data);
  Tensor* result = tensor_create(result_data, a->requires_grad);

  if (a->requires_grad) {
    result->op = OP_SOFTMAX;
    result->parent1 = a;
  }

  return result;
}

// Backward operations
void tensor_backward(Tensor* t) {
  if (!t->requires_grad) {
    return;
  }

  // Initialize gradient if this is the root (loss)
  if (t->grad->rows == 1 && t->grad->cols == 1 && t->grad->data[0] == 0.0f) {
    t->grad->data[0] = 1.0f;
  }

  switch (t->op) {
    case OP_ADD:
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
        matrix_add_inplace(t->parent2->grad, t->grad);
      }
      break;

    case OP_SUB:
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
        Matrix* neg_grad = matrix_scale(t->grad, -1.0f);
        matrix_add_inplace(t->parent2->grad, neg_grad);
        matrix_free(neg_grad);
      }
      break;

    case OP_MUL:
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
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
        Matrix* grad2 = matrix_mul(t->grad, t->parent1->data);
        matrix_add_inplace(t->parent2->grad, grad2);
        matrix_free(grad2);
      }
      break;

    case OP_MATMUL:
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        // grad_a = grad_output @ b^T
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
        // grad_b = a^T @ grad_output
        Matrix* a_t = matrix_transpose(t->parent1->data);
        Matrix* grad2 = matrix_matmul(a_t, t->grad);
        matrix_add_inplace(t->parent2->grad, grad2);
        matrix_free(a_t);
        matrix_free(grad2);
      }
      break;

    case OP_RELU:
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        Matrix* relu_grad = matrix_relu_derivative(t->parent1->data);
        Matrix* grad = matrix_mul(t->grad, relu_grad);
        matrix_add_inplace(t->parent1->grad, grad);
        matrix_free(relu_grad);
        matrix_free(grad);
      }
      break;

    case OP_SIGMOID:
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        Matrix* sig_grad = matrix_sigmoid_derivative(t->parent1->data);
        Matrix* grad = matrix_mul(t->grad, sig_grad);
        matrix_add_inplace(t->parent1->grad, grad);
        matrix_free(sig_grad);
        matrix_free(grad);
      }
      break;

    case OP_SOFTMAX:
      // For softmax with cross-entropy, gradient is typically computed directly
      // Here we implement simplified version
      if (t->parent1 && t->parent1->requires_grad) {
        if (!t->parent1->grad) {
          t->parent1->grad =
              matrix_create(t->parent1->data->rows, t->parent1->data->cols);
          matrix_zeros(t->parent1->grad);
        }
        matrix_add_inplace(t->parent1->grad, t->grad);
      }
      break;

    case OP_NONE:
      // Leaf node, no backward pass needed
      break;
  }

  // Recursively backward through parents
  if (t->parent1) {
    tensor_backward(t->parent1);
  }
  if (t->parent2) {
    tensor_backward(t->parent2);
  }
}

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
