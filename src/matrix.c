#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Matrix creation and destruction
Matrix* matrix_create(size_t rows, size_t cols) {
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->data = (float*)calloc(rows * cols, sizeof(float));
  return m;
}

Matrix* matrix_create_from_data(float* data, size_t rows, size_t cols) {
  Matrix* m = matrix_create(rows, cols);
  memcpy(m->data, data, rows * cols * sizeof(float));
  return m;
}

void matrix_free(Matrix* m) {
  if (m) {
    if (m->data) {
      free(m->data);
    }
    free(m);
  }
}

Matrix* matrix_copy(const Matrix* m) {
  Matrix* copy = matrix_create(m->rows, m->cols);
  memcpy(copy->data, m->data, m->rows * m->cols * sizeof(float));
  return copy;
}

// Matrix initialization
void matrix_fill(Matrix* m, float value) {
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = value;
  }
}

void matrix_random(Matrix* m, float min, float max) {
  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  for (size_t i = 0; i < m->rows * m->cols; i++) {
    float random = (float)rand() / (float)RAND_MAX;
    m->data[i] = min + random * (max - min);
  }
}

void matrix_zeros(Matrix* m) { matrix_fill(m, 0.0f); }

void matrix_ones(Matrix* m) { matrix_fill(m, 1.0f); }

// Matrix operations
Matrix* matrix_add(const Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions don't match for addition\n");
    return NULL;
  }

  Matrix* result = matrix_create(a->rows, a->cols);
  for (size_t i = 0; i < a->rows * a->cols; i++) {
    result->data[i] = a->data[i] + b->data[i];
  }
  return result;
}

Matrix* matrix_sub(const Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions don't match for subtraction\n");
    return NULL;
  }

  Matrix* result = matrix_create(a->rows, a->cols);
  for (size_t i = 0; i < a->rows * a->cols; i++) {
    result->data[i] = a->data[i] - b->data[i];
  }
  return result;
}

Matrix* matrix_mul(const Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr,
            "Error: Matrix dimensions don't match for element-wise "
            "multiplication\n");
    return NULL;
  }

  Matrix* result = matrix_create(a->rows, a->cols);
  for (size_t i = 0; i < a->rows * a->cols; i++) {
    result->data[i] = a->data[i] * b->data[i];
  }
  return result;
}

Matrix* matrix_matmul(const Matrix* a, const Matrix* b) {
  if (a->cols != b->rows) {
    fprintf(stderr,
            "Error: Matrix dimensions incompatible for matrix multiplication "
            "(%zux%zu) x (%zux%zu)\n",
            a->rows, a->cols, b->rows, b->cols);
    return NULL;
  }

  Matrix* result = matrix_create(a->rows, b->cols);

  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < b->cols; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < a->cols; k++) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      result->data[i * result->cols + j] = sum;
    }
  }

  return result;
}

Matrix* matrix_transpose(const Matrix* m) {
  Matrix* result = matrix_create(m->cols, m->rows);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      result->data[j * result->cols + i] = m->data[i * m->cols + j];
    }
  }

  return result;
}

Matrix* matrix_scale(const Matrix* m, float scalar) {
  Matrix* result = matrix_create(m->rows, m->cols);
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    result->data[i] = m->data[i] * scalar;
  }
  return result;
}

// In-place operations
void matrix_add_inplace(Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr,
            "Error: Matrix dimensions don't match for in-place addition\n");
    return;
  }

  for (size_t i = 0; i < a->rows * a->cols; i++) {
    a->data[i] += b->data[i];
  }
}

void matrix_scale_inplace(Matrix* m, float scalar) {
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    m->data[i] *= scalar;
  }
}

// Activation functions
Matrix* matrix_relu(const Matrix* m) {
  Matrix* result = matrix_create(m->rows, m->cols);
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    result->data[i] = m->data[i] > 0 ? m->data[i] : 0;
  }
  return result;
}

Matrix* matrix_relu_derivative(const Matrix* m) {
  Matrix* result = matrix_create(m->rows, m->cols);
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    result->data[i] = m->data[i] > 0 ? 1.0f : 0.0f;
  }
  return result;
}

Matrix* matrix_sigmoid(const Matrix* m) {
  Matrix* result = matrix_create(m->rows, m->cols);
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    result->data[i] = 1.0f / (1.0f + expf(-m->data[i]));
  }
  return result;
}

Matrix* matrix_sigmoid_derivative(const Matrix* m) {
  Matrix* result = matrix_create(m->rows, m->cols);
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    float sig = 1.0f / (1.0f + expf(-m->data[i]));
    result->data[i] = sig * (1.0f - sig);
  }
  return result;
}

// Loss functions
float matrix_mse(const Matrix* pred, const Matrix* target) {
  if (pred->rows != target->rows || pred->cols != target->cols) {
    fprintf(stderr, "Error: Matrix dimensions don't match for MSE\n");
    return -1.0f;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < pred->rows * pred->cols; i++) {
    float diff = pred->data[i] - target->data[i];
    sum += diff * diff;
  }

  return sum / (pred->rows * pred->cols);
}

Matrix* matrix_mse_derivative(const Matrix* pred, const Matrix* target) {
  if (pred->rows != target->rows || pred->cols != target->cols) {
    fprintf(stderr,
            "Error: Matrix dimensions don't match for MSE derivative\n");
    return NULL;
  }

  Matrix* result = matrix_create(pred->rows, pred->cols);
  float scale = 2.0f / (pred->rows * pred->cols);

  for (size_t i = 0; i < pred->rows * pred->cols; i++) {
    result->data[i] = scale * (pred->data[i] - target->data[i]);
  }

  return result;
}

float matrix_cross_entropy(const Matrix* pred, const Matrix* target) {
  if (pred->rows != target->rows || pred->cols != target->cols) {
    fprintf(stderr, "Error: Matrix dimensions don't match for cross entropy\n");
    return -1.0f;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < pred->rows * pred->cols; i++) {
    // Clip predictions to avoid log(0)
    float p = fmaxf(pred->data[i], 1e-7f);
    p = fminf(p, 1.0f - 1e-7f);
    sum -= target->data[i] * logf(p);
  }

  return sum / pred->rows;
}

// Softmax
Matrix* matrix_softmax(const Matrix* m) {
  Matrix* result = matrix_create(m->rows, m->cols);

  for (size_t i = 0; i < m->rows; i++) {
    // Find max for numerical stability
    float max_val = m->data[i * m->cols];
    for (size_t j = 1; j < m->cols; j++) {
      if (m->data[i * m->cols + j] > max_val) {
        max_val = m->data[i * m->cols + j];
      }
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t j = 0; j < m->cols; j++) {
      result->data[i * m->cols + j] = expf(m->data[i * m->cols + j] - max_val);
      sum += result->data[i * m->cols + j];
    }

    // Normalize
    for (size_t j = 0; j < m->cols; j++) {
      result->data[i * m->cols + j] /= sum;
    }
  }

  return result;
}

// Utility functions
void matrix_print(const Matrix* m) {
  printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      printf("%8.4f ", m->data[i * m->cols + j]);
    }
    printf("\n");
  }
}

float matrix_sum(const Matrix* m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    sum += m->data[i];
  }
  return sum;
}

float matrix_get(const Matrix* m, size_t row, size_t col) {
  return m->data[row * m->cols + col];
}

void matrix_set(Matrix* m, size_t row, size_t col, float value) {
  m->data[row * m->cols + col] = value;
}
