#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
  float* data;
  size_t rows;
  size_t cols;
} Matrix;

// Matrix creation and destruction
Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_create_from_data(float* data, size_t rows, size_t cols);
void matrix_free(Matrix* m);
Matrix* matrix_copy(const Matrix* m);

// Matrix initialization
void matrix_fill(Matrix* m, float value);
void matrix_random(Matrix* m, float min, float max);
void matrix_zeros(Matrix* m);
void matrix_ones(Matrix* m);

// Matrix operations
Matrix* matrix_add(const Matrix* a, const Matrix* b);
Matrix* matrix_sub(const Matrix* a, const Matrix* b);
Matrix* matrix_mul(const Matrix* a, const Matrix* b);  // Element-wise
Matrix* matrix_matmul(const Matrix* a,
                      const Matrix* b);  // Matrix multiplication
Matrix* matrix_transpose(const Matrix* m);
Matrix* matrix_scale(const Matrix* m, float scalar);

// In-place operations
void matrix_add_inplace(Matrix* a, const Matrix* b);
void matrix_scale_inplace(Matrix* m, float scalar);

// Activation functions
Matrix* matrix_relu(const Matrix* m);
Matrix* matrix_relu_derivative(const Matrix* m);
Matrix* matrix_sigmoid(const Matrix* m);
Matrix* matrix_sigmoid_derivative(const Matrix* m);

// Loss functions
float matrix_mse(const Matrix* pred, const Matrix* target);
Matrix* matrix_mse_derivative(const Matrix* pred, const Matrix* target);
float matrix_cross_entropy(const Matrix* pred, const Matrix* target);

// Softmax
Matrix* matrix_softmax(const Matrix* m);

// Utility functions
void matrix_print(const Matrix* m);
float matrix_sum(const Matrix* m);
float matrix_get(const Matrix* m, size_t row, size_t col);
void matrix_set(Matrix* m, size_t row, size_t col, float value);

#endif  // MATRIX_H
