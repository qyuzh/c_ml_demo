#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include "autograd.h"
#include "matrix.h"
#include "nn.h"

// Simple MLP for MNIST using Autograd
typedef struct {
  Linear* fc1;
  Linear* fc2;
  Linear* fc3;
} MLPAutograd;

// Model lifecycle
MLPAutograd* mlp_create(size_t input_size, size_t hidden1_size,
                        size_t hidden2_size, size_t output_size);
void mlp_free(MLPAutograd* model);

// Forward pass for training (keeps intermediate results)
typedef struct {
  Tensor* h1;
  Tensor* a1;
  Tensor* h2;
  Tensor* a2;
  Tensor* h3;
  Tensor* output;
} ForwardResult;

ForwardResult* mlp_forward_train(MLPAutograd* model, Tensor* x);
void forward_result_free(ForwardResult* result);

// Forward pass for inference (cleans up intermediates automatically)
Tensor* mlp_forward_inference(MLPAutograd* model, Tensor* x);

// Gradient operations
void mlp_zero_grad(MLPAutograd* model);

// Get layers array (useful for optimizer and weights I/O)
void mlp_get_layers(MLPAutograd* model, Linear** layers, size_t* num_layers);

// ============================================================================
// Loss Function for MLP (Cross-Entropy with Softmax)
// ============================================================================
// This loss function is specifically designed for the MLP model's softmax output.
// 
// Mathematical background:
// ========================
// Loss: L = -Σ yᵢ * log(softmax(zᵢ))
// 
// The beautiful property:
// When computing ∂L/∂z (gradient w.r.t. logits before softmax),
// the derivative simplifies to:
//   ∂L/∂zᵢ = softmax(zᵢ) - yᵢ = predictions - labels
// 
// This elegant result comes from the chain rule and softmax properties,
// avoiding numerical instability from separate softmax + log operations.

typedef struct {
  float loss_value;      // Scalar loss value
  Matrix* gradients;     // Gradient matrix (same shape as predictions)
} MLPLossResult;

// Compute Cross-Entropy loss with Softmax for MLP training
// 
// Parameters:
//   predictions: Output from softmax activation [batch_size x num_classes]
//   labels: One-hot encoded true labels [batch_size x num_classes]
// 
// Returns:
//   MLPLossResult containing:
//     - loss_value: Average cross-entropy loss over the batch
//     - gradients: ∂L/∂z = predictions - labels [batch_size x num_classes]
// 
// Note: The gradient is w.r.t. pre-softmax logits, even though we pass
//       post-softmax predictions. This is the mathematically correct gradient
//       for the fused CrossEntropy+Softmax operation.
MLPLossResult* mlp_loss(const Matrix* predictions, const Matrix* labels);

// Free MLP loss result
void mlp_loss_result_free(MLPLossResult* result);

#endif // MLP_MODEL_H