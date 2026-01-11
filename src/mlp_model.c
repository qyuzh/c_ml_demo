#include "mlp_model.h"

#include <math.h>
#include <stdlib.h>

// Create MLP model with specified architecture
MLPAutograd* mlp_create(size_t input_size, size_t hidden1_size,
                        size_t hidden2_size, size_t output_size) {
  MLPAutograd* model = (MLPAutograd*)malloc(sizeof(MLPAutograd));

  model->fc1 = linear_create(input_size, hidden1_size);
  model->fc2 = linear_create(hidden1_size, hidden2_size);
  model->fc3 = linear_create(hidden2_size, output_size);

  // Initialize weights
  linear_init_he(model->fc1);
  linear_init_he(model->fc2);
  linear_init_xavier(model->fc3);

  return model;
}

// Free MLP model
void mlp_free(MLPAutograd* model) {
  if (model) {
    if (model->fc1) linear_free(model->fc1);
    if (model->fc2) linear_free(model->fc2);
    if (model->fc3) linear_free(model->fc3);
    free(model);
  }
}

// Forward pass for training - builds computational graph and keeps intermediates
ForwardResult* mlp_forward_train(MLPAutograd* model, Tensor* x) {
  ForwardResult* result = (ForwardResult*)malloc(sizeof(ForwardResult));
  
  // Layer 1: fc1 -> ReLU
  // The computational graph is built automatically here
  result->h1 = linear_forward(model->fc1, x);
  result->a1 = tensor_relu(result->h1);

  // Layer 2: fc2 -> ReLU
  result->h2 = linear_forward(model->fc2, result->a1);
  result->a2 = tensor_relu(result->h2);

  // Layer 3: fc3 -> Softmax
  result->h3 = linear_forward(model->fc3, result->a2);
  result->output = tensor_softmax(result->h3);

  return result;
}

// Free forward result
void forward_result_free(ForwardResult* result) {
  if (result) {
    if (result->h1) tensor_free(result->h1);
    if (result->a1) tensor_free(result->a1);
    if (result->h2) tensor_free(result->h2);
    if (result->a2) tensor_free(result->a2);
    if (result->h3) tensor_free(result->h3);
    if (result->output) tensor_free(result->output);
    free(result);
  }
}

// Forward pass for inference - no need to keep intermediates
Tensor* mlp_forward_inference(MLPAutograd* model, Tensor* x) {
  // Layer 1: fc1 -> ReLU
  Tensor* h1 = linear_forward(model->fc1, x);
  Tensor* a1 = tensor_relu(h1);
  tensor_free(h1);

  // Layer 2: fc2 -> ReLU
  Tensor* h2 = linear_forward(model->fc2, a1);
  Tensor* a2 = tensor_relu(h2);
  tensor_free(h2);
  tensor_free(a1);

  // Layer 3: fc3 -> Softmax
  Tensor* h3 = linear_forward(model->fc3, a2);
  Tensor* output = tensor_softmax(h3);
  tensor_free(h3);
  tensor_free(a2);

  return output;
}

// Zero gradients for all layers
void mlp_zero_grad(MLPAutograd* model) {
  linear_zero_grad(model->fc1);
  linear_zero_grad(model->fc2);
  linear_zero_grad(model->fc3);
}

// Get layers array (useful for optimizer and weights I/O)
void mlp_get_layers(MLPAutograd* model, Linear** layers, size_t* num_layers) {
  if (layers) {
    layers[0] = model->fc1;
    layers[1] = model->fc2;
    layers[2] = model->fc3;
  }
  if (num_layers) {
    *num_layers = 3;
  }
}

// ============================================================================
// Loss Function Implementation
// ============================================================================

// Compute Cross-Entropy Loss with Softmax and its gradient
// 
// This function encapsulates the mathematical property that makes
// training neural networks with softmax classification efficient.
MLPLossResult* mlp_loss(const Matrix* predictions, const Matrix* labels) {
  // Validate inputs
  if (!predictions || !labels) {
    return NULL;
  }
  
  if (predictions->rows != labels->rows || 
      predictions->cols != labels->cols) {
    return NULL;
  }

  // Allocate result structure
  MLPLossResult* result = (MLPLossResult*)malloc(sizeof(MLPLossResult));
  if (!result) {
    return NULL;
  }

  size_t batch_size = predictions->rows;
  size_t num_classes = predictions->cols;
  
  // ============================================================================
  // PART 1: Compute Loss Value
  // ============================================================================
  // Cross-Entropy: L = -Σ yᵢ * log(ŷᵢ)
  // where:
  //   yᵢ = true label (one-hot: 1 for correct class, 0 for others)
  //   ŷᵢ = predicted probability from softmax
  
  float total_loss = 0.0f;
  
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < num_classes; j++) {
      size_t idx = i * num_classes + j;
      
      float pred = predictions->data[idx];
      float label = labels->data[idx];
      
      // Clamp prediction to avoid log(0) → -∞
      if (pred < 1e-7f) pred = 1e-7f;
      if (pred > 1.0f - 1e-7f) pred = 1.0f - 1e-7f;
      
      // Only non-zero labels contribute to loss (sparse one-hot)
      if (label > 0.0f) {
        total_loss -= label * logf(pred);
      }
    }
  }
  
  // Average loss over batch
  result->loss_value = total_loss / batch_size;
  
  // ============================================================================
  // PART 2: Compute Gradient
  // ============================================================================
  // The gradient of CrossEntropy(Softmax(z)) w.r.t. logits z is:
  //
  //   ∂L/∂zᵢ = ŷᵢ - yᵢ
  //
  // Derivation (for the curious):
  // ----------------------------
  // Let z = logits (pre-softmax), ŷ = softmax(z), y = labels
  //
  // 1. Softmax: ŷᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
  //
  // 2. Cross-Entropy: L = -Σᵢ yᵢ log(ŷᵢ)
  //
  // 3. Chain rule: ∂L/∂zᵢ = Σⱼ (∂L/∂ŷⱼ) * (∂ŷⱼ/∂zᵢ)
  //
  // 4. Softmax derivative has special property:
  //    ∂ŷⱼ/∂zᵢ = ŷⱼ(δᵢⱼ - ŷᵢ)  where δᵢⱼ is Kronecker delta
  //
  // 5. After substitution and simplification:
  //    ∂L/∂zᵢ = ŷᵢ Σⱼ yⱼ - yᵢ = ŷᵢ * 1 - yᵢ = ŷᵢ - yᵢ
  //    (because Σⱼ yⱼ = 1 for one-hot encoding)
  //
  // This elegant result:
  // - Avoids numerical instability (no log derivatives)
  // - Is computationally efficient (simple subtraction)
  // - Provides natural error signal (larger when more wrong)
  
  result->gradients = matrix_create(batch_size, num_classes);
  if (!result->gradients) {
    free(result);
    return NULL;
  }
  
  // Compute: gradient = predictions - labels
  for (size_t i = 0; i < batch_size * num_classes; i++) {
    result->gradients->data[i] = predictions->data[i] - labels->data[i];
  }
  
  return result;
}

// Free MLP loss result
void mlp_loss_result_free(MLPLossResult* result) {
  if (result) {
    if (result->gradients) {
      matrix_free(result->gradients);
    }
    free(result);
  }
}