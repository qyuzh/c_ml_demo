#include "optimizer.h"

#include <stdio.h>
#include <stdlib.h>

// Create optimizer
SGD* sgd_create(float learning_rate, float momentum, float weight_decay) {
  SGD* optimizer = (SGD*)malloc(sizeof(SGD));
  optimizer->learning_rate = learning_rate;
  optimizer->momentum = momentum;
  optimizer->weight_decay = weight_decay;
  return optimizer;
}

void sgd_free(SGD* optimizer) {
  free(optimizer);
}

// Update a single tensor's parameters using its gradient
void sgd_update_tensor(SGD* optimizer, Tensor* param) {
  if (!param || !param->requires_grad || !param->grad) {
    return;
  }
  
  Matrix* data = param->data;
  Matrix* grad = param->grad;
  
  // Update rule: param = param - learning_rate * (grad + weight_decay * param)
  for (size_t i = 0; i < data->rows * data->cols; i++) {
    // Add weight decay (L2 regularization)
    float gradient = grad->data[i];
    if (optimizer->weight_decay > 0.0f) {
      gradient += optimizer->weight_decay * data->data[i];
    }
    
    // SGD update: param -= learning_rate * gradient
    data->data[i] -= optimizer->learning_rate * gradient;
  }
}

// Update parameters for all layers
void sgd_step(SGD* optimizer, Linear** layers, size_t num_layers) {
  for (size_t i = 0; i < num_layers; i++) {
    Linear* layer = layers[i];
    
    // Update weights
    sgd_update_tensor(optimizer, layer->weights);
    
    // Update bias
    sgd_update_tensor(optimizer, layer->bias);
  }
}