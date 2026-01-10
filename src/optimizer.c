#include "optimizer.h"

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
  if (optimizer) {
    free(optimizer);
  }
}

// Update parameters using SGD
void sgd_step(SGD* optimizer, Linear** layers, size_t num_layers) {
  for (size_t i = 0; i < num_layers; i++) {
    Linear* layer = layers[i];

    // Update weights: w = w - lr * (grad + weight_decay * w)
    for (size_t j = 0; j < layer->weights->rows * layer->weights->cols; j++) {
      float grad = layer->weight_grad->data[j];

      // Add weight decay
      if (optimizer->weight_decay > 0) {
        grad += optimizer->weight_decay * layer->weights->data[j];
      }

      // Update weight
      layer->weights->data[j] -= optimizer->learning_rate * grad;
    }

    // Update biases: b = b - lr * grad
    for (size_t j = 0; j < layer->bias->cols; j++) {
      layer->bias->data[j] -=
          optimizer->learning_rate * layer->bias_grad->data[j];
    }
  }
}
