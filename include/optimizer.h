#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "autograd.h"
#include "nn.h"

// SGD optimizer for neural network layers
typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
} SGD;

// Create optimizer
SGD* sgd_create(float learning_rate, float momentum, float weight_decay);
void sgd_free(SGD* optimizer);

// Update parameters using gradients from autograd
void sgd_step(SGD* optimizer, Linear** layers, size_t num_layers);

// Update a single tensor's parameters
void sgd_update_tensor(SGD* optimizer, Tensor* param);

#endif // OPTIMIZER_H