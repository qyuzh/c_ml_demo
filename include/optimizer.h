#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn.h"

// SGD optimizer
typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
} SGD;

// Create optimizer
SGD* sgd_create(float learning_rate, float momentum, float weight_decay);
void sgd_free(SGD* optimizer);

// Update parameters
void sgd_step(SGD* optimizer, Linear** layers, size_t num_layers);

#endif // OPTIMIZER_H
