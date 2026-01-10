#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "nn.h"

// Save model weights to file
int save_weights(const char* filename, Linear** layers, size_t num_layers);

// Load model weights from file
int load_weights(const char* filename, Linear** layers, size_t num_layers);

#endif // WEIGHTS_H
