#include "weights.h"

#include <stdio.h>
#include <stdlib.h>

// Save model weights to file
int save_weights(const char* filename, Linear** layers, size_t num_layers) {
  FILE* fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
    return -1;
  }

  // Write number of layers
  fwrite(&num_layers, sizeof(size_t), 1, fp);

  // Write each layer's weights and biases
  for (size_t i = 0; i < num_layers; i++) {
    Linear* layer = layers[i];

    // Write dimensions
    fwrite(&layer->weights->data->rows, sizeof(size_t), 1, fp);
    fwrite(&layer->weights->data->cols, sizeof(size_t), 1, fp);

    // Write weights
    size_t weight_size = layer->weights->data->rows * layer->weights->data->cols;
    fwrite(layer->weights->data->data, sizeof(float), weight_size, fp);

    // Write bias
    size_t bias_size = layer->bias->data->cols;
    fwrite(layer->bias->data->data, sizeof(float), bias_size, fp);
  }

  fclose(fp);
  return 0;
}

// Load model weights from file
int load_weights(const char* filename, Linear** layers, size_t num_layers) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
    return -1;
  }

  // Read number of layers
  size_t saved_num_layers;
  if (fread(&saved_num_layers, sizeof(size_t), 1, fp) != 1) {
    fprintf(stderr, "Error: Failed to read number of layers\n");
    fclose(fp);
    return -1;
  }

  if (saved_num_layers != num_layers) {
    fprintf(
        stderr,
        "Error: Model architecture mismatch (expected %zu layers, got %zu)\n",
        num_layers, saved_num_layers);
    fclose(fp);
    return -1;
  }

  // Read each layer's weights and biases
  for (size_t i = 0; i < num_layers; i++) {
    Linear* layer = layers[i];

    // Read dimensions
    size_t rows, cols;
    if (fread(&rows, sizeof(size_t), 1, fp) != 1 ||
        fread(&cols, sizeof(size_t), 1, fp) != 1) {
      fprintf(stderr, "Error: Failed to read layer %zu dimensions\n", i);
      fclose(fp);
      return -1;
    }

    // Check dimensions match
    if (rows != layer->weights->data->rows || cols != layer->weights->data->cols) {
      fprintf(stderr, "Error: Layer %zu dimension mismatch\n", i);
      fclose(fp);
      return -1;
    }

    // Read weights
    size_t weight_size = rows * cols;
    if (fread(layer->weights->data->data, sizeof(float), weight_size, fp) !=
        weight_size) {
      fprintf(stderr, "Error: Failed to read layer %zu weights\n", i);
      fclose(fp);
      return -1;
    }

    // Read bias
    size_t bias_size = cols;
    if (fread(layer->bias->data->data, sizeof(float), bias_size, fp) != bias_size) {
      fprintf(stderr, "Error: Failed to read layer %zu bias\n", i);
      fclose(fp);
      return -1;
    }
  }

  fclose(fp);
  return 0;
}
