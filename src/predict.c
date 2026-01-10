#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "mnist.h"
#include "nn.h"
#include "weights.h"

// Simple MLP for MNIST (same as train.c)
typedef struct {
  Linear* fc1;
  Linear* fc2;
  Linear* fc3;
} MLP;

MLP* mlp_create(size_t input_size, size_t hidden1_size, size_t hidden2_size,
                size_t output_size) {
  MLP* model = (MLP*)malloc(sizeof(MLP));
  model->fc1 = linear_create(input_size, hidden1_size);
  model->fc2 = linear_create(hidden1_size, hidden2_size);
  model->fc3 = linear_create(hidden2_size, output_size);
  linear_init_he(model->fc1);
  linear_init_he(model->fc2);
  linear_init_xavier(model->fc3);
  return model;
}

void mlp_free(MLP* model) {
  if (model) {
    if (model->fc1) linear_free(model->fc1);
    if (model->fc2) linear_free(model->fc2);
    if (model->fc3) linear_free(model->fc3);
    free(model);
  }
}

Matrix* mlp_forward(MLP* model, Matrix* x) {
  Matrix* h1 = linear_forward(model->fc1, x);
  Matrix* a1 = matrix_relu(h1);
  matrix_free(h1);

  Matrix* h2 = linear_forward(model->fc2, a1);
  Matrix* a2 = matrix_relu(h2);
  matrix_free(h2);
  matrix_free(a1);

  Matrix* h3 = linear_forward(model->fc3, a2);
  Matrix* output = matrix_softmax(h3);
  matrix_free(h3);
  matrix_free(a2);

  return output;
}

// Render MNIST image to terminal using ANSI grayscale colors
void render_image(const Matrix* img) {
  printf("\n");

  for (size_t i = 0; i < 28; i++) {
    for (size_t j = 0; j < 28; j++) {
      float pixel = img->data[i * 28 + j];

      // Map pixel value (0-1) to grayscale (232-255 are grayscale colors in
      // 256-color mode)
      int gray = 232 + (int)(pixel * 23);
      if (gray < 232) gray = 232;
      if (gray > 255) gray = 255;

      // Print using ANSI 256-color background
      printf("\033[48;5;%dm  \033[0m", gray);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <test-images> <test-labels> [image_index]\n", argv[0]);
    printf(
        "Example: %s data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte "
        "42\n",
        argv[0]);
    return 1;
  }

  // Load test data
  printf("Loading test data...\n");
  MNISTDataset* test_data = mnist_load(argv[1], argv[2]);
  if (!test_data) {
    return 1;
  }

  // Get image index
  int img_idx = 0;
  if (argc >= 4) {
    img_idx = atoi(argv[3]);
  }

  if (img_idx < 0 || img_idx >= (int)test_data->num_samples) {
    fprintf(stderr, "Error: Image index %d out of range [0, %zu)\n", img_idx,
            test_data->num_samples);
    mnist_free(test_data);
    return 1;
  }

  // Create model (we'll initialize with random weights for demo)
  printf("Creating model...\n");
  MLP* model = mlp_create(784, 128, 64, 10);

  // Try to load trained weights
  Linear* layers[] = {model->fc1, model->fc2, model->fc3};
  int weights_loaded = 0;
  if (load_weights("model.weights", layers, 3) == 0) {
    printf("Loaded trained weights from 'model.weights'\n");
    weights_loaded = 1;
  } else {
    printf("Warning: Could not load trained weights. Using random weights.\n");
  }

  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║              MNIST Image Prediction                        ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n");

  printf("\nImage Index: %d\n", img_idx);
  printf("True Label: %d\n", test_data->labels[img_idx]);

  // Render the image
  printf("\n28x28 Image Visualization:");
  render_image(test_data->images[img_idx]);

  // Make prediction
  Matrix* input = matrix_copy(test_data->images[img_idx]);
  Matrix* output = mlp_forward(model, input);

  // Find predicted class and confidence
  size_t pred_class = 0;
  float max_prob = output->data[0];
  for (size_t i = 1; i < 10; i++) {
    if (output->data[i] > max_prob) {
      max_prob = output->data[i];
      pred_class = i;
    }
  }

  // Display prediction results
  printf("Prediction Results:\n");
  printf("─────────────────────────────────────────────────────────────\n");
  printf("Predicted Digit: %zu (Confidence: %.2f%%)\n", pred_class,
         max_prob * 100.0f);
  printf("\nProbability Distribution:\n");

  for (size_t i = 0; i < 10; i++) {
    float prob = output->data[i];
    int bar_len = (int)(prob * 50);

    printf("  %zu: [", i);
    for (int j = 0; j < 50; j++) {
      if (j < bar_len) {
        printf("█");
      } else {
        printf(" ");
      }
    }
    printf("] %.2f%%", prob * 100.0f);

    if (i == pred_class) {
      printf(" ← PREDICTED");
    }
    if (i == test_data->labels[img_idx]) {
      printf(" ← TRUE LABEL");
    }
    printf("\n");
  }

  printf("─────────────────────────────────────────────────────────────\n");

  if (pred_class == test_data->labels[img_idx]) {
    printf("✓ CORRECT PREDICTION!\n");
  } else {
    printf("✗ INCORRECT (Expected: %d, Got: %zu)\n", test_data->labels[img_idx],
           pred_class);
  }

  if (!weights_loaded) {
    printf("\nNote: Model uses random weights (not trained).\n");
    printf("      Train the model first: ./train [args]\n");
    printf("      This will create 'model.weights' file.\n");
  }

  // Cleanup
  matrix_free(input);
  matrix_free(output);
  mlp_free(model);
  mnist_free(test_data);

  return 0;
}
