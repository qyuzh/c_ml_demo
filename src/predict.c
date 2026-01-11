#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "autograd.h"
#include "matrix.h"
#include "mlp_model.h"
#include "mnist.h"
#include "nn.h"
#include "weights.h"

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

// Prediction result
typedef struct {
  size_t predicted_class;
  float confidence;
  float probabilities[10];
} PredictionResult;

// Load model and weights
int load_model(MLPAutograd** model_out, int* weights_loaded_out) {
  printf("Creating model...\n");
  *model_out = mlp_create(784, 128, 64, 10);

  // Try to load trained weights
  Linear* layers[3];
  size_t num_layers;
  mlp_get_layers(*model_out, layers, &num_layers);
  if (load_weights("model.weights", layers, num_layers) == 0) {
    printf("Loaded trained weights from 'model.weights'\n");
    *weights_loaded_out = 1;
  } else {
    printf("Warning: Could not load trained weights. Using random weights.\n");
    *weights_loaded_out = 0;
  }

  return 1;
}

// Make prediction on an image
PredictionResult predict_image(MLPAutograd* model, Matrix* image) {
  PredictionResult result;

  // Forward pass
  Tensor* input = tensor_from_matrix(image, 0);
  Tensor* output = mlp_forward_inference(model, input);

  // Find predicted class and extract probabilities
  result.predicted_class = 0;
  result.confidence = output->data->data[0];

  for (size_t i = 0; i < 10; i++) {
    result.probabilities[i] = output->data->data[i];
    if (output->data->data[i] > result.confidence) {
      result.confidence = output->data->data[i];
      result.predicted_class = i;
    }
  }

  // Cleanup
  tensor_free(input);
  tensor_free(output);

  return result;
}

// Display prediction results
void display_results(const PredictionResult* result, int true_label,
                     int weights_loaded) {
  printf("Prediction Results:\n");
  printf("─────────────────────────────────────────────────────────────\n");
  printf("Predicted Digit: %zu (Confidence: %.2f%%)\n",
         result->predicted_class, result->confidence * 100.0f);
  printf("\nProbability Distribution:\n");

  for (size_t i = 0; i < 10; i++) {
    float prob = result->probabilities[i];
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

    if (i == result->predicted_class) {
      printf(" ← PREDICTED");
    }
    if (i == (size_t)true_label) {
      printf(" ← TRUE LABEL");
    }
    printf("\n");
  }

  printf("─────────────────────────────────────────────────────────────\n");

  if (result->predicted_class == (size_t)true_label) {
    printf("✓ CORRECT PREDICTION!\n");
  } else {
    printf("✗ INCORRECT (Expected: %d, Got: %zu)\n", true_label,
           result->predicted_class);
  }

  if (!weights_loaded) {
    printf("\nNote: Model uses random weights (not trained).\n");
    printf("      Train the model first: ./bin/train [args]\n");
    printf("      This will create 'model.weights' file.\n");
  }
}

// Print header
void print_header(int img_idx, int true_label) {
  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║              MNIST Image Prediction                        ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n");
  printf("\nImage Index: %d\n", img_idx);
  printf("True Label: %d\n", true_label);
}

int main(int argc, char* argv[]) {
  // Parse arguments
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

  // Get and validate image index
  int img_idx = (argc >= 4) ? atoi(argv[3]) : 0;
  if (img_idx < 0 || img_idx >= (int)test_data->num_samples) {
    fprintf(stderr, "Error: Image index %d out of range [0, %zu)\n", img_idx,
            test_data->num_samples);
    mnist_free(test_data);
    return 1;
  }

  // Load model
  MLPAutograd* model = NULL;
  int weights_loaded = 0;
  if (!load_model(&model, &weights_loaded)) {
    mnist_free(test_data);
    return 1;
  }

  // Display header and image
  print_header(img_idx, test_data->labels[img_idx]);
  printf("\n28x28 Image Visualization:");
  render_image(test_data->images[img_idx]);

  // Make prediction
  PredictionResult result = predict_image(model, test_data->images[img_idx]);

  // Display results
  display_results(&result, test_data->labels[img_idx], weights_loaded);

  // Cleanup
  mlp_free(model);
  mnist_free(test_data);

  return 0;
}