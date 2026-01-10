#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"
#include "mnist.h"
#include "nn.h"
#include "optimizer.h"
#include "weights.h"

// Simple MLP for MNIST
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

  // Initialize weights
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

// Forward pass
Matrix* mlp_forward(MLP* model, Matrix* x) {
  // Layer 1: fc1 -> ReLU
  Matrix* h1 = linear_forward(model->fc1, x);
  Matrix* a1 = matrix_relu(h1);
  matrix_free(h1);

  // Layer 2: fc2 -> ReLU
  Matrix* h2 = linear_forward(model->fc2, a1);
  Matrix* a2 = matrix_relu(h2);
  matrix_free(h2);
  matrix_free(a1);

  // Layer 3: fc3 -> Softmax
  Matrix* h3 = linear_forward(model->fc3, a2);
  Matrix* output = matrix_softmax(h3);
  matrix_free(h3);
  matrix_free(a2);

  return output;
}

// Backward pass
void mlp_backward(MLP* model, Matrix* x, Matrix* grad_output) {
  // Forward pass to cache activations
  Matrix* h1 = linear_forward(model->fc1, x);
  Matrix* a1 = matrix_relu(h1);

  Matrix* h2 = linear_forward(model->fc2, a1);
  Matrix* a2 = matrix_relu(h2);

  // Backward through fc3
  Matrix* grad_a2 = NULL;
  linear_backward(model->fc3, grad_output, &grad_a2);

  // Backward through ReLU2
  Matrix* relu2_mask = matrix_relu_derivative(h2);
  Matrix* grad_h2 = matrix_mul(grad_a2, relu2_mask);
  matrix_free(relu2_mask);
  matrix_free(grad_a2);

  // Backward through fc2
  Matrix* grad_a1 = NULL;
  linear_backward(model->fc2, grad_h2, &grad_a1);
  matrix_free(grad_h2);

  // Backward through ReLU1
  Matrix* relu1_mask = matrix_relu_derivative(h1);
  Matrix* grad_h1 = matrix_mul(grad_a1, relu1_mask);
  matrix_free(relu1_mask);
  matrix_free(grad_a1);

  // Backward through fc1
  linear_backward(model->fc1, grad_h1, NULL);
  matrix_free(grad_h1);

  // Clean up
  matrix_free(h1);
  matrix_free(a1);
  matrix_free(h2);
  matrix_free(a2);
}

void mlp_zero_grad(MLP* model) {
  linear_zero_grad(model->fc1);
  linear_zero_grad(model->fc2);
  linear_zero_grad(model->fc3);
}

// Calculate accuracy
float calculate_accuracy(Matrix* predictions, Matrix* labels) {
  size_t correct = 0;

  for (size_t i = 0; i < predictions->rows; i++) {
    // Find predicted class
    size_t pred_class = 0;
    float max_pred = predictions->data[i * predictions->cols];
    for (size_t j = 1; j < predictions->cols; j++) {
      if (predictions->data[i * predictions->cols + j] > max_pred) {
        max_pred = predictions->data[i * predictions->cols + j];
        pred_class = j;
      }
    }

    // Find true class
    size_t true_class = 0;
    for (size_t j = 0; j < labels->cols; j++) {
      if (labels->data[i * labels->cols + j] > 0.5f) {
        true_class = j;
        break;
      }
    }

    if (pred_class == true_class) {
      correct++;
    }
  }

  return (float)correct / predictions->rows;
}

int main(int argc, char* argv[]) {
  // Check command line arguments
  if (argc != 5) {
    printf(
        "Usage: %s <train-images> <train-labels> <test-images> <test-labels>\n",
        argv[0]);
    printf(
        "Example: %s train-images-idx3-ubyte train-labels-idx1-ubyte "
        "t10k-images-idx3-ubyte t10k-labels-idx1-ubyte\n",
        argv[0]);
    return 1;
  }

  // Load MNIST dataset
  printf("Loading training data...\n");
  MNISTDataset* train_data = mnist_load(argv[1], argv[2]);
  if (!train_data) {
    return 1;
  }

  printf("Loading test data...\n");
  MNISTDataset* test_data = mnist_load(argv[3], argv[4]);
  if (!test_data) {
    mnist_free(train_data);
    return 1;
  }

  // Create model
  printf("\nCreating model...\n");
  MLP* model = mlp_create(784, 128, 64, 10);

  // Create optimizer
  float learning_rate = 0.01f;
  SGD* optimizer = sgd_create(learning_rate, 0.0f, 0.0001f);

  // Training parameters
  size_t epochs = 10;
  size_t batch_size = 32;
  size_t num_batches = train_data->num_samples / batch_size;

  printf("Starting training...\n");
  printf("Epochs: %zu, Batch size: %zu, Learning rate: %.4f\n\n", epochs,
         batch_size, learning_rate);

  clock_t start_time = clock();

  // Training loop
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // Shuffle training data
    mnist_shuffle(train_data);

    float epoch_loss = 0.0f;
    float epoch_accuracy = 0.0f;

    for (size_t batch = 0; batch < num_batches; batch++) {
      // Get batch
      Matrix* batch_images = NULL;
      Matrix* batch_labels = NULL;
      mnist_get_batch(train_data, batch * batch_size, batch_size, &batch_images,
                      &batch_labels);

      // Zero gradients
      mlp_zero_grad(model);

      // Forward pass
      Matrix* predictions = mlp_forward(model, batch_images);

      // Calculate loss
      float loss = matrix_cross_entropy(predictions, batch_labels);
      epoch_loss += loss;

      // Calculate accuracy
      float acc = calculate_accuracy(predictions, batch_labels);
      epoch_accuracy += acc;

      // Backward pass (gradient = predictions - labels for softmax + cross
      // entropy)
      Matrix* grad = matrix_sub(predictions, batch_labels);
      mlp_backward(model, batch_images, grad);
      matrix_free(grad);

      // Update weights
      Linear* layers[] = {model->fc1, model->fc2, model->fc3};
      sgd_step(optimizer, layers, 3);

      // Clean up
      matrix_free(batch_images);
      matrix_free(batch_labels);
      matrix_free(predictions);

      // Print progress
      if ((batch + 1) % 100 == 0) {
        printf("Epoch %zu/%zu, Batch %zu/%zu, Loss: %.4f, Accuracy: %.2f%%\r",
               epoch + 1, epochs, batch + 1, num_batches,
               epoch_loss / (batch + 1), 100.0f * epoch_accuracy / (batch + 1));
        fflush(stdout);
      }
    }

    printf("\nEpoch %zu/%zu - Avg Loss: %.4f, Avg Accuracy: %.2f%%\n",
           epoch + 1, epochs, epoch_loss / num_batches,
           100.0f * epoch_accuracy / num_batches);

    // Evaluate on test set
    printf("Evaluating on test set...\n");
    size_t test_batches = test_data->num_samples / batch_size;
    float test_accuracy = 0.0f;

    for (size_t batch = 0; batch < test_batches; batch++) {
      Matrix* batch_images = NULL;
      Matrix* batch_labels = NULL;
      mnist_get_batch(test_data, batch * batch_size, batch_size, &batch_images,
                      &batch_labels);

      Matrix* predictions = mlp_forward(model, batch_images);
      float acc = calculate_accuracy(predictions, batch_labels);
      test_accuracy += acc;

      matrix_free(batch_images);
      matrix_free(batch_labels);
      matrix_free(predictions);
    }

    printf("Test Accuracy: %.2f%%\n\n", 100.0f * test_accuracy / test_batches);
  }

  clock_t end_time = clock();
  double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Training completed in %.2f seconds\n", elapsed);

  // Save trained weights
  printf("\nSaving model weights to 'model.weights'...\n");
  Linear* layers[] = {model->fc1, model->fc2, model->fc3};
  if (save_weights("model.weights", layers, 3) == 0) {
    printf("Model weights saved successfully!\n");
  } else {
    fprintf(stderr, "Failed to save model weights\n");
  }

  // Clean up
  mnist_free(train_data);
  mnist_free(test_data);
  mlp_free(model);
  sgd_free(optimizer);

  return 0;
}
