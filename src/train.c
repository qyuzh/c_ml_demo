#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "autograd.h"
#include "matrix.h"
#include "mlp_model.h"
#include "mnist.h"
#include "nn.h"
#include "optimizer.h"
#include "weights.h"

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

// Training configuration
typedef struct {
  size_t epochs;
  size_t batch_size;
  float learning_rate;
  float momentum;
  float weight_decay;
} TrainingConfig;

// Create default training configuration
TrainingConfig training_config_default() {
  TrainingConfig config;
  config.epochs = 4;
  config.batch_size = 32;
  config.learning_rate = 0.01f;
  config.momentum = 0.0f;
  config.weight_decay = 0.0001f;
  return config;
}

// Load datasets
int load_datasets(const char* train_images, const char* train_labels,
                  const char* test_images, const char* test_labels,
                  MNISTDataset** train_data, MNISTDataset** test_data) {
  printf("Loading training data...\n");
  *train_data = mnist_load(train_images, train_labels);
  if (!*train_data) {
    return 0;
  }

  printf("Loading test data...\n");
  *test_data = mnist_load(test_images, test_labels);
  if (!*test_data) {
    mnist_free(*train_data);
    return 0;
  }

  return 1;
}

// Train single batch
void train_batch(MLPAutograd* model, SGD* optimizer, Matrix* batch_images,
                 Matrix* batch_labels, float* loss_out, float* acc_out) {
  // Convert to Tensors for autograd
  Tensor* input_tensor = tensor_from_matrix(batch_images, 0);

  // Zero gradients before forward pass
  mlp_zero_grad(model);

  // Forward pass: Builds computational graph automatically
  ForwardResult* forward = mlp_forward_train(model, input_tensor);

  // ============================================================================
  // Compute Loss and Gradients using MLP's Loss Function
  // ============================================================================
  // The loss function encapsulates the mathematical relationship:
  //   Loss: L = CrossEntropy(Softmax(z), y)
  //   Gradient: ∂L/∂z = Softmax(z) - y = predictions - labels
  //
  // This computes both:
  // 1. Cross-Entropy loss value (for monitoring training progress)
  // 2. Gradient w.r.t. pre-softmax logits (for backpropagation)
  MLPLossResult* loss_result = mlp_loss(
      forward->output->data, batch_labels);

  *loss_out = loss_result->loss_value;
  *acc_out = calculate_accuracy(forward->output->data, batch_labels);

  // Copy gradients from loss function to tensor's grad buffer
  // This is the starting point (∂L/∂output) for backpropagation through the network
  for (size_t i = 0; i < loss_result->gradients->rows * loss_result->gradients->cols; i++) {
    forward->output->grad->data[i] = loss_result->gradients->data[i];
  }

  // Backward pass: Automatic differentiation computes all gradients
  // This propagates gradients backward through: softmax -> fc3 -> relu -> fc2 -> relu -> fc1
  tensor_backward(forward->output);

  // Handle bias gradients (sum across batch dimension)
  linear_bias_backward(forward->h3->grad, model->fc3);
  linear_bias_backward(forward->h2->grad, model->fc2);
  linear_bias_backward(forward->h1->grad, model->fc1);

  // Update weights using gradients
  Linear* layers[3];
  size_t num_layers;
  mlp_get_layers(model, layers, &num_layers);
  sgd_step(optimizer, layers, num_layers);

  // Clean up
  tensor_free(input_tensor);
  mlp_loss_result_free(loss_result);
  forward_result_free(forward);
}

// Train one epoch
void train_epoch(MLPAutograd* model, SGD* optimizer, MNISTDataset* train_data,
                 size_t epoch, size_t epochs, size_t batch_size) {
  size_t num_batches = train_data->num_samples / batch_size;
  float epoch_loss = 0.0f;
  float epoch_accuracy = 0.0f;

  // Shuffle training data
  mnist_shuffle(train_data);

  for (size_t batch = 0; batch < num_batches; batch++) {
    // Get batch
    Matrix* batch_images = NULL;
    Matrix* batch_labels = NULL;
    mnist_get_batch(train_data, batch * batch_size, batch_size,
                    &batch_images, &batch_labels);

    // Train batch
    float loss, acc;
    train_batch(model, optimizer, batch_images, batch_labels, &loss, &acc);
    epoch_loss += loss;
    epoch_accuracy += acc;

    // Clean up batch
    matrix_free(batch_images);
    matrix_free(batch_labels);

    // Print progress
    if ((batch + 1) % 100 == 0) {
      printf("Epoch %zu/%zu, Batch %zu/%zu, Loss: %.4f, Accuracy: %.2f%%\r",
             epoch + 1, epochs, batch + 1, num_batches,
             epoch_loss / (batch + 1),
             100.0f * epoch_accuracy / (batch + 1));
      fflush(stdout);
    }
  }

  printf("\nEpoch %zu/%zu - Avg Loss: %.4f, Avg Accuracy: %.2f%%\n",
         epoch + 1, epochs, epoch_loss / num_batches,
         100.0f * epoch_accuracy / num_batches);
}

// Evaluate model on test set
float evaluate_model(MLPAutograd* model, MNISTDataset* test_data,
                     size_t batch_size) {
  printf("Evaluating on test set...\n");
  size_t test_batches = test_data->num_samples / batch_size;
  float test_accuracy = 0.0f;

  for (size_t batch = 0; batch < test_batches; batch++) {
    Matrix* batch_images = NULL;
    Matrix* batch_labels = NULL;
    mnist_get_batch(test_data, batch * batch_size, batch_size,
                    &batch_images, &batch_labels);

    // Forward pass only (no gradients needed for evaluation)
    Tensor* input_tensor = tensor_from_matrix(batch_images, 0);
    ForwardResult* forward = mlp_forward_train(model, input_tensor);

    float acc = calculate_accuracy(forward->output->data, batch_labels);
    test_accuracy += acc;

    // Clean up
    tensor_free(input_tensor);
    forward_result_free(forward);
    matrix_free(batch_images);
    matrix_free(batch_labels);
  }

  float avg_accuracy = 100.0f * test_accuracy / test_batches;
  printf("Test Accuracy: %.2f%%\n\n", avg_accuracy);
  return avg_accuracy;
}

// Print training summary
void print_training_summary(double elapsed_time) {
  printf("Training completed in %.2f seconds\n", elapsed_time);
  printf("\n========================================\n");
  printf("AUTOGRAD SYSTEM SUCCESSFULLY USED!\n");
  printf("========================================\n");
  printf("- Forward pass built computational graph\n");
  printf("- Loss function computed gradients\n");
  printf("- Single tensor_backward() propagated all gradients\n");
  printf("- No manual gradient computation required!\n");
  printf("========================================\n\n");
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

  // Load datasets
  MNISTDataset* train_data = NULL;
  MNISTDataset* test_data = NULL;
  if (!load_datasets(argv[1], argv[2], argv[3], argv[4], &train_data, &test_data)) {
    return 1;
  }

  // Get training configuration
  TrainingConfig config = training_config_default();

  // Create model and optimizer
  printf("\nCreating model with autograd...\n");
  MLPAutograd* model = mlp_create(784, 128, 64, 10);
  SGD* optimizer = sgd_create(config.learning_rate, config.momentum, config.weight_decay);

  // Print training configuration
  printf("Starting training with AUTOGRAD system...\n");
  printf("Epochs: %zu, Batch size: %zu, Learning rate: %.4f\n\n",
         config.epochs, config.batch_size, config.learning_rate);

  // Training loop
  clock_t start_time = clock();
  for (size_t epoch = 0; epoch < config.epochs; epoch++) {
    train_epoch(model, optimizer, train_data, epoch, config.epochs, config.batch_size);
    evaluate_model(model, test_data, config.batch_size);
  }
  clock_t end_time = clock();
  double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  // Print summary
  print_training_summary(elapsed);

  // Save trained weights
  printf("Saving model weights to 'model.weights'...\n");
  Linear* layers[3];
  size_t num_layers;
  mlp_get_layers(model, layers, &num_layers);
  if (save_weights("model.weights", layers, num_layers) == 0) {
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
