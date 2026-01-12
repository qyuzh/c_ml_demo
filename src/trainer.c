#include "trainer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "autograd.h"
#include "nn.h"
#include "weights.h"

Trainer* trainer_create(const TrainingConfig* config) {
  // Validate configuration
  if (!config->train_images_path || !config->train_labels_path ||
      !config->test_images_path || !config->test_labels_path) {
    fprintf(stderr, "Error: Missing data paths in configuration\n");
    return NULL;
  }

  Trainer* trainer = (Trainer*)malloc(sizeof(Trainer));
  if (!trainer) {
    fprintf(stderr, "Error: Failed to allocate trainer\n");
    return NULL;
  }

  // Load datasets
  printf("Loading training data...\n");
  trainer->train_data =
      mnist_load(config->train_images_path, config->train_labels_path);
  if (!trainer->train_data) {
    fprintf(stderr, "Error: Failed to load training data\n");
    free(trainer);
    return NULL;
  }

  printf("Loading test data...\n");
  trainer->test_data =
      mnist_load(config->test_images_path, config->test_labels_path);
  if (!trainer->test_data) {
    fprintf(stderr, "Error: Failed to load test data\n");
    mnist_free(trainer->train_data);
    free(trainer);
    return NULL;
  }

  // Create model
  printf("Creating model with autograd...\n");
  trainer->model = mlp_create(config->input_size, config->hidden1_size,
                              config->hidden2_size, config->output_size);
  if (!trainer->model) {
    fprintf(stderr, "Error: Failed to create model\n");
    mnist_free(trainer->train_data);
    mnist_free(trainer->test_data);
    free(trainer);
    return NULL;
  }

  // Create optimizer
  trainer->optimizer =
      sgd_create(config->learning_rate, config->momentum, config->weight_decay);
  if (!trainer->optimizer) {
    fprintf(stderr, "Error: Failed to create optimizer\n");
    mlp_free(trainer->model);
    mnist_free(trainer->train_data);
    mnist_free(trainer->test_data);
    free(trainer);
    return NULL;
  }

  // Copy configuration
  trainer->config = *config;

  printf("Trainer created successfully!\n");
  return trainer;
}

void trainer_free(Trainer* trainer) {
  if (trainer) {
    if (trainer->model) mlp_free(trainer->model);
    if (trainer->optimizer) sgd_free(trainer->optimizer);
    if (trainer->train_data) mnist_free(trainer->train_data);
    if (trainer->test_data) mnist_free(trainer->test_data);
    free(trainer);
  }
}

BatchResult trainer_train_batch(Trainer* trainer, Matrix* batch_images,
                                Matrix* batch_labels) {
  BatchResult result;

  // ============================================================================
  // INPUT PREPARATION
  // ============================================================================
  // batch_images: [batch_size, 784] - Flattened MNIST images (28x28 = 784)
  // batch_labels: [batch_size, 10]  - One-hot encoded labels (10 classes)
  // Convert to Tensors for autograd
  Tensor* input_tensor = tensor_from_matrix(batch_images, 0);
  // input_tensor->data: [batch_size, 784]

  // Zero gradients before forward pass
  mlp_zero_grad(trainer->model);

  // ============================================================================
  // FORWARD PASS - Data flows through the network
  // ============================================================================
  // Model architecture (default: 784 → 128 → 64 → 10):
  //   fc1: input [batch_size, 784]  @ weights [784, 128]  + bias [1, 128]
  //        → h1 [batch_size, 128]
  //   relu: h1 [batch_size, 128] → a1 [batch_size, 128] (element-wise)
  //   fc2: a1 [batch_size, 128]  @ weights [128, 64]   + bias [1, 64]
  //        → h2 [batch_size, 64]
  //   relu: h2 [batch_size, 64]  → a2 [batch_size, 64]  (element-wise)
  //   fc3: a2 [batch_size, 64]   @ weights [64, 10]    + bias [1, 10]
  //        → h3 [batch_size, 10]
  //   softmax: h3 [batch_size, 10] → output [batch_size, 10] (row-wise)
  //
  // All operations automatically build computational graph for backprop
  ForwardResult* forward = mlp_forward_train(trainer->model, input_tensor);
  // forward->output->data: [batch_size, 10] - Probability distribution over
  // classes

  // ============================================================================
  // LOSS COMPUTATION
  // ============================================================================
  // Cross-entropy loss: L = -Σ labels * log(predictions)
  // Loss gradient: ∂L/∂logits = predictions - labels (elegant simplification!)
  MLPLossResult* loss_result = mlp_loss(forward->output->data, batch_labels);
  // loss_result->loss_value: scalar - Average loss over batch
  // loss_result->gradients: [batch_size, 10] - Gradient w.r.t. pre-softmax
  // logits
  result.loss = loss_result->loss_value;
  result.accuracy = calculate_accuracy(forward->output->data, batch_labels);

  // ============================================================================
  // BACKWARD PASS PREPARATION
  // ============================================================================
  // Copy loss gradients to output tensor's gradient buffer
  // This is the "seed" gradient that backpropagation starts from
  // forward->output->grad: [batch_size, 10] ← loss_result->gradients
  for (size_t i = 0;
       i < loss_result->gradients->rows * loss_result->gradients->cols; i++) {
    forward->output->grad->data[i] = loss_result->gradients->data[i];
  }

  // ============================================================================
  // BACKWARD PASS - Gradients flow backward through the network
  // ============================================================================
  // Automatic differentiation computes ALL gradients via chain rule:
  //   output [batch_size, 10] grad flows back through:
  //   ← softmax (gradient passes through, simplified with loss)
  //   ← fc3: accumulates into weights [64, 10] and bias [1, 10]
  //          (bias uses broadcast_add, automatically sums across batch)
  //   ← relu: gates gradient (zeros out where input was negative)
  //   ← fc2: accumulates into weights [128, 64] and bias [1, 64]
  //   ← relu: gates gradient
  //   ← fc1: accumulates into weights [784, 128] and bias [1, 128]
  //
  // Single call - no manual gradient computation needed!
  tensor_backward(forward->output);

  // ============================================================================
  // OPTIMIZER STEP - Update parameters using gradients
  // ============================================================================
  // After backward pass, all parameter gradients are ready:
  //   fc1->weights->grad: [784, 128]
  //   fc1->bias->grad:    [1, 128]
  //   fc2->weights->grad: [128, 64]
  //   fc2->bias->grad:    [1, 64]
  //   fc3->weights->grad: [64, 10]
  //   fc3->bias->grad:    [1, 10]
  //
  // SGD update rule: θ_new = θ_old - learning_rate * gradient
  //                  (with optional momentum and weight decay)
  Linear* layers[3];
  size_t num_layers;
  mlp_get_layers(trainer->model, layers, &num_layers);
  sgd_step(trainer->optimizer, layers, num_layers);

  // Clean up
  tensor_free(input_tensor);
  mlp_loss_result_free(loss_result);
  forward_result_free(forward);

  return result;
}

void trainer_train_epoch(Trainer* trainer, size_t epoch) {
  size_t num_batches =
      trainer->train_data->num_samples / trainer->config.batch_size;
  Metrics metrics = metrics_create();

  // Shuffle training data
  mnist_shuffle(trainer->train_data);

  for (size_t batch = 0; batch < num_batches; batch++) {
    // ============================================================================
    // BATCH LOADING
    // ============================================================================
    // Extract mini-batch from training data
    // batch_images: [batch_size, 784] - Flattened grayscale pixel values [0, 1]
    // batch_labels: [batch_size, 10]  - One-hot encoded (e.g., digit 5 →
    // [0,0,0,0,0,1,0,0,0,0])
    Matrix* batch_images = NULL;
    Matrix* batch_labels = NULL;
    mnist_get_batch(trainer->train_data, batch * trainer->config.batch_size,
                    trainer->config.batch_size, &batch_images, &batch_labels);

    // Train batch
    BatchResult result =
        trainer_train_batch(trainer, batch_images, batch_labels);
    metrics_update(&metrics, result.loss, result.accuracy,
                   trainer->config.batch_size);

    // Clean up batch
    matrix_free(batch_images);
    matrix_free(batch_labels);

    // Print progress
    if ((batch + 1) % 100 == 0) {
      metrics_print_batch(&metrics, batch, num_batches, epoch,
                          trainer->config.epochs);
    }
  }

  printf("\n");
  metrics_average(&metrics, num_batches);
  metrics_print_epoch(&metrics, epoch, trainer->config.epochs);
}

void trainer_train(Trainer* trainer) {
  printf("\nStarting training with AUTOGRAD system...\n");
  printf(
      "Note: Bias gradients now computed automatically through computational "
      "graph!\n\n");

  clock_t start_time = clock();

  for (size_t epoch = 0; epoch < trainer->config.epochs; epoch++) {
    trainer_train_epoch(trainer, epoch);

    // Evaluate after each epoch
    float test_accuracy = trainer_evaluate(trainer);
    printf("Test Accuracy: %.2f%%\n\n", test_accuracy);
  }

  clock_t end_time = clock();
  double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("\n========================================\n");
  printf("TRAINING COMPLETED!\n");
  printf("========================================\n");
  printf("Time: %.2f seconds\n", elapsed);
  printf("✓ Forward pass built computational graph\n");
  printf("✓ Loss function computed gradients\n");
  printf("✓ Single tensor_backward() propagated ALL gradients\n");
  printf("✓ Bias gradients computed automatically via broadcast operation\n");
  printf("✓ No manual gradient computation required!\n");
  printf("========================================\n\n");
}

float trainer_evaluate(Trainer* trainer) {
  size_t test_batches =
      trainer->test_data->num_samples / trainer->config.batch_size;
  Metrics metrics = metrics_create();

  for (size_t batch = 0; batch < test_batches; batch++) {
    Matrix* batch_images = NULL;
    Matrix* batch_labels = NULL;
    mnist_get_batch(trainer->test_data, batch * trainer->config.batch_size,
                    trainer->config.batch_size, &batch_images, &batch_labels);
    // batch_images: [batch_size, 784] - Test images
    // batch_labels: [batch_size, 10]  - True labels for accuracy calculation

    // ============================================================================
    // EVALUATION FORWARD PASS (inference mode)
    // ============================================================================
    // Forward pass only (no gradients needed for evaluation)
    // Input: [batch_size, 784] → fc1 → relu → fc2 → relu → fc3 → softmax
    // Output: [batch_size, 10] probability distributions
    Tensor* input_tensor = tensor_from_matrix(batch_images, 0);
    ForwardResult* forward = mlp_forward_train(trainer->model, input_tensor);
    // forward->output->data: [batch_size, 10] - Predicted probabilities

    // Calculate accuracy by comparing predicted class (argmax) with true class
    float acc = calculate_accuracy(forward->output->data, batch_labels);
    metrics_update(&metrics, 0.0f, acc, trainer->config.batch_size);

    // Clean up
    tensor_free(input_tensor);
    forward_result_free(forward);
    matrix_free(batch_images);
    matrix_free(batch_labels);
  }

  metrics_average(&metrics, test_batches);
  return 100.0f * metrics.accuracy;
}

int trainer_save_weights(Trainer* trainer) {
  printf("Saving model weights to '%s'...\n", trainer->config.weights_path);

  Linear* layers[3];
  size_t num_layers;
  mlp_get_layers(trainer->model, layers, &num_layers);

  int result = save_weights(trainer->config.weights_path, layers, num_layers);

  if (result == 0) {
    printf("Model weights saved successfully!\n");
  } else {
    fprintf(stderr, "Failed to save model weights\n");
  }

  return result;
}
