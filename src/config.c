#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TrainingConfig config_default(void) {
  TrainingConfig config;

  // Model architecture
  config.input_size = 784;  // 28x28 MNIST images
  config.hidden1_size = 128;
  config.hidden2_size = 64;
  config.output_size = 10;  // 10 digit classes

  // Training hyperparameters
  config.epochs = 4;
  config.batch_size = 32;
  config.learning_rate = 0.01f;
  config.momentum = 0.0f;
  config.weight_decay = 0.0001f;

  // Default paths (can be overridden)
  config.train_images_path = NULL;
  config.train_labels_path = NULL;
  config.test_images_path = NULL;
  config.test_labels_path = NULL;
  config.weights_path = "model.weights";

  return config;
}

TrainingConfig config_from_args(int argc, char** argv) {
  TrainingConfig config = config_default();

  // Require exactly 4 arguments: train_images, train_labels, test_images,
  // test_labels
  if (argc >= 5) {
    config.train_images_path = argv[1];
    config.train_labels_path = argv[2];
    config.test_images_path = argv[3];
    config.test_labels_path = argv[4];
  }

  // TODO: Add support for additional optional arguments (--epochs, --lr, etc.)

  return config;
}

void config_print(const TrainingConfig* config) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║              Training Configuration                           ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");
  printf("\n");
  printf("Model Architecture:\n");
  printf("  Input size:      %zu\n", config->input_size);
  printf("  Hidden layer 1:  %zu neurons\n", config->hidden1_size);
  printf("  Hidden layer 2:  %zu neurons\n", config->hidden2_size);
  printf("  Output size:     %zu classes\n", config->output_size);
  printf("\n");
  printf("Hyperparameters:\n");
  printf("  Epochs:          %zu\n", config->epochs);
  printf("  Batch size:      %zu\n", config->batch_size);
  printf("  Learning rate:   %.4f\n", config->learning_rate);
  printf("  Momentum:        %.4f\n", config->momentum);
  printf("  Weight decay:    %.6f\n", config->weight_decay);
  printf("\n");
  printf("Data Paths:\n");
  printf("  Train images:    %s\n",
         config->train_images_path ? config->train_images_path : "Not set");
  printf("  Train labels:    %s\n",
         config->train_labels_path ? config->train_labels_path : "Not set");
  printf("  Test images:     %s\n",
         config->test_images_path ? config->test_images_path : "Not set");
  printf("  Test labels:     %s\n",
         config->test_labels_path ? config->test_labels_path : "Not set");
  printf("  Weights file:    %s\n", config->weights_path);
  printf("\n");
}
