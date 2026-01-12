#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "trainer.h"

/**
 * Simplified training program using modular design
 * All logic delegated to config and trainer modules
 */
int main(int argc, char* argv[]) {
  // Check command line arguments
  if (argc != 5) {
    printf(
        "Usage: %s <train-images> <train-labels> <test-images> <test-labels>\n",
        argv[0]);
    printf(
        "Example: %s data/train-images.idx3-ubyte data/train-labels.idx1-ubyte "
        "data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte\n",
        argv[0]);
    return 1;
  }

  // Parse configuration from command line
  TrainingConfig config = config_from_args(argc, argv);
  config_print(&config);

  // Create trainer (handles data loading, model creation, optimizer setup)
  Trainer* trainer = trainer_create(&config);
  if (!trainer) {
    fprintf(stderr, "Failed to create trainer\n");
    return 1;
  }

  // Train model (handles all epoch loops, batch processing, metrics)
  trainer_train(trainer);

  // Save trained weights
  trainer_save_weights(trainer);

  // Cleanup
  trainer_free(trainer);

  return 0;
}
