#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

/**
 * Training configuration structure
 * Centralizes all hyperparameters and paths for training
 */
typedef struct {
    // Model architecture
    size_t input_size;
    size_t hidden1_size;
    size_t hidden2_size;
    size_t output_size;
    
    // Training hyperparameters
    size_t epochs;
    size_t batch_size;
    float learning_rate;
    float momentum;
    float weight_decay;
    
    // Data paths
    const char* train_images_path;
    const char* train_labels_path;
    const char* test_images_path;
    const char* test_labels_path;
    
    // Model paths
    const char* weights_path;
} TrainingConfig;

/**
 * Create default training configuration
 */
TrainingConfig config_default(void);

/**
 * Create configuration from command line arguments
 */
TrainingConfig config_from_args(int argc, char** argv);

/**
 * Print configuration summary
 */
void config_print(const TrainingConfig* config);

#endif // CONFIG_H
