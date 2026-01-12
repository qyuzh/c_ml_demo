#ifndef TRAINER_H
#define TRAINER_H

#include "mlp_model.h"
#include "optimizer.h"
#include "mnist.h"
#include "config.h"
#include "metrics.h"

/**
 * Trainer encapsulates all training logic
 * Separates concerns: data, model, optimization, and metrics
 */
typedef struct {
    MLPAutograd* model;
    SGD* optimizer;
    MNISTDataset* train_data;
    MNISTDataset* test_data;
    TrainingConfig config;
} Trainer;

/**
 * Batch training result
 */
typedef struct {
    float loss;
    float accuracy;
} BatchResult;

/**
 * Create trainer with configuration
 */
Trainer* trainer_create(const TrainingConfig* config);

/**
 * Free trainer and all resources
 */
void trainer_free(Trainer* trainer);

/**
 * Train model for configured number of epochs
 */
void trainer_train(Trainer* trainer);

/**
 * Train single batch
 */
BatchResult trainer_train_batch(Trainer* trainer, Matrix* batch_images, Matrix* batch_labels);

/**
 * Train single epoch
 */
void trainer_train_epoch(Trainer* trainer, size_t epoch);

/**
 * Evaluate model on test set
 */
float trainer_evaluate(Trainer* trainer);

/**
 * Save model weights to file
 */
int trainer_save_weights(Trainer* trainer);

#endif // TRAINER_H
