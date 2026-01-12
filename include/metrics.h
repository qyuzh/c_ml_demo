#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"
#include <stddef.h>

/**
 * Training metrics structure
 * Tracks loss and accuracy during training
 */
typedef struct {
    float loss;
    float accuracy;
    size_t correct;
    size_t total;
} Metrics;

/**
 * Create empty metrics structure
 */
Metrics metrics_create(void);

/**
 * Update metrics with batch results
 */
void metrics_update(Metrics* metrics, float loss, float accuracy, size_t batch_size);

/**
 * Calculate average metrics
 */
void metrics_average(Metrics* metrics, size_t num_batches);

/**
 * Print metrics for current epoch
 */
void metrics_print_epoch(const Metrics* metrics, size_t epoch, size_t total_epochs);

/**
 * Print batch progress
 */
void metrics_print_batch(const Metrics* metrics, size_t batch, size_t total_batches,
                         size_t epoch, size_t total_epochs);

/**
 * Reset metrics to zero
 */
void metrics_reset(Metrics* metrics);

/**
 * Calculate accuracy from predictions and labels
 */
float calculate_accuracy(const Matrix* predictions, const Matrix* labels);

#endif // METRICS_H
