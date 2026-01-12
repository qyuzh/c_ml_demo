#include "metrics.h"

#include <stdio.h>
#include <string.h>

Metrics metrics_create(void) {
  Metrics m;
  m.loss = 0.0f;
  m.accuracy = 0.0f;
  m.correct = 0;
  m.total = 0;
  return m;
}

void metrics_update(Metrics* metrics, float loss, float accuracy,
                    size_t batch_size) {
  metrics->loss += loss;
  metrics->accuracy += accuracy;
  metrics->correct += (size_t)(accuracy * batch_size);
  metrics->total += batch_size;
}

void metrics_average(Metrics* metrics, size_t num_batches) {
  if (num_batches > 0) {
    metrics->loss /= num_batches;
    metrics->accuracy /= num_batches;
  }
}

void metrics_print_epoch(const Metrics* metrics, size_t epoch,
                         size_t total_epochs) {
  printf("Epoch %zu/%zu - Loss: %.4f, Accuracy: %.2f%% (%zu/%zu correct)\n",
         epoch + 1, total_epochs, metrics->loss, 100.0f * metrics->accuracy,
         metrics->correct, metrics->total);
}

void metrics_print_batch(const Metrics* metrics, size_t batch,
                         size_t total_batches, size_t epoch,
                         size_t total_epochs) {
  float avg_loss = metrics->loss / (batch + 1);
  float avg_accuracy = metrics->accuracy / (batch + 1);

  printf("Epoch %zu/%zu, Batch %zu/%zu, Loss: %.4f, Accuracy: %.2f%%\r",
         epoch + 1, total_epochs, batch + 1, total_batches, avg_loss,
         100.0f * avg_accuracy);
  fflush(stdout);
}

void metrics_reset(Metrics* metrics) {
  metrics->loss = 0.0f;
  metrics->accuracy = 0.0f;
  metrics->correct = 0;
  metrics->total = 0;
}

float calculate_accuracy(const Matrix* predictions, const Matrix* labels) {
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
