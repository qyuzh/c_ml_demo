#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"
#include <stdint.h>

// MNIST dataset structure
typedef struct {
    Matrix** images;      // Array of image matrices (28x28 each)
    uint8_t* labels;      // Array of labels (0-9)
    size_t num_samples;
} MNISTDataset;

// Load MNIST data from files
MNISTDataset* mnist_load(const char* images_path, const char* labels_path);
void mnist_free(MNISTDataset* dataset);

// Get a batch of data
void mnist_get_batch(MNISTDataset* dataset, size_t batch_start, size_t batch_size,
                     Matrix** batch_images, Matrix** batch_labels);

// Utility functions
void mnist_shuffle(MNISTDataset* dataset);
Matrix* mnist_one_hot_encode(uint8_t label, size_t num_classes);

#endif // MNIST_H
