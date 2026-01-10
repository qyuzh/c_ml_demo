#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Helper function to read big-endian 32-bit integer
static uint32_t read_int32(FILE* fp) {
  uint8_t bytes[4];
  if (fread(bytes, 1, 4, fp) != 4) {
    fprintf(stderr, "Error: Failed to read 4 bytes from file\n");
    return 0;
  }
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Load MNIST data from files
MNISTDataset* mnist_load(const char* images_path, const char* labels_path) {
  // Open image file
  FILE* images_fp = fopen(images_path, "rb");
  if (!images_fp) {
    fprintf(stderr, "Error: Could not open image file: %s\n", images_path);
    return NULL;
  }

  // Open label file
  FILE* labels_fp = fopen(labels_path, "rb");
  if (!labels_fp) {
    fprintf(stderr, "Error: Could not open label file: %s\n", labels_path);
    fclose(images_fp);
    return NULL;
  }

  // Read image file header
  uint32_t magic_number = read_int32(images_fp);
  if (magic_number != 0x00000803) {
    fprintf(stderr, "Error: Invalid image file magic number: 0x%x\n",
            magic_number);
    fclose(images_fp);
    fclose(labels_fp);
    return NULL;
  }

  uint32_t num_images = read_int32(images_fp);
  uint32_t num_rows = read_int32(images_fp);
  uint32_t num_cols = read_int32(images_fp);

  // Read label file header
  magic_number = read_int32(labels_fp);
  if (magic_number != 0x00000801) {
    fprintf(stderr, "Error: Invalid label file magic number: 0x%x\n",
            magic_number);
    fclose(images_fp);
    fclose(labels_fp);
    return NULL;
  }

  uint32_t num_labels = read_int32(labels_fp);

  if (num_images != num_labels) {
    fprintf(
        stderr,
        "Error: Number of images (%u) doesn't match number of labels (%u)\n",
        num_images, num_labels);
    fclose(images_fp);
    fclose(labels_fp);
    return NULL;
  }

  // Allocate dataset
  MNISTDataset* dataset = (MNISTDataset*)malloc(sizeof(MNISTDataset));
  dataset->num_samples = num_images;
  dataset->images = (Matrix**)malloc(num_images * sizeof(Matrix*));
  dataset->labels = (uint8_t*)malloc(num_images * sizeof(uint8_t));

  // Read images and labels
  uint8_t* buffer = (uint8_t*)malloc(num_rows * num_cols);

  for (size_t i = 0; i < num_images; i++) {
    // Read image data
    size_t bytes_read = fread(buffer, 1, num_rows * num_cols, images_fp);
    if (bytes_read != num_rows * num_cols) {
      fprintf(stderr, "Error: Failed to read image %zu\n", i);
      // Continue with partial data or handle error
    }

    // Create matrix and normalize to [0, 1]
    dataset->images[i] = matrix_create(1, num_rows * num_cols);
    for (size_t j = 0; j < num_rows * num_cols; j++) {
      dataset->images[i]->data[j] = buffer[j] / 255.0f;
    }

    // Read label
    if (fread(&dataset->labels[i], 1, 1, labels_fp) != 1) {
      fprintf(stderr, "Error: Failed to read label %zu\n", i);
    }
  }

  free(buffer);
  fclose(images_fp);
  fclose(labels_fp);

  printf("Loaded %zu MNIST samples (%ux%u images)\n", dataset->num_samples,
         num_rows, num_cols);

  return dataset;
}

void mnist_free(MNISTDataset* dataset) {
  if (dataset) {
    if (dataset->images) {
      for (size_t i = 0; i < dataset->num_samples; i++) {
        if (dataset->images[i]) {
          matrix_free(dataset->images[i]);
        }
      }
      free(dataset->images);
    }
    if (dataset->labels) {
      free(dataset->labels);
    }
    free(dataset);
  }
}

// Get a batch of data
void mnist_get_batch(MNISTDataset* dataset, size_t batch_start,
                     size_t batch_size, Matrix** batch_images,
                     Matrix** batch_labels) {
  // Ensure we don't go out of bounds
  if (batch_start + batch_size > dataset->num_samples) {
    batch_size = dataset->num_samples - batch_start;
  }

  // Allocate batch matrices
  *batch_images = matrix_create(batch_size, 784);  // 28x28 = 784
  *batch_labels = matrix_create(batch_size, 10);   // 10 classes
  matrix_zeros(*batch_labels);

  // Copy data to batch
  for (size_t i = 0; i < batch_size; i++) {
    size_t idx = batch_start + i;

    // Copy image
    memcpy((*batch_images)->data + i * 784, dataset->images[idx]->data,
           784 * sizeof(float));

    // One-hot encode label
    (*batch_labels)->data[i * 10 + dataset->labels[idx]] = 1.0f;
  }
}

// Shuffle dataset
void mnist_shuffle(MNISTDataset* dataset) {
  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  for (size_t i = dataset->num_samples - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);

    // Swap images
    Matrix* temp_img = dataset->images[i];
    dataset->images[i] = dataset->images[j];
    dataset->images[j] = temp_img;

    // Swap labels
    uint8_t temp_label = dataset->labels[i];
    dataset->labels[i] = dataset->labels[j];
    dataset->labels[j] = temp_label;
  }
}

// One-hot encode a single label
Matrix* mnist_one_hot_encode(uint8_t label, size_t num_classes) {
  Matrix* encoded = matrix_create(1, num_classes);
  matrix_zeros(encoded);
  encoded->data[label] = 1.0f;
  return encoded;
}
