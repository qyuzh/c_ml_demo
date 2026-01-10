#!/bin/bash

# Script to download and extract MNIST dataset

echo "Downloading MNIST dataset..."
echo ""

# Create data directory
mkdir -p data
cd data

# Download training images
if [ ! -f "train-images-idx3-ubyte" ]; then
    echo "Downloading training images..."
    wget -q --show-progress http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
    echo "✓ Training images downloaded"
else
    echo "✓ Training images already exist"
fi

# Download training labels
if [ ! -f "train-labels-idx1-ubyte" ]; then
    echo "Downloading training labels..."
    wget -q --show-progress http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
    echo "✓ Training labels downloaded"
else
    echo "✓ Training labels already exist"
fi

# Download test images
if [ ! -f "t10k-images-idx3-ubyte" ]; then
    echo "Downloading test images..."
    wget -q --show-progress http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gunzip t10k-images-idx3-ubyte.gz
    echo "✓ Test images downloaded"
else
    echo "✓ Test images already exist"
fi

# Download test labels
if [ ! -f "t10k-labels-idx1-ubyte" ]; then
    echo "Downloading test labels..."
    wget -q --show-progress http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
    echo "✓ Test labels downloaded"
else
    echo "✓ Test labels already exist"
fi

cd ..

echo ""
echo "MNIST dataset ready!"
echo "You can now run: ./train data/train-images-idx3-ubyte data/train-labels-idx1-ubyte data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte"
