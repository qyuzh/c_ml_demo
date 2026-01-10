# Project Structure

## Directory Layout

```
c_ml_demo/
├── include/                # Header files
│   ├── matrix.h           # Matrix operations
│   ├── autograd.h         # Automatic differentiation
│   ├── nn.h               # Neural network layers
│   ├── optimizer.h        # SGD optimizer
│   ├── mnist.h            # MNIST data loader
│   └── weights.h          # Model persistence
│
├── src/                    # Source code
│   ├── matrix.c           # Matrix operations
│   ├── autograd.c         # Automatic differentiation
│   ├── nn.c               # Neural network layers
│   ├── optimizer.c        # SGD optimizer
│   ├── mnist.c            # MNIST data loader
│   ├── weights.c          # Model persistence
│   ├── train.c            # Training program
│   ├── test.c             # Test suite
│   ├── example.c          # Usage examples
│   └── predict.c          # Prediction tool
│
├── bin/                    # Compiled binaries
│   ├── train              # Training executable
│   ├── test               # Test executable
│   ├── example            # Example executable
│   ├── predict            # Prediction executable
│   └── obj/               # Object files
│
├── data/                   # MNIST dataset
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── docs/                   # Documentation
│   └── AUTOGRAD.md        # Autograd principles explained
│
├── Makefile               # Build configuration
├── README.md              # Complete documentation
├── STRUCTURE.md           # This file
├── download_mnist.sh      # Dataset downloader
├── .gitignore            # Git ignore rules
└── model.weights         # Trained model (if exists)
```

## Build System

The Makefile organizes everything:

- **Headers**: All `.h` files in `include/`
- **Source**: All `.c` files in `src/`
- **Objects**: Compiled `.o` files in `bin/obj/`
- **Binaries**: Executables in `bin/`

### Build Commands

```bash
make                # Build all
make clean          # Remove bin/
make rebuild        # Clean + build
make test_run       # Build and run tests
make example_run    # Build and run examples
```

## Usage

All executables are in `bin/`:

```bash
bin/test                                          # Run tests
bin/example                                       # Run examples
bin/train data/*.idx*-ubyte data/*.idx*-ubyte    # Train model
bin/predict data/*.idx*-ubyte data/*.idx*-ubyte 0  # Predict
```

## Clean Separation

- **Headers**: Public interfaces in `include/`
- **Implementation**: Work in `src/`
- **Building**: Happens in `bin/obj/`
- **Running**: Execute from `bin/`
- **Data**: Stored in `data/`

This structure keeps everything organized and clean!
