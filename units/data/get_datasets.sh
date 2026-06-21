#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "cifar-10-batches-py" ]; then
  wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
fi

if [ ! -f "imagenet_val_25.npz" ]; then
  wget http://cs231n.stanford.edu/imagenet_val_25.npz
fi
