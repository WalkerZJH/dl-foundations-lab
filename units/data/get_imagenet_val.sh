#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "imagenet_val_25.npz" ]; then
  wget http://cs231n.stanford.edu/imagenet_val_25.npz
fi
