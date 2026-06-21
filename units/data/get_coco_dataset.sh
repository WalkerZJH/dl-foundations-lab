#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "coco_captioning" ]; then
  wget http://cs231n.stanford.edu/coco_captioning.zip
  unzip coco_captioning.zip
  rm coco_captioning.zip
fi
