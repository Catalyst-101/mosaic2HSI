#!/bin/bash
# Usage: bash scripts/run_infer.sh mosaic.npy checkpoints/best.pth output.h5
python infer.py "$1" "$2" "$3"