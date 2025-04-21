#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y git-lfs

# Initialize Git LFS
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt