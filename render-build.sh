#!/bin/bash
# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs -y

# Pull LFS files
git lfs install
git lfs pull

# Install Python deps
pip install -r requirements.txt