FROM python:3.10-slim

# System packages
RUN apt-get update && apt-get install -y \
    git wget vim curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install helpful tools
RUN pip install matplotlib tqdm
