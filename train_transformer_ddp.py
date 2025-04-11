import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, distributed
import random
from datetime import datetime
import socket

# Utility to add timestamp to logs
def log(rank, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [Rank {rank}] {message}")

# ----------------------------
# Minimal Transformer Encoder
# ----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, nhead=4, num_layers=2, seq_len=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        logits = self.fc(x)
        return logits

# ----------------------------
# Fake Token Dataset
# ----------------------------
class FakeTokenDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=16, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y

# ----------------------------
# Distributed Setup
# ----------------------------
def setup(rank, world_size):
    import datetime
    log(rank, "Initializing process group")
    run_id = os.environ.get("RUN_ID", "default")
    store_path = f"{os.getcwd()}/ddp_store_{run_id}"
    init_method = f"file://{store_path}"

    # All ranks will use the same init_method. It's important to not delete the file once it's used.
    # Deleting it can cause race conditions where one process is initializing and another is removing the file.
    # So the file must be created once and left untouched for the duration of the run.

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30)
    )
    log(rank, "Process group initialized")

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch']

# ----------------------------
# Main Training Function Per Rank
# ----------------------------
def demo_ddp(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cpu")

    dataset = FakeTokenDataset(num_samples=128)
    train_sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    model = MiniTransformer().to(device)
    model = DDP(model)

    # ----------------------------
    # Optimizer explanation:
    # ----------------------------
    # The optimizer updates model weights based on gradients from loss.
    # Optimizers track internal state (e.g. momentum, learning rates).
    # Each rank creates its own optimizer instance.
    # Optimizer state is NOT broadcasted, because it's rank-local and lazily created after first forward+backward.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    checkpoint_path = "checkpoint_shared.pt"

    if rank == 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        log(rank, f"Resumed from checkpoint at epoch {start_epoch}")

        # 🧠 Sample layout of a checkpoint dict:
        # {
        #     'epoch': 2,
        #     'model_state': { 'module.encoder.layers.0.self_attn.in_proj_weight': tensor(...), ... },
        #     'optimizer_state': { 'state': {...}, 'param_groups': [...] }
        # }

    start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.int)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()

    # Only broadcast model weights. Do NOT broadcast optimizer state manually.
    # DDP will internally sync model grads via AllReduce during backward().
    for param in model.state_dict().values():
        dist.broadcast(param, src=0)

    for epoch in range(start_epoch, 2):
        model.train()
        log(rank, f"Starting epoch {epoch}")

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # -----------------------------
            # What happens during training?
            # -----------------------------
            # Each rank runs a forward pass on its own data shard.
            # Then it computes loss locally and calls backward().
            # DDP intercepts backward() and does AllReduce for gradients.
            # Every rank ends up with the same gradients => consistent model.

            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 2 == 0:
                log(rank, f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        log(rank, f"Epoch {epoch} complete")

        if rank == 0:
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
            log(rank, f"Checkpoint saved at epoch {epoch+1}")

    cleanup()

# ----------------------------
# Entry Point for torchrun
# ----------------------------
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    demo_ddp(rank, world_size)
