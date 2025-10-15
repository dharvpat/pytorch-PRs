#!/usr/bin/env python3
"""
MPS Graph Cache Growth Simulation with Regularization, Dropout, and Pruning

This version reproduces linear MPSGraphCache (G-Cache).
It does so by simulating a training loop with:
  - L1 regularization
  - Random dropout
  - Periodic pruning of small weights
  - Randomized activations and variable input shapes
  - G-Cache monitoring and tracking

Usage:
    When Testing with no cache clearing:
    python3 test_comprehensive_memory_training.py --epochs 5 --batches 100 --mode baseline
    When Testing with with cache clearing:
    python3 test_comprehensive_memory_training.py --epochs 5 --batches 100 --mode modified
    To plot results:
    python3 test_comprehensive_memory_training.py --plot
    Best to run this in a fresh Python environment with only the modified torch (need utils for tracking cache object count) and matplotlib installed.
"""

import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List
import json
import matplotlib.pyplot as plt
import numpy as np

# Memory Tracking
class GCacheTracker:
    """Track G-Cache (MPSGraphCache) growth across training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_mps = device.type == 'mps'
        self.history = []

    def record(self, epoch: int, batch: int, loss: float, batch_time: float = 0.0, clear_time: float = 0.0) -> Dict:
        """Record current memory state, batch runtime, and cache clear time."""
        metrics = {
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'batch_time': batch_time,
            'clear_time': clear_time,
        }

        if self.is_mps:
            if hasattr(torch.mps, 'driver_allocated_memory'):
                metrics['driver_mem_mb'] = torch.mps.driver_allocated_memory() / (1024**2)
            if hasattr(torch.mps, 'current_allocated_memory'):
                metrics['current_mem_mb'] = torch.mps.current_allocated_memory() / (1024**2)
            if hasattr(torch.mps, 'graph_cache_size'):
                metrics['graph_cache'] = torch.mps.graph_cache_size()
            if hasattr(torch.mps, 'kernel_cache_size'):
                metrics['kernel_cache'] = torch.mps.kernel_cache_size()

        self.history.append(metrics)
        return metrics

    def format_current(self, metrics: Dict) -> str:
        """Format current metrics for display."""
        if not self.is_mps:
            return "CPU mode"

        parts = []
        if 'driver_mem_mb' in metrics:
            parts.append(f"Driver: {metrics['driver_mem_mb']:.1f}MB")
        if 'graph_cache' in metrics:
            parts.append(f"G-Cache: {metrics['graph_cache']}")
        if 'kernel_cache' in metrics:
            parts.append(f"K-Cache: {metrics['kernel_cache']}")

        return " | ".join(parts)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.history:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        summary = {
            'initial_gcache': initial.get('graph_cache', 0),
            'final_gcache': final.get('graph_cache', 0),
            'gcache_growth': final.get('graph_cache', 0) - initial.get('graph_cache', 0),
            'batches_tracked': len(self.history),
        }

        return summary

# Models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(F.relu(self.conv2(x)))
        x = x.mean(dim=[2, 3], keepdim=True)      # safe pooling
        x = F.interpolate(x, size=(8, 8), mode="nearest")
        x = x.flatten(1)
        return self.fc(x)


class SimpleTransformer(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.embed = nn.Linear(64, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=0.2)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class SimpleRNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class VAE(nn.Module):
    """Variational Autoencoder for image data - simplified for variable input sizes."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder - uses global pooling to handle variable sizes
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Use global average pooling instead of flattening
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, padding=1)

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 10)  # Classify from encoded features

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # Global average pooling - works with any spatial size
        h = h.mean(dim=[2, 3])
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # For classification, use the latent representation
        z_dropped = self.dropout(z)
        features = F.relu(self.decoder_fc(z_dropped))
        return self.fc(features)


class ResNet(nn.Module):
    """Simple ResNet-style model with residual connections."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Second residual block
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 10)  # Use global pooling instead

    def forward(self, x):
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))

        # First residual block
        identity = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = F.relu(out)

        # Downsample
        out = F.relu(self.bn4(self.conv4(out)))

        # Second residual block
        identity = out
        out = F.relu(self.bn5(self.conv5(out)))
        out = self.bn6(self.conv6(out))
        out += identity
        out = F.relu(out)

        # Classification with global average pooling
        out = out.mean(dim=[2, 3])  # Global average pooling
        out = self.dropout(out)
        return self.fc(out)


class UNet(nn.Module):
    """U-Net style architecture with skip connections - simplified for variable sizes."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(16, 10)  # Use global pooling

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        # Decoder with skip connections
        d1 = self.up1(e3)
        # Handle size mismatch from variable input sizes
        if d1.shape[2:] != e2.shape[2:]:
            d1 = F.interpolate(d1, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        # Handle size mismatch
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        # Classification with global average pooling
        out = d2.mean(dim=[2, 3])
        out = self.dropout(out)
        return self.fc(out)


class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism."""
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Take mean over sequence
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)

# Helper functions
def activation_function(x):
    fn = F.relu
    return fn(x)


def random_tensor(device, base_shape=(8, 3, 32, 32), model_type="CNN"):
    '''Placeholder to generate random tensors with variable shapes based on model type, used this to check if caching problem gets worse if input tensors are completely different each time. (it did not, set delta to a random int to verify)'''
    '''Thinking was that if input shapes are different each time, then the graphs would be impossible to be reused, but it did not make a difference. Showcasing that we are already recomputing graphs each time.'''
    delta = 0

    # Image-based models (CNN-like architectures)
    if model_type in ["CNN", "VAE", "ResNet", "UNet"]:
        size = base_shape[2] + delta
        return torch.randn(base_shape[0], 3, size, size, device=device)

    # Sequence models with transformer-style input
    elif model_type == "Transformer":
        seq_len = 32 + delta
        return torch.randn(base_shape[0], seq_len, 64, device=device)

    # Sequence models with smaller input dimension (LSTM, RNN, GRU, AttentionLSTM)
    elif model_type in ["LSTM", "RNN", "GRU", "AttentionLSTM"]:
        seq_len = 16 + delta
        return torch.randn(base_shape[0], seq_len, 32, device=device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def add_per_sample_noise(inputs, use_continuous=True):
    batch_size = inputs.size(0)
    device = inputs.device

    # Determine number of channels to add for noise map
    if inputs.dim() == 4:  # CNN: (batch, channels, height, width)
        output_batch = torch.zeros(batch_size, inputs.shape[1] + 1,
                                   inputs.shape[2], inputs.shape[3], device=device)

        # Per-sample loop with index assignment
        for i in range(batch_size):
            if use_continuous:
                noise_scale = random.uniform(0.01, 0.1)
            else:
                noise_scale = random.choice([0.02, 0.04, 0.06, 0.08])

            # Extract single sample (line 293)
            sample = inputs[i].unsqueeze(0)

            # Add noise map channel
            noise_map = torch.full((1, 1, inputs.shape[2], inputs.shape[3]),
                                  noise_scale, device=device)
            sample_with_noise = torch.cat([sample, noise_map], dim=1)

            # Index assignment
            output_batch[i] = sample_with_noise.squeeze(0)

        # Return without the extra channel to match original shape
        return output_batch[:, :-1, :, :]

    elif inputs.dim() == 3:  # Transformer/LSTM: (batch, seq, features)
        # Pre-allocate
        output_batch = torch.zeros(batch_size, inputs.shape[1],
                                   inputs.shape[2] + 1, device=device)

        # Per-sample loop
        for i in range(batch_size):
            if use_continuous:
                noise_scale = random.uniform(0.01, 0.1)
            else:
                noise_scale = random.choice([0.02, 0.04, 0.06, 0.08])

            sample = inputs[i].unsqueeze(0)
            noise_map = torch.full((1, inputs.shape[1], 1), noise_scale, device=device)
            sample_with_noise = torch.cat([sample, noise_map], dim=2)

            # Index assignment
            output_batch[i] = sample_with_noise.squeeze(0)

        return output_batch[:, :, :-1]

    else:
        return inputs


def apply_pruning(model, threshold=1e-3):
    """Zero out small weights (structured pruning-like behavior)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                mask = param.abs() > threshold
                param.mul_(mask)


def l1_regularization(model, weight=1e-5):
    """Compute L1 penalty for all parameters."""
    l1 = 0.0
    for p in model.parameters():
        l1 += p.abs().sum()
    return weight * l1

# Training loop with dropout, L1, and pruning + G-Cache tracking
def train_model(model, model_type, device, epochs=3, batches=200, prune_every=20, mode='baseline', batch_size=8):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Initialize G-Cache tracker
    tracker = GCacheTracker(device)

    # Determine clearing strategy
    clear_strategy = "none"
    if mode in ['modified', 'modified_batch', 'modified_epoch']:
        if mode == 'modified':
            clear_strategy = "every_batch"
        elif mode == 'modified_batch':
            clear_strategy = "every_20_batches"
        elif mode == 'modified_epoch':
            clear_strategy = "every_epoch"

    print(f"\n{'='*60}")
    print(f"Training {model_type} with G-Cache tracking")
    print(f"Mode: {mode.upper()}")
    if clear_strategy != "none":
        print(f"Clearing strategy: {clear_strategy}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        for b in range(batches):
            batch_start_time = time.time()

            # Variable input shapes
            inputs = random_tensor(device, model_type=model_type)

            # KEY: DVDNet's exact per-sample loop pattern (trainer.py lines 291-295)
            # This causes ~1 graph per sample = batch_size graphs per batch
            inputs = add_per_sample_noise(inputs, use_continuous=True)

            targets = torch.randint(0, 10, (inputs.size(0),), device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = activation_function(outputs)
            loss = criterion(outputs, targets)

            # Add L1 regularization
            loss = loss + l1_regularization(model, weight=1e-5)
            loss.backward()
            optimizer.step()

            # Apply pruning periodically
            if (b + 1) % prune_every == 0:
                apply_pruning(model, threshold=random.choice([1e-3, 5e-4, 1e-4]))

            # Memory clearing based on mode
            clear_time = 0.0
            should_clear = False

            if device.type == 'mps':
                if mode == 'modified':  # Clear every batch
                    should_clear = True
                elif mode == 'modified_batch':  # Clear every 20 batches
                    should_clear = (b + 1) % 20 == 0
                elif mode == 'modified_epoch':  # Clear at end of epoch (handled separately)
                    should_clear = False

                if should_clear:
                    clear_start = time.perf_counter()
                    torch.mps.empty_graph_cache()
                    clear_time = time.perf_counter() - clear_start
                    if b % 50 == 0:
                        print(f'  └─ Cache cleared in {clear_time:.4f}s')

            batch_time = time.time() - batch_start_time

            if b % 50 == 0:
                metrics = tracker.record(epoch, b, loss.item(), batch_time, clear_time)
                growth = metrics.get('graph_cache', 0) - tracker.history[0].get('graph_cache', 0)
                print(f"[{model_type}] Epoch {epoch+1}, Batch {b} | "
                      f"Loss: {loss.item():.4f} | Time: {batch_time:.3f}s | "
                      f"{tracker.format_current(metrics)} | Growth: +{growth}")
            else:
                # Still record for plotting, just don't print
                tracker.record(epoch, b, loss.item(), batch_time, clear_time)

        # Clear at end of epoch for epoch-based clearing
        if mode == 'modified_epoch' and device.type == 'mps':
            clear_start = time.perf_counter()
            torch.mps.empty_graph_cache()
            clear_time = time.perf_counter() - clear_start
            print(f"  └─ End of epoch cache clear: {clear_time:.4f}s")

    # Print summary
    summary = tracker.get_summary()
    batches_processed = epochs * batches
    growth = summary.get('gcache_growth', 0)
    growth_per_batch = growth / batches_processed if batches_processed > 0 else 0
    growth_per_sample = growth / (batches_processed * 8) if batches_processed > 0 else 0

    print(f"\n{'='*60}")
    print(f"{model_type} Training Summary:")
    print(f"  Initial G-Cache: {summary.get('initial_gcache', 0)}")
    print(f"  Final G-Cache: {summary.get('final_gcache', 0)}")
    print(f"  G-Cache Growth: {growth}")
    print(f"  Batches Processed: {batches_processed}")
    print(f"  Growth Rate: {growth_per_batch:.1f} graphs/batch")
    print(f"  Per-Sample Rate: {growth_per_sample:.2f} graphs/sample")
    print(f"{'='*60}\n")

    return tracker

# CLI
def plot_results(baseline_file='baseline.json', modified_file='modified.json'):
    """Plot comparison of baseline vs all modified modes."""
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ {baseline_file} not found. Run with --mode baseline first.")
        return

    try:
        with open(modified_file, 'r') as f:
            modified_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ {modified_file} not found. Run with --mode modified first.")
        return

    models = list(baseline_data.keys())

    colors = ['red', 'green', 'blue', 'black']
    strategy_names = ['every_batch', 'every_20_batches', 'every_epoch']
    strategy_labels = ['Every Batch', 'Every 20 Batches', 'Every Epoch']

    for model_name in models:
        if model_name not in modified_data:
            continue

        baseline = baseline_data[model_name]
        modified = modified_data[model_name]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{model_name} - Baseline vs Cache Clearing Strategies', fontsize=16, fontweight='bold')

        # Extract baseline data
        baseline_batches = [h['batch_number'] for h in baseline['history']]
        baseline_gcache = [h['graph_cache'] for h in baseline['history']]
        baseline_gcache = np.array(baseline_gcache) - baseline_gcache[0]  # Normalize to start at 0
        baseline_driver = [h['driver_mem_mb'] for h in baseline['history']]
        baseline_time = [h['batch_time'] for h in baseline['history']]

        # Plot 1: G-Cache Growth
        axes[0, 0].plot(baseline_batches, baseline_gcache, label='Baseline (No Clear)',
                       color='red', linewidth=2, linestyle='--', alpha=0.8)
        for i, (strategy_name, strategy_label) in enumerate(zip(strategy_names, strategy_labels)):
            if strategy_name in modified:
                data = modified[strategy_name]
                batches = [h['batch_number'] for h in data['history']]
                gcache = [h['graph_cache'] for h in data['history']]
                gcache = np.array(gcache) - gcache[0]  # Normalize to start at 0
                axes[0, 0].plot(batches, gcache, label=f'Clear {strategy_label}',
                              color=colors[i+1], linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Batch Number', fontsize=10)
        axes[0, 0].set_ylabel('G-Cache Growth', fontsize=10)
        axes[0, 0].set_title('Graph Cache Accumulation', fontsize=11, fontweight='bold')
        axes[0, 0].legend(fontsize=8, loc='upper left')
        axes[0, 0].grid(True, alpha=0.2, linewidth=0.5)

        # Plot 2: Driver Memory
        axes[0, 1].plot(baseline_batches, baseline_driver, label='Baseline (No Clear)',
                       color='red', linewidth=2, linestyle='--', alpha=0.8)
        for i, (strategy_name, strategy_label) in enumerate(zip(strategy_names, strategy_labels)):
            if strategy_name in modified:
                data = modified[strategy_name]
                batches = [h['batch_number'] for h in data['history']]
                driver = [h['driver_mem_mb'] for h in data['history']]
                axes[0, 1].plot(batches, driver, label=f'Clear {strategy_label}',
                              color=colors[i+1], linewidth=1.5, alpha=0.7)
        axes[0, 1].set_xlabel('Batch Number', fontsize=10)
        axes[0, 1].set_ylabel('Driver Memory (MB)', fontsize=10)
        axes[0, 1].set_title('Driver Allocated Memory', fontsize=11, fontweight='bold')
        axes[0, 1].legend(fontsize=8, loc='upper left')
        axes[0, 1].grid(True, alpha=0.2, linewidth=0.5)

        # Plot 3: Batch Runtime (excluding clear time)
        axes[0, 2].plot(baseline_batches, baseline_time, label='Baseline',
                       color='red', alpha=0.3, linewidth=0.8)
        for i, (strategy_name, strategy_label) in enumerate(zip(strategy_names, strategy_labels)):
            if strategy_name in modified:
                data = modified[strategy_name]
                batches = [h['batch_number'] for h in data['history']]
                times = [h['batch_time'] for h in data['history']]
                axes[0, 2].plot(batches, times, label=f'Clear {strategy_label}',
                              color=colors[i+1], alpha=0.4, linewidth=0.8)
        axes[0, 2].set_xlabel('Batch Number', fontsize=10)
        axes[0, 2].set_ylabel('Batch Time (seconds)', fontsize=10)
        axes[0, 2].set_title('Batch Processing Time (excl. clear)', fontsize=11, fontweight='bold')
        axes[0, 2].legend(fontsize=8, loc='upper left')
        axes[0, 2].grid(True, alpha=0.2, linewidth=0.5)

        # Plot 4: Cache Clear Time
        axes[1, 0].set_xlabel('Batch Number', fontsize=10)
        axes[1, 0].set_ylabel('Clear Time (seconds)', fontsize=10)
        axes[1, 0].set_title('Cache Clearing Overhead', fontsize=11, fontweight='bold')
        for i, (strategy_name, strategy_label) in enumerate(zip(strategy_names, strategy_labels)):
            if strategy_name in modified:
                data = modified[strategy_name]
                batches = [h['batch_number'] for h in data['history']]
                clear_times = [h['clear_time'] for h in data['history']]
                axes[1, 0].plot(batches, clear_times, label=f'Clear {strategy_label}',
                              color=colors[i+1], alpha=0.5, linewidth=0.8)
        axes[1, 0].legend(fontsize=8, loc='upper left')
        axes[1, 0].grid(True, alpha=0.2, linewidth=0.5)

        # Plot 5: Total Time (batch + clear)
        axes[1, 1].plot(baseline_batches, baseline_time, label='Baseline',
                       color='red', linewidth=0.8, alpha=0.3)
        for i, (strategy_name, strategy_label) in enumerate(zip(strategy_names, strategy_labels)):
            if strategy_name in modified:
                data = modified[strategy_name]
                batches = [h['batch_number'] for h in data['history']]
                total_times = [h['batch_time'] + h['clear_time'] for h in data['history']]
                axes[1, 1].plot(batches, total_times, label=f'Clear {strategy_label}',
                              color=colors[i+1], linewidth=0.8, alpha=0.4)
        axes[1, 1].set_xlabel('Batch Number', fontsize=10)
        axes[1, 1].set_ylabel('Total Time (seconds)', fontsize=10)
        axes[1, 1].set_title('Total Time per Batch (batch + clear)', fontsize=11, fontweight='bold')
        axes[1, 1].legend(fontsize=8, loc='upper left')
        axes[1, 1].grid(True, alpha=0.2, linewidth=0.5)

        # Plot 6: Summary Statistics Table
        summary_text = "Baseline (No Clear):\n"
        summary_text += f"  G-Cache Growth: {baseline['summary']['gcache_growth']}\n"
        summary_text += f"  Avg Batch Time: {np.mean(baseline_time):.3f}s\n"
        summary_text += f"  Total Time: {baseline['summary']['total_time']:.1f}s\n\n"

        for strategy_name, strategy_label in zip(strategy_names, strategy_labels):
            if strategy_name in modified:
                data = modified[strategy_name]
                batch_times = [h['batch_time'] for h in data['history']]
                clear_times = [h['clear_time'] for h in data['history']]
                total_times = [h['batch_time'] + h['clear_time'] for h in data['history']]

                summary_text += f"Clear {strategy_label}:\n"
                summary_text += f"  G-Cache Growth: {data['summary']['gcache_growth']}\n"
                summary_text += f"  Avg Batch Time: {np.mean(batch_times):.3f}s\n"
                summary_text += f"  Avg Clear Time: {np.mean(clear_times):.4f}s\n"
                summary_text += f"  Avg Total Time: {np.mean(total_times):.3f}s\n"
                summary_text += f"  Total Time: {data['summary']['total_time']:.1f}s\n\n"

        axes[1, 2].text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Statistics')

        plt.tight_layout()
        output_file = f'{model_name.lower()}_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot: {output_file}")
        plt.close()

    print(f"\n{'='*60}")
    print(f"All plots generated successfully!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline",
                       choices=["baseline", "modified", "plot"],
                       help="baseline: no clearing, modified: test all 3 clearing strategies, plot: generate plots")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--models", nargs="+",
                       default=["CNN", "Transformer", "LSTM", "RNN", "GRU", "VAE", "ResNet", "UNet", "AttentionLSTM"],
                       help="Models to train. Available: CNN, Transformer, LSTM, RNN, GRU, VAE, ResNet, UNet, AttentionLSTM")
    args = parser.parse_args()

    if args.mode == "plot":
        plot_results()
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'#'*60}")
    print(f"# Comprehensive Memory Training with G-Cache Tracking")
    print(f"# Device: {device}")
    print(f"# Epochs: {args.epochs}, Batches: {args.batches}")
    print(f"{'#'*60}")

    model_map = {
        "CNN": CNN,
        "Transformer": SimpleTransformer,
        "LSTM": SimpleLSTM,
        "RNN": SimpleRNN,
        "GRU": SimpleGRU,
        "VAE": VAE,
        "ResNet": ResNet,
        "UNet": UNet,
        "AttentionLSTM": AttentionLSTM,
    }

    results = {}
    detailed_results = {}

    if args.mode == 'baseline':
        # Run baseline only
        start = time.time()
        for name in args.models:
            print(f"\n{'#'*60}")
            print(f"# Model: {name}")
            print(f"{'#'*60}")
            model = model_map[name]()
            tracker = train_model(model, name, device, args.epochs, args.batches, mode='baseline', batch_size=8)

            summary = tracker.get_summary()
            results[name] = summary

            detailed_results[name] = {
                'summary': summary,
                'history': [
                    {
                        'batch_number': i,
                        'epoch': h['epoch'],
                        'batch': h['batch'],
                        'loss': h['loss'],
                        'batch_time': h['batch_time'],
                        'clear_time': h.get('clear_time', 0.0),
                        'driver_mem_mb': h.get('driver_mem_mb', 0),
                        'current_mem_mb': h.get('current_mem_mb', 0),
                        'graph_cache': h.get('graph_cache', 0),
                        'kernel_cache': h.get('kernel_cache', 0),
                    }
                    for i, h in enumerate(tracker.history)
                ],
                'config': {
                    'model_type': name,
                    'epochs': args.epochs,
                    'batches': args.batches,
                    'batch_size': 8,
                    'mode': 'baseline',
                }
            }

            del model
            if device.type == 'mps':
                torch.mps.empty_graph_cache()

        elapsed = time.time() - start
        for name in detailed_results:
            detailed_results[name]['summary']['total_time'] = elapsed / len(args.models)

    else:  # args.mode == 'modified'
        # Run all 3 clearing strategies
        clearing_strategies = [
            ('every_batch', 'modified'),
            ('every_20_batches', 'modified_batch'),
            ('every_epoch', 'modified_epoch')
        ]

        for name in args.models:
            print(f"\n{'#'*60}")
            print(f"# Model: {name}")
            print(f"{'#'*60}")

            detailed_results[name] = {}

            for strategy_name, strategy_mode in clearing_strategies:
                print(f"\n{'='*60}")
                print(f"Running strategy: {strategy_name}")
                print(f"{'='*60}")

                model = model_map[name]()
                start = time.time()
                tracker = train_model(model, name, device, args.epochs, args.batches,
                                    mode=strategy_mode, batch_size=8)
                elapsed = time.time() - start

                summary = tracker.get_summary()
                summary['total_time'] = elapsed

                detailed_results[name][strategy_name] = {
                    'summary': summary,
                    'history': [
                        {
                            'batch_number': i,
                            'epoch': h['epoch'],
                            'batch': h['batch'],
                            'loss': h['loss'],
                            'batch_time': h['batch_time'],
                            'clear_time': h.get('clear_time', 0.0),
                            'driver_mem_mb': h.get('driver_mem_mb', 0),
                            'current_mem_mb': h.get('current_mem_mb', 0),
                            'graph_cache': h.get('graph_cache', 0),
                            'kernel_cache': h.get('kernel_cache', 0),
                        }
                        for i, h in enumerate(tracker.history)
                    ],
                    'config': {
                        'model_type': name,
                        'epochs': args.epochs,
                        'batches': args.batches,
                        'batch_size': 8,
                        'strategy': strategy_name,
                    }
                }

                del model
                if device.type == 'mps':
                    torch.mps.empty_graph_cache()

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")

    if args.mode == 'baseline':
        print(f"Mode: BASELINE (No Cache Clearing)")
        print(f"Epochs: {args.epochs}, Batches: {args.batches}, Batch Size: 8")
        print(f"\nG-Cache Growth by Model:")
        for name in detailed_results:
            summary = detailed_results[name]['summary']
            growth = summary.get('gcache_growth', 0)
            initial = summary.get('initial_gcache', 0)
            final = summary.get('final_gcache', 0)
            print(f"  {name:15s}: {growth:4d} graphs ({initial} → {final})")
    else:
        print(f"Mode: MODIFIED (All 3 Clearing Strategies)")
        print(f"Epochs: {args.epochs}, Batches: {args.batches}, Batch Size: 8")
        for name in detailed_results:
            print(f"\n{name}:")
            for strategy_name in ['every_batch', 'every_20_batches', 'every_epoch']:
                if strategy_name in detailed_results[name]:
                    summary = detailed_results[name][strategy_name]['summary']
                    growth = summary.get('gcache_growth', 0)
                    total_time = summary.get('total_time', 0)
                    print(f"  {strategy_name:20s}: G-Cache growth = {growth:4d}, Time = {total_time:.1f}s")

    print(f"\n{'='*60}")
    print(f"All models completed")
    print(f"{'='*60}")

    # Save results to JSON
    output_file = f"{args.mode}.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Run with --mode plot to generate comparison graphs")


if __name__ == "__main__":
    main()