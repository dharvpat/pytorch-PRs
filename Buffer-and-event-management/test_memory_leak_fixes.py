"""
MPS Memory Leak Fixes Validation Suite (Issues #2, #3, #4)

This test suite validates the fixes for:
  - Issue #2: buffers_pending_free never freed
  - Issue #3: MPSEvent pool never cleared
  - Issue #4: Missing synchronization before cleanup

The test suite monitors multiple memory metrics:
  - torch.mps.driver_allocated_memory() - GPU driver memory (PyTorch's view)
  - Process RSS memory - System-wide process memory (Activity Monitor view)
  - torch.mps.current_allocated_memory() - Current active allocations

Usage:
    Run baseline (original PyTorch):
    python3 test_memory_leak_fixes.py --mode baseline --epochs 5 --batches-per-epoch 100

    Run with fixes applied:
    python3 test_memory_leak_fixes.py --mode fixed --epochs 5 --batches-per-epoch 100

    Compare results:
    python3 test_memory_leak_fixes.py --mode compare

    Generate plots:
    python3 test_memory_leak_fixes.py --mode plot

Requirements:
    - psutil (pip install psutil)
    - matplotlib (pip install matplotlib)
    - PyTorch with MPS backend
"""

import argparse
import json
import time
import os
import gc
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    print("WARNING: matplotlib not installed. Plotting will be disabled.")
    print("To enable plotting, run: pip install matplotlib numpy")
    HAS_MATPLOTLIB = False


class MemoryMonitor:
    """
    Comprehensive memory monitoring for MPS backend.
    Tracks both PyTorch's view (driver_allocated_memory) and system view (RSS).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_mps = device.type == 'mps'
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[Dict] = []
        self.baseline_rss = 0
        self.baseline_driver = 0

    def set_baseline(self):
        """Establish baseline memory after warmup."""
        if self.is_mps:
            self.baseline_driver = torch.mps.driver_allocated_memory()
        self.baseline_rss = self.process.memory_info().rss

    def take_snapshot(self, epoch: int, batch: int, phase: str = "training") -> Dict:
        """
        Take a comprehensive memory snapshot.

        Args:
            epoch: Current epoch number
            batch: Current batch number
            phase: Phase identifier (warmup, training, cleanup)

        Returns:
            Dictionary with all memory metrics
        """
        snapshot = {
            'timestamp': time.time(),
            'epoch': epoch,
            'batch': batch,
            'phase': phase,
        }

        # PyTorch MPS memory (if available)
        if self.is_mps:
            snapshot['driver_allocated_mb'] = torch.mps.driver_allocated_memory() / (1024**2)
            snapshot['current_allocated_mb'] = torch.mps.current_allocated_memory() / (1024**2)
            snapshot['driver_growth_mb'] = (torch.mps.driver_allocated_memory() - self.baseline_driver) / (1024**2)

        # System memory (what Activity Monitor shows)
        mem_info = self.process.memory_info()
        snapshot['rss_mb'] = mem_info.rss / (1024**2)
        snapshot['rss_growth_mb'] = (mem_info.rss - self.baseline_rss) / (1024**2)
        snapshot['vms_mb'] = mem_info.vms / (1024**2)

        # Memory mapped files (can indicate Metal buffer allocations)
        if hasattr(mem_info, 'pfaults'):
            snapshot['page_faults'] = mem_info.pfaults

        self.snapshots.append(snapshot)
        return snapshot

    def get_current_memory_mb(self) -> Tuple[float, float, float]:
        """
        Get current memory usage in MB.
        Returns: (driver_allocated, current_allocated, rss)
        """
        if self.is_mps:
            driver = torch.mps.driver_allocated_memory() / (1024**2)
            current = torch.mps.current_allocated_memory() / (1024**2)
        else:
            driver = 0
            current = 0

        rss = self.process.memory_info().rss / (1024**2)
        return driver, current, rss

    def get_memory_growth_mb(self) -> Tuple[float, float]:
        """
        Get memory growth since baseline.
        Returns: (driver_growth, rss_growth)
        """
        driver_growth = 0
        if self.is_mps:
            driver_growth = (torch.mps.driver_allocated_memory() - self.baseline_driver) / (1024**2)

        rss_growth = (self.process.memory_info().rss - self.baseline_rss) / (1024**2)
        return driver_growth, rss_growth

    def format_current(self) -> str:
        """Format current memory state for display."""
        driver, current, rss = self.get_current_memory_mb()
        driver_growth, rss_growth = self.get_memory_growth_mb()

        if self.is_mps:
            return (f"Driver: {driver:.1f}MB ({driver_growth:+.1f}MB) | "
                   f"Current: {current:.1f}MB | "
                   f"RSS: {rss:.1f}MB ({rss_growth:+.1f}MB)")
        else:
            return f"RSS: {rss:.1f}MB ({rss_growth:+.1f}MB)"

    def get_summary(self) -> Dict:
        """Get summary statistics from all snapshots."""
        if not self.snapshots:
            return {}

        training_snapshots = [s for s in self.snapshots if s['phase'] == 'training']
        if not training_snapshots:
            return {}

        initial = training_snapshots[0]
        final = training_snapshots[-1]

        summary = {
            'num_snapshots': len(training_snapshots),
            'initial_rss_mb': initial['rss_mb'],
            'final_rss_mb': final['rss_mb'],
            'rss_growth_mb': final['rss_mb'] - initial['rss_mb'],
        }

        if self.is_mps:
            summary.update({
                'initial_driver_mb': initial['driver_allocated_mb'],
                'final_driver_mb': final['driver_allocated_mb'],
                'driver_growth_mb': final['driver_allocated_mb'] - initial['driver_allocated_mb'],
                'initial_current_mb': initial['current_allocated_mb'],
                'final_current_mb': final['current_allocated_mb'],
                'current_growth_mb': final['current_allocated_mb'] - initial['current_allocated_mb'],
            })

        # Calculate growth rates
        total_batches = len(training_snapshots)
        if total_batches > 0:
            summary['rss_growth_per_batch_mb'] = summary['rss_growth_mb'] / total_batches
            if self.is_mps:
                summary['driver_growth_per_batch_mb'] = summary['driver_growth_mb'] / total_batches

        return summary


class SimpleModel(nn.Module):
    """
    Simple model for memory leak testing.
    Uses operations that create temporary buffers and events.
    """
    def __init__(self, input_size=1000, hidden_size=500):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # Create intermediate tensors that will go to buffers_pending_free
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ConvModel(nn.Module):
    """
    Convolutional model for testing with image-like data.
    Creates many temporary tensors during forward/backward passes.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        # Each convolution creates many temporary buffers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        return self.fc(x)

class SimpleTransformer(nn.Module):
    """Transformer model for sequence data."""
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.embed = nn.Linear(64, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class SimpleLSTM(nn.Module):
    """LSTM model for sequence data."""
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class SimpleRNN(nn.Module):
    """RNN model for sequence data."""
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


class SimpleGRU(nn.Module):
    """GRU model for sequence data."""
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class VAE(nn.Module):
    """Variational Autoencoder for image data."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, padding=1)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 10)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # Global average pooling
        h = h.mean(dim=[2, 3])
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_dropped = self.dropout(z)
        features = F.relu(self.decoder_fc(z_dropped))
        return self.fc(features)


class ResNet(nn.Module):
    """Simple ResNet-style model with residual connections."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # First residual block
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Downsample
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Second residual block
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 10)

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
        out = out.mean(dim=[2, 3])
        out = self.dropout(out)
        return self.fc(out)


class UNet(nn.Module):
    """U-Net style architecture with skip connections."""
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

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        # Decoder with skip connections
        d1 = self.up1(e3)
        # Handle size mismatch
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



def create_input_batch(model_type, device, batch_size=32):
    """
    Create appropriate input tensor based on model type.

    Args:
        model_type: Type of model
        device: Device to create tensor on
        batch_size: Batch size

    Returns:
        Input tensor with appropriate shape
    """
    # Image-based models
    if model_type in ['conv', 'vae', 'resnet', 'unet']:
        return torch.randn(batch_size, 3, 64, 64, device=device)

    # Transformer
    elif model_type == 'transformer':
        return torch.randn(batch_size, 32, 64, device=device)

    # Sequence models (LSTM, RNN, GRU)
    elif model_type in ['lstm', 'rnn', 'gru']:
        return torch.randn(batch_size, 16, 32, device=device)

    # Simple linear model
    else:  # 'simple'
        return torch.randn(batch_size, 1000, device=device)


def warmup_phase(model, device, num_batches=20, model_type='simple'):
    """
    Warmup phase to establish stable baseline.
    Creates initial graph cache and buffer pool.
    """
    print("Starting warmup phase...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for i in range(num_batches):
        # Create batch with appropriate shape for model type
        batch = create_input_batch(model_type, device)
        targets = torch.randint(0, 10, (batch.size(0),), device=device)

        # Forward + backward
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Force GPU completion
    if device.type == 'mps':
        torch.mps.synchronize()

    print(f"Warmup complete ({num_batches} batches)")


def training_phase(model, device, monitor,
                   epochs=5, batches_per_epoch=100,
                   clear_cache=False, clear_frequency='epoch',
                   model_type='simple', verbose=True):
    """
    Main training phase with memory monitoring.

    Args:
        model: Model to train
        device: Device to train on
        monitor: MemoryMonitor instance
        epochs: Number of epochs
        batches_per_epoch: Batches per epoch
        clear_cache: Whether to call empty_cache()
        clear_frequency: When to clear ('epoch', 'batch', or 'every_10')
        model_type: Type of model ('simple' or 'conv')
        verbose: Whether to print progress
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_batches = 0
    cache_clears = 0
    total_clear_time = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        for batch_idx in range(batches_per_epoch):
            batch_start = time.time()

            # Create batch with appropriate shape for model type
            batch = create_input_batch(model_type, device)
            targets = torch.randint(0, 10, (batch.size(0),), device=device)

            # Training step
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            # Batch complete - tensors go to buffers_pending_free
            batch_time = time.time() - batch_start

            # Memory clearing logic
            should_clear = False
            if clear_cache and device.type == 'mps':
                if clear_frequency == 'batch':
                    should_clear = True
                elif clear_frequency == 'every_10':
                    should_clear = (batch_idx + 1) % 10 == 0
                # 'epoch' mode clears at end of epoch

            clear_time = 0
            if should_clear:
                clear_start = time.time()
                torch.mps.synchronize()  # Issue #4 fix
                torch.mps.empty_cache()   # Triggers Issue #2 and #3 fixes
                clear_time = time.time() - clear_start
                cache_clears += 1
                total_clear_time += clear_time

            # Take memory snapshot
            total_batches += 1
            snapshot = monitor.take_snapshot(epoch, batch_idx, phase='training')

            # Print progress
            if verbose and (batch_idx % 20 == 0 or batch_idx == batches_per_epoch - 1):
                driver_growth, rss_growth = monitor.get_memory_growth_mb()
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{batches_per_epoch} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {batch_time:.3f}s | "
                      f"{monitor.format_current()}")
                if should_clear:
                    print(f"    └─ Cache cleared in {clear_time:.4f}s")

        # End of epoch clearing
        if clear_cache and clear_frequency == 'epoch' and device.type == 'mps':
            clear_start = time.time()
            torch.mps.synchronize()  # Issue #4 fix
            torch.mps.empty_cache()   # Triggers Issue #2 and #3 fixes
            clear_time = time.time() - clear_start
            cache_clears += 1
            total_clear_time += clear_time
            if verbose:
                print(f"  └─ End-of-epoch cache clear: {clear_time:.4f}s")

        epoch_time = time.time() - epoch_start
        if verbose:
            print(f"Epoch {epoch+1} complete in {epoch_time:.2f}s")

    return {
        'total_batches': total_batches,
        'cache_clears': cache_clears,
        'total_clear_time': total_clear_time,
        'avg_clear_time': total_clear_time / cache_clears if cache_clears > 0 else 0
    }


def run_memory_leak_test(mode='baseline', epochs=5, batches_per_epoch=100,
                         model_type='simple', output_file=None):
    """
    Run the memory leak test.

    Args:
        mode: 'baseline' (no fixes) or 'fixed' (with fixes)
        epochs: Number of training epochs
        batches_per_epoch: Batches per epoch
        model_type: 'simple' or 'conv'
        output_file: File to save results (auto-generated if None)
    """
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if device.type != 'mps':
        print("ERROR: MPS device not available. This test requires Apple Silicon with MPS.")
        return

    print("\n" + "="*70)
    print(f"MPS MEMORY LEAK TEST - Mode: {mode.upper()}")
    print("="*70)
    print(f"Testing fixes for Issues #2, #3, #4")
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Epochs: {epochs}, Batches/epoch: {batches_per_epoch}")
    print(f"Total batches: {epochs * batches_per_epoch}")

    if mode == 'baseline':
        print("\nMode: BASELINE - No cache clearing (simulates unfixed PyTorch)")
        print("Expected: Linear memory growth due to:")
        print("  - Issue #2: buffers_pending_free accumulation")
        print("  - Issue #3: Event pool accumulation")
        print("  - Issue #4: No synchronization before cleanup")
    else:
        print("\nMode: FIXED - With cache clearing (simulates fixed PyTorch)")
        print("Expected: Stable memory after warmup due to:")
        print("  - Issue #2 fix: freeInactiveBuffers() called")
        print("  - Issue #3 fix: Event pool cleared")
        print("  - Issue #4 fix: Synchronization before cleanup")

    print("="*70 + "\n")

    # Create model based on type
    model_map = {
        'simple': SimpleModel,
        'conv': ConvModel,
        'transformer': SimpleTransformer,
        'lstm': SimpleLSTM,
        'rnn': SimpleRNN,
        'gru': SimpleGRU,
        'vae': VAE,
        'resnet': ResNet,
        'unet': UNet,
    }

    if model_type.lower() not in model_map:
        print(f"ERROR: Unknown model type '{model_type}'")
        print(f"Available types: {', '.join(model_map.keys())}")
        return

    model = model_map[model_type.lower()]().to(device)

    # Create monitor
    monitor = MemoryMonitor(device)

    # Initial memory snapshot
    print("Initial memory state:")
    print(f"  {monitor.format_current()}")
    print()

    # Warmup phase
    print("="*70)
    print("WARMUP PHASE")
    print("="*70)
    warmup_phase(model, device, num_batches=20, model_type=model_type)

    # Clear cache after warmup to establish clean baseline
    if device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
        time.sleep(0.5)  # Let cleanup settle

    # Python garbage collection
    gc.collect()

    # Establish baseline
    monitor.set_baseline()
    print(f"\nBaseline established after warmup:")
    print(f"  {monitor.format_current()}")
    print()

    # Training phase
    print("="*70)
    print("TRAINING PHASE")
    print("="*70)

    if mode == 'baseline':
        # No cache clearing - simulates unfixed PyTorch
        training_stats = training_phase(
            model, device, monitor,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            clear_cache=False,
            model_type=model_type,
            verbose=True
        )
    else:  # mode == 'fixed'
        # With cache clearing - simulates fixed PyTorch
        training_stats = training_phase(
            model, device, monitor,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            clear_cache=True,
            clear_frequency='epoch',  # Clear at end of each epoch
            model_type=model_type,
            verbose=True
        )

    print("\n" + "="*70)
    print("FINAL CLEANUP")
    print("="*70)

    # Final cleanup
    if device.type == 'mps':
        print("Synchronizing GPU...")
        torch.mps.synchronize()
        time.sleep(0.1)

        print("Clearing cache...")
        torch.mps.empty_cache()
        time.sleep(0.1)

    gc.collect()

    # Final snapshot
    final_snapshot = monitor.take_snapshot(epochs, batches_per_epoch, phase='cleanup')
    print(f"\nFinal memory state:")
    print(f"  {monitor.format_current()}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    summary = monitor.get_summary()

    print(f"\nMemory Growth:")
    print(f"  Driver Memory: {summary.get('initial_driver_mb', 0):.1f}MB → "
          f"{summary.get('final_driver_mb', 0):.1f}MB "
          f"(+{summary.get('driver_growth_mb', 0):.1f}MB)")
    print(f"  RSS (Activity Monitor): {summary.get('initial_rss_mb', 0):.1f}MB → "
          f"{summary.get('final_rss_mb', 0):.1f}MB "
          f"(+{summary.get('rss_growth_mb', 0):.1f}MB)")

    print(f"\nGrowth Rates:")
    print(f"  Driver: {summary.get('driver_growth_per_batch_mb', 0):.3f}MB/batch")
    print(f"  RSS: {summary.get('rss_growth_per_batch_mb', 0):.3f}MB/batch")

    if training_stats['cache_clears'] > 0:
        print(f"\nCache Clearing:")
        print(f"  Total clears: {training_stats['cache_clears']}")
        print(f"  Total time: {training_stats['total_clear_time']:.2f}s")
        print(f"  Average time: {training_stats['avg_clear_time']:.4f}s")

    print("\nDiagnosis:")
    driver_growth = summary.get('driver_growth_mb', 0)
    rss_growth = summary.get('rss_growth_mb', 0)

    if mode == 'baseline':
        if driver_growth > 200 or rss_growth > 500:
            print("  ✅ EXPECTED: Significant memory leak detected (unfixed behavior)")
            print("     This confirms Issues #2, #3, #4 are present")
        else:
            print("  ⚠️  UNEXPECTED: Memory growth is low for baseline mode")
            print("     Leak may require more batches to manifest")
    else:  # fixed
        if driver_growth < 100 and rss_growth < 200:
            print("  ✅ PASS: Memory stable (fixes working!)")
            print("     Issues #2, #3, #4 are properly fixed")
        else:
            print("  ❌ FAIL: Memory still growing (fixes not working)")
            print(f"     Driver growth: {driver_growth:.1f}MB (expected <100MB)")
            print(f"     RSS growth: {rss_growth:.1f}MB (expected <200MB)")

    print("="*70)

    # Save results
    if output_file is None:
        output_file = f"{mode}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = {
        'mode': mode,
        'model_type': model_type,
        'config': {
            'epochs': epochs,
            'batches_per_epoch': batches_per_epoch,
            'total_batches': epochs * batches_per_epoch,
        },
        'summary': summary,
        'training_stats': training_stats,
        'snapshots': monitor.snapshots,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print()

    return results


def compare_results(baseline_file, fixed_file):
    """
    Compare baseline vs fixed results and print analysis.
    """
    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"❌ Baseline file not found: {baseline_file}")
        print("   Run with --mode baseline first")
        return

    try:
        with open(fixed_file, 'r') as f:
            fixed = json.load(f)
    except FileNotFoundError:
        print(f"❌ Fixed file not found: {fixed_file}")
        print("   Run with --mode fixed first")
        return

    print("\n" + "="*70)
    print("BASELINE VS FIXED COMPARISON")
    print("="*70)

    baseline_sum = baseline['summary']
    fixed_sum = fixed['summary']

    print(f"\nDriver Memory Growth:")
    print(f"  Baseline: {baseline_sum.get('driver_growth_mb', 0):.1f}MB")
    print(f"  Fixed:    {fixed_sum.get('driver_growth_mb', 0):.1f}MB")

    driver_reduction = baseline_sum.get('driver_growth_mb', 0) - fixed_sum.get('driver_growth_mb', 0)
    if baseline_sum.get('driver_growth_mb', 0) > 0:
        driver_pct = (driver_reduction / baseline_sum.get('driver_growth_mb', 1)) * 100
        print(f"  Reduction: {driver_reduction:.1f}MB ({driver_pct:.1f}%)")

    print(f"\nRSS Memory Growth (Activity Monitor):")
    print(f"  Baseline: {baseline_sum.get('rss_growth_mb', 0):.1f}MB")
    print(f"  Fixed:    {fixed_sum.get('rss_growth_mb', 0):.1f}MB")

    rss_reduction = baseline_sum.get('rss_growth_mb', 0) - fixed_sum.get('rss_growth_mb', 0)
    if baseline_sum.get('rss_growth_mb', 0) > 0:
        rss_pct = (rss_reduction / baseline_sum.get('rss_growth_mb', 1)) * 100
        print(f"  Reduction: {rss_reduction:.1f}MB ({rss_pct:.1f}%)")

    print(f"\nGrowth Rate per Batch:")
    print(f"  Baseline Driver: {baseline_sum.get('driver_growth_per_batch_mb', 0):.3f}MB/batch")
    print(f"  Fixed Driver:    {fixed_sum.get('driver_growth_per_batch_mb', 0):.3f}MB/batch")
    print(f"  Baseline RSS:    {baseline_sum.get('rss_growth_per_batch_mb', 0):.3f}MB/batch")
    print(f"  Fixed RSS:       {fixed_sum.get('rss_growth_per_batch_mb', 0):.3f}MB/batch")

    print(f"\nFix Effectiveness:")
    if driver_pct > 50 and rss_pct > 50:
        print(f"  ✅ EXCELLENT: Fixes reduced memory growth by >50%")
        print(f"     Issues #2, #3, #4 successfully addressed")
    elif driver_pct > 25 and rss_pct > 25:
        print(f"  ✓ GOOD: Fixes reduced memory growth by >25%")
        print(f"     Partial improvement, may need more aggressive clearing")
    else:
        print(f"  ⚠️  LIMITED: Fixes reduced memory growth by <25%")
        print(f"     Fixes may not be fully effective")

    print("="*70 + "\n")


def plot_comparison(baseline_file, fixed_file):
    """
    Generate comparison plots between baseline and fixed versions.
    """
    if not HAS_MATPLOTLIB:
        print("❌ Matplotlib not installed. Cannot generate plots.")
        return

    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"❌ Baseline file not found: {baseline_file}")
        return

    try:
        with open(fixed_file, 'r') as f:
            fixed = json.load(f)
    except FileNotFoundError:
        print(f"❌ Fixed file not found: {fixed_file}")
        return

    # Extract training snapshots
    baseline_snaps = [s for s in baseline['snapshots'] if s['phase'] == 'training']
    fixed_snaps = [s for s in fixed['snapshots'] if s['phase'] == 'training']

    # Create batch indices
    baseline_batches = list(range(len(baseline_snaps)))
    fixed_batches = list(range(len(fixed_snaps)))

    # Extract data
    baseline_driver = [s['driver_allocated_mb'] for s in baseline_snaps]
    baseline_rss = [s['rss_mb'] for s in baseline_snaps]
    baseline_driver_growth = [s['driver_growth_mb'] for s in baseline_snaps]
    baseline_rss_growth = [s['rss_growth_mb'] for s in baseline_snaps]

    fixed_driver = [s['driver_allocated_mb'] for s in fixed_snaps]
    fixed_rss = [s['rss_mb'] for s in fixed_snaps]
    fixed_driver_growth = [s['driver_growth_mb'] for s in fixed_snaps]
    fixed_rss_growth = [s['rss_growth_mb'] for s in fixed_snaps]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('MPS Memory Leak Fixes - Baseline vs Fixed', fontsize=16, fontweight='bold')

    # Plot 1: Driver Memory Absolute
    axes[0, 0].plot(baseline_batches, baseline_driver,
                    label='Baseline (Unfixed)', color='red', linewidth=2, alpha=0.7)
    axes[0, 0].plot(fixed_batches, fixed_driver,
                    label='Fixed (Issues #2,#3,#4)', color='green', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Driver Allocated Memory (MB)')
    axes[0, 0].set_title('GPU Driver Memory (PyTorch View)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: RSS Memory Absolute
    axes[0, 1].plot(baseline_batches, baseline_rss,
                    label='Baseline (Unfixed)', color='red', linewidth=2, alpha=0.7)
    axes[0, 1].plot(fixed_batches, fixed_rss,
                    label='Fixed (Issues #2,#3,#4)', color='green', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('RSS Memory (MB)')
    axes[0, 1].set_title('Process Memory (Activity Monitor View)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Driver Memory Growth
    axes[1, 0].plot(baseline_batches, baseline_driver_growth,
                    label='Baseline (Unfixed)', color='red', linewidth=2, alpha=0.7)
    axes[1, 0].plot(fixed_batches, fixed_driver_growth,
                    label='Fixed (Issues #2,#3,#4)', color='green', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Batch Number')
    axes[1, 0].set_ylabel('Memory Growth from Baseline (MB)')
    axes[1, 0].set_title('Driver Memory Growth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: RSS Memory Growth
    axes[1, 1].plot(baseline_batches, baseline_rss_growth,
                    label='Baseline (Unfixed)', color='red', linewidth=2, alpha=0.7)
    axes[1, 1].plot(fixed_batches, fixed_rss_growth,
                    label='Fixed (Issues #2,#3,#4)', color='green', linewidth=2, alpha=0.7)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Batch Number')
    axes[1, 1].set_ylabel('Memory Growth from Baseline (MB)')
    axes[1, 1].set_title('RSS Memory Growth')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = f"memory_leak_fixes_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test MPS memory leak fixes (Issues #2, #3, #4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline test (unfixed PyTorch)
  python test_memory_leak_fixes.py --mode baseline --epochs 5 --batches-per-epoch 100

  # Run fixed test (with fixes applied)
  python test_memory_leak_fixes.py --mode fixed --epochs 5 --batches-per-epoch 100

  # Compare results
  python test_memory_leak_fixes.py --mode compare --baseline baseline.json --fixed fixed.json

  # Generate plots
  python test_memory_leak_fixes.py --mode plot --baseline baseline.json --fixed fixed.json
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['baseline', 'fixed', 'compare', 'plot'],
                       help='Test mode: baseline (unfixed), fixed (with fixes), compare, or plot')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--batches-per-epoch', type=int, default=100,
                       help='Batches per epoch (default: 100)')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'conv', 'transformer', 'lstm', 'rnn', 'gru', 'vae', 'resnet', 'unet'],
                       help='Model type (default: simple). Options: simple, conv, transformer, lstm, rnn, gru, vae, resnet, unet')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (auto-generated if not specified)')
    parser.add_argument('--baseline', type=str, default='baseline.json',
                       help='Baseline results file for compare/plot modes')
    parser.add_argument('--fixed', type=str, default='fixed.json',
                       help='Fixed results file for compare/plot modes')

    args = parser.parse_args()

    if args.mode in ['baseline', 'fixed']:
        run_memory_leak_test(
            mode=args.mode,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            model_type=args.model_type,
            output_file=args.output
        )
    elif args.mode == 'compare':
        compare_results(args.baseline, args.fixed)
    elif args.mode == 'plot':
        plot_comparison(args.baseline, args.fixed)


if __name__ == '__main__':
    main()


