import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import numpy as np

import sys
sys.path.append("../..")

OSCILLATE_THRESHOLD = False
DELTA_BASE_THRESHOLD = 0.4
DELTA_WAVE_AMPLITUDE = 0.3
DELTA_WAVE_FREQUENCY = 28 * 2 # In terms of time steps (i.e. one full oscillation completed at this timestep)
NEGATIVE_AT_TROUGH = True

def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor
):
    tensor = tensor.view(batch_size_, sequence_length_, input_size_)  # BxTxC
    tensor = tensor.permute(1, 0, 2)  # TxBxC
    tensor = tensor[permuted_idx_, :, :]

    # Delta between time steps
    tensor = tensor - tensor.roll(1, 0)
    tensor[0] = 0

    if NEGATIVE_AT_TROUGH:
        if OSCILLATE_THRESHOLD:
            wave = torch.sin(
                2 * torch.pi *
                torch.arange(
                    sequence_length_,
                    dtype=tensor.dtype
                ) / DELTA_WAVE_FREQUENCY
            )
            pos_threshold = DELTA_BASE_THRESHOLD - DELTA_WAVE_AMPLITUDE * wave
            neg_threshold = DELTA_BASE_THRESHOLD + DELTA_WAVE_AMPLITUDE * wave
            pos_threshold = pos_threshold[:, None, None].expand(-1, batch_size_, input_size_)
            neg_threshold = neg_threshold[:, None, None].expand(-1, batch_size_, input_size_)
        else:
            pos_threshold = DELTA_BASE_THRESHOLD
            neg_threshold = DELTA_BASE_THRESHOLD
        pos_spike = torch.where(
            tensor > pos_threshold,
            torch.ones_like(tensor),
            torch.zeros_like(tensor)
        )
        neg_spike = torch.where(
            tensor < -neg_threshold,
            -torch.ones_like(tensor),
            torch.zeros_like(tensor)
        )

    else:
        if OSCILLATE_THRESHOLD:
            wave = torch.sin(
                2 * torch.pi *
                torch.arange(
                    sequence_length_,
                    dtype=tensor.dtype
                ) / DELTA_WAVE_FREQUENCY
            )

            threshold = DELTA_BASE_THRESHOLD - DELTA_WAVE_AMPLITUDE * wave
            threshold = threshold[:, None, None].expand(-1, batch_size_, input_size_)
        else:
            threshold = DELTA_BASE_THRESHOLD

        pos_spike = torch.where(
            tensor > threshold,
            torch.ones_like(tensor),
            torch.zeros_like(tensor)
        )

        neg_spike = torch.where(
            tensor < -threshold,
            -torch.ones_like(tensor),
            torch.zeros_like(tensor)
        )

    return pos_spike + neg_spike
    

sequence_length = 28 * 28
input_size = 1
num_classes = 10
batch_size = 256  # (256 from Yin et al. 2021)

train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
    shuffle=True
)

# test_tensor = next(iter(train_loader))[0][0].unsqueeze(0)  # Get first image from first batch
test_tensor = torch.rand(1, 1, 28, 28)# test_tensor = train_dataset[0][0].unsqueeze(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
fig.suptitle(f"Noise Sample (Base threshold = {DELTA_BASE_THRESHOLD})", fontsize=16)

for ax, oscillate, title in zip(
    axes,
    [False, True],
    ["Flat threshold", f"Oscillating threshold (Amplitude = {DELTA_WAVE_AMPLITUDE}, Frequency = {DELTA_WAVE_FREQUENCY})"]
):
    OSCILLATE_THRESHOLD = oscillate

    y = smnist_transform_input_batch(
        test_tensor,
        sequence_length,
        1,
        1,
        torch.arange(sequence_length)
    ).numpy()
    y[y == 0] = np.nan

    ax.scatter(torch.arange(sequence_length), y.squeeze(), s=8)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Spike value")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, sequence_length)

    if oscillate:
        wave = DELTA_BASE_THRESHOLD + DELTA_WAVE_AMPLITUDE * torch.sin(
            2 * torch.pi * torch.arange(sequence_length) / DELTA_WAVE_FREQUENCY
        )
        ax.plot(wave.numpy())
    else:
        ax.axhline(DELTA_BASE_THRESHOLD, linestyle="--")
plt.tight_layout()
plt.savefig("plot.svg")
plt.show()