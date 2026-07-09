import torch
import matplotlib.pyplot as plt

OSCILLATE_THRESHOLD = True
DELTA_BASE_THRESHOLD = 0.3
DELTA_WAVE_AMPLITUDE = 0.3
DELTA_WAVE_FREQUENCY = 20 # In terms of time steps (i.e. one full oscillation completed at this timestep)

def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor
):
    tensor = tensor.view(batch_size_, sequence_length_, input_size_) # BxTxC
    tensor = tensor.permute(1, 0, 2) # TxBxC
    tensor = tensor[permuted_idx_, :, :]
    # Get delta between time steps
    tensor = tensor - tensor.roll(1, 0)
    tensor[0, :, :] = 0
    if OSCILLATE_THRESHOLD:
        sin = torch.sin(2 * torch.pi * torch.arange(sequence_length_, dtype=torch.float64) / DELTA_WAVE_FREQUENCY)
        sin = DELTA_WAVE_AMPLITUDE * sin
        sin = sin[:, None, None].expand(-1, batch_size_, input_size_)
        tensor = tensor - sin
    tensor = torch.where(tensor > DELTA_BASE_THRESHOLD, 1, 0)
    return tensor
length = 100
test_tensor = torch.rand(length).unsqueeze(dim=0).unsqueeze(dim=0)
y = smnist_transform_input_batch(test_tensor, length, 1, 1, torch.arange(length))

plt.scatter(torch.arange(length), y.squeeze())
plt.show()