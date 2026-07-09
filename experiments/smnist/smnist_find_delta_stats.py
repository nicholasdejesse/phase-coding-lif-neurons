import torch.nn
import torchvision
from torch.utils.data import DataLoader, random_split
# import tools
# from datetime import datetime
# import math

import sys
sys.path.append("../..")
# import snn
# import random
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

label_last = True

sequence_length = 28 * 28
encoded_sequence_length = sequence_length
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
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True
)

def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
):
    tensor = tensor.view(-1, sequence_length_, input_size_) # BxTxC
    tensor = tensor.permute(1, 0, 2) # TxBxC
    # Get delta between time steps
    tensor = tensor - tensor.roll(1, 0)
    tensor[0, :, :] = 0
    return tensor

tens = torch.empty((0))
avg = 0
max_val = -9999
for input, target in train_loader:
    tens = torch.cat((tens, smnist_transform_input_batch(input, sequence_length, batch_size, input_size).flatten()))

vals = tens[tens > 0]
print(f"Mean (of nonzero items): {vals.mean()}")
print(f"Max: {tens.max()}")
print(torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75])))