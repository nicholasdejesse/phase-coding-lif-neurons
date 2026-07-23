import torch.nn
import torchvision
import tools
from datetime import datetime
import math
import os
import sys
sys.path.append("../..")
import snn
import argparse

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(device)

################################################################
# Argparse
################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, help="Path to the model to load")
parser.add_argument("--delta-base-threshold", type=float, default=0.2)
parser.add_argument("--delta-wave-amplitude", type=float, default=0.15)
parser.add_argument("--delta-wave-frequency", type=int, default=28 * 2) # In terms of time steps (i.e. one full oscillation completed at this timestep)
parser.add_argument("--oscillate-threshold", action="store_true", help="Whether to oscillate the threshold or not.")
parser.add_argument("--negative-at-trough", action="store_true")
args = parser.parse_args()

################################################################
# Data loading and preparation, logging
################################################################

# if True, S-MNIST, if False, PS-MNIST
PERMUTED = False
DELTA_BASE_THRESHOLD = args.delta_base_threshold
DELTA_WAVE_AMPLITUDE = args.delta_wave_amplitude
DELTA_WAVE_FREQUENCY = args.delta_wave_frequency
OSCILLATE_THRESHOLD = args.oscillate_threshold
NEGATIVE_AT_TROUGH = args.negative_at_trough

# Change neuron type manually for different models
# VRF: vanilla RF neuron set to: no reset mechanisms.
# Manually change rf_update at snn/modules/rf for other reset types
neuron = "alif"  # "vrf", "brf" or "alif"

sequence_length = 28 * 28
input_size = 1
hidden_size = 256
num_classes = 10
test_batch_size = 10000

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_dataset_size = len(test_dataset)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False
)


# def smnist_transform_input_batch(
#         tensor: torch.Tensor,
#         sequence_length_: int,
#         batch_size_: int,
#         input_size_: int,
#         permuted_idx_: torch.Tensor
# ):
#     tensor = tensor.to(device=device).view(batch_size_, sequence_length_, input_size_)
#     tensor = tensor.permute(1, 0, 2)
#     tensor = tensor[permuted_idx_, :, :]
#     return tensor

def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor
):
    tensor = tensor.view(batch_size_, sequence_length_, input_size_)  # BxTxC
    tensor = tensor.permute(1, 0, 2)  # TxBxC
    tensor = tensor[permuted_idx_.to(device), :, :]

    # Delta between time steps
    tensor = tensor - tensor.roll(1, 0)
    tensor[0] = 0

    if NEGATIVE_AT_TROUGH:
        if OSCILLATE_THRESHOLD:
            wave = torch.sin(
                2 * torch.pi *
                torch.arange(
                    sequence_length_,
                    device=device,
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
                    device=device,
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


################################################################
# Model setup
################################################################

if "vrf" in neuron:

    if PERMUTED:
        # vrf psmnist: 9.8 % nan, due to divergence
        label_last = False
        comment = "Adam(0.1),NLL,LinearLR,LabelLast(False),PERMUTED(True),RFSNN(1,256,10,bs=256,ep=300)," \
                  "RF(abs(omega_uni(15.0,85.0)),nothing,abs(b(uni(0.1,1.0)),linearMask(0.0))LI(norm_20.0,1.0)"
    else:
        # vrf smnist: test acc. of saved model 98.14 %
        label_last = True
        comment = "Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300)," \
                  "RF(abs(omega_uni(15.0,50.0)),no_sust_osc,-abs(b(uni(0.1,1.0)),linearMask(0.0))LI(norm_20.0,5.0)"

    model = snn.models.SimpleVanillaRFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        pruning=True,  # setting of the model, whether pruning tensor saved within the model (even if no pruning)
    ).to(device)

elif "brf" in neuron:

    if PERMUTED:
        label_last = False
        # brf psmnist: test acc. of saved model 95.22 %
        comment = "Adam(0.1),NLL,LinearLR,LabelLast(False),PERMUTED(True),RFSNN(1,256,10,bs=256,ep=300)," \
                  "RF(abs(omega_uni(15.0,85.0)),sust_osc,abs(b_offset(uni(0.1,1.0))-q,theta(1),linearMask(0.0))" \
                  "LI(norm_20.0,1.0)"
    else:
        label_last = True
        # brf smnist: test acc. of saved model 99.14 %
        comment = 'Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300),' \
                  'RF(abs(omega_uni(15.0,50.0)),sust_osc,abs(b_offset(uni(0.1,1.0))-q,theta(1),linearMask(0.0))' \
                  'LI(norm_20.0,5.0)'

    model = snn.models.SimpleResRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        pruning=True
    ).to(device)

else:

    if PERMUTED:
        label_last = False
        # alif psmnist: test acc. of saved model 83.8 %
        comment = 'Adam(0.001),PERMUTED(True),LinearLR,NLL,LabelLast(False),TBPTT(50),RSNN(1,256,10,bs_256,ep_300,' \
                  'no_bias),ALIF(tau_m(20.0,5.0),tau_a(200.0,50.0),linearMask(0.0))LI(tau_m(20.0,5.0))'
    else:
        label_last = True
        # alif smnist: test acc. of saved model 92.44 %
        comment = 'Adam(0.001),PERMUTED(False),LinearLR,NLL,LabelLast(True),TBPTT(50),RSNN(1,256,10,bs_256,ep_300,' \
                  'no_bias),ALIF(tau_m(20.0,5.0),tau_a(200.0,50.0),linearMask(0.0))LI(tau_m(20.0,5.0))'


    # recorded into comment
    # fraction of the elements in the hidden.linear.weight to be zero
    mask_prob = 0.0

    # ALIF alpha tau_mem init normal dist.
    adaptive_tau_mem_mean = 20.
    adaptive_tau_mem_std = 5.

    # ALIF rho tau_adp init normal dist.
    adaptive_tau_adp_mean = 200.
    adaptive_tau_adp_std = 50.

    # LI alpha tau_mem init normal distribution
    out_adaptive_tau_mem_mean = 20.
    out_adaptive_tau_mem_std = 5.

    hidden_bias = True
    output_bias = True

    tbptt_steps = 50

    criterion = torch.nn.NLLLoss()

    model = snn.models.SimpleALIFRNNTbptt(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        mask_prob=0.0,
        criterion=criterion,
        adaptive_tau_mem_mean=20.,
        adaptive_tau_mem_std=5.,
        adaptive_tau_adp_mean=200.,
        adaptive_tau_adp_std=50.,
        out_adaptive_tau_mem_mean=20.,
        out_adaptive_tau_mem_std=5.,
        label_last=True,
        hidden_bias=True,
        output_bias=True,
        tbptt_steps=50
    ).to(device)

criterion = torch.nn.NLLLoss()

path = './experiments/smnist/models/'

models_str = [f for f in os.listdir(path) if comment in f]

# take out initial model and permuted idx
models_str = [f for f in models_str if '_init_' not in f]
models_str = [f for f in models_str if 'permuted_' not in f]

print(models_str)

# PSMNIST fixed random permutation
if PERMUTED and "vrf" not in neuron:
    permuted_idx = torch.load('./experiments/smnist/models/{}'.format(models_str[0][:14]) + comment + '_permuted_idx.pt')
else:
    permuted_idx = torch.arange(sequence_length)

if args.load:
    PATH = args.load
else:
    PATH = "./experiments/smnist/models/" + models_str[0]

checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Go into eval mode
model.eval()

with torch.no_grad():

    test_loss = 0
    test_correct = 0
    total_spikes = 0

    # Perform Inference
    inputs, targets = next(iter(test_loader))

    # Reshape inputs in [sequence_length, batch_size, data_size].
    current_batch_size = len(inputs)

    inputs = smnist_transform_input_batch(
        tensor=inputs.to(device=device),
        sequence_length_=sequence_length,
        batch_size_=current_batch_size,
        input_size_=input_size,
        permuted_idx_=permuted_idx
    )

    # Reshape targets (for MNIST it's a single pattern).
    target = targets.to(device=device)

    outputs, _, num_spikes = model(inputs, target.repeat((sequence_length, 1)))

total_spikes += num_spikes

# Apply loss sequentially against single pattern.
loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

# for Label Last
if label_last:
    test_loss_value = loss.item()
else:
    test_loss_value = loss.item() / sequence_length

test_loss += test_loss_value

# Calculate batch accuracy
batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
test_correct += batch_correct

test_loss /= len(test_loader)
test_accuracy = (test_correct / test_dataset_size) * 100.0

# total average SOP (spike operations per sample)
SOP = total_spikes / test_dataset_size

# total average SOP per sequence step
SOP_per_step = SOP / sequence_length

# firing rate per neuron
firing_rate = total_spikes / (test_dataset_size * sequence_length * hidden_size)


print(
    'Test loss: {:.6f}, Test acc: {:.4f}, SOP: {:.2f}, SOP per step: {:.2f}, mean firing rate per neuron: {:.2f}'
    .format(test_loss, test_accuracy, SOP, SOP_per_step, firing_rate))


