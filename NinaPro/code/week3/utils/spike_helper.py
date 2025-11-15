import torch
import snntorch as snn
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt

# 1. Poisson Spike Encoding
def poisson_encoder(inputs, time_steps):
    """
    Encodes input into Poisson spike trains.

    Args:
        inputs (torch.Tensor): Input tensor (batch, ...)
        time_steps (int): Number of time steps

    Returns:
        torch.Tensor: Poisson spike trains (time_steps, batch, ...)
    """
    return torch.rand((time_steps,) + inputs.shape) < inputs

# 2. Run SNN step by step
def run_snn_over_time(model, inputs, time_steps):
    """
    Feeds input through an SNN over multiple time steps.

    Args:
        model (torch.nn.Module): SNN model
        inputs (torch.Tensor): Input (batch, channels, height, width)
        time_steps (int): Number of time steps

    Returns:
        list: List of spike outputs over time
    """
    mem = None
    spk_rec = []

    for t in range(time_steps):
        spk_out, mem = model(inputs, mem)
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

# 3. Spike train visualization
def plot_spike_train(spike_record, example_idx=0, neuron_idx=0):
    """
    Plots spike train of a single neuron over time.

    Args:
        spike_record (torch.Tensor): (time_steps, batch, neurons)
        example_idx (int): Which sample in batch to visualize
        neuron_idx (int): Which neuron to visualize
    """
    spk = spike_record[:, example_idx, neuron_idx]
    splt.raster(spk)
    plt.show()
