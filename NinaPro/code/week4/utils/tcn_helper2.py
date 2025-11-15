import torch
import torch.nn as nn
from typing import Tuple, List
from torch.nn.utils.parametrizations import weight_norm
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikegen as spikegen
from snntorch import surrogate

class Chomp1d(nn.Module):
    """
    Remove extra padding from the right side
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    Temporal Block for TCN implementation
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2, beta: float = 0.5, slope: int = 25):
        super(TemporalBlock, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.spike1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=1.5)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.spike2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, threshold=1.5)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, mem1, mem2):

        x1 = self.conv1(x)
        x1 = self.chomp1(x1)
        mem1, spk1 = self.spike1(x1, mem1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1)
        x2 = self.chomp2(x2)
        mem2, spk2 = self.spike2(x2, mem2)
        out = self.dropout2(x2)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res), mem1, mem2


class TCN(nn.Module):
    """
    Temporal Convolutional Network for NinaPro dataset
    """
    def __init__(self, num_inputs: int, num_channels: List[int], num_classes: int, 
                 kernel_size: int = 2, dropout: float = 0.2, timestamp=2, beta=0.5, slope=25):
        super(TCN, self).__init__()
        self.timesteps = timestamp

        self.blocks = nn.ModuleList()
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout, beta=0.5, slope=slope))
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        mem_states = [(blk.spike1.init_leaky(), blk.spike2.init_leaky()) for blk in self.blocks]
        x = x.unsqueeze(2) 

        out_spikes = 0
        for t in range(self.timesteps):
            cur_input = x  # optional: use rate-coded spikes or input[t]
            for i, block in enumerate(self.blocks):
                mem1, mem2 = mem_states[i]
                cur_input, mem1, mem2 = block(cur_input, mem1, mem2)
                mem_states[i] = (mem1, mem2)

            pooled = torch.mean(cur_input, dim=2)
            out_spikes += self.classifier(pooled)

        return out_spikes / self.timesteps


__all__ = ["Chomp1d", "TemporalBlock", "TCN"]