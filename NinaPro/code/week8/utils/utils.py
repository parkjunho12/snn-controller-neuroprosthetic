import torch
import torch.nn as nn
from typing import Tuple, List
from torch.nn.utils.parametrizations import weight_norm
import snntorch as snn
from snntorch import surrogate
from snntorch import spikeplot as splt
import snntorch.functional as SF

class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(MLPClassifier, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last time step
        return self.fc(out)


class TemporalBlock(nn.Module):
    """
    Temporal Block for TCN implementation
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove extra padding from the right side
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCN_Only(nn.Module):
    """
    Temporal Convolutional Network for NinaPro dataset
    """
    def __init__(self, num_inputs: int, num_channels: List[int], num_classes: int, 
                 kernel_size: int = 2, dropout: float = 0.2):
        super(TCN_Only, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # TCN expects: (batch_size, num_features, seq_len)
        x = x.transpose(1, 2)
        # Pass through TCN
        y = self.network(x)
        
        # Global average pooling
        y = torch.mean(y, dim=2)  # (batch_size, num_channels[-1])
        
        # Classification
        return self.classifier(y)

class TCN_RMS_Only(nn.Module):
    """
    Temporal Convolutional Network for NinaPro dataset
    """
    def __init__(self, num_inputs: int, num_channels: List[int], num_classes: int, 
                 kernel_size: int = 2, dropout: float = 0.2):
        super(TCN_RMS_Only, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # TCN expects: (batch_size, num_features, seq_len)
        x = x.unsqueeze(2) 
        # x = x.permute(0, 2, 1)
        
        # Pass through TCN
        y = self.network(x)
        
        # Global average pooling
        y = torch.mean(y, dim=2)  # (batch_size, num_channels[-1])
        
        # Classification
        return self.classifier(y)

class SNN_Only(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(SNN_Only, self).__init__()

        self.beta = 0.5
        self.spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

    def forward(self, x, num_steps=20):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)

        spk_out_rec = torch.stack(spk_out_rec, dim=0)
        out = spk_out_rec.sum(dim=0)

        return out

class SNN_beta_Only(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, beta: float):
        super(SNN_beta_Only, self).__init__()

        self.beta = beta
        self.spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

    def forward(self, x, num_steps=20):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)

        spk_out_rec = torch.stack(spk_out_rec, dim=0)
        out = spk_out_rec.sum(dim=0)

        return out
        

class SNN_beta_encode(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, beta: float):
        super(SNN_beta_encode, self).__init__()

        self.beta = beta
        self.spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

    def forward(self, x, num_steps=20):
        x_seq = rate_encode(x, num_steps)  # (num_steps, batch, input_size)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        for step in range(num_steps):
            cur1 = self.fc1(x_seq[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

        return mem_out  # final potential for classification

def rate_encode(x, num_steps):
    # x shape: (batch, features)
    x = x.unsqueeze(0).repeat(num_steps, 1, 1)  # replicate over time
    rand_threshold = torch.rand_like(x)
    return (x > rand_threshold).float()


__all__ = ["MLPClassifier", "LSTMClassifier", "TemporalBlock", "TCN_Only", "TCN_RMS_Only", "SNN_Only", "SNN_beta_Only", "SNN_beta_encode"]