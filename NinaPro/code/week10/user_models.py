import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import butter, filtfilt

# snnTorch imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import snntorch.spikegen as spikegen
import os
from torch.nn.utils.parametrizations import weight_norm
from sklearn.metrics import f1_score


class EMGDataset(Dataset):
    """EMG 데이터셋 클래스"""
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# ======================
# TCN Components
# ======================

class TemporalBlock(nn.Module):
    """TCN의 기본 빌딩 블록"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        if out.shape != residual.shape:
            min_len = min(out.shape[2], residual.shape[2])
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len] 
            
        out += residual
        return self.relu(out)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal padding
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Back to (batch_size, seq_len, features)
        return out.transpose(1, 2)

# ======================
# SNN Components using snnTorch
# ======================

class SpikeEncoder(nn.Module):
    """연속 신호를 스파이크로 변환하는 인코더"""
    def __init__(self, encoding_type='rate', num_steps=10):
        super(SpikeEncoder, self).__init__()
        self.encoding_type = encoding_type
        self.num_steps = num_steps
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        if self.encoding_type == 'delta':
            # 1) (B,T,C) 그대로 delta에 투입
            #    snntorch.spikegen.delta는 (B,T,C) -> (B,T,C) 형태 반환(버전에 따라 다를 수 있으니 아래 체크 권장)
            spikes = spikegen.delta(x, threshold=0.1)   # (B,T,C)

            # 2) 시간 차원 첫 번째로: (T,B,C)
            spikes = spikes.transpose(0, 1).contiguous()

            return spikes  # (T, B, C)
        if self.encoding_type == 'rate':
            # Rate encoding: 입력 크기에 비례하는 스파이크 확률
            # 입력을 0-1 범위로 정규화
            p = torch.sigmoid(x)  # (B,T,C)
                                            
            spikes = spikegen.rate(p, num_steps=self.num_steps)   # (num_steps, B, T, C)
            return spikes
        
        elif self.encoding_type == 'latency':
            # Latency encoding
            spikes = spikegen.latency(x, num_steps=seq_len, normalize=True, linear=True)
            return spikes
        
        else:  # 'delta'
            # Delta modulation
            spikes = spikegen.delta(x, threshold=0.1)   # (B,T,C)

            # 2) 시간 차원 첫 번째로: (T,B,C)
            spikes = spikes.transpose(0, 1).contiguous()

            return spikes  # (T, B, C)

class SNNBlock(nn.Module):
    """SNN 블록 (LIF 뉴런 사용)"""
    def __init__(self, input_size, hidden_size, num_steps=10, beta=0.9, threshold=1.0):
        super(SNNBlock, self).__init__()
        
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        
        # Linear layer
        self.fc = nn.Linear(input_size, hidden_size)
        
        # LIF neuron
        self.lif = snn.Leaky(beta=beta, threshold=threshold, 
                            spike_grad=surrogate.fast_sigmoid())
        
    def forward(self, x):                    # x: (T,B,input_size)
        T, B, _ = x.shape                    # ❗ self.num_steps 대신 T 사용
        mem = self.lif.init_leaky()
        spk_rec, mem_rec = [], []
        for t in range(T):
            cur = self.fc(x[t])              # (B,H)
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk); mem_rec.append(mem)
        return torch.stack(spk_rec, 0), torch.stack(mem_rec, 0)

class SpikingNeuralNetwork(nn.Module):
    """Multi-layer SNN"""
    def __init__(self, input_size, hidden_sizes, num_steps=10, beta=0.9, threshold=1.0, encoding_type='rate'):
        super(SpikingNeuralNetwork, self).__init__()
        
        self.num_steps = num_steps
        self.layers = nn.ModuleList()
        
        # Input encoding
        self.encoder = SpikeEncoder(encoding_type=encoding_type, num_steps=num_steps)
        
        # SNN layers
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.layers.append(
                SNNBlock(layer_sizes[i], layer_sizes[i+1], 
                        num_steps=num_steps, beta=beta, threshold=threshold)
            )
    
    def forward(self, x):
        B, T, F = x.shape
        # num_steps 고정 사용
        num_steps = self.num_steps

        spikes = self.encoder(x)  # (num_steps, B, T_seq, C)  # latency 기준
        
        spikes = spikes.sum(dim=2) * 2

        for layer in self.layers:
            spikes, _ = layer(spikes)  # (num_steps, B, H)
        return spikes



# ======================
# Model Architectures
# ======================
class Chomp1d(nn.Module):
    """
    Remove extra padding from the right side
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

spike_grad = surrogate.fast_sigmoid(slope=25)

class SpikingTemporalBlock(nn.Module):
    """Spiking 버전 Temporal Block (막전위 유지)"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 beta=0.9, v_th=1.0):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.lif1  = snn.Leaky(beta=beta, threshold=v_th, spike_grad=spike_grad, init_hidden=False)
        self.do1   = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.lif2  = snn.Leaky(beta=beta, threshold=v_th, spike_grad=spike_grad, init_hidden=False)
        self.do2   = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x, mem1, mem2, return_spk=False):
        x1 = self.chomp1(self.conv1(x))
        spk1, mem1 = self.lif1(x1, mem1)
        x1 = self.do1(spk1)

        x2 = self.chomp2(self.conv2(x1))
        spk2, mem2 = self.lif2(x2, mem2)
        out = self.do2(spk2)

        res = x if self.downsample is None else self.downsample(x)
        y = out + res

        if return_spk:
            # spk2: (B, C, T_seq) — 마지막 LIF의 바이너리 스파이크
            return y, mem1, mem2, spk2
        else:
            return y, mem1, mem2



# ======================
# Spiking Temporal ConvNet
# ======================
class SpikingTCN(nn.Module):
    """에너지 효율적인 Spiking-TCN (막전위 유지)"""
    def __init__(self, num_inputs, num_channels, num_classes,
                 kernel_size=2, dropout=0.2, timesteps=10, beta=0.9, v_th=1.0):
        super().__init__()
        self.timesteps = timesteps
        self.encoder = SpikeEncoder(encoding_type='rate', num_steps=timesteps)

        self.blocks = nn.ModuleList()
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_c  = num_inputs if i == 0 else num_channels[i-1]
            out_c = num_channels[i]
            pad   = (kernel_size - 1) * dilation
            self.blocks.append(
                SpikingTemporalBlock(in_c, out_c, kernel_size, 1, dilation, pad,
                              dropout=dropout, beta=beta, v_th=v_th)
            )

        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )

    @torch.no_grad()
    def collect_spike_stats(self, spike_buffers, T_total):
        """타임스텝 동안 누적된 스파이크로 간단 통계 산출"""
        stats = {}
        for bidx, buf in spike_buffers.items():
            stats[f'block{bidx}_lif1_rate_per_ch'] = (buf['lif1_sum'] / T_total)  # (C,)
            stats[f'block{bidx}_lif2_rate_per_ch'] = (buf['lif2_sum'] / T_total)  # (C,)
            stats[f'block{bidx}_lif1_total'] = buf['lif1_sum'].sum().item()
            stats[f'block{bidx}_lif2_total'] = buf['lif2_sum'].sum().item()
        return stats

    def forward(self, x, return_spikes=False):
        spikes = self.encoder(x)  # (T, B, F_in)
        mem_states = [(blk.lif1.init_leaky(), blk.lif2.init_leaky()) for blk in self.blocks]

        logits_sum = 0.0
        spk_tbC_list = []  # 여기 모아 T×B×C_last 로 반환

        for t in range(self.timesteps):
            cur = spikes[t]                  # (B, T_seq, C_in)
            cur = cur.transpose(1, 2).contiguous()  # (B, C_in, T_seq)

            for i, blk in enumerate(self.blocks):
                last_block = (i == len(self.blocks) - 1) and return_spikes
                if last_block:
                    cur, m1, m2, spk2 = blk(cur, *mem_states[i], return_spk=True)  # spk2: (B,C_last,T_seq)
                else:
                    cur, m1, m2 = blk(cur, *mem_states[i])
                mem_states[i] = (m1, m2)

            # 분류용 풀링
            pooled = cur.mean(dim=2)
            logits_sum += self.classifier(pooled)

            # 비교용 스파이크 시퀀스: (T,B,C_last)
            if return_spikes:
                # 래스터/히스토그램 비교를 위해 conv 시간축을 이진 합성:
                # conv 시간축 중 "한 번이라도 쏘면 1" 로 축약  => (B, C_last)
                spk_frame = (spk2 > 0).float().mean(dim=2) 
                spk_tbC_list.append(spk_frame)

        logits = logits_sum / self.timesteps

        if return_spikes:
            spk_tbC = torch.stack(spk_tbC_list, dim=0)  # (T, B, C_last)
            # 원래 통계들에 같이 넣어서 보낼 수도 있고,
            # 최소한 spk_tbC만 넘겨도 시각화는 동일 루틴 재사용 가능
            stats = {"spk_tbC": spk_tbC}
            return logits, stats

        return logits

class TCNClassifier(nn.Module):
    """TCN 기반 EMG 분류기"""
    def __init__(self, input_size, num_classes, tcn_channels=[64, 128, 256], 
                 kernel_size=3, dropout=0.2):
        super(TCNClassifier, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size, dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(tcn_channels[-1], num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # TCN feature extraction
        tcn_out = self.tcn(x)  # (batch_size, seq_len, features)
        
        # Self-attention
        attn_out, _ = self.attention(tcn_out, tcn_out, tcn_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, features)
        
        # Classification
        output = self.classifier(pooled)
        return output

class SNNClassifier(nn.Module):
    """SNN 기반 EMG 분류기"""
    def __init__(self, input_size, num_classes, hidden_sizes=[64, 128, 256], 
                 num_steps=10, beta=0.9, threshold=1.0, encoding_type='rate'):
        super(SNNClassifier, self).__init__()
        
        self.num_steps = num_steps
        self.snn = SpikingNeuralNetwork(input_size, hidden_sizes, 
                                       num_steps, beta, threshold, encoding_type)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Get spike outputs from SNN
        spikes = self.snn(x)  # (num_steps, batch_size, hidden_size)

        spike_rates = spikes.mean(dim=0) # (B, H)  <-- 시간축 평균

        output = self.output_layer(spike_rates)
        
        return output

class HybridTCNSNN(nn.Module):
    """TCN과 SNN을 결합한 하이브리드 모델"""
    def __init__(self, input_size, num_classes, tcn_channels=[64, 128, 256], 
                 snn_hidden_sizes=[64, 128, 256], num_steps=10, kernel_size=3, dropout=0.2, encoding_type='rate', beta=0.9, threshold=1.0):
        super(HybridTCNSNN, self).__init__()
        
        self.num_steps = num_steps
        
        # TCN branch
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size, dropout)
        
        # SNN branch
        self.snn = SpikingNeuralNetwork(input_size, snn_hidden_sizes, num_steps=num_steps, beta=beta, encoding_type=encoding_type, threshold=threshold)
        
        # Feature fusion
        combined_size = tcn_channels[-1] + snn_hidden_sizes[-1]
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # TCN branch
        tcn_out = self.tcn(x)
        # print(tcn_out.shape)
        tcn_pooled = tcn_out.mean(dim=1)  # Global average pooling
       
        # SNN branch
        snn_spikes = self.snn(x)
        # print(snn_spikes.shape)
        snn_rates = snn_spikes.mean(dim=0)   
        
        # Combine features
        combined = torch.cat([tcn_pooled, snn_rates], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        return output
class LSTMClassifier(nn.Module):
    """LSTM 기반 EMG 분류기 (batch_first)"""
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C)
        out, (h_n, c_n) = self.lstm(x)     # h_n: (num_layers*dir, B, H)
        # 최종 층의 hidden만 취함
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1]                               # (B, H)
        logits = self.classifier(h_last)
        return logits


def build_model(model_name: str) -> nn.Module:
    encoding_type = 'rate'
    num_steps = 20
    if model_name == "TCN":
        return TCNClassifier(input_size=16, num_classes=7)   # B,T,C 또는 B,C,T 입력 지원
    elif model_name == "Hybrid":
        return HybridTCNSNN(input_size=16, num_classes=7, encoding_type=encoding_type, num_steps=num_steps, beta=0.95, threshold=0.6)
    elif model_name == "SpikingTCN":
        return SpikingTCN(
        num_inputs=16,   # 채널 수(=특징 수)
        num_channels=[64, 128, 256],    # 기존 TCN 채널 구성 그대로
        num_classes=7,
        kernel_size=3,
        dropout=0.2,
        timesteps=num_steps,   # 6~16 권장 (낮출수록 지연/연산↓)
        beta=0.9,      # EMG는 0.9~0.99가 전이 유지에 유리
        v_th=1.25
    )
    elif model_name == "SNN":
        return SNNClassifier(input_size=16, num_classes=7, encoding_type=encoding_type, num_steps=num_steps, beta=0.95, threshold=0.45)
    elif model_name == "LSTM":
        return LSTMClassifier(
        input_size=16,
        num_classes=7,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
    )
    else:
        raise ValueError(f"Unknown model: {model_name}")
