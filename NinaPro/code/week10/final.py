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

# snnTorch imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import snntorch.spikegen as spikegen
import os
from torch.nn.utils.parametrizations import weight_norm
# ======================
# Dataset Class
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    def __init__(self, encoding_type='delta', num_steps=10):
        super(SpikeEncoder, self).__init__()
        self.encoding_type = encoding_type
        self.num_steps = num_steps
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        if self.encoding_type == 'delta':
            spikes = spikegen.delta(x, threshold=0.2)   # (B, T_seq, C)

            # (B, T_seq, C) -> (1, B, T_seq, C)
            spikes = spikes.unsqueeze(0)  

            # T_sim만큼 복제: (T_sim, B, T_seq, C)
            spikes = spikes.repeat(self.num_steps, 1, 1, 1)

            return spikes
        if self.encoding_type == 'rate':
            # Rate encoding: 입력 크기에 비례하는 스파이크 확률
            # 입력을 0-1 범위로 정규화
            x_norm = torch.sigmoid(x)
            # Poisson spike generation
            spikes = spikegen.rate(x_norm, num_steps=seq_len)
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
    def __init__(self, input_size, hidden_sizes, num_steps=10, beta=0.9, threshold=1.0, encoding_type='delta'):
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
        spikes = spikes.mean(dim=2).contiguous()  # (num_steps, B, C)

        for layer in self.layers:
            spikes, _ = layer(spikes)  # (num_steps, B, H)
        return spikes

class SNNCore(nn.Module):
    """Encoder 없이 스파이크 시퀀스를 입력으로 받는 SNN (입력: (T,B,F))"""
    def __init__(self, input_size, hidden_sizes, num_steps=10, beta=0.9, threshold=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.layers.append(
                SNNBlock(layer_sizes[i], layer_sizes[i+1],
                         num_steps=num_steps, beta=beta, threshold=threshold)
            )

    def forward(self, spikes):               # spikes: (T, B, F)
        x = spikes
        for layer in self.layers:
            x, _ = layer(x)                   # (T, B, H)
        return x                              # (T, B, H_last)


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
        self.encoder = SpikeEncoder(encoding_type='delta', num_steps=timesteps)

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

        self.classifier = nn.Linear(num_channels[-1], num_classes)

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
                spk_frame = (spk2 > 0).amax(dim=2).float()
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
    def __init__(self, input_size, num_classes, hidden_sizes=[256], 
                 num_steps=10, beta=0.9, threshold=1.0, encoding_type='delta'):
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
    def __init__(self, input_size, num_classes, tcn_channels=[32, 64], 
                 snn_hidden_sizes=[64, 32], num_steps=10, kernel_size=3, dropout=0.2, encoding_type='delta'):
        super(HybridTCNSNN, self).__init__()
        
        self.num_steps = num_steps

        # TCN branch
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.encoder = SpikeEncoder(encoding_type=encoding_type, num_steps=num_steps)

        self.snn = SNNCore(
            input_size=tcn_channels[-1],
            hidden_sizes=snn_hidden_sizes,
            num_steps=num_steps
        )

        # 최종 분류 헤드
        self.head = nn.Sequential(
            nn.Linear(snn_hidden_sizes[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):                    # x: (B,T,F_in)
        tcn_out = self.tcn(x)                # (B,T,F_tcn)
        # num_steps 동기화(선택): self.encoder.num_steps = tcn_out.shape[1]
        spikes  = self.encoder(tcn_out)      # (T,B,F_tcn)
        spk     = self.snn(spikes)           # (T,B,H_last)
        h       = spk.mean(dim=0).float()    # (B,H_last)
        return self.head(h)

# ======================
# Data Loading Functions
# ======================

def load_ninapro_data(file_path):
    """NinaPro 데이터를 로드하고 전처리하는 함수"""
    try:
        # .mat 파일 로드
        data = sio.loadmat(file_path)
        
        print("Available keys in the data:", list(data.keys()))
        
        # EMG 데이터 추출
        if 'emg' in data:
            emg_data = data['emg']
        elif 'data' in data:
            emg_data = data['data']
        else:
            data_keys = [k for k in data.keys() if not k.startswith('__')]
            emg_data = data[data_keys[0]]
        
        # 라벨 데이터 추출
        if 'stimulus' in data:
            labels = data['stimulus'].flatten()
        elif 'restimulus' in data:
            labels = data['restimulus'].flatten()
        elif 'glove' in data:
            labels = data['glove']
            if labels.ndim > 1:
                labels = labels[:, 0]
        else:
            label_keys = [k for k in data.keys() if 'stimulus' in k.lower() or 'label' in k.lower()]
            if label_keys:
                labels = data[label_keys[0]].flatten()
            else:
                data_keys = [k for k in data.keys() if not k.startswith('__')]
                labels = data[data_keys[1]].flatten() if len(data_keys) > 1 else np.zeros(emg_data.shape[0])
        
        print(f"EMG data shape: {emg_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
        return emg_data, labels
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating sample data for demonstration...")
        
        # 샘플 데이터 생성
        n_samples = 10000
        n_channels = 12
        emg_data = np.random.randn(n_samples, n_channels) * 0.1
        
        # EMG 신호처럼 보이도록 시간적 상관관계 추가
        for i in range(n_channels):
            emg_data[:, i] += np.sin(np.linspace(0, 100*np.pi, n_samples) + i) * 0.05
            emg_data[:, i] += np.convolve(np.random.randn(n_samples), 
                                        np.ones(5)/5, mode='same') * 0.02
        
        labels = np.random.randint(0, 7, n_samples)  # 0-6 클래스
        return emg_data, labels

def preprocess_data_for_networks(emg_data, labels, window_size=200, overlap=100):
    """네트워크를 위한 EMG 데이터 전처리"""
    # 레이블이 0인 rest 구간 제거 (선택사항)
    non_zero_mask = labels != 0
    emg_data = emg_data[non_zero_mask]
    labels = labels[non_zero_mask]
    
    # 윈도우 기반 시퀀스 생성
    windowed_sequences = []
    windowed_labels = []
    
    step_size = window_size - overlap
    
    for i in range(0, len(emg_data) - window_size + 1, step_size):
        window = emg_data[i:i+window_size]
        window_label = labels[i:i+window_size]
        
        # 윈도우 내에서 가장 빈번한 라벨 사용
        unique_labels, counts = np.unique(window_label, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        
        windowed_sequences.append(window)
        windowed_labels.append(dominant_label)
    
    return np.array(windowed_sequences), np.array(windowed_labels)

# ======================
# Training Functions
# ======================
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def train_model_warmup(
    model, train_loader, val_loader,
    num_epochs=1, lr=1e-3, warmup_epochs=8, snn_lr_scale=0.3, patience=20
):
    """하이브리드 Warmup(동결)→Finetune(합동) 학습"""
    global device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # -------- Warmup: SNN 동결, TCN+Head만 학습 --------
    if hasattr(model, 'snn'):   # 하이브리드 전용
        set_requires_grad(model.snn, False)
    # encoder는 파라미터 없음

    optim_warm = optim.Adam(
        list(model.tcn.parameters()) + list(model.head.parameters()), lr=lr
    )
    sched_warm = ReduceLROnPlateau(optim_warm, mode='max', factor=0.5, patience=5, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, patience_counter = 0.0, 0

    def run_epoch(phase, optimizer=None):
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss, correct, total = 0.0, 0, 0
        loader = train_loader if phase == 'train' else val_loader
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if phase == 'train':
                optimizer.zero_grad()
            output = model(data)               # (B, C)
            loss = criterion(output, target)
            if phase == 'train':
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        avg_loss = running_loss / len(loader)
        acc = 100.0 * correct / total
        return avg_loss, acc

    # --- Warmup loop ---
    for epoch in range(warmup_epochs):
        tr_loss, tr_acc = run_epoch('train', optim_warm)
        val_loss, val_acc = run_epoch('val')

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss);   history['val_acc'].append(val_acc)

        sched_warm.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            os.makedirs("./output/delta", exist_ok=True)
            torch.save(model.state_dict(), "./output/delta/best_model_delta.pth")
        else:
            patience_counter += 1
        print(f"[Warmup {epoch+1}/{warmup_epochs}] "
              f"Train {tr_acc:.2f}% / Val {val_acc:.2f}% (loss {val_loss:.4f})")

    # -------- Finetune: SNN 해제, 전부 함께 학습 --------
    if hasattr(model, 'snn'):
        set_requires_grad(model.snn, True)

    # 파라미터 그룹: SNN은 더 작은 lr로 미세조정(원하면 1.0으로 동일하게)
    params = [
        {'params': model.tcn.parameters(), 'lr': lr},
        {'params': model.head.parameters(), 'lr': lr},
    ]
    if hasattr(model, 'snn'):
        params.append({'params': model.snn.parameters(), 'lr': lr * snn_lr_scale})

    optim_ft = optim.Adam(params)
    sched_ft = ReduceLROnPlateau(optim_ft, mode='max', factor=0.5, patience=5, verbose=True)

    # 남은 epoch 학습
    for epoch in range(num_epochs - warmup_epochs):
        tr_loss, tr_acc = run_epoch('train', optim_ft)
        val_loss, val_acc = run_epoch('val')

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss);   history['val_acc'].append(val_acc)

        sched_ft.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            torch.save(model.state_dict(), "./output/delta/best_model_delta.pth")
        else:
            patience_counter += 1

        print(f"[Finetune {epoch+1}/{num_epochs - warmup_epochs}] "
              f"Train {tr_acc:.2f}% / Val {val_acc:.2f}% (loss {val_loss:.4f})")

        if patience_counter >= patience:
            print(f"Early stopping at finetune epoch {epoch+1}")
            break

    # Best 로드
    model.load_state_dict(torch.load("./output/delta/best_model_delta.pth"))
    return model, history

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """모델 훈련 함수"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), './output/delta/best_model_delta.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Load best model
    model.load_state_dict(torch.load('./output/delta/best_model_delta.pth'))
    
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }
    
    return model, history

def evaluate_model(model, test_loader):
    """모델 평가 함수"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    return test_acc, np.array(all_predictions), np.array(all_targets)

# ======================
# Visualization Functions
# ======================

def plot_training_history(histories, model_names):
    """훈련 과정 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = colors[i % len(colors)]
        
        # Training accuracy
        axes[0, 0].plot(history['train_acc'], label=f'{name}', 
                       color=color, linewidth=2)
        
        # Validation accuracy
        axes[0, 1].plot(history['val_acc'], label=f'{name}', 
                       color=color, linewidth=2)
        
        # Training loss
        axes[1, 0].plot(history['train_loss'], label=f'{name}', 
                       color=color, linewidth=2)
        
        # Validation loss
        axes[1, 1].plot(history['val_loss'], label=f'{name}', 
                       color=color, linewidth=2)
    
    axes[0, 0].set_title('Training Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Training Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    count_path = os.path.join("./output/delta", "model_trainint_delta4.png")
    plt.savefig(count_path)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    count_path = os.path.join("./output/delta", "rasterplot_delta4.png")
    plt.savefig(count_path)
    plt.close(fig)

def visualize_snn_spikes(model, x, title_prefix="Model"):
    model.eval()
    with torch.no_grad():
        # 공통: (T, B, N) 형태의 spike 시퀀스를 확보
        if isinstance(model, SpikingTCN):
            logits, stats = model(x, return_spikes=True)
            spk = stats["spk_tbC"]  # (T, B, C_last)
        elif isinstance(model, SNNClassifier):
            spk = model.snn(x)      # (T, B, H)
        else:
            raise TypeError("SpikingTCN 또는 SNNClassifier만 지원")

        T, B, N = spk.shape
        # 발화율/카운트
        counts_per_neuron = spk.sum(dim=(0,1))        # (N,)
        rate_per_neuron   = counts_per_neuron / T     # 배치 평균된 타임스텝당 발화율
        total_spikes      = counts_per_neuron.sum().item()

        print(f"=== {title_prefix} Spike Stats ===")
        print(f"T={T}, B={B}, N={N}")
        print(f"Total spikes: {total_spikes:.0f}")
        print(f"Mean firing rate per neuron: {rate_per_neuron.mean():.4f}")
        print(f"Max firing rate per neuron:  {rate_per_neuron.max():.4f}")

        os.makedirs("./output/delta", exist_ok=True)

        # 히스토그램 (두 모델 동일)
        fig = plt.figure()
        plt.hist(rate_per_neuron.cpu().numpy(), bins=40)
        plt.title(f"{title_prefix} Neuron Firing Rate Distribution")
        plt.xlabel("Firing rate (spikes / timestep)")
        plt.ylabel("Neuron count")
        plt.savefig(os.path.join("./output/delta", f"{title_prefix}_rate_hist.png"))
        plt.close(fig)

        # 래스터 (두 모델 동일)
        fig = plt.figure(figsize=(10, 5))
        spk_cpu = spk.cpu()
        max_neurons = min(N, 100)
        for h in range(max_neurons):
            # (T, B)에서 nonzero: [t, b]
            nz = (spk_cpu[:, :, h] > 0).nonzero(as_tuple=False)
            if nz.numel() == 0:
                continue
            # 배치 인덱스를 살짝 오프셋으로 그려 겹침 방지
            plt.scatter(nz[:,0], h + nz[:,1]*0.1, 
                marker='|', s=500, alpha=0.9, linewidths=5, color='black')
        plt.title(f"{title_prefix} Raster (first {max_neurons} neurons)")
        plt.xlabel("Time step")
        plt.ylabel("Neuron index")
        plt.savefig(os.path.join("./output/delta", f"{title_prefix}_raster.png"))
        plt.close(fig)

def plot_model_comparison_results(results):
    """모델 성능 비교 결과 시각화"""
    model_names = list(results.keys())
    accuracies = [results[name]['test_acc'] for name in model_names]
    
    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, alpha=0.7, 
                   color=['blue', 'red', 'green', 'orange'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    count_path = os.path.join("./output/delta", "model_performance_com_delta4.png")
    plt.savefig(count_path)
    plt.close(fig)


# ======================
# Main Function
# ======================
def main():
    # Check if CUDA is available
    

    # 데이터 경로 설정
    DATA_PATH = "/users/acp24jhp/com6012/Dissertation/data/S1_D1_T1.mat"
    encoding_type = 'delta'
    num_steps = 10
    print("=== EMG Classification with TCN and SNN using PyTorch ===")
    
    # 1. 데이터 로드
    print("\n1. Loading data...")
    emg_data, labels = load_ninapro_data(DATA_PATH)
    
    # 2. 데이터 전처리
    print("\n2. Preprocessing data...")
    X, y = preprocess_data_for_networks(emg_data, labels, window_size=200, overlap=100)
    
    print(f"Sequence data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # 3. 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    class_names = [f"Gesture {i}" for i in range(num_classes)]
    
    # 4. 데이터 분할
    print("\n3. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. 데이터 정규화
    print("\n4. Normalizing data...")
    scaler = StandardScaler()

    # 2D로 변형 후 스케일링, 다시 3D로 복원
    X_train_res = X_train.reshape(-1, X_train.shape[-1])
    X_val_res = X_val.reshape(-1, X_val.shape[-1])
    X_test_res = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_res).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_res).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_res).reshape(X_test.shape)

    # 6. DataLoader 생성
    print("\n5. Creating DataLoaders...")
    train_dataset = EMGDataset(X_train_scaled, y_train)
    val_dataset = EMGDataset(X_val_scaled, y_val)
    test_dataset = EMGDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 7. 모델 학습 및 평가
    print("\n6. Training Models...")
    results = {}
    histories = []

    # SNN
    print("\n--- Training SNN Model ---")
    snn_model = SNNClassifier(input_size=X_train.shape[-1], num_classes=num_classes, encoding_type=encoding_type, num_steps=num_steps)
    snn_model, snn_history = train_model(snn_model, train_loader, val_loader)
    snn_test_acc, snn_preds, snn_targets = evaluate_model(snn_model, test_loader)
    results['SNN'] = {'test_acc': snn_test_acc, 'preds': snn_preds, 'targets': snn_targets}
    histories.append(snn_history)

    # TCN
    print("\n--- Training TCN Model ---")
    tcn_model = TCNClassifier(input_size=X_train.shape[-1], num_classes=num_classes)
    tcn_model, tcn_history = train_model(tcn_model, train_loader, val_loader)
    tcn_test_acc, tcn_preds, tcn_targets = evaluate_model(tcn_model, test_loader)
    results['TCN'] = {'test_acc': tcn_test_acc, 'preds': tcn_preds, 'targets': tcn_targets}
    histories.append(tcn_history)

    # Hybrid
    print("\n--- Training Hybrid TCN-SNN Model ---")
    hybrid_model = SpikingTCN(
        num_inputs=X_train.shape[-1],   # 채널 수(=특징 수)
        num_channels=[64, 128, 256],    # 기존 TCN 채널 구성 그대로
        num_classes=num_classes,
        kernel_size=3,
        dropout=0.2,
        timesteps=num_steps,   # 6~16 권장 (낮출수록 지연/연산↓)
        beta=0.9,      # EMG는 0.9~0.99가 전이 유지에 유리
        v_th=1.0
    )
    hybrid_model, hybrid_history = train_model(hybrid_model, train_loader, val_loader)
    hybrid_test_acc, hybrid_preds, hybrid_targets = evaluate_model(hybrid_model, test_loader)
    results['Hybrid'] = {'test_acc': hybrid_test_acc, 'preds': hybrid_preds, 'targets': hybrid_targets}
    histories.append(hybrid_history)

    # 8. 학습 결과 시각화
    print("\n7. Plotting Training History...")
    plot_training_history(histories, ['SNN', 'TCN', 'Hybrid'])

    print("\n8. Plotting Test Accuracy Comparison...")
    plot_model_comparison_results(results)

    print("\n9. Confusion Matrices")
    for model_name, result in results.items():
        plot_confusion_matrix(result['targets'], result['preds'], class_names, model_name)

    # 10. SNN Spike 시각화
    print("\n10. Visualizing Spikes (Hybrid Model)...")
    sample_input = torch.FloatTensor(X_test_scaled[:1])
    visualize_snn_spikes(hybrid_model, sample_input, title_prefix="SpikingTCN")

    visualize_snn_spikes(snn_model, sample_input, title_prefix="SNNClassifier")

    print("\n=== All Done ===")
    return hybrid_model, hybrid_history, scaler, label_encoder

if __name__ == "__main__":
    model, history, scaler, label_encoder = main()
