# 📚 TCN (Temporal Convolutional Network) Study Notes

# 1. Goals 🎯

---

- Understand the basic structure and working principles of TCN
- Practice applying TCN for time series data processing

---

## 2. Summary of Learned Concepts ✍️

- **Topic:** Temporal Convolutional Network (TCN)
- **Key Concepts:**
    - Dilated causal convolutions
    - Residual connections
    - Sequence modeling without recurrence
- **Important Formulas / Code Snippets:**
    
    ```python
    # Simple TCN block example (PyTorch)
    import torch
      import torch.nn as nn
      import torch.nn.functional as F
    
      class TemporalBlock(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
              super(TemporalBlock, self).__init__()
              self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation)
              self.relu1 = nn.ReLU()
              self.dropout1 = nn.Dropout(dropout)
    
              self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation)
              self.relu2 = nn.ReLU()
              self.dropout2 = nn.Dropout(dropout)
    
              self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
              self.relu = nn.ReLU()
    
          def forward(self, x):
              out = self.conv1(x)
              out = self.relu1(out)
              out = self.dropout1(out)
              out = self.conv2(out)
              out = self.relu2(out)
              out = self.dropout2(out)
    
              res = x if self.downsample is None else self.downsample(x)
              return self.relu(out + res)
    
    ```
    

- **Reference Links / Resources**
    - [Original TCN Paper](https://arxiv.org/abs/1803.01271)
    - [Colah's Blog on LSTM and Sequence Models](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    - [PyTorch TCN GitHub Repository](https://github.com/locuslab/TCN)

---

## 3. Detailed Notes 📖

- TCNs replace recurrent connections with 1D dilated causal convolutions to capture temporal dependencies
- Dilation allows exponential growth of receptive field size, which helps in learning long sequences efficiently
- Causal convolutions prevent “future leakage” by ensuring output at time t only depends on inputs from time ≤ t
- Residual connections mitigate the vanishing gradient problem and stabilize training for deeper networks
- In my experiment with NinaPro dataset, TCN effectively classified hand movements with ~85% accuracy
- Important hyperparameters include dilation rate (e.g., 1, 2, 4, 8), kernel size (usually 3 or 5), number of temporal blocks, and dropout rates to reduce overfitting

---

## 4. Questions & Challenges ❓

- What are the practical limits on dilation rate before performance degrades due to sparse receptive fields?
- How does TCN’s training time and resource usage compare with LSTM and Transformer models on large datasets?
- Best strategies for tuning dropout and regularization specifically for TCN architectures?

---

## 5. Next Steps 🔜

- Experiment with different dilation schedules and kernel sizes to optimize model accuracy
- Compare TCN’s performance on NinaPro with Transformer-based sequence models
- Explore multi-channel input handling in TCN for multimodal sensor data
- Implement visualization of learned filters and activations to better understand feature extraction

---

## 6. References & Resources 📚

- Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” arXiv preprint arXiv:1803.01271 (2018).
- [TCN PyTorch Implementation](https://github.com/locuslab/TCN)
- [Deep Learning Book - Chapter on CNNs](https://www.deeplearningbook.org/)

---

## 7. Personal Insights & Thoughts 💡

- TCN’s ability to process entire sequences in parallel drastically reduces training time compared to RNNs
- Residual connections seem critical for avoiding degradation in very deep TCNs
- Choosing dilation and kernel size carefully is key — too large dilation can skip important temporal features
- Overall, TCN is a strong alternative to RNNs for many sequence modeling tasks, especially when long-range dependencies matter

```

필요하면 실험 코드, 그래프, 모델 결과 등 더 덧붙여서 완성도 높여보자!

```
