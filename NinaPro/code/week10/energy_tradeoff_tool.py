
#!/usr/bin/env python3
"""
Energy–Accuracy Trade-off Tool for sEMG Models

Usage (CLI examples):
  python energy_tradeoff_tool.py \
      --stats_csv spike_stats.csv \
      --alpha 1.0 --beta 0.01 --gamma 0.001 \
      --input_shape "1,200,8" \
      --model_builder "user_models:build_model" \
      --weights_map "LSTM:checkpoints/lstm.pth,TCN:checkpoints/tcn.pth,Hybrid:checkpoints/hybrid.pth,SpikingTCN:checkpoints/spikingtcn.pth,SNN:checkpoints/snn.pth" \
      --out_csv "tradeoff_results.csv" \
      --out_plot "tradeoff_plot.png"

What you need to provide:
- A CSV (spike_stats.csv) with columns:
    model_name,encoding,Ts,accuracy,macro_f1,total_spikes,mean_firing_rate,fan_in_avg(optional),syn_events(optional)
- A Python module with a function `build_model(model_name: str) -> torch.nn.Module`
  referenced as "module_path:function_name" in --model_builder. Example: "user_models:build_model".
- A map of model_name to .pth paths via --weights_map.

This script will:
- Load models and estimate dense MACs (Conv1d, Linear, LSTM) given an example input shape (B,T,C).
- If syn_events is not provided in CSV, approximate syn_events = total_spikes * fan_in_avg.
- Compute E_hat = alpha*MACs_dense + beta*total_spikes + gamma*syn_events
- Save a scatter plot of Macro-F1 vs E_hat and an output CSV.
"""
from adjustText import adjust_text
import argparse
import importlib
import math
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Lightweight MACs Estimator
# ---------------------------
class MACsCounter:
    def __init__(self):
        self.macs = 0
        self.hooks = []

    def _conv1d_macs(self, module: nn.Conv1d, inp, out):
        # inp: tuple(tensor), out: tensor; shapes (B, Cin, Lin), (B, Cout, Lout)
        x = inp[0]
        B, Cin, Lin = x.shape
        Cout, _, K = module.weight.shape  # (Cout, Cin/groups, K)
        Lout = out.shape[-1]
        groups = module.groups if hasattr(module, "groups") else 1
        macs = B * Cout * Lout * (Cin // groups) * K
        self.macs += macs

    def _linear_macs(self, module: nn.Linear, inp, out):
        x = inp[0]
        # Support (B, *, in_features)
        in_features = module.in_features
        out_features = module.out_features
        # Flatten batch dims except last
        num_elements = x.numel() // in_features
        macs = num_elements * in_features * out_features
        self.macs += macs

    def _lstm_macs(self, module: nn.LSTM, inp, out):
        # Rough estimate for LSTM: per time step 4*(in*hid + hid*hid) per layer, times T and B
        # inp[0]: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
        x = inp[0]
        batch_first = module.batch_first if hasattr(module, "batch_first") else False
        if batch_first:
            B, T, I = x.shape
        else:
            T, B, I = x.shape

        total = 0
        num_layers = module.num_layers
        H = module.hidden_size
        D = 2 if module.bidirectional else 1

        # For simplicity, approximate each layer
        in_size = I
        for layer in range(num_layers):
            # gates = 4*(in*H + H*H) per direction
            per_dir = 4 * (in_size * H + H * H)
            total_layer = D * per_dir * B * T
            total += total_layer
            in_size = H * D  # next layer input

        self.macs += total

    def add_hooks(self, model: nn.Module):
        def register(module):
            if isinstance(module, nn.Conv1d):
                h = module.register_forward_hook(self._conv1d_macs)
                self.hooks.append(h)
            elif isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._linear_macs)
                self.hooks.append(h)
            elif isinstance(module, nn.LSTM):
                h = module.register_forward_hook(self._lstm_macs)
                self.hooks.append(h)

        model.apply(register)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def count(self, model: nn.Module, example_input: torch.Tensor) -> int:
        self.macs = 0
        self.add_hooks(model)
        with torch.no_grad():
            _ = model(example_input)
        self.remove_hooks()
        return int(self.macs)


def parse_shape(shape_str: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in shape_str.split(",")]
    if len(parts) != 3:
        raise ValueError("input_shape must be 'B,T,C'")
    return parts[0], parts[1], parts[2]


# 1) 함수 시그니처 변경
def load_models_and_count_macs(builder_ref: str,
                               model_names: list,
                               weights_map: Dict[str, str],
                               input_shape: Tuple[int, int, int]) -> Dict[str, int]:
    module_path, func_name = builder_ref.split(":")
    builder_module = importlib.import_module(module_path)
    builder_fn = getattr(builder_module, func_name)

    macs_counter = MACsCounter()
    macs_dict = {}
    B, T, C = input_shape

    for model_name in model_names:
        model = builder_fn(model_name)

        # (옵션) 가중치 로드 — 없거나 실패해도 계속
        wpath = weights_map.get(model_name) if weights_map else None
        if wpath:
            try:
                state = torch.load(wpath, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[WARN] weights load skipped for {model_name}: {e}")

        model.eval()
        example_input = torch.randn(B, T, C)
        macs = macs_counter.count(model, example_input)
        macs_dict[model_name] = int(macs)
        print(f"[MACs] {model_name}: {macs:,} MACs (for input {(B, T, C)})")
    return macs_dict



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_csv", required=True, help="CSV with spike stats + accuracy")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.001)
    ap.add_argument("--input_shape", default="1,200,8", help="B,T,C for MACs probe")
    ap.add_argument("--model_builder", required=False, help='e.g. "user_models:build_model"')
    ap.add_argument("--weights_map", required=False,
                    help='Comma list "Name:path,Name2:path2" to load .pth and count MACs')
    ap.add_argument("--out_csv", default="tradeoff_results.csv")
    ap.add_argument("--out_plot", default="tradeoff_plot.png")
    args = ap.parse_args()

    df = pd.read_csv(args.stats_csv)

    # Optional: load models and compute dense MACs
    macs_dict = {}
    if args.model_builder:
        # weights_map이 없어도 빈 dict로 처리
        weights_map = {}
        if args.weights_map:
            for kv in args.weights_map.split(","):
                name, path = kv.split(":")
                weights_map[name.strip()] = path.strip()
        B, T, C = parse_shape(args.input_shape)
        model_names = df["model_name"].unique().tolist()
        macs_dict = load_models_and_count_macs(args.model_builder, model_names, weights_map, (B, T, C))

    # Attach MACs to rows by model_name (fallback 0 if not provided/loaded)
    df["dense_macs"] = df["model_name"].map(lambda n: macs_dict.get(n, 0))

    # Approximate synaptic events if not present
    if "syn_events" not in df.columns or df["syn_events"].isna().all():
        if "fan_in_avg" in df.columns:
            df["syn_events"] = df["total_spikes"] * df["fan_in_avg"]
        else:
            # conservative fallback: syn_events ~= total_spikes * 50
            df["syn_events"] = df["total_spikes"] * 50

    # Compute E_hat
    df["E_hat"] = args.alpha * df["dense_macs"] + args.beta * df["total_spikes"] + args.gamma * df["syn_events"]

    # Normalise E_hat for nicer plotting (min-max to [0,1], but keep raw too)
    Emin, Emax = df["E_hat"].min(), df["E_hat"].max()
    if Emax > Emin:
        df["E_hat_norm"] = (df["E_hat"] - Emin) / (Emax - Emin)
    else:
        df["E_hat_norm"] = 0.0

    # Save results
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved: {args.out_csv}")

    # Plot Macro-F1 vs Mean firing rate
    plt.figure()
    texts = []
    for name, g in df.groupby("model_name"):
        plt.scatter(g["mean_firing_rate"], g["macro_f1"], label=name)
        # annotate points with (encoding, Ts)
        for j, (_, row) in enumerate(g.iterrows()):
            label = f'{row["encoding"]}, Ts={row["Ts"]}'

    plt.xlabel("Mean firing rate (%)")
    plt.ylabel("Macro-F1 (%)")
    plt.title("Accuracy–Firing Rate Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    print(f"[OK] Saved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
