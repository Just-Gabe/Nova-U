"""
Nova-U: PyTorch training script for Google Colab.
Trains the U-Net and exports weights in a format directly loadable by the Go
inference engine (cmd/generate).

Usage in Colab:
  !pip install torch numpy
  # Optional: produce a deterministic vocab from the Go side so both stacks
  # share the exact same char-to-id map:
  #   go run ./cmd/train -data text.txt -vocab-out vocab.json
  !python train_colab.py --data text.txt --vocab vocab.json --export model.bin
"""

import argparse
import json
import math
import os
import re
import struct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- model ---


class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = DoubleConv1D(in_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x.clone()
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, cat_ch, out_ch, kernel_size=3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConv1D(cat_ch, out_ch, kernel_size)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_levels=3, kernel_size=3, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoder", self._make_pos_encoding(max_seq_len, d_model))
        self.in_conv = DoubleConv1D(d_model, d_model, kernel_size)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch = d_model
        for i in range(num_levels):
            out_ch = d_model * (2 ** (i + 1))
            self.down_blocks.append(DownBlock(ch, out_ch, kernel_size))
            ch = out_ch

        self.bottleneck = DoubleConv1D(ch, ch, kernel_size)

        for i in range(num_levels):
            level = num_levels - 1 - i
            up_in_ch = d_model * (2 ** (level + 1))
            cat_ch = 2 * up_in_ch
            out_ch = d_model * (2 ** level)
            self.up_blocks.append(UpBlock(cat_ch, out_ch, kernel_size))

        self.out_conv = nn.Conv1d(d_model, vocab_size, 1)

    def _make_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(1, d_model, max_len)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pe[0, i, pos] = math.sin(angle)
                if i + 1 < d_model:
                    pe[0, i + 1, pos] = math.cos(angle)
        return pe

    def forward(self, x):
        b, l = x.shape
        x = self.embed(x).transpose(1, 2)
        x = x + self.pos_encoder[:, :, :l]

        x = self.in_conv(x)

        skips = []
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            x = up(x, skip)

        x = self.out_conv(x)
        return x


# --- vocab ---


def build_or_load_vocab(text, vocab_path):
    """Return (char_to_id, id_to_char). If vocab_path exists, load it; else
    build a deterministic one and write it to vocab_path."""
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            v = json.load(f)
        if v.get("version") != 1:
            raise ValueError(f"unsupported vocab version: {v.get('version')}")
        char_to_id = {ch: int(i) for ch, i in v["chars"].items()}
    else:
        chars = sorted(set(text))
        char_to_id = {ch: i + 3 for i, ch in enumerate(chars)}
        if vocab_path:
            os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
            with open(vocab_path, "w") as f:
                json.dump({"version": 1, "chars": char_to_id}, f, ensure_ascii=False, indent=2)
            print(f"wrote vocab to {vocab_path}")
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    return char_to_id, id_to_char


def vocab_size_from(char_to_id):
    # PAD/BOS/EOS are reserved at 0/1/2 and not in char_to_id.
    return len(char_to_id) + 3


# --- weight name mapping (PyTorch → Go) ---


_NAME_REWRITES = [
    (re.compile(r"^down_blocks\."), "down."),
    (re.compile(r"^up_blocks\."), "up."),
]


def to_go_name(name):
    for pat, repl in _NAME_REWRITES:
        name = pat.sub(repl, name)
    return name


# --- export ---


def export_weights_json(model, path):
    state = model.state_dict()
    out = {}
    for py_name, tensor in state.items():
        go_name = to_go_name(py_name)
        out[go_name] = {
            "data": tensor.detach().cpu().numpy().flatten().tolist(),
            "shape": list(tensor.shape),
        }
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"weights exported (json) to {path}")


def export_weights_binary(model, path):
    """Binary format compatible with internal/infer/weights.go LoadWeightsBinary."""
    state = model.state_dict()
    entries = []
    for py_name, tensor in state.items():
        go_name = to_go_name(py_name)
        data = tensor.detach().cpu().numpy().astype("<f4").flatten()
        shape = list(tensor.shape)
        entries.append((go_name, shape, data))

    with open(path, "wb") as f:
        f.write(b"NOVA")
        f.write(struct.pack("<I", 1))             # version
        f.write(struct.pack("<I", len(entries)))  # nEntries
        for name, shape, data in entries:
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > 0xFFFF:
                raise ValueError(f"name too long: {name}")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            if len(shape) > 255:
                raise ValueError(f"too many dims for {name}")
            f.write(struct.pack("<B", len(shape)))
            for d in shape:
                f.write(struct.pack("<i", int(d)))
            f.write(data.tobytes())
    print(f"weights exported (binary) to {path}")


def export_weights(model, path):
    if path.endswith(".bin"):
        export_weights_binary(model, path)
    else:
        export_weights_json(model, path)


# --- train loop ---


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}: loss={loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Nova-U Colab Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--data", type=str, required=True, help="path to training text")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                        help="path to vocab.json (loaded if exists, else built)")
    parser.add_argument("--export", type=str, default="model.bin",
                        help=".bin (Go binary format) or .json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    with open(args.data, "r") as f:
        text = f.read()

    char_to_id, _ = build_or_load_vocab(text, args.vocab)
    vocab_size = vocab_size_from(char_to_id)
    tokens = [char_to_id.get(ch, 0) for ch in text]
    print(f"vocab size: {vocab_size}, total tokens: {len(tokens)}")

    sequences = []
    targets = []
    stride = args.seq_len // 2
    for i in range(0, len(tokens) - args.seq_len - 1, stride):
        sequences.append(tokens[i: i + args.seq_len])
        targets.append(tokens[i + 1: i + args.seq_len + 1])

    if not sequences:
        raise SystemExit("not enough tokens for the requested --seq-len")

    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(sequences, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet1D(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_levels=3,
        max_seq_len=args.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {param_count:,}")

    for epoch in range(args.epochs):
        start = time.time()
        loss = train_epoch(model, dataloader, optimizer, device)
        elapsed = time.time() - start
        ppl = math.exp(loss)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, ppl={ppl:.2f}, time={elapsed:.1f}s")

    export_weights(model, args.export)
    print("done!")


if __name__ == "__main__":
    main()
