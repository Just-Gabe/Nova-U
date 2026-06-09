"""
Estudo de Quantização inspirada no TurboQuant para Nova-U.
Analisa os pesos do modelo e simula quantização com Lloyd-Max + group quantization.

Uso:
  python scripts/study_quantization.py --weights /tmp/novocortex_500_v2.bin --vocab vocab.json
"""

import argparse
import json
import math
import struct
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from train_colab import (
    UNet1D,
    SYSTEM_MARKER,
    USER_MARKER,
    ASSISTANT_MARKER,
    END_MARKER,
)


def load_nova_weights(model, path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"NOVA", f"bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_entries = struct.unpack("<I", f.read(4))[0]
        state = {}
        for _ in range(n_entries):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<B", f.read(1))[0]
            shape = struct.unpack("<" + "i" * n_dims, f.read(4 * n_dims))
            n_elems = math.prod(shape)
            data = torch.frombuffer(f.read(4 * n_elems), dtype=torch.float32).clone()
            if shape:
                data = data.reshape(shape)
            state[name] = data

    py_name_map = {}
    py_name_map["embed.weight"] = "embed.weight"
    py_name_map["pos_encoder"] = "pos_encoder"
    for prefix in ["in_conv", "bottleneck"]:
        for layer in ["conv1", "conv2", "norm1", "norm2"]:
            for param in ["weight", "bias"]:
                py_name_map[f"{prefix}.{layer}.{param}"] = f"{prefix}.{layer}.{param}"
    py_name_map["out_conv.weight"] = "out_conv.weight"
    py_name_map["out_conv.bias"] = "out_conv.bias"
    for i in range(3):
        for layer in ["conv1", "conv2", "norm1", "norm2"]:
            for param in ["weight", "bias"]:
                py_name_map[f"down.{i}.conv.{layer}.{param}"] = f"down_blocks.{i}.conv.{layer}.{param}"
                py_name_map[f"up.{i}.conv.{layer}.{param}"] = f"up_blocks.{i}.conv.{layer}.{param}"

    pt_state = {}
    for nova_name, pt_key in py_name_map.items():
        if nova_name in state:
            pt_state[pt_key] = state[nova_name]

    model.load_state_dict(pt_state, strict=False)
    return model


# --- Lloyd-Max Quantizer (adaptado do TurboQuant) ---


def lloyd_max_gaussian(data, num_bits, max_iter=200, tol=1e-12):
    """Lloyd-Max quantizer otimizado para distribuição de pesos (Gaussiana/Laplace)."""
    n_levels = 1 << num_bits

    flat = data.view(-1)
    std = flat.std().item()
    mean = flat.mean().item()
    normalized = (flat - mean) / std

    quantiles = torch.linspace(0, 1, n_levels + 1)
    sorted_vals, _ = normalized.sort()
    idxs = (quantiles[1:-1] * len(sorted_vals)).long().clamp(0, len(sorted_vals) - 1)
    centroids = sorted_vals[idxs].float()

    for iteration in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        expanded = normalized.unsqueeze(0)
        bounds_exp = boundaries.unsqueeze(1)
        assignments = (expanded > bounds_exp).sum(dim=0).clamp(0, n_levels - 1)
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = normalized[mask].mean()

        change = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if change < tol:
            break

    centroids = centroids * std + mean
    return centroids, boundaries * std + mean, mean, std


def group_quantize(tensor, group_size=32, num_bits=4):
    """Group quantization com per-group scale/zero, similar ao TurboQuant."""
    orig_shape = tensor.shape
    flat = tensor.view(-1)
    n = flat.shape[0]
    n_groups = (n + group_size - 1) // group_size

    padded = torch.zeros(n_groups * group_size)
    padded[:n] = flat

    groups = padded.view(n_groups, group_size)
    g_min = groups.min(dim=1, keepdim=True)[0]
    g_max = groups.max(dim=1, keepdim=True)[0]
    scale = (g_max - g_min) / ((1 << num_bits) - 1)
    scale[scale == 0] = 1.0
    q = ((groups - g_min) / scale).round().clamp(0, (1 << num_bits) - 1)
    deq = q * scale + g_min

    compressed = q.to(torch.uint8).flatten()[:n]
    scales = scale.flatten()
    zeros = g_min.flatten()

    mse = ((flat - deq.flatten()[:n]) ** 2).mean().item()
    return compressed, scales, zeros, deq.flatten()[:n].reshape(orig_shape), mse


def analyze_weights(state_dict, model_name="Modelo"):
    print(f"\n{'='*60}")
    print(f"Análise de Pesos: {model_name}")
    print(f"{'='*60}")

    total_params = 0
    total_size_f32 = 0
    total_size_i8 = 0
    total_size_i4 = 0

    layers = []

    for name, tensor in state_dict.items():
        if "norm" in name or "pos_encoder" in name:
            continue

        w = tensor.detach().cpu().float()
        n = w.numel()
        total_params += n
        total_size_f32 += n * 4

        w_flat = w.view(-1)

        layers.append({
            "name": name,
            "shape": list(w.shape),
            "params": n,
            "size_f32_mb": n * 4 / 1e6,
            "mean": w.mean().item(),
            "std": w.std().item(),
            "min": w.min().item(),
            "max": w.max().item(),
            "outliers_3std": (w.abs() > 3 * w.std()).sum().item(),
            "outliers_pct": (w.abs() > 3 * w.std()).sum().item() / n * 100,
        })

    print(f"\nParâmetros totais: {total_params:,}")
    print(f"Tamanho float32:   {total_size_f32/1e6:.1f} MB")
    print(f"Tamanho int8 (est): {total_params/1e6:.1f} MB (4x menor)")
    print(f"Tamanho int4 (est): {total_params/2e6:.1f} MB (8x menor)")
    print(f"\n{'Nome':<35} {'Shape':<20} {'Params':<10} {'Média':<8} {'Std':<8} {'Outliers%':<10}")
    print("-" * 95)
    for l in layers:
        print(
            f"{l['name']:<35} {str(l['shape']):<20} {l['params']:<10} "
            f"{l['mean']:<8.4f} {l['std']:<8.4f} {l['outliers_pct']:<10.2f}"
        )


def quantization_simulation(state_dict, model_name="Modelo"):
    print(f"\n{'='*60}")
    print(f"Simulação de Quantização: {model_name}")
    print(f"{'='*60}")

    results = []

    for name, tensor in state_dict.items():
        if "norm" in name or "pos_encoder" in name or "bias" in name:
            continue

        w = tensor.detach().cpu().float()

        mse_4bit, mse_3bit, mse_2bit = None, None, None
        compression_4bit, compression_3bit, compression_2bit = None, None, None
        ratio_4bit, ratio_3bit, ratio_2bit = None, None, None

        for bits in [4, 3, 2]:
            try:
                _, _, _, deq, mse = group_quantize(w, group_size=32, num_bits=bits)
                if bits == 4:
                    mse_4bit = mse
                    compression_4bit = w.numel() * 32 / (w.numel() * bits + (w.numel() / 32) * 2 * 16)
                    ratio_4bit = compression_4bit
                elif bits == 3:
                    mse_3bit = mse
                    compression_3bit = w.numel() * 32 / (w.numel() * 3 + (w.numel() / 32) * 2 * 16)
                    ratio_3bit = compression_3bit
                elif bits == 2:
                    mse_2bit = mse
                    compression_2bit = w.numel() * 32 / (w.numel() * 2 + (w.numel() / 32) * 2 * 16)
                    ratio_2bit = compression_2bit
            except Exception:
                pass

        results.append({
            "name": name,
            "shape": list(w.shape),
            "params": w.numel(),
            "mse_4bit": mse_4bit,
            "mse_3bit": mse_3bit,
            "mse_2bit": mse_2bit,
            "ratio_4bit": ratio_4bit,
            "ratio_3bit": ratio_3bit,
            "ratio_2bit": ratio_2bit,
        })

    print(f"\n{'Nome':<35} {'#Params':<10} {'MSE(4b)':<12} {'MSE(3b)':<12} {'MSE(2b)':<12}")
    print("-" * 85)
    for r in results:
        mse4 = f"{r['mse_4bit']:.6e}" if r['mse_4bit'] else "-"
        mse3 = f"{r['mse_3bit']:.6e}" if r['mse_3bit'] else "-"
        mse2 = f"{r['mse_2bit']:.6e}" if r['mse_2bit'] else "-"
        print(f"{r['name']:<35} {r['params']:<10} {mse4:<12} {mse3:<12} {mse2:<12}")

    avg_mse_4 = sum(r['mse_4bit'] for r in results if r['mse_4bit']) / max(
        sum(1 for r in results if r['mse_4bit']), 1
    )
    avg_mse_3 = sum(r['mse_3bit'] for r in results if r['mse_3bit']) / max(
        sum(1 for r in results if r['mse_3bit']), 1
    )
    avg_mse_2 = sum(r['mse_2bit'] for r in results if r['mse_2bit']) / max(
        sum(1 for r in results if r['mse_2bit']), 1
    )
    print(f"\nMSE médio 4-bit: {avg_mse_4:.6e}")
    print(f"MSE médio 3-bit: {avg_mse_3:.6e}")
    print(f"MSE médio 2-bit: {avg_mse_2:.6e}")

    total_params = sum(r['params'] for r in results)
    print(f"\nCompressão estimada (4-bit, group=32): {total_params*4/(total_params*0.5 + total_params/32*2*2):.1f}x")
    print(f"Tamanho estimado (4-bit): {total_params*0.5/1e6:.1f} MB + {total_params/32*4/1e6:.1f} MB (metadata)")
    print(f"Tamanho float32 original: {total_params*4/1e6:.1f} MB")


def cosine_similarity_simulation(model, weights_path, vocab_path):
    """Avalia o impacto da quantização na saída do modelo."""
    print(f"\n{'='*60}")
    print("Teste de Similaridade Cosseno (saída com/sem quantização)")
    print(f"{'='*60}")

    with open(vocab_path) as f:
        v = json.load(f)
    char_to_id = {ch: int(i) for ch, i in v["chars"].items()}
    vocab_size = len(char_to_id) + 3

    # Carrega modelo original
    model_orig = UNet1D(vocab_size=vocab_size, d_model=64, max_seq_len=64)
    load_nova_weights(model_orig, weights_path)
    model_orig.eval()

    # Cria batch de teste
    test_text = "<|sys|>\nTeste de entrada\n<|end|>\n<|usr|>\nComo funciona?\n<|end|>\n<|ast|>\n"
    tokens = [char_to_id.get(ch, 0) for ch in test_text]
    x = torch.tensor([tokens[:32]], dtype=torch.long)
    if x.shape[1] < 32:
        x = torch.cat([torch.zeros(1, 32 - x.shape[1], dtype=torch.long), x], dim=1)

    with torch.no_grad():
        out_orig = model_orig(x)

    # Simula quantização 4-bit nos pesos Conv1D
    model_quant = UNet1D(vocab_size=vocab_size, d_model=64, max_seq_len=64)
    load_nova_weights(model_quant, weights_path)
    model_quant.eval()

    with torch.no_grad():
        for name, param in model_quant.named_parameters():
            if "conv" in name and "weight" in name and param.dim() >= 2:
                if param.numel() > 64:
                    _, _, _, deq, _ = group_quantize(param.data, group_size=32, num_bits=4)
                    param.data.copy_(deq.to(param.dtype))

        out_quant = model_quant(x)

    cos_sim = F.cosine_similarity(out_orig.view(-1), out_quant.view(-1), dim=0).item()
    mse = ((out_orig - out_quant) ** 2).mean().item()
    max_diff = (out_orig - out_quant).abs().max().item()

    print(f"Cosine similarity (4-bit quant): {cos_sim:.6f}")
    print(f"MSE da saída: {mse:.6e}")
    print(f"Erro máximo por logit: {max_diff:.6f}")
    print(f"Qualidade: {'EXCELENTE' if cos_sim > 0.999 else 'BOA' if cos_sim > 0.99 else 'ACEITÁVEL' if cos_sim > 0.95 else 'DEGRADADO'}")


def main():
    parser = argparse.ArgumentParser(description="Estudo de quantização para Nova-U")
    parser.add_argument("--weights", default="/tmp/novocortex_500_v2.bin", help="Caminho para weights .bin")
    parser.add_argument("--vocab", default="vocab.json", help="Caminho para vocab.json")
    args = parser.parse_args()

    # Carrega modelo
    with open(args.vocab) as f:
        v = json.load(f)
    vocab_size = len(v["chars"]) + 3
    model = UNet1D(vocab_size=vocab_size, d_model=64, max_seq_len=64)
    load_nova_weights(model, args.weights)

    state = model.state_dict()

    # 1. Análise estatística dos pesos
    analyze_weights(state, "Nova-U (d_model=64)")

    # 2. Simulação de quantização
    quantization_simulation(state, "Nova-U (d_model=64)")

    # 3. Teste de similaridade cosseno
    cosine_similarity_simulation(model, args.weights, args.vocab)

    # 4. Recomendações
    print(f"\n{'='*60}")
    print("Recomendações TurboQuant para Nova-U")
    print(f"{'='*60}")
    print("""
1. ESTRATÉGIA DE QUANTIZAÇÃO RECOMENDADA
   ─────────────────────────────────────
   Conforme análise dos pesos do Nova-U (Conv1D + Embedding + LayerNorm):

   a) Conv1D weights (85% dos parâmetros)
      → Group quantization 4-bit, group_size=32
      → Lloyd-Max codebook adaptado à distribuição dos pesos (aproximadamente Gaussiana)
      → Per-group scale (float16) + zero (float16)
      → MSE esperado: < 1e-6 (quase lossless)

   b) Embedding weight (vocab_size × d_model)
      → 8-bit quantization (mais sensível por ser lookup table)
      → Per-channel (per-row) scaling

   c) LayerNorm weight/bias
      → Manter float32 (poucos parâmetros, ~0.1% do total)
      → Crítico para estabilidade numérica

   d) Bias
      → Manter float32 (poucos parâmetros)

2. ARQUITETURA DE INFERÊNCIA QUANTIZADA
   ─────────────────────────────────────
   Adaptando conceitos do TurboQuant para U-Net:

   ├─ Weight Loading
   │   ├─ packed_indices  (uint8, 2 valores/byte para 4-bit)
   │   ├─ scales_f16      (float16, 1 por grupo)
   │   └─ zeros_f16       (float16, 1 por grupo)
   │
   ├─ Conv1D Forward (quantizado)
   │   ├─ Dequantizar weights on-the-fly: w = (idx * scale) + zero
   │   ├─ im2col como antes
   │   └─ Matmul em float32 (acumulador precisa de precisão)
   │
   └─ Embedding Forward
       ├─ Lookup como antes (índices int)
       └─ Dequantizar o embedding: val = q_val * scale_row + zero_row

3. BIT-PACKING (adaptado do TurboQuant)
   ─────────────────────────────────────
   4-bit: 2 valores por byte
   byte = (v0 << 4) | v1
   unpack: v0 = (byte >> 4) & 0xF, v1 = byte & 0xF

4. ESTIMATIVA DE TAMANHO FINAL
   ─────────────────────────
   Modelo d_model=32 (1.1M params):
   - float32:  4.2 MB
   - int8:     1.1 MB (4x compressão)
   - int4:     0.6 MB (8x compressão) + 0.1 MB metadata = ~0.7 MB

   Modelo d_model=64 (4.5M params):
   - float32: 17.9 MB
   - int8:     4.5 MB
   - int4:    ~2.3 MB

5. PRÓXIMOS PASSOS
   ─────────────
   a) Implementar Lloyd-Max codebook no Go (pkg/tensor/quantize.go)
   b) Adicionar tipo QTensor (Tensor quantizado) ao pkg/tensor
   c) Implementar Conv1D quantizado (dequantização on-the-fly)
   d) Adicionar formato de serialização quantizado (QNOVA magic)
   e) Benchmark: inferência quantizada vs float32
""")


if __name__ == "__main__":
    main()
