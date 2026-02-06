import torch
import time
import argparse
import numpy as np
from generate_optimized import sample_gumbel_chunked, add_gumbel_noise_original


def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_memory_stats():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def test_logic_strict_cpu(batch_size, seq_len, vocab_size, temperature=1.0):
    """
    Runs on CPU to verify the output is identical.
    CPU RNG is serial, so this should match exactly.
    """
    print(f"\n>>> [Logic Check: CPU] Batch={batch_size}, Temp={temperature}")
    device = 'cpu'
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)  # float32 for CPU speed

    torch.manual_seed(42)
    start_t = time.time()
    noisy_logits_orig = add_gumbel_noise_original(logits, temperature)
    indices_orig = torch.argmax(noisy_logits_orig, dim=-1)
    print (f'Original:  Time = {time.time() - start_t:.4f}s')

    torch.manual_seed(42)
    start_t = time.time()
    indices_eff = sample_gumbel_chunked(logits, temperature)
    print (f'Efficient:  Time = {time.time() - start_t:.4f}s')
    mismatch = (indices_orig != indices_eff).sum().item()
    if mismatch == 0:
        print("SUCCESS: CPU outputs are identical. Logic is correct.")
    else:
        print(f"FAILURE: CPU outputs differ by {mismatch} elements.")


def test_statistical_gpu(batch_size, seq_len, vocab_size, temperature=1.0, device='cuda'):
    """
    Runs on GPU to verify the distribution matches.
    We cannot expect identity due to Philox RNG.
    """
    print(f"\n>>> [Distribution Check: GPU] Batch={batch_size}, Temp={temperature}")

    logits = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=torch.float16)

    torch.manual_seed(42)
    indices_eff = sample_gumbel_chunked(logits, temperature)

    # Since logits are 0, we are essentially sampling purely from Gumbel noise.
    # Uniform distribution of indices means the noise is working.
    # We check if the distribution of selected indices is roughly uniform across vocab.

    # Expected mean index should be approx vocab_size / 2
    mean_idx = indices_eff.float().mean().item()
    expected_idx = vocab_size / 2.0

    # We allow a loose tolerance because random is random
    diff = abs(mean_idx - expected_idx)

    print(f"   Sampled Mean Index: {mean_idx:.2f}")
    print(f"   Expected Mean Index: {expected_idx:.2f}")

    if diff < (vocab_size * 0.05):  # within 5%
        print("SUCCESS: GPU output distribution looks statistically valid.")
    else:
        print("WARNING: Distribution might be skewed (or sample size too small).")


def test_performance(batch_size, seq_len, vocab_size, temperature=0.7, device='cuda'):
    print(f"\n>>> [Performance Test] Batch={batch_size}, Seq={seq_len}, Vocab={vocab_size}")

    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.bfloat16)

    reset_memory_stats()
    start_t = time.time()

    peak_mem_orig = 0
    try:
        _ = add_gumbel_noise_original(logits, temperature)
        torch.cuda.synchronize()
        end_t = time.time()
        peak_mem_orig = get_peak_memory_mb()
        time_orig = end_t - start_t
        print(f"Original:  Time = {time_orig:.4f}s | Peak VRAM = {peak_mem_orig:.2f} MB")
    except RuntimeError:
        print(f"Original:  CRASHED")
        peak_mem_orig = float('inf')

    reset_memory_stats()
    start_t = time.time()

    _ = sample_gumbel_chunked(logits, temperature)

    torch.cuda.synchronize()
    end_t = time.time()

    peak_mem_eff = get_peak_memory_mb()
    time_eff = end_t - start_t

    print(f"Efficient: Time = {time_eff:.4f}s | Peak VRAM = {peak_mem_eff:.2f} MB")

    if peak_mem_orig != float('inf'):
        saved = peak_mem_orig - peak_mem_eff
        print(f"VRAM Saved: {saved:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=128000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if torch.cuda.is_available():
        test_statistical_gpu(8, 128, 1000, device='cuda')
        test_performance(args.batch_size, args.seq_len, args.vocab_size, device='cuda')

        print("\n>>> Batch=32 (High VRAM)")
        test_performance(32, 128, args.vocab_size, device='cuda')
    else:
        print("Skipping GPU tests.")

    # Show the algorithms are identical with CPU RNG 
    
    test_logic_strict_cpu(2, 32, 100)
