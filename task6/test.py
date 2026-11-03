import torch
import torch.distributed as dist
import os
import time
from tqdm import tqdm

from .model import customModel

NUM_WARMUP = 3
NUM_RUNS = 10

def test_model(batch_size, seq_len, hidden_dim, num_heads, num_layers, vocab_size):
    # Generate random token IDs (shape: [batch_size, seq_len])
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    model = customModel(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, vocab_size=vocab_size).eval().to("cuda")

    with torch.no_grad():
        # Warmup prefill
        print("Warming up...")
        for _ in range(NUM_WARMUP):
            _, _ = model(x)
        torch.cuda.synchronize()

        # Timed prefill runs (keep last kv_cache for decode)
        print("Testing prefill mode...")
        prefill_times = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            start_time = time.time()
            out, kv_cache = model(x)
            torch.cuda.synchronize()
            prefill_times.append(time.time() - start_time)

        prefill_time = sorted(prefill_times)[NUM_RUNS // 2]
        current_pos = seq_len
        generated_tokens = [x]

        # Get first token from prefill
        next_token = torch.argmax(out[:, -1:, :], dim=-1)
        generated_tokens.append(next_token)
        print(f"Prefill time (median of {NUM_RUNS}): {prefill_time * 1000 :.2f} ms")

        # Decode mode: generate 100 tokens using KV cache
        print("Testing decode mode (generating 100 tokens with KV cache)...")
        decode_times = []

        # Generate remaining tokens
        for step in tqdm(range(99), desc="Decode"):
            torch.cuda.synchronize()
            start_time = time.time()
            logits, kv_cache = model(next_token, kv_cache=kv_cache, current_pos=current_pos)
            torch.cuda.synchronize()
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            generated_tokens.append(next_token)
            current_pos += 1
            decode_time = time.time() - start_time
            decode_times.append(decode_time)

        decode_time = sorted(decode_times)[len(decode_times) // 2]
        print(f"Median decode time: {decode_time * 1000 :.2f} ms")

        final_tokens = torch.cat(generated_tokens, dim=1)
        print(f"Final sequence length after decode: {final_tokens.shape[1]} (original {seq_len} + 100 generated)")

if __name__ == "__main__":
    # For single GPU test
    test_model(batch_size=4, seq_len=1024, hidden_dim=4096, num_heads=32, num_layers=24, vocab_size=8192)
