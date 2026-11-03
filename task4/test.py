import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from task2.attention import CustomSelfAttention
from .attention import CustomFlashAttention

NUM_WARMUP = 3
NUM_RUNS = 10

def test_attention(causal=False):
    hidden_dim = 4096
    num_heads = 32

    torch_self_attention = (
        torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, bias=False, batch_first=True)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    all_param = dict(torch_self_attention.named_parameters())
    w_q, w_k, w_v = all_param["in_proj_weight"].chunk(3)
    w_o = all_param["out_proj.weight"]

    naive_attention = (
        CustomSelfAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    flash_attention = (
        CustomFlashAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    batch_size = 4
    seq_len = 1024

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        # Warmup
        for _ in range(NUM_WARMUP):
            _ = naive_attention(x, causal=causal)
            _ = flash_attention(x, causal=causal)
        torch.cuda.synchronize()

        # Correctness check
        naive_out = naive_attention(x, causal=causal)
        flash_out = flash_attention(x, causal=causal)
        torch.cuda.synchronize()
        assert torch.allclose(naive_out, flash_out, atol=1e-6)

        # Benchmark naive (task2) vs flash (task4)
        naive_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Naive (small)"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = naive_attention(x, causal=causal)
            torch.cuda.synchronize()
            naive_times.append(time.time() - start_time)

        flash_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Flash (small)"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = flash_attention(x, causal=causal)
            torch.cuda.synchronize()
            flash_times.append(time.time() - start_time)

        naive_time = sorted(naive_times)[NUM_RUNS // 2]
        flash_time = sorted(flash_times)[NUM_RUNS // 2]

        print(f"{'' if causal else 'Non-'}Causal Self-Attention: batch_size = {batch_size}, seq_len = {seq_len}")
        print(f"Naive time (ms): {naive_time * 1000 :.2f}, Flash time (ms): {flash_time * 1000 :.2f}")
        print(f"Speedup: {naive_time / flash_time :.2f}")

    seq_len = 4096

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        # Warmup
        for _ in range(NUM_WARMUP):
            _ = naive_attention(x, causal=causal)
            _ = flash_attention(x, causal=causal)
        torch.cuda.synchronize()

        # Correctness check
        naive_out = naive_attention(x, causal=causal)
        flash_out = flash_attention(x, causal=causal)
        torch.cuda.synchronize()
        assert torch.allclose(naive_out, flash_out, atol=1e-6)

        # Benchmark naive (task2) vs flash (task4)
        naive_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Naive (large)"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = naive_attention(x, causal=causal)
            torch.cuda.synchronize()
            naive_times.append(time.time() - start_time)

        flash_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Flash (large)"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = flash_attention(x, causal=causal)
            torch.cuda.synchronize()
            flash_times.append(time.time() - start_time)

        naive_time = sorted(naive_times)[NUM_RUNS // 2]
        flash_time = sorted(flash_times)[NUM_RUNS // 2]

        print(f"{'' if causal else 'Non-'}Causal Self-Attention: batch_size = {batch_size}, seq_len = {seq_len}")
        print(f"Naive time (ms): {naive_time * 1000 :.2f}, Flash time (ms): {flash_time * 1000 :.2f}")
        print(f"Speedup: {naive_time / flash_time :.2f}")

    if causal:
        print("Causal Self-Attention test passed!")
    else:
        print("Non-causal Self-Attention test passed!")

if __name__ == "__main__":
    torch.manual_seed(41)
    test_attention()
    test_attention(causal=True)

    print("All tests passed!")
