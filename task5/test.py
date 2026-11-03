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
    max_seq_len = 2048

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

    custom_self_attention = (
        CustomFlashAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    batch_size = 16
    seq_len = 1024

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")
        k_cache = torch.zeros(batch_size, max_seq_len, hidden_dim).to(torch.float).to("cuda")
        v_cache = torch.zeros(batch_size, max_seq_len, hidden_dim).to(torch.float).to("cuda")

        # Warmup
        for _ in range(NUM_WARMUP):
            _ = naive_attention(x, causal=causal)
            _ = custom_self_attention(x, kv_cache=(k_cache, v_cache), current_pos=0, causal=causal)
        torch.cuda.synchronize()

        # Correctness check
        naive_out = naive_attention(x, causal=causal)
        custom_out = custom_self_attention(x, kv_cache=(k_cache, v_cache), current_pos=0, causal=causal)
        torch.cuda.synchronize()
        assert torch.allclose(naive_out, custom_out, atol=1e-6), "Naive and custom prefill outputs do not match"

        # Benchmark prefill (naive)
        naive_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Naive prefill"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = naive_attention(x, causal=causal)
            torch.cuda.synchronize()
            naive_times.append(time.time() - start_time)

        # Benchmark prefill (custom)
        custom_times = []
        for _ in tqdm(range(NUM_RUNS), desc="Custom prefill"):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = custom_self_attention(x, kv_cache=(k_cache, v_cache), current_pos=0, causal=causal)
            torch.cuda.synchronize()
            custom_times.append(time.time() - start_time)

        naive_time = sorted(naive_times)[NUM_RUNS // 2]
        custom_time = sorted(custom_times)[NUM_RUNS // 2]

        print(f"{'' if causal else 'Non-'}Causal Self-Attention Prefill")
        print(f"Naive time (ms): {naive_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup on time to first token (TTFT): {naive_time / custom_time :.2f}")

        all_tokens = x

        naive_times = []
        custom_times = []

        for i in tqdm(range(101), desc="Decode"):
            new_token = custom_out[:, -1:, :]
            all_tokens = torch.cat([all_tokens, new_token], dim=1)

            torch.cuda.synchronize()
            start_time = time.time()
            naive_out = naive_attention(all_tokens, causal=causal)
            torch.cuda.synchronize()
            naive_time = time.time() - start_time
            start_time = time.time()
            custom_out = custom_self_attention(new_token, kv_cache=(k_cache, v_cache), decode=True, current_pos=seq_len + i, causal=causal)
            torch.cuda.synchronize()
            custom_time = time.time() - start_time

            assert torch.allclose(naive_out[:, -1:, :], custom_out[:, -1:, :], atol=1e-6), "Naive and custom decode outputs do not match at step %d" % (i + 1)

            naive_times.append(naive_time)
            custom_times.append(custom_time)

        # Remove the first iteration to exclude warmup effects, take median
        naive_time = sorted(naive_times[1:])[len(naive_times[1:]) // 2]
        custom_time = sorted(custom_times[1:])[len(custom_times[1:]) // 2]

        print(f"{'' if causal else 'Non-'}Causal Self-Attention Decode ({len(naive_times) - 1} tokens)")
        print(f"Naive time (ms): {naive_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup on time between tokens (TBT): {naive_time / custom_time :.2f}")

    if causal:
        print("Causal Self-Attention test passed!")
    else:
        print("Non-causal Self-Attention test passed!")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_attention(causal=True)

    print("All tests passed!")
