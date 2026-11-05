#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>



using namespace torch::indexing;

// FlashAttention kernel
template <int B>
__global__ void __launch_bounds__(128) flash_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles (padded by +1 to avoid bank conflicts)
    __shared__ __align__(16) float q_tile[128];  // Max head_dim = 128
    __shared__ __align__(16) float k_tile[B][128 + 4];
    __shared__ __align__(16) float v_tile[B][128 + 4];
    __shared__ float m_tile;
    __shared__ float l_tile;
    __shared__ float alpha_tile;
    __shared__ float p[B];

    // ########################################################

    // TODO: Implement the FlashAttention operation 

    constexpr int NT = 128;
    constexpr int TPK = NT / B;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bh = blockIdx.x;

    const float* qb = q + (size_t)bh * head_dim;
    const float* kb = k + (size_t)bh * seq_len * head_dim;
    const float* vb = v + (size_t)bh * seq_len * head_dim;
    float* ob = out + (size_t)bh * head_dim;

    const float scale = rsqrtf((float)head_dim);
    const int d4 = head_dim >> 2;

    for (int d = tid; d < head_dim; d += NT) q_tile[d] = qb[d] * scale;
    if (tid == 0) {
        m_tile = -FLT_MAX;
        l_tile = 0.f;
    }

    float o_acc = 0.f;
    const int key = tid / TPK;
    const int sub = tid % TPK;
    const int num_tiles = (seq_len + B - 1) / B;

    for (int j = 0; j < num_tiles; ++j) {
        const int col0 = j * B;
        __syncthreads();
        for (int idx = tid; idx < B * d4; idx += NT) {
            int r = idx / d4;
            int c = (idx - r * d4) << 2;
            int gr = col0 + r;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv4 = kv4;
            if (gr < seq_len) {
                kv4 = *reinterpret_cast<const float4*>(kb + (size_t)gr * head_dim + c);
                vv4 = *reinterpret_cast<const float4*>(vb + (size_t)gr * head_dim + c);
            }
            *reinterpret_cast<float4*>(&k_tile[r][c]) = kv4;
            *reinterpret_cast<float4*>(&v_tile[r][c]) = vv4;
        }
        __syncthreads();

        float sc = 0.f;
        for (int d = sub * 4; d < head_dim; d += TPK * 4) {
            float4 qv = *reinterpret_cast<const float4*>(&q_tile[d]);
            float4 kv = *reinterpret_cast<const float4*>(&k_tile[key][d]);
            sc += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        #pragma unroll
        for (int off = TPK / 2; off > 0; off >>= 1)
            sc += __shfl_xor_sync(0xFFFFFFFFu, sc, off);
        if (sub == 0) p[key] = (col0 + key < seq_len) ? sc : -INFINITY;
        __syncthreads();

        if (warp == 0) {
            float mx = -FLT_MAX;
            for (int t = lane; t < B; t += 32) mx = fmaxf(mx, p[t]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
            const float m_old = m_tile;
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
            for (int t = lane; t < B; t += 32) {
                float e = __expf(p[t] - m_new);
                p[t] = e;
                sum += e;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
            if (lane == 0) {
                const float a = __expf(m_old - m_new);
                alpha_tile = a;
                l_tile = l_tile * a + sum;
                m_tile = m_new;
            }
        }
        __syncthreads();

        if (tid < head_dim) {
            float acc = o_acc * alpha_tile;
            #pragma unroll 8
            for (int t = 0; t < B; ++t) acc += p[t] * v_tile[t][tid];
            o_acc = acc;
        }
    }

    if (tid < head_dim) ob[tid] = o_acc / l_tile;

    // ########################################################
}

torch::Tensor custom_flash_attention_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = k.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, 1, head_dim)
    auto q_bh = q.view({batch_size, 1, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, 1, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    
    auto out_bh = torch::zeros_like(q_bh, options);
    
    // ########################################################

    // TODO: launch the kernel
    constexpr int B = 32;
    flash_attention_decode_kernel<B><<<batch_size * num_heads, 128>>>(
        q_bh.data_ptr<float>(), k_bh.data_ptr<float>(), v_bh.data_ptr<float>(),
        out_bh.data_ptr<float>(), seq_len, head_dim, causal);
    
    // ########################################################

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, 1, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, 1, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, 1, hidden_dim});
    
    return out;
}

__global__ void update_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int current_pos,
    int max_seq_len,
    int hidden_dim
) {
    // ########################################################

    // TODO: Implement the update kv cache operation 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_dim) return;
    int b = idx / hidden_dim;
    int h = idx - b * hidden_dim;
    size_t dst = ((size_t)b * max_seq_len + current_pos) * hidden_dim + h;
    k_cache[dst] = k[idx];
    v_cache[dst] = v[idx];

    // ########################################################
}

void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, int current_pos) {
    auto options = k_cache.options();

    int batch_size = k_cache.size(0);
    int max_seq_len = k_cache.size(1);
    int hidden_dim = k_cache.size(2);
    
    int total_threads = batch_size * hidden_dim;

    int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks((total_threads + threads_per_block - 1) / threads_per_block);

    update_cache_kernel<<<blocks, threads>>>(
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        batch_size,
        current_pos,
        max_seq_len,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_decode", &custom_flash_attention_decode, "Custom FlashAttention in CUDA");
    m.def("update_kv_cache", &update_kv_cache, "Update KV cache in CUDA");
}
