#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>
//#include <cooperative_groups>
using namespace torch::indexing;
//#define BLOCK_SIZE 1024
//we can just have each block size be the row size
#define warpSize 32





__device__ __inline__ float warpReduceMax(float val){
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFU, val, offset));
    return val;
}

__device__ __inline__ float warpReduceSum(float val){
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFFU, val, offset);
    return val;
}

__global__ void softmax_kernel_batched(float *input, float *output, int NUM_ROW, int NUM_COL)
{
    // ########################################################

    extern __shared__ float buffer[];

    int b = blockIdx.x;
    int row = blockIdx.y;
    int tx = threadIdx.x;
    if (row >= NUM_ROW) return;

    int lane = tx % warpSize;
    int warpID = tx / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    size_t offset = ((size_t)b * NUM_ROW + row) * NUM_COL;
    float *in_row = input + offset;
    float *out_row = output + offset;

    float val = -FLT_MAX;
    for (int j = tx; j < NUM_COL; j += blockDim.x)
        val = fmaxf(val, in_row[j]);

    val = warpReduceMax(val);
    if (lane == 0) buffer[warpID] = val;
    __syncthreads();
    if (warpID == 0) {
        val = (lane < numWarps) ? buffer[lane] : -FLT_MAX;
        val = warpReduceMax(val);
        if (lane == 0) buffer[0] = val;
    }
    __syncthreads();
    float row_max = buffer[0];
    __syncthreads();

    val = 0.0f;
    for (int j = tx; j < NUM_COL; j += blockDim.x)
        val += expf(in_row[j] - row_max);

    val = warpReduceSum(val);
    if (lane == 0) buffer[warpID] = val;
    __syncthreads();
    if (warpID == 0) {
        val = (lane < numWarps) ? buffer[lane] : 0.0f;
        val = warpReduceSum(val);
        if (lane == 0) buffer[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / buffer[0];

    for (int j = tx; j < NUM_COL; j += blockDim.x)
        out_row[j] = expf(in_row[j] - row_max) * inv_sum;

    // ########################################################
}

template <int TILE_SIZE>
__global__ void GEMM_NT_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // ########################################################

    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float *A = a_mat + (size_t)batch * M * K;
    float *B = b_mat + (size_t)batch * N * K;
    float *C = out_mat + (size_t)batch * M * N;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        int a_k = t * TILE_SIZE + tx;
        a_tile[ty][tx] = (row < M && a_k < K) ? A[(size_t)row * K + a_k] : 0.0f;

        int b_n = blockIdx.x * TILE_SIZE + ty;
        int b_k = t * TILE_SIZE + tx;
        b_tile[tx][ty] = (b_n < N && b_k < K) ? B[(size_t)b_n * K + b_k] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += a_tile[ty][k] * b_tile[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * N + col] = acc;

    // ########################################################
}

template <int TILE_SIZE>
__global__ void GEMM_NN_kernel_batched(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // ########################################################

    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float *A = a_mat + (size_t)batch * M * K;
    float *B = b_mat + (size_t)batch * K * N;
    float *C = out_mat + (size_t)batch * M * N;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        int a_k = t * TILE_SIZE + tx;
        a_tile[ty][tx] = (row < M && a_k < K) ? A[(size_t)row * K + a_k] : 0.0f;

        int b_k = t * TILE_SIZE + ty;
        b_tile[ty][tx] = (b_k < K && col < N) ? B[(size_t)b_k * N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += a_tile[ty][k] * b_tile[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * N + col] = acc;

    // ########################################################
}

__global__ void scale_and_causal_mask_batched(float *mat, int rows, int cols, float scale, bool causal)
{
    int b = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= rows) return;

    int stride = rows * cols;
    float *row_ptr = mat + b * stride + row * cols;
    for (int j = tid; j < cols; j += blockDim.x)
    {
        float val = row_ptr[j] * scale;
        if (causal && j > row)
        {
            val = -FLT_MAX;
        }
        row_ptr[j] = val;
    }
}

torch::Tensor custom_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal)
{
    auto options = q.options();

    const int TILE_SIZE = 16;

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;

    // Reshape to (batch_size * num_heads, seq_len, head_dim) for parallel per-head processing
    auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, seq_len, head_dim});

    auto qk = torch::empty({batch_size * num_heads, seq_len, seq_len}, options);
    auto s = torch::empty_like(qk);
    auto o_bh = torch::empty({batch_size * num_heads, seq_len, head_dim}, options);

    // ########################################################

    int BH = batch_size * num_heads;
    dim3 gemm_threads(TILE_SIZE, TILE_SIZE);

    // TODO: Run QK^T batched
    dim3 qk_blocks((seq_len + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE, BH);
    GEMM_NT_kernel_batched<TILE_SIZE><<<qk_blocks, gemm_threads>>>(
        q_bh.data_ptr<float>(), k_bh.data_ptr<float>(), qk.data_ptr<float>(), seq_len, seq_len, head_dim);
                
    // TODO: Run Scale + causal mask batched (over BH, rows)
    float scale = 1.0f / sqrtf((float)head_dim);
    scale_and_causal_mask_batched<<<dim3(BH, seq_len), dim3(256)>>>(
        qk.data_ptr<float>(), seq_len, seq_len, scale, causal);
    
    // TODO: Run Softmax batched along last dim
    softmax_kernel_batched<<<dim3(BH, seq_len), dim3(256), 256 * sizeof(float)>>>(
        qk.data_ptr<float>(), s.data_ptr<float>(), seq_len, seq_len);
    
    // TODO: Run (S @ V) batched
    dim3 sv_blocks((head_dim + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE, BH);
    GEMM_NN_kernel_batched<TILE_SIZE><<<sv_blocks, gemm_threads>>>(
        s.data_ptr<float>(), v_bh.data_ptr<float>(), o_bh.data_ptr<float>(), seq_len, head_dim, seq_len);
    
    // ########################################################
    
    // Reshape back to (batch_size, seq_len, hidden_dim)
    auto o = o_bh.view({batch_size, num_heads, seq_len, head_dim}).permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, hidden_dim});
    return o;
}

torch::Tensor test_batched_softmax(torch::Tensor inp, int dim) {
    int N_DIM = inp.dim();
    if (dim < 0) {
        dim += N_DIM;
    }
    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
    }
    // Ensure contiguous memory layout for linear indexing in the CUDA kernel
    inp = inp.contiguous();

    auto inp_sizes = inp.sizes();
    int N_BATCH = inp_sizes[0];
    int N_ROW = inp_sizes[1];
    int N_COL = inp_sizes[2];
    dim3 threads_per_block(256);
    dim3 blocks_per_grid(N_BATCH, N_ROW);
    auto outp = torch::empty(inp_sizes, inp.options());
    
    // Allocate shared memory for each thread in the block

    //I want to do a cooperative groupm launch here
    softmax_kernel_batched<<<blocks_per_grid, threads_per_block, threads_per_block.x * sizeof(float)>>>(
        inp.data_ptr<float>(), outp.data_ptr<float>(), N_ROW, N_COL);
    
    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
        outp = outp.transpose(dim, N_DIM - 1);
    }

    return outp;
}

torch::Tensor test_batched_GEMM_NN(torch::Tensor a, torch::Tensor b)
{
    a = a.contiguous();
    b = b.contiguous();
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[2], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NN_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

torch::Tensor test_batched_GEMM_NT(torch::Tensor a, torch::Tensor b)
{
    a = a.contiguous();
    b = b.contiguous();
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[1], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NT_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_batched_softmax", &test_batched_softmax, "Testing interface for custom softmax in CUDA");
    m.def("test_batched_GEMM_NN", &test_batched_GEMM_NN, "Testing interface for custom matmul (A @ B) in CUDA");
    m.def("test_batched_GEMM_NT", &test_batched_GEMM_NT, "Testing interface for custom matmul (A @ B^T) in CUDA");
    m.def("custom_attention", &custom_attention, "Custom attention in CUDA");
}