#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <iostream>
#include <stdexcept>

// CUDA error checking macro (consistent with paged_kv_cache.cu)
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "[CUDA_ERROR] " << __FILE__ << ":"              \
                      << __LINE__ << " " << cudaGetErrorString(err)      \
                      << " (code " << err << ")" << std::endl;           \
            throw std::runtime_error(std::string("CUDA error at ") +     \
                std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        }                                                                \
    } while (0)

// K=8 draft tokens per speculation step
constexpr uint32_t SPEC_K = 8;

// ---- Device-side constants ----
static constexpr int MAX_DRAFT_DEPTH = SPEC_K;
static constexpr int MAX_SEQ_LEN     = 4096;

// ---- Kernel: Tree-aware parallel forward pass for K candidates ----
// This kernel evaluates all K candidate tokens from the draft tree in a
// single attention pass by processing them as a batched sequence.
__global__ void tree_forward_pass_kernel(
    const float* __restrict__ embedding_table,  // [vocab_size x hidden_dim]
    const float* __restrict__ context_hidden,   // [seq_len x hidden_dim], current context
    const int32_t* __restrict__ draft_tokens,   // [K], drafted token ids (linearized)
    const uint32_t* __restrict__ tree_positions,  // [K], position in context for each draft
    float* __restrict__ candidate_logits,       // [K x vocab_size] output logits
    int hidden_dim,
    int vocab_size,
    int seq_len) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = SPEC_K * vocab_size;

    if (tid >= total_tokens) return;

    // Each thread handles one (draft_candidate, vocab_position) pair
    int candidate_idx  = tid / vocab_size;
    int vocab_pos      = tid % vocab_size;

    if (candidate_idx >= SPEC_K || vocab_pos >= vocab_size) return;

    // Load the hidden state at this draft position
    int ctx_pos = tree_positions[candidate_idx];
    const float* h_state = context_hidden + ctx_pos * hidden_dim;

    // Compute logit for this vocabulary position via embedding dot product
    float logit = 0.0f;
    #pragma unroll
    for (int d = 0; d < hidden_dim; ++d) {
        logit += h_state[d] * embedding_table[vocab_pos * hidden_dim + d];
    }

    // Store logits (unnormalized, softmax applied on host or in next kernel)
    candidate_logits[tid] = logit / sqrtf(static_cast<float>(hidden_dim));
}

// ---- Kernel: Rejection sampling for tree verification ----
__global__ void tree_rejection_kernel(
    const float* __restrict__ draft_log_probs,   // [K] from draft model
    const float* __restrict__ target_log_probs,  // [K] from target model (pre-softmax)
    int32_t* __restrict__ acceptance_flags,      // [K], 1=accepted, 0=rejected
    uint32_t* __restrict__ accepted_count,       // atomic output: total consecutive accepted
    float rng_seed) {

    int k = threadIdx.x;
    if (k >= SPEC_K) return;

    // Compute acceptance probability: min(1, exp(target - draft))
    float log_accept_ratio = target_log_probs[k] - draft_log_probs[k];
    float accept_prob = fminf(1.0f, expf(fmaxf(log_accept_ratio, -10.0f)));

    // Deterministic pseudo-random for reproducibility in testing
    float r = fmodf(sin(rng_seed + static_cast<float>(k) * 127.1f) * 43758.5453f, 1.0f);
    if (r < 0.0f) r += 1.0f;

    acceptance_flags[k] = (r < accept_prob) ? 1 : 0;

    // Atomic: compute consecutive accepted count from the start
    uint32_t count = 0;
    for (int i = 0; i <= k; ++i) {
        if (acceptance_flags[i] == 1) {
            count++;
        } else {
            break; // Stop at first rejection
        }
    }
    atomicMax(accepted_count, count);
}

// ---- Host-side API: Launch tree verification kernel pair ----

/**
 * Execute parallel tree verification for K draft tokens.
 */
void tree_verify_kernel_launch(
    const float* d_embedding_table,
    const float* d_context_hidden,
    const int32_t* d_draft_tokens,
    const uint32_t* d_tree_positions,
    float* d_candidate_logits,
    const float* d_draft_log_probs,
    const float* d_target_log_probs,
    int32_t* d_acceptance_flags,
    uint32_t* d_accepted_count,
    int hidden_dim,
    int vocab_size,
    int seq_len,
    float rng_seed = 42.0f) {

    // Step 1: Parallel forward pass to compute logits for all K candidates
    int threads_per_block = 256;
    int total_logits = SPEC_K * vocab_size;
    int num_blocks   = (total_logits + threads_per_block - 1) / threads_per_block;

    tree_forward_pass_kernel<<<num_blocks, threads_per_block>>>(
        d_embedding_table, d_context_hidden, d_draft_tokens,
        d_tree_positions, d_candidate_logits, hidden_dim, vocab_size, seq_len);

    CUDA_CHECK(cudaGetLastError());

    // Step 2: Rejection sampling on K candidates
    tree_rejection_kernel<<<1, SPEC_K>>>(
        d_draft_log_probs, d_target_log_probs,
        d_acceptance_flags, d_accepted_count, rng_seed);

    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure kernel completion before host reads results
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Host-side wrapper for tree verification with memory management.
 * Allocates device buffers, launches kernels, and copies results back.
 */
uint32_t tree_verify(
    const int32_t* h_draft_tokens,
    const uint32_t* h_tree_positions,
    const float* h_draft_log_probs,
    const float* h_target_log_probs,
    const float* d_embedding_table,
    const float* d_context_hidden,
    int hidden_dim,
    int vocab_size,
    int seq_len) {

    int32_t*  d_draft_tokens     = nullptr;
    uint32_t* d_tree_positions   = nullptr;
    float*    d_candidate_logits = nullptr;
    float*    d_draft_log_p      = nullptr;
    float*    d_target_log_p     = nullptr;
    int32_t*  d_accept_flags     = nullptr;
    uint32_t* d_accepted_count   = nullptr;

    // Allocate device buffers
    CUDA_CHECK(cudaMalloc(&d_draft_tokens,       SPEC_K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tree_positions,      SPEC_K * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_logits,    SPEC_K * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft_log_p,         SPEC_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target_log_p,        SPEC_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accept_flags,        SPEC_K * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_accepted_count,      sizeof(uint32_t)));

    // Zero the accepted count
    uint32_t h_zero = 0;
    CUDA_CHECK(cudaMemcpy(d_accepted_count, &h_zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_draft_tokens,       h_draft_tokens,      SPEC_K * sizeof(int32_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree_positions,      h_tree_positions,    SPEC_K * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_log_p,         h_draft_log_probs,   SPEC_K * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_log_p,        h_target_log_probs,  SPEC_K * sizeof(float),   cudaMemcpyHostToDevice));

    // Launch verification kernels
    tree_verify_kernel_launch(
        d_embedding_table, d_context_hidden,
        d_draft_tokens, d_tree_positions,
        d_candidate_logits,
        d_draft_log_p, d_target_log_p,
        d_accept_flags, d_accepted_count,
        hidden_dim, vocab_size, seq_len);

    // Read back results
    uint32_t accepted = 0;
    CUDA_CHECK(cudaMemcpy(&accepted, d_accepted_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Cleanup device allocations
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_tree_positions));
    CUDA_CHECK(cudaFree(d_candidate_logits));
    CUDA_CHECK(cudaFree(d_draft_log_p));
    CUDA_CHECK(cudaFree(d_target_log_p));
    CUDA_CHECK(cudaFree(d_accept_flags));
    CUDA_CHECK(cudaFree(d_accepted_count));

    return accepted;
}
