#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>

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

namespace A11 {
namespace Memory {

// ---- Constants ----
constexpr size_t HOT_EXPERT_CACHE_HOST_GB = 12;       // 12GB pinned host RAM
constexpr size_t EXPERT_WEIGHT_ALIGNMENT   = 256;      // Cache-line alignment for async copy
constexpr uint32_t DEFAULT_NUM_EXPERTS     = 8;        // Default MoE expert count

struct ExpertMetadata {
    uint32_t expert_id;
    size_t   weight_bytes;
    int32_t  access_count;       // LRU frequency counter
    float*   host_buffer;        // Pinned host RAM buffer (nullptr if not loaded)
    float*   device_buffer;      // VRAM copy (nullptr if not in VRAM)
    bool     pinned_allocated;   // Whether cudaHostAlloc was called for this expert
};

class HotExpertCache {
public:
    struct Config {
        uint32_t num_experts       = DEFAULT_NUM_EXPERTS;
        size_t   expert_weight_bytes = 4 * 1024 * 1024; // 4MB per expert (default)
        size_t   host_cache_gb     = HOT_EXPERT_CACHE_HOST_GB;
        int      async_stream_count = 2;                // CUDA streams for pipelining
    };

    explicit HotExpertCache(const Config& cfg);
    ~HotExpertCache();

    HotExpertCache(const HotExpertCache&) = delete;
    HotExpertCache& operator=(const HotExpertCache&) = delete;

    /// Pin the entire host RAM cache buffer using cudaHostAlloc (4KB page aligned).
    bool allocate_pinned_cache();

    /// Swap a specific expert from host RAM to VRAM via async copy.
    void swap_expert_to_device(uint32_t expert_id);

    /// Swap an expert from VRAM back to host RAM.
    void swap_expert_to_host(uint32_t expert_id);

    /// Get the device buffer pointer for a given expert (swap if needed).
    float* get_expert_device_buffer(uint32_t expert_id);

    /// Record an access to an expert for LRU tracking.
    void record_access(uint32_t expert_id);

    /// Evict least-recently-used experts from VRAM to make room.
    void evict_lru_experts(size_t min_free_bytes = 0);

    /// Get current memory usage statistics.
    struct MemoryStats {
        size_t host_pinned_bytes   = 0;
        size_t device_used_bytes   = 0;
        size_t device_total_bytes  = 0;
        size_t active_experts      = 0; // Experts currently in VRAM
        size_t total_experts       = 0;
    };

    MemoryStats get_memory_stats() const;

    /// Get the CUDA stream for async operations (round-robin).
    cudaStream_t get_async_stream();

private:
    Config config_;
    std::vector<ExpertMetadata> experts_;
    std::vector<cudaStream_t> streams_;
    size_t total_host_pinned_bytes_ = 0;
    size_t total_device_used_bytes_ = 0;

    /// LRU eviction using access frequency + timestamp.
    uint32_t find_lru_expert() const;

    /// Reallocate pinned memory for a single expert if needed.
    void ensure_host_allocation(ExpertMetadata& meta);
};

// ---- Implementation ----

HotExpertCache::HotExpertCache(const Config& cfg)
    : config_(cfg), streams_{} {

    experts_.resize(config_.num_experts);

    // Create async CUDA streams for pipelining expert swaps
    for (uint32_t i = 0; i < static_cast<uint32_t>(config_.async_stream_count); ++i) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        streams_.push_back(stream);
    }

    std::cerr << "[HotExpertCache] Created with " << config_.num_experts
              << " experts, " << config_.expert_weight_bytes << " bytes/expert"
              << ", " << config_.async_stream_count << " async streams" << std::endl;
}

HotExpertCache::~HotExpertCache() {
    for (auto& exp : experts_) {
        if (exp.device_buffer) CUDA_CHECK(cudaFree(exp.device_buffer));
        if (exp.host_buffer && exp.pinned_allocated) {
            CUDA_CHECK(cudaFreeHost(exp.host_buffer));
        }
    }

    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }

    std::cerr << "[HotExpertCache] Freed all expert buffers" << std::endl;
}

bool HotExpertCache::allocate_pinned_cache() {
    size_t total_bytes = config_.num_experts * config_.expert_weight_bytes;
    total_host_pinned_bytes_ = ((total_bytes + 1024 * 1024 - 1) / (1024 * 1024))
                               * (1024 * 1024); // Round up to MB

    std::cerr << "[HotExpertCache] Allocating " << total_host_pinned_bytes_ / (1024*1024)
              << " MB pinned host RAM via cudaHostAlloc..." << std::endl;

    CUDA_CHECK(cudaSetDevice(0));

    // Allocate contiguous pinned memory for all experts
    float* base_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&base_ptr),
                              total_host_pinned_bytes_,
                              cudaHostAllocDefault));

    if (!base_ptr) {
        std::cerr << "[HotExpertCache ERROR] cudaHostAlloc failed for "
                  << total_host_pinned_bytes_ / (1024*1024) << " MB" << std::endl;
        return false;
    }

    // Split contiguous allocation into per-expert buffers
    size_t expert_stride = (config_.expert_weight_bytes + EXPERT_WEIGHT_ALIGNMENT - 1)
                           / EXPERT_WEIGHT_ALIGNMENT * EXPERT_WEIGHT_ALIGNMENT;

    for (uint32_t i = 0; i < config_.num_experts; ++i) {
        experts_[i].host_buffer      = base_ptr + i * expert_stride / sizeof(float);
        experts_[i].pinned_allocated = true;
        experts_[i].access_count     = 0;
    }

    std::cerr << "[HotExpertCache] Pinned cache allocated: "
              << total_host_pinned_bytes_ / (1024*1024) << " MB" << std::endl;
    return true;
}

void HotExpertCache::ensure_host_allocation(ExpertMetadata& meta) {
    if (!meta.host_buffer && meta.pinned_allocated) {
        // This should not happen after allocate_pinned_cache(), but handle defensively
        size_t aligned_size = (meta.weight_bytes + EXPERT_WEIGHT_ALIGNMENT - 1)
                              / EXPERT_WEIGHT_ALIGNMENT * EXPERT_WEIGHT_ALIGNMENT;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&meta.host_buffer),
                                  aligned_size, cudaHostAllocDefault));
    }
}

void HotExpertCache::swap_expert_to_device(uint32_t expert_id) {
    if (expert_id >= config_.num_experts) {
        throw std::runtime_error("HotExpertCache: expert_id out of range");
    }

    ExpertMetadata& meta = experts_[expert_id];
    ensure_host_allocation(meta);

    // Allocate device memory if not already done
    if (!meta.device_buffer) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&meta.device_buffer),
                              meta.weight_bytes));
        total_device_used_bytes_ += meta.weight_bytes;
    }

    // Async copy from pinned host to device VRAM
    cudaStream_t stream = get_async_stream();
    CUDA_CHECK(cudaMemcpyAsync(meta.device_buffer,
                                meta.host_buffer,
                                meta.weight_bytes,
                                cudaMemcpyHostToDevice,
                                stream));

    std::cerr << "[HotExpertCache] Swapped expert " << expert_id
              << " to device (" << meta.weight_bytes / 1024 << " KB)" << std::endl;
}

void HotExpertCache::swap_expert_to_host(uint32_t expert_id) {
    if (expert_id >= config_.num_experts) {
        throw std::runtime_error("HotExpertCache: expert_id out of range");
    }

    ExpertMetadata& meta = experts_[expert_id];

    if (!meta.device_buffer) return; // Not in VRAM, nothing to evict

    cudaStream_t stream = get_async_stream();
    CUDA_CHECK(cudaMemcpyAsync(meta.host_buffer,
                                meta.device_buffer,
                                meta.weight_bytes,
                                cudaMemcpyDeviceToHost,
                                stream));

    total_device_used_bytes_ -= meta.weight_bytes;
    std::cerr << "[HotExpertCache] Swapped expert " << expert_id
              << " back to host" << std::endl;
}

float* HotExpertCache::get_expert_device_buffer(uint32_t expert_id) {
    if (expert_id >= config_.num_experts) {
        throw std::runtime_error("HotExpertCache: expert_id out of range");
    }

    ExpertMetadata& meta = experts_[expert_id];

    if (!meta.device_buffer) {
        swap_expert_to_device(expert_id);
    }

    record_access(expert_id);
    return meta.device_buffer;
}

void HotExpertCache::record_access(uint32_t expert_id) {
    if (expert_id >= config_.num_experts) return;
    experts_[expert_id].access_count++;
}

uint32_t HotExpertCache::find_lru_expert() const {
    uint32_t lru_id = 0;
    int32_t min_access = std::numeric_limits<int32_t>::max();

    for (uint32_t i = 0; i < config_.num_experts; ++i) {
        if (!experts_[i].device_buffer) continue; // Not in VRAM, skip
        if (experts_[i].access_count < min_access) {
            min_access = experts_[i].access_count;
            lru_id = i;
        }
    }

    return lru_id;
}

void HotExpertCache::evict_lru_experts(size_t min_free_bytes) {
    // Find how much VRAM we need to free
    size_t current_used = total_device_used_bytes_;
    size_t bytes_to_free = (min_free_bytes > 0 && min_free_bytes > current_used)
                           ? min_free_bytes - current_used : 0;

    if (bytes_to_free == 0 && current_used == 0) return;

    std::vector<bool> evicted(config_.num_experts, false);

    while ((min_free_bytes > 0 && total_device_used_bytes_ > min_free_bytes) ||
           (min_free_bytes == 0 && bytes_to_free > 0)) {

        uint32_t lru = find_lru_expert();
        if (!experts_[lru].device_buffer) break; // No more to evict

        size_t expert_size = experts_[lru].weight_bytes;

        // Copy back to host first (async, but we sync for correctness in eviction)
        cudaStream_t stream = get_async_stream();
        CUDA_CHECK(cudaMemcpyAsync(experts_[lru].host_buffer,
                                    experts_[lru].device_buffer,
                                    expert_size,
                                    cudaMemcpyDeviceToHost,
                                    stream));

        // Free device memory
        CUDA_CHECK(cudaFree(experts_[lru].device_buffer));
        total_device_used_bytes_ -= expert_size;
        bytes_to_free -= expert_size;
        experts_[lru].device_buffer = nullptr;

        std::cerr << "[HotExpertCache] Evicted LRU expert " << lru
                  << " (" << expert_size / 1024 << " KB)" << std::endl;

        if (evicted[lru]) break; // Safety: avoid infinite loop
        evicted[lru] = true;
    }

    // Sync to ensure all async copies complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

HotExpertCache::MemoryStats HotExpertCache::get_memory_stats() const {
    MemoryStats stats;
    stats.host_pinned_bytes   = total_host_pinned_bytes_;
    stats.device_used_bytes   = total_device_used_bytes_;

    // Use cudaMemGetInfo for broader CUDA version compatibility
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    stats.device_total_bytes = total_mem;

    for (const auto& exp : experts_) {
        if (exp.device_buffer) ++stats.active_experts;
    }
    stats.total_experts = config_.num_experts;
    return stats;
}

cudaStream_t HotExpertCache::get_async_stream() {
    // Round-robin stream selection for load balancing
    static thread_local uint32_t stream_idx = 0;
    cudaStream_t stream = streams_[stream_idx % streams_.size()];
    ++stream_idx;
    return stream;
}

// ---- External API (C-compatible) ----

extern "C" {

/// Create a new HotExpertCache instance.
void* hot_expert_cache_create(size_t num_experts, size_t expert_weight_bytes,
                               size_t host_cache_gb, int async_streams) {
    HotExpertCache::Config cfg;
    cfg.num_experts       = static_cast<uint32_t>(num_experts);
    cfg.expert_weight_bytes = expert_weight_bytes;
    cfg.host_cache_gb     = host_cache_gb;
    cfg.async_stream_count = async_streams;

    auto* cache = new HotExpertCache(cfg);
    return static_cast<void*>(cache);
}

/// Allocate pinned host RAM for the cache.
int hot_expert_cache_allocate(void* handle) {
    try {
        auto* cache = static_cast<HotExpertCache*>(handle);
        return cache->allocate_pinned_cache() ? 0 : -1;
    } catch (const std::exception& e) {
        std::cerr << "[hot_expert_cache_allocate] Exception: " << e.what() << std::endl;
        return -1;
    }
}

/// Get device buffer for an expert, swapping from host if needed.
float* hot_expert_cache_get_buffer(void* handle, uint32_t expert_id) {
    try {
        auto* cache = static_cast<HotExpertCache*>(handle);
        return cache->get_expert_device_buffer(expert_id);
    } catch (const std::exception& e) {
        std::cerr << "[hot_expert_cache_get_buffer] Exception: " << e.what() << std::endl;
        return nullptr;
    }
}

/// Record expert access for LRU tracking.
void hot_expert_cache_record_access(void* handle, uint32_t expert_id) {
    try {
        auto* cache = static_cast<HotExpertCache*>(handle);
        cache->record_access(expert_id);
    } catch (const std::exception& e) {
        std::cerr << "[hot_expert_cache_record_access] Exception: " << e.what() << std::endl;
    }
}

/// Evict LRU experts from VRAM.
void hot_expert_cache_evict_lru(void* handle, size_t min_free_bytes) {
    try {
        auto* cache = static_cast<HotExpertCache*>(handle);
        cache->evict_lru_experts(min_free_bytes);
    } catch (const std::exception& e) {
        std::cerr << "[hot_expert_cache_evict_lru] Exception: " << e.what() << std::endl;
    }
}

/// Get memory usage statistics.
void hot_expert_cache_get_stats(void* handle, size_t* host_bytes,
                                 size_t* dev_used, size_t* dev_total,
                                 uint32_t* active, uint32_t* total) {
    try {
        auto* cache = static_cast<HotExpertCache*>(handle);
        auto stats = cache->get_memory_stats();
        if (host_bytes)   *host_bytes   = stats.host_pinned_bytes;
        if (dev_used)     *dev_used     = stats.device_used_bytes;
        if (dev_total)    *dev_total    = stats.device_total_bytes;
        if (active)       *active       = stats.active_experts;
        if (total)        *total        = stats.total_experts;
    } catch (const std::exception& e) {
        std::cerr << "[hot_expert_cache_get_stats] Exception: " << e.what() << std::endl;
    }
}

/// Destroy the cache and free all resources.
void hot_expert_cache_destroy(void* handle) {
    delete static_cast<HotExpertCache*>(handle);
}

} // extern "C"

} // namespace Memory
} // namespace A11
