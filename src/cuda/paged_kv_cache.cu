#include "cuda/paged_kv_cache.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "[CUDA_ERROR] " << __FILE__ << ":"              \
                      << __LINE__ << " " << cudaGetErrorString(err)      \
                      << " (code " << err << ")" << std::endl;           \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__));\
        }                                                                \
    } while (0)

size_t PagedKVCacheManager::block_size_bytes() const {
    return config_.num_layers * config_.head_dim * BLOCK_TOKENS * sizeof(float);
}

PagedKVCacheManager::PagedKVCacheManager(const Config& cfg) : config_(cfg) {
    sequences_.resize(cfg.max_sequences);
    init_pool();
}

PagedKVCacheManager::~PagedKVCacheManager() {
    for (auto& blk : pool_) {
        if (blk->key_cache)   CUDA_CHECK(cudaFree(blk->key_cache));
        if (blk->value_cache) CUDA_CHECK(cudaFree(blk->value_cache));
    }
}

PagedKVCacheManager::PagedKVCacheManager(PagedKVCacheManager&& other) noexcept
    : config_(other.config_), block_size_bytes_(other.block_size_bytes_),
      free_stack_(std::move(other.free_stack_)), sequences_(std::move(other.sequences_)),
      pool_(std::move(other.pool_)), num_used_blocks_(other.num_used_blocks_),
      total_tokens_(other.total_tokens_) {
    other.num_used_blocks_ = 0;
    other.total_tokens_    = 0;
}

PagedKVCacheManager& PagedKVCacheManager::operator=(PagedKVCacheManager&& other) noexcept {
    if (this != &other) {
        for (auto& blk : pool_) {
            if (blk->key_cache)   CUDA_CHECK(cudaFree(blk->key_cache));
            if (blk->value_cache) CUDA_CHECK(cudaFree(blk->value_cache));
        }
        config_          = other.config_;
        block_size_bytes_ = other.block_size_bytes_;
        free_stack_      = std::move(other.free_stack_);
        sequences_       = std::move(other.sequences_);
        pool_            = std::move(other.pool_);
        num_used_blocks_ = other.num_used_blocks_;
        total_tokens_    = other.total_tokens_;
        other.num_used_blocks_ = 0;
        other.total_tokens_    = 0;
    }
    return *this;
}

void PagedKVCacheManager::init_pool() {
    size_t vram_cap_bytes = config_.vram_cap_gb * 1024ULL * 1024ULL * 1024ULL;
    size_t block_mem = block_size_bytes() * 2; // key + value
    if (block_mem == 0) block_mem = 1;
    size_t max_blocks = vram_cap_bytes / block_mem;

    pool_.resize(max_blocks);
    free_stack_.reserve(max_blocks);

    for (uint32_t i = 0; i < max_blocks; ++i) {
        auto blk = std::make_unique<PhysicalBlock>();
        CUDA_CHECK(cudaMalloc(&blk->key_cache,   block_size_bytes()));
        CUDA_CHECK(cudaMalloc(&blk->value_cache, block_size_bytes()));
        CUDA_CHECK(cudaMemset(blk->key_cache, 0, block_size_bytes()));
        CUDA_CHECK(cudaMemset(blk->value_cache, 0, block_size_bytes()));
        pool_[i] = std::move(blk);
        free_stack_.push_back(i);
    }

    std::cerr << "[PagedKVCache] Pool: " << max_blocks << " blocks, cap="
              << config_.vram_cap_gb << " GB" << std::endl;
}

uint32_t PagedKVCacheManager::alloc_block() {
    if (free_stack_.empty()) {
        throw std::runtime_error("PagedKVCache: out of physical blocks");
    }
    uint32_t idx = free_stack_.back();
    free_stack_.pop_back();
    ++num_used_blocks_;
    return idx;
}

void PagedKVCacheManager::dealloc_block(uint32_t block_idx) {
    if (block_idx >= pool_.size()) return;
    CUDA_CHECK(cudaMemset(pool_[block_idx]->key_cache, 0, block_size_bytes()));
    CUDA_CHECK(cudaMemset(pool_[block_idx]->value_cache, 0, block_size_bytes()));
    free_stack_.push_back(block_idx);
    --num_used_blocks_;
}

std::vector<uint32_t> PagedKVCacheManager::allocate_sequence(uint32_t seq_id) {
    if (seq_id >= sequences_.size()) {
        throw std::runtime_error("PagedKVCache: seq_id out of range");
    }
    if (sequences_[seq_id].active) {
        throw std::runtime_error("PagedKVCache: sequence already allocated");
    }

    uint32_t blk = alloc_block();
    sequences_[seq_id].active       = true;
    sequences_[seq_id].num_tokens   = 0;
    sequences_[seq_id].block_table  = {blk};

    std::cerr << "[PagedKVCache] Allocated seq " << seq_id << " blocks=" << sequences_[seq_id].block_table.size() << std::endl;
    return sequences_[seq_id].block_table;
}

void PagedKVCacheManager::append_token(uint32_t seq_id) {
    if (!sequences_[seq_id].active) {
        throw std::runtime_error("PagedKVCache: sequence not active");
    }

    auto& state = sequences_[seq_id];
    ++state.num_tokens;
    ++total_tokens_;

    if (state.num_tokens > state.block_table.size() * BLOCK_TOKENS) {
        if (free_stack_.empty()) {
            throw std::runtime_error("PagedKVCache: out of blocks during append");
        }
        uint32_t blk = alloc_block();
        state.block_table.push_back(blk);
    }
}

void PagedKVCacheManager::free_sequence(uint32_t seq_id) {
    if (seq_id >= sequences_.size() || !sequences_[seq_id].active) return;

    auto& state = sequences_[seq_id];
    for (uint32_t blk : state.block_table) {
        dealloc_block(blk);
    }
    total_tokens_ -= state.num_tokens;
    state.active       = false;
    state.num_tokens   = 0;
    state.block_table.clear();

    std::cerr << "[PagedKVCache] Freed seq " << seq_id << std::endl;
}

const std::vector<uint32_t>& PagedKVCacheManager::get_block_table(uint32_t seq_id) const {
    if (!sequences_[seq_id].active) {
        throw std::runtime_error("PagedKVCache: sequence not active");
    }
    return sequences_[seq_id].block_table;
}

size_t PagedKVCacheManager::vram_usage_bytes() const {
    return num_used_blocks_ * block_size_bytes() * 2;
}
