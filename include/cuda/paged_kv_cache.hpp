#pragma once

#ifndef A11_PAGED_KV_CACHE_HPP
#define A11_PAGED_KV_CACHE_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>

constexpr size_t KV_CACHE_VRAM_GB = 8;
constexpr size_t BLOCK_TOKENS = 16;

struct PhysicalBlock {
    float* key_cache   = nullptr;
    float* value_cache = nullptr;
};

class PagedKVCacheManager {
public:
    struct Config {
        uint32_t num_layers     = 80;
        uint32_t head_dim       = 128;
        uint32_t max_sequences  = 64;
        size_t   vram_cap_gb    = KV_CACHE_VRAM_GB;
    };

    explicit PagedKVCacheManager(const Config& cfg);
    ~PagedKVCacheManager();

    PagedKVCacheManager(const PagedKVCacheManager&) = delete;
    PagedKVCacheManager& operator=(const PagedKVCacheManager&) = delete;
    PagedKVCacheManager(PagedKVCacheManager&&) noexcept;
    PagedKVCacheManager& operator=(PagedKVCacheManager&&) noexcept;

    std::vector<uint32_t> allocate_sequence(uint32_t seq_id);
    void append_token(uint32_t seq_id);
    void free_sequence(uint32_t seq_id);
    const std::vector<uint32_t>& get_block_table(uint32_t seq_id) const;

    size_t used_blocks() const { return num_used_blocks_; }
    size_t total_tokens() const { return total_tokens_; }
    size_t vram_usage_bytes() const;

private:
    Config config_;
    size_t block_size_elements_ = 0;
    size_t block_size_bytes_    = 0;

    std::vector<uint32_t> free_stack_;

    struct SeqState {
        bool active       = false;
        size_t num_tokens = 0;
        std::vector<uint32_t> block_table;
    };
    std::vector<SeqState> sequences_;
    std::vector<std::unique_ptr<PhysicalBlock>> pool_;
    size_t num_used_blocks_ = 0;
    size_t total_tokens_    = 0;

    size_t block_size_bytes() const;
    uint32_t alloc_block();
    void dealloc_block(uint32_t block_idx);
    void init_pool();
};

#endif // A11_PAGED_KV_CACHE_HPP
