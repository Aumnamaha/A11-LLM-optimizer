#include "catch.hpp"
#include "cuda/paged_kv_cache.hpp"
#include <iostream>

TEST_CASE("PagedKVCacheManager basic allocation and block table", "[paged-kv][cache]") {
    PagedKVCacheManager::Config cfg;
    cfg.num_layers = 8;
    cfg.head_dim   = 64;
    cfg.max_sequences = 16;

    PagedKVCacheManager mgr(cfg);

    // Allocate a sequence
    auto bt = mgr.allocate_sequence(0);
    REQUIRE(bt.size() == 1);
    REQUIRE(mgr.used_blocks() == 1);
    REQUIRE(mgr.total_tokens() == 0);

    // Append tokens — should stay in same block until BLOCK_TOKENS reached
    for (int i = 0; i < 15; ++i) {
        mgr.append_token(0);
    }
    REQUIRE(mgr.used_blocks() == 1);
    REQUIRE(mgr.total_tokens() == 15);

    // Append BLOCK_TOKENS-th token — fills the first block exactly, no new block yet
    mgr.append_token(0);
    REQUIRE(mgr.used_blocks() == 1);
    REQUIRE(mgr.total_tokens() == 16);
    auto bt2 = mgr.get_block_table(0);
    REQUIRE(bt2.size() == 1);

    // Append one more — overflows block, triggers new allocation
    mgr.append_token(0);
    REQUIRE(mgr.used_blocks() == 2);
    auto bt3 = mgr.get_block_table(0);
    REQUIRE(bt3.size() == 2);
}

TEST_CASE("PagedKVCacheManager free and reallocate", "[paged-kv][cache]") {
    PagedKVCacheManager::Config cfg;
    cfg.num_layers = 4;
    cfg.head_dim   = 32;
    cfg.max_sequences = 8;

    PagedKVCacheManager mgr(cfg);

    auto bt0 = mgr.allocate_sequence(0);
    REQUIRE(mgr.used_blocks() == 1);

    // Free sequence — blocks should return to free stack
    mgr.free_sequence(0);
    REQUIRE(mgr.used_blocks() == 0);

    // Reallocate same seq_id — should reuse freed block (O(1) from free stack)
    auto bt1 = mgr.allocate_sequence(0);
    REQUIRE(bt1.size() == 1);
    REQUIRE(mgr.used_blocks() == 1);
}

TEST_CASE("PagedKVCacheManager VRAM cap enforcement", "[paged-kv][cache]") {
    PagedKVCacheManager::Config cfg;
    cfg.num_layers = 2;
    cfg.head_dim   = 32;
    cfg.max_sequences = 64;
    cfg.vram_cap_gb = 1;

    PagedKVCacheManager mgr(cfg);

    size_t initial_used = mgr.used_blocks();
    int seq_count = 0;
    try {
        while (true) {
            mgr.allocate_sequence(static_cast<uint32_t>(seq_count));
            ++seq_count;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "[TEST] VRAM cap hit after " << seq_count
                  << " sequences: " << e.what() << std::endl;
    }

    REQUIRE(seq_count > 0);
    REQUIRE(mgr.used_blocks() == initial_used + static_cast<size_t>(seq_count));
}

TEST_CASE("PagedKVCacheManager multi-sequence fragmentation", "[paged-kv][cache]") {
    PagedKVCacheManager::Config cfg;
    cfg.num_layers = 4;
    cfg.head_dim   = 64;
    cfg.max_sequences = 16;

    PagedKVCacheManager mgr(cfg);

    // Allocate 3 sequences, each with multiple blocks (48 tokens = 3 blocks)
    for (int s = 0; s < 3; ++s) {
        mgr.allocate_sequence(static_cast<uint32_t>(s));
        for (int t = 0; t < 48; ++t) {
            mgr.append_token(static_cast<uint32_t>(s));
        }
    }

    REQUIRE(mgr.used_blocks() == 9);
    REQUIRE(mgr.total_tokens() == 144);

    // Free middle sequence — blocks go to free stack
    mgr.free_sequence(1);
    REQUIRE(mgr.used_blocks() == 6);
    REQUIRE(mgr.total_tokens() == 96);

    // Allocate new sequence — should reuse freed block from top of free stack
    auto bt = mgr.allocate_sequence(3);
    REQUIRE(bt.size() == 1);
    REQUIRE(mgr.used_blocks() == 7);
}

TEST_CASE("PagedKVCacheManager VRAM usage reporting", "[paged-kv][cache]") {
    PagedKVCacheManager::Config cfg;
    cfg.num_layers     = 2;
    cfg.head_dim       = 32;
    cfg.max_sequences  = 8;
    cfg.vram_cap_gb    = 1; // Small cap to avoid OOM in CI/test runs

    PagedKVCacheManager mgr(cfg);
    REQUIRE(mgr.vram_usage_bytes() == 0);

    mgr.allocate_sequence(0);
    size_t usage_after_alloc = mgr.vram_usage_bytes();
    REQUIRE(usage_after_alloc > 0);

    // Each block: key + value, each element is float (4 bytes)
    size_t expected_block_bytes = cfg.num_layers * cfg.head_dim * BLOCK_TOKENS * sizeof(float) * 2;
    REQUIRE(usage_after_alloc == expected_block_bytes);
}
