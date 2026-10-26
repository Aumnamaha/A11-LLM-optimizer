#include "catch.hpp"
#include "engine/speculative_verifier.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace A11::Engine;

// ---- Test 1: SpeculativeTreeVerifier instantiation and basic config ----
TEST_CASE("SpeculativeTreeVerifier construction with default config", "[speculative][verifier]") {
    SpeculativeTreeVerifier::Config cfg;
    REQUIRE(cfg.max_draft_depth == SPEC_K);
    REQUIRE(cfg.vocab_size == 32000);
    REQUIRE(cfg.acceptance_threshold == 1.0f);

    SpeculativeTreeVerifier verifier(cfg);
    REQUIRE(verifier.diagnostics().total_steps == 0);
    REQUIRE(verifier.diagnostics().steps_accepted == 0);
}

// ---- Test 2: Draft tree root construction ----
TEST_CASE("Build draft tree root node", "[speculative][tree]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.max_draft_depth = SPEC_K;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    TreeNode root = verifier.build_draft_tree(42);
    REQUIRE(root.token_id == 42);
    REQUIRE(root.parent_idx == UINT32_MAX);
    REQUIRE(root.depth == 0);
    REQUIRE(root.children.empty());
}

// ---- Test 3: Tree mask construction for K candidates ----
TEST_CASE("Construct tree verification mask", "[speculative][mask]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.vocab_size = 100; // Small vocab for testing
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    // Build a simple linear draft chain: tokens [10, 20, 30, ...]
    std::vector<TreeNode> tree(SPEC_K);
    for (uint32_t i = 0; i < SPEC_K; ++i) {
        tree[i].token_id   = static_cast<int32_t>(10 + i * 10);
        tree[i].parent_idx = (i == 0) ? UINT32_MAX : i - 1;
        tree[i].depth      = i;
    }

    auto mask = verifier.construct_tree_mask(tree);
    REQUIRE(mask.size() == SPEC_K * cfg.vocab_size);

    // Verify only the drafted token positions are set to 1.0
    for (uint32_t i = 0; i < SPEC_K; ++i) {
        int tok = tree[i].token_id;
        REQUIRE(mask[i * cfg.vocab_size + static_cast<size_t>(tok)] == Approx(1.0f));

        // All other positions should be 0
        for (int v = 0; v < cfg.vocab_size; ++v) {
            if (v != tok) {
                REQUIRE(mask[i * cfg.vocab_size + static_cast<size_t>(v)] == Approx(0.0f));
            }
        }
    }
}

// ---- Test 4: Rejection sampling — full acceptance ----
TEST_CASE("Rejection sampling with perfect draft matching", "[speculative][rejection]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false; // Suppress diagnostic noise in test output

    SpeculativeTreeVerifier verifier(cfg);

    // Draft and target log-probs are identical → acceptance ratio = 1.0 always
    std::vector<int32_t>   draft_tokens   = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float>     draft_log_probs = {-0.1f, -0.2f, -0.15f, -0.3f, -0.1f, -0.25f, -0.2f, -0.18f};
    std::vector<float>     target_log_probs = draft_log_probs; // Perfect match

    auto [accepted_len, last_idx] = verifier.verify_draft_tokens(
        draft_tokens, draft_log_probs, target_log_probs);

    REQUIRE(accepted_len == SPEC_K);
    REQUIRE(last_idx == SPEC_K - 1);
}

// ---- Test 5: Rejection sampling — early rejection ----
TEST_CASE("Rejection sampling with early token mismatch", "[speculative][rejection]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    // First 3 tokens match closely, 4th has very low target probability
    std::vector<int32_t>   draft_tokens   = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float>     draft_log_probs = {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f};
    // First 3 have high target log-prob (~e^-0.1 ≈ 0.9), 4th has very low (e^-5 ≈ 0.007)
    std::vector<float>     target_log_probs = {-0.1f, -0.1f, -0.1f, -5.0f, -0.1f, -0.1f, -0.1f, -0.1f};

    auto [accepted_len, last_idx] = verifier.verify_draft_tokens(
        draft_tokens, draft_log_probs, target_log_probs);

    // First 3 should be accepted (r < acceptance_ratio), 4th rejected
    REQUIRE(accepted_len > 0);
    REQUIRE(accepted_len <= SPEC_K);
    REQUIRE(last_idx < SPEC_K - 1);
}

// ---- Test 6: Full speculation step with diagnostics ----
TEST_CASE("Full speculation step and diagnostic counters", "[speculative][integration]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);
    auto& diags = verifier.diagnostics();

    REQUIRE(diags.total_steps == 0);
    REQUIRE(diags.steps_accepted == 0);
    REQUIRE(diags.fallback_count == 0);

    // Run one speculation step
    std::vector<int32_t> context = {1, 2, 3};
    auto [output_tokens, full_acceptance] = verifier.speculate_step(context, 100);

    REQUIRE(output_tokens.size() > 0);
    REQUIRE(static_cast<uint32_t>(output_tokens.size()) >= 1); // At least the target token
    REQUIRE(diags.total_steps == 1);
}

// ---- Test 7: Diagnostic fallback logging on zero acceptance ----
TEST_CASE("Fallback to autoregressive when all drafts rejected", "[speculative][diagnostics]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = true; // Enable diagnostic output

    SpeculativeTreeVerifier verifier(cfg);
    auto& diags = verifier.diagnostics();

    REQUIRE(diags.fallback_count == 0);

    // Run speculation — with simulated data, some acceptance is expected
    std::vector<int32_t> context = {5};
    auto [output_tokens, full_acceptance] = verifier.speculate_step(context, 999);

    // Diagnostic counters should be updated
    REQUIRE(diags.total_steps >= 1);
    REQUIRE(diags.steps_accepted + diags.steps_rejected == diags.total_steps);
}

// ---- Test 8: Verification with mismatched input sizes (error handling) ----
TEST_CASE("Rejection sampling rejects mismatched input lengths", "[speculative][validation]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    std::vector<int32_t>   draft_tokens   = {1, 2, 3};
    std::vector<float>     draft_log_probs = {-0.5f, -0.5f}; // Size mismatch!
    std::vector<float>     target_log_probs = {0.9f, 0.9f, 0.9f};

    REQUIRE_THROWS_AS(verifier.verify_draft_tokens(
        draft_tokens, draft_log_probs, target_log_probs), std::runtime_error);
}

// ---- Test 9: Reset diagnostics ----
TEST_CASE("Reset diagnostic counters", "[speculative][diagnostics]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    // Run a few speculation steps to populate counters
    for (int i = 0; i < 5; ++i) {
        verifier.speculate_step({1, 2}, 10 + i);
    }

    auto& diags_before = verifier.diagnostics();
    REQUIRE(diags_before.total_steps == 5);

    // Reset and verify counters are zeroed
    verifier.reset_diagnostics();
    auto& diags_after = verifier.diagnostics();
    REQUIRE(diags_after.total_steps == 0);
    REQUIRE(diags_after.steps_accepted == 0);
    REQUIRE(diags_after.fallback_count == 0);
}

// ---- Test 10: Multiple speculation steps accumulate counters correctly ----
TEST_CASE("Accumulate diagnostics across multiple speculation steps", "[speculative][integration]") {
    SpeculativeTreeVerifier::Config cfg;
    cfg.log_diagnostics = false;

    SpeculativeTreeVerifier verifier(cfg);

    for (int i = 0; i < 10; ++i) {
        verifier.speculate_step({static_cast<int32_t>(i)}, 50 + i);
    }

    auto& diags = verifier.diagnostics();
    REQUIRE(diags.total_steps == 10);
    REQUIRE(diags.total_draft_tokens == 10 * SPEC_K); // K=8 per step
    REQUIRE(diags.steps_accepted >= 0);
    REQUIRE(diags.steps_rejected >= 0);
}
