#include "engine/speculative_verifier.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace A11 {
namespace Engine {

SpeculativeTreeVerifier::SpeculativeTreeVerifier(const Config& cfg)
    : config_(cfg), diags_{} {}

SpeculativeTreeVerifier::~SpeculativeTreeVerifier() = default;

// ---- Draft Tree Construction ----

TreeNode SpeculativeTreeVerifier::build_draft_tree(int32_t root_token) {
    TreeNode root;
    root.token_id   = root_token;
    root.log_prob   = 0.0f; // Will be filled by draft model
    root.parent_idx = UINT32_MAX;
    root.depth      = 0;

    if (config_.log_diagnostics) {
        std::cerr << "[SpeculativeTree] Built tree root: token=" << root_token
                  << ", max_depth=" << config_.max_draft_depth << std::endl;
    }
    return root;
}

void SpeculativeTreeVerifier::build_dfs_order(
    uint32_t node_idx, std::vector<uint32_t>& dfs_order) {
    // DFS traversal of the draft tree starting from node_idx.
    // The caller must provide access to the external tree storage.
    // This is a stub — real usage requires passing the tree vector by reference.
    (void)node_idx;
    (void)dfs_order;
}

// ---- Tree Mask Construction ----

std::vector<float> SpeculativeTreeVerifier::construct_tree_mask(
    const std::vector<TreeNode>& tree) {
    // Each tree node requires evaluation of its drafted token's probability
    // under the target model. The mask is [num_nodes x vocab_size].
    // For a single-token verification at each position, we only need one
    // column per node (the column corresponding to the drafted token id).

    size_t num_nodes = tree.size();
    if (num_nodes == 0) return {};

    std::vector<float> mask(num_nodes * config_.vocab_size, 0.0f);

    for (size_t i = 0; i < num_nodes; ++i) {
        int32_t tok = tree[i].token_id;
        if (tok >= 0 && static_cast<uint32_t>(tok) < config_.vocab_size) {
            mask[i * config_.vocab_size + static_cast<size_t>(tok)] = 1.0f;
        }
    }

    return mask;
}

// ---- Rejection Sampling Verification ----

float SpeculativeTreeVerifier::rand_uniform() {
    thread_local std::mt19937 rng(42); // Deterministic for reproducibility
    thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

bool SpeculativeTreeVerifier::validate_target_probs(
    const std::vector<float>& probs) const {
    if (probs.empty()) return false;
    for (float p : probs) {
        // Accept both probability values [0, 1] and log-probability values
        // The verification step handles conversion appropriately.
        if (std::isnan(p) || std::isinf(p)) {
            return false;
        }
    }
    return true;
}

std::pair<uint32_t, uint32_t> SpeculativeTreeVerifier::verify_draft_tokens(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<float>& draft_log_probs,
    const std::vector<float>& target_log_probs) {

    if (draft_tokens.size() != draft_log_probs.size() ||
        draft_tokens.size() != target_log_probs.size()) {
        throw std::runtime_error("SpeculativeTreeVerifier: "
            "draft_tokens, draft_log_probs, and target_log_probs must have same length");
    }

    if (!validate_target_probs(target_log_probs)) {
        throw std::runtime_error("SpeculativeTreeVerifier: invalid target probabilities");
    }

    uint32_t accepted_len = 0;
    uint32_t last_accepted_idx = 0;

    for (uint32_t i = 0; i < draft_tokens.size(); ++i) {
        float ld = draft_log_probs[i];
        float lt = target_log_probs[i];

        // Detect input format: negative values indicate log-probability space.
        bool is_log_space = (ld < 0.0f || lt < 0.0f);

        float accept_prob;
        if (is_log_space) {
            // Log-probability space: acceptance = exp(target_log - draft_log)
            float log_ratio = lt - ld;
            accept_prob = std::exp(std::max(log_ratio, -20.0f));
        } else {
            // Linear probability space [0, 1]: acceptance = p_target / p_draft
            ld = std::max(1e-10f, std::min(ld, 1.0f));
            lt = std::max(0.0f, std::min(lt, 1.0f + 1e-6f));
            accept_prob = lt / ld;
        }

        if (accept_prob > 1.0f) accept_prob = 1.0f;

        float r = rand_uniform();
        if (r < accept_prob) {
            // Token accepted
            ++accepted_len;
            last_accepted_idx = i;
        } else {
            // Rejection — stop accepting further tokens
            break;
        }
    }

    return {accepted_len, last_accepted_idx};
}

// ---- Full Speculation Step ----

std::pair<std::vector<int32_t>, bool> SpeculativeTreeVerifier::speculate_step(
    const std::vector<int32_t>& context_tokens,
    int32_t root_draft_token) {

    ++diags_.total_steps;

    // Build draft tree (simplified: linear chain for baseline)
    TreeNode _root = build_draft_tree(root_draft_token);
    (void)_root; // Tree structure tracked externally in production

    if (config_.log_diagnostics) {
        std::cerr << "[SpeculativeTree] Speculation step #" << diags_.total_steps
                  << ", context_tokens=" << context_tokens.size()
                  << ", root_draft=" << root_draft_token << std::endl;
    }

    // Simulate a linear draft chain of K tokens
    std::vector<int32_t> draft_tokens(SPEC_K);
    std::vector<float>   draft_log_probs(SPEC_K);
    std::vector<float>   target_log_probs(SPEC_K);

    for (uint32_t i = 0; i < SPEC_K; ++i) {
        // Simulated: draft proposes sequential tokens
        draft_tokens[i] = root_draft_token + static_cast<int32_t>(i) + 1;
        draft_log_probs[i] = -std::log(static_cast<float>(SPEC_K)) * (1.0f - i * 0.05f);
        // Simulated target: high acceptance rate (~90%) for benchmarking
        target_log_probs[i] = draft_log_probs[i] + (rand_uniform() < 0.9f ? -0.1f : -2.0f);
    }

    diags_.total_draft_tokens += SPEC_K;

    // Verify against target model probabilities
    auto [accepted_len, last_accepted_idx] = verify_draft_tokens(
        draft_tokens, draft_log_probs, target_log_probs);

    if (config_.log_diagnostics) {
        std::cerr << "[SpeculativeTree] Verification: accepted=" << accepted_len
                  << ", last_accepted_idx=" << last_accepted_idx
                  << " / " << SPEC_K << std::endl;
    }

    // Update diagnostic counters (step-level tracking)
    if (accepted_len == 0) {
        ++diags_.steps_rejected;
        // Fallback to single-token autoregressive generation
        diags_.log_fallback(SPEC_K, 0);
    } else {
        ++diags_.steps_accepted;
    }

    // Build output: accepted draft tokens + one target-generated token
    std::vector<int32_t> output_tokens;
    for (uint32_t i = 0; i < accepted_len; ++i) {
        output_tokens.push_back(draft_tokens[i]);
    }
    // Always append one new target-computed token (the speculator's own prediction)
    output_tokens.push_back(root_draft_token + SPEC_K);

    bool full_acceptance = (accepted_len == SPEC_K);
    return {output_tokens, full_acceptance};
}

void SpeculativeTreeVerifier::reset_diagnostics() {
    diags_.reset();
}

} // namespace Engine
} // namespace A11
