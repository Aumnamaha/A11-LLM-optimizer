#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <vector>
#include <string>
#include <utility>

// Define NOMINMAX before any Windows headers (including via pybind11)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Include all C++ library headers
#include "memory/nvme_loader.hpp"
#include "cuda/paged_kv_cache.hpp"
#include "engine/speculative_verifier.hpp"

namespace py = pybind11;

PYBIND11_MODULE(a11_optimizer, m) {
    m.doc() = "A11 Language Model Optimizer — Pybind11 C++ extension module";

    // Module-level constants
    m.attr("SPEC_K") = SPEC_K;
    m.attr("BLOCK_TOKENS") = BLOCK_TOKENS;
    m.attr("KV_CACHE_VRAM_GB") = KV_CACHE_VRAM_GB;

    // ===================================================================
    // 1. NvmeAsyncLoader binding
    // ===================================================================
    py::class_<A11::Memory::NvmeAsyncLoader> nvme_cls(m, "NvmeAsyncLoader");

    nvme_cls
        .def(py::init<>())
        .def("open_file", [](A11::Memory::NvmeAsyncLoader& self, const std::wstring& path) {
                return self.open_file(path); })
        .def("close_file", [](A11::Memory::NvmeAsyncLoader& self) { self.close_file(); })
        .def("is_open", [](const A11::Memory::NvmeAsyncLoader& self) { return self.is_open(); });

    // ===================================================================
    // 2. PagedKVCacheManager binding
    // ===================================================================
    py::class_<PagedKVCacheManager::Config> kv_config_cls(m, "KVCacheConfig");

    kv_config_cls
        .def(py::init<>())
        .def_readwrite("num_layers", &PagedKVCacheManager::Config::num_layers)
        .def_readwrite("head_dim", &PagedKVCacheManager::Config::head_dim)
        .def_readwrite("max_sequences", &PagedKVCacheManager::Config::max_sequences)
        .def_readwrite("vram_cap_gb", &PagedKVCacheManager::Config::vram_cap_gb);

    py::class_<PagedKVCacheManager> kv_cls(m, "PagedKVCacheManager");

    // Use lambda wrappers to avoid MSVC namespace resolution issues with member pointers
    kv_cls
        .def(py::init<const PagedKVCacheManager::Config&>())
        .def("allocate_sequence", [](PagedKVCacheManager& self, uint32_t seq_id) {
                return self.allocate_sequence(seq_id); })
        .def("append_token", [](PagedKVCacheManager& self, uint32_t seq_id) {
                self.append_token(seq_id); })
        .def("free_sequence", [](PagedKVCacheManager& self, uint32_t seq_id) {
                self.free_sequence(seq_id); })
        .def("get_block_table", [](const PagedKVCacheManager& self, uint32_t seq_id) {
                return self.get_block_table(seq_id); })
        .def("used_blocks", [](const PagedKVCacheManager& self) {
                return self.used_blocks(); })
        .def("total_tokens", [](const PagedKVCacheManager& self) {
                return self.total_tokens(); })
        .def("vram_usage_bytes", [](const PagedKVCacheManager& self) {
                return self.vram_usage_bytes(); });

    // ===================================================================
    // 3. SpeculativeTreeVerifier binding
    // ===================================================================
    py::class_<A11::Engine::TreeNode> tree_node_cls(m, "TreeNode");

    tree_node_cls
        .def_readwrite("token_id", &A11::Engine::TreeNode::token_id)
        .def_readwrite("log_prob", &A11::Engine::TreeNode::log_prob)
        .def_readwrite("parent_idx", &A11::Engine::TreeNode::parent_idx)
        .def_readwrite("depth", &A11::Engine::TreeNode::depth)
        .def_readwrite("children", &A11::Engine::TreeNode::children);

    py::class_<A11::Engine::SpecDiagnosticCounters> diag_cls(m, "SpecDiagnostics");

    diag_cls
        .def_readonly("steps_accepted", &A11::Engine::SpecDiagnosticCounters::steps_accepted)
        .def_readonly("steps_rejected", &A11::Engine::SpecDiagnosticCounters::steps_rejected)
        .def_readonly("fallback_count", &A11::Engine::SpecDiagnosticCounters::fallback_count)
        .def_readonly("total_draft_tokens", &A11::Engine::SpecDiagnosticCounters::total_draft_tokens)
        .def_readonly("total_steps", &A11::Engine::SpecDiagnosticCounters::total_steps);

    py::class_<A11::Engine::SpeculativeTreeVerifier::Config> spec_config_cls(m, "SpecConfig");

    spec_config_cls
        .def(py::init<>())
        .def_readwrite("max_draft_depth", &A11::Engine::SpeculativeTreeVerifier::Config::max_draft_depth)
        .def_readwrite("vocab_size", &A11::Engine::SpeculativeTreeVerifier::Config::vocab_size)
        .def_readwrite("acceptance_threshold", &A11::Engine::SpeculativeTreeVerifier::Config::acceptance_threshold)
        .def_readwrite("log_diagnostics", &A11::Engine::SpeculativeTreeVerifier::Config::log_diagnostics);

    py::class_<A11::Engine::SpeculativeTreeVerifier> spec_cls(m, "SpeculativeTreeVerifier");

    spec_cls
        .def(py::init<const A11::Engine::SpeculativeTreeVerifier::Config&>())
        .def("build_draft_tree", [](A11::Engine::SpeculativeTreeVerifier& self, int32_t root_token) {
                return self.build_draft_tree(root_token); })
        .def("construct_tree_mask", [](A11::Engine::SpeculativeTreeVerifier& self, const std::vector<A11::Engine::TreeNode>& tree) {
                return self.construct_tree_mask(tree); });

    spec_cls
        .def("verify_draft_tokens",
             [](A11::Engine::SpeculativeTreeVerifier& self,
                const std::vector<int32_t>& draft_tokens,
                const std::vector<float>& draft_log_probs,
                const std::vector<float>& target_log_probs) {
                 auto [accepted_len, last_idx] = self.verify_draft_tokens(
                     draft_tokens, draft_log_probs, target_log_probs);
                 return std::make_pair(accepted_len, last_idx); },
             py::arg("draft_tokens"),
             py::arg("draft_log_probs"),
             py::arg("target_log_probs"));

    spec_cls
        .def("speculate_step",
             [](A11::Engine::SpeculativeTreeVerifier& self,
                const std::vector<int32_t>& context_tokens,
                int32_t root_draft_token) {
                 auto [output_tokens, full_acceptance] = self.speculate_step(
                     context_tokens, root_draft_token);
                 return std::make_pair(output_tokens, full_acceptance); },
             py::arg("context_tokens"),
             py::arg("root_draft_token"));

    spec_cls
        .def("diagnostics", [](A11::Engine::SpeculativeTreeVerifier& self) -> const A11::Engine::SpecDiagnosticCounters& {
                return self.diagnostics(); },
             py::return_value_policy::reference_internal)
        .def("reset_diagnostics", [](A11::Engine::SpeculativeTreeVerifier& self) {
                self.reset_diagnostics(); });
}
