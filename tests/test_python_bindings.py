"""Integration tests for a11_optimizer Python bindings (pybind11).

Tests:
  - Module imports and attribute access
  - NvmeAsyncLoader class instantiation and method invocation
  - PagedKVCacheManager class with KV cache operations
  - SpeculativeTreeVerifier with token acceptance logic
"""

import sys
import os
import pytest


# ------------------------------------------------------------------
# Helper: ensure the extension module is importable
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def a11():
    """Import the a11_optimizer binding module once per session."""
    # Add python/ directory to sys.path for local development builds
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    python_pkg = os.path.join(project_root, "python", "a11_optimizer")

    if python_pkg not in sys.path:
        sys.path.insert(0, python_pkg)

    try:
        import a11_optimizer as a11
        return a11
    except ImportError as e:
        pytest.skip(f"a11_optimizer extension not built: {e}")


# ==================================================================
# Module-level tests
# ==================================================================

class TestModuleConstants:
    """Verify module exposes correct constants."""

    def test_spec_k(self, a11):
        assert hasattr(a11, "SPEC_K")
        assert a11.SPEC_K == 8

    def test_block_tokens(self, a11):
        assert hasattr(a11, "BLOCK_TOKENS")
        assert a11.BLOCK_TOKENS == 16

    def test_kv_cache_vram_gb(self, a11):
        assert hasattr(a11, "KV_CACHE_VRAM_GB")
        assert a11.KV_CACHE_VRAM_GB == 8


# ==================================================================
# NvmeAsyncLoader tests
# ==================================================================

class TestNvmeAsyncLoader:
    """Test NvmeAsyncLoader Python bindings."""

    def test_instantiation(self, a11):
        loader = a11.NvmeAsyncLoader()
        assert loader is not None
        assert not loader.is_open()

    def test_open_nonexistent_file(self, a11):
        """Opening a non-existent file should return False (not raise)."""
        loader = a11.NvmeAsyncLoader()
        result = loader.open_file(r"C:\nonexistent\path\model.gguf")
        assert not result

    def test_open_and_close(self, tmp_path, a11):
        """Create a small file, open it, verify is_open(), then close."""
        test_file = tmp_path / "test_model.bin"
        test_file.write_bytes(b"\x00" * 256)

        loader = a11.NvmeAsyncLoader()
        result = loader.open_file(str(test_file))
        assert result is True
        assert loader.is_open()

        loader.close_file()
        assert not loader.is_open()


# ==================================================================
# PagedKVCacheManager tests
# ==================================================================

class TestPagedKVCacheManager:
    """Test PagedKVCacheManager Python bindings."""

    def _make_config(self, a11, num_layers=4, head_dim=32, max_seqs=8, vram_gb=1):
        cfg = a11.KVCacheConfig()
        cfg.num_layers = num_layers
        cfg.head_dim = head_dim
        cfg.max_sequences = max_seqs
        cfg.vram_cap_gb = vram_gb
        return cfg

    def test_config_creation(self, a11):
        cfg = a11.KVCacheConfig()
        assert cfg.num_layers == 80   # default
        assert cfg.head_dim == 128     # default
        assert cfg.max_sequences == 64  # default
        assert cfg.vram_cap_gb == 8    # default

    def test_custom_config(self, a11):
        cfg = self._make_config(a11, num_layers=8, head_dim=64)
        assert cfg.num_layers == 8
        assert cfg.head_dim == 64

    def test_manager_instantiation(self, a11):
        cfg = self._make_config(a11)
        mgr = a11.PagedKVCacheManager(cfg)
        assert mgr is not None
        assert mgr.used_blocks() == 0
        assert mgr.total_tokens() == 0

    def test_allocate_sequence(self, a11):
        cfg = self._make_config(a11)
        mgr = a11.PagedKVCacheManager(cfg)

        block_table = mgr.allocate_sequence(0)
        assert isinstance(block_table, list)
        assert len(block_table) == 1
        assert mgr.used_blocks() == 1

    def test_append_token(self, a11):
        cfg = self._make_config(a11)
        mgr = a11.PagedKVCacheManager(cfg)
        mgr.allocate_sequence(0)

        for _ in range(5):
            mgr.append_token(0)

        assert mgr.total_tokens() == 5
        # With BLOCK_TOKENS=16 and small config, 5 tokens fit in one block
        assert mgr.used_blocks() == 1

    def test_free_sequence(self, a11):
        cfg = self._make_config(a11)
        mgr = a11.PagedKVCacheManager(cfg)
        mgr.allocate_sequence(0)
        assert mgr.used_blocks() == 1

        mgr.free_sequence(0)
        assert mgr.used_blocks() == 0

    def test_vram_usage(self, a11):
        cfg = self._make_config(a11)
        mgr = a11.PagedKVCacheManager(cfg)
        assert mgr.vram_usage_bytes() == 0

        mgr.allocate_sequence(0)
        usage = mgr.vram_usage_bytes()
        assert usage > 0


# ==================================================================
# SpeculativeTreeVerifier tests
# ==================================================================

class TestSpeculativeTreeVerifier:
    """Test SpeculativeTreeVerifier Python bindings."""

    def _make_config(self, a11):
        cfg = a11.SpecConfig()
        cfg.max_draft_depth = 8
        cfg.vocab_size = 32000
        cfg.acceptance_threshold = 1.0
        cfg.log_diagnostics = False  # Suppress diagnostic noise in tests
        return cfg

    def test_config_creation(self, a11):
        cfg = a11.SpecConfig()
        assert cfg.max_draft_depth == 8
        assert cfg.vocab_size == 32000
        assert cfg.acceptance_threshold == 1.0
        assert cfg.log_diagnostics is True

    def test_manager_instantiation(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)
        assert verifier is not None

    def test_build_draft_tree(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        tree_node = verifier.build_draft_tree(42)
        assert tree_node.token_id == 42
        assert tree_node.parent_idx == 0xFFFFFFFF  # UINT32_MAX in Python
        assert tree_node.depth == 0

    def test_verify_draft_tokens_perfect_match(self, a11):
        """When draft and target log-probs match, all tokens should be accepted."""
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        draft_tokens = [10, 20, 30, 40]
        draft_log_probs = [-0.5, -0.6, -0.7, -0.8]
        target_log_probs = draft_log_probs[:]  # Identical → perfect match

        accepted_len, last_idx = verifier.verify_draft_tokens(
            draft_tokens, draft_log_probs, target_log_probs)

        assert accepted_len == len(draft_tokens)
        assert last_idx == len(draft_tokens) - 1

    def test_verify_draft_tokens_early_rejection(self, a11):
        """When target log-prob is much lower, tokens should be rejected."""
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        draft_tokens = [10, 20, 30, 40]
        draft_log_probs = [-0.5, -0.5, -0.5, -0.5]
        # Target is much worse for token at index 2
        target_log_probs = [-0.6, -0.6, -10.0, -0.6]

        accepted_len, last_idx = verifier.verify_draft_tokens(
            draft_tokens, draft_log_probs, target_log_probs)

        # First two tokens may be accepted (high acceptance ratio), third rejected
        assert 0 <= accepted_len < len(draft_tokens)
        assert last_idx < len(draft_tokens) - 1 or accepted_len == 0

    def test_speculate_step(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        context = [1, 2, 3]
        output_tokens, full_acceptance = verifier.speculate_step(context, 100)

        assert isinstance(output_tokens, list)
        assert len(output_tokens) >= 1  # At least the target's own prediction token
        assert isinstance(full_acceptance, bool)

    def test_diagnostics_counters(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        diag = verifier.diagnostics()
        assert diag.total_steps == 0
        assert diag.steps_accepted == 0
        assert diag.fallback_count == 0

    def test_diagnostics_after_steps(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        for _ in range(5):
            verifier.speculate_step([1], 42 + _)

        diag = verifier.diagnostics()
        assert diag.total_steps == 5
        # steps_accepted + steps_rejected should equal total_steps
        assert diag.steps_accepted + diag.steps_rejected == diag.total_steps

    def test_reset_diagnostics(self, a11):
        cfg = self._make_config(a11)
        verifier = a11.SpeculativeTreeVerifier(cfg)

        for _ in range(3):
            verifier.speculate_step([1], 42 + _)

        assert verifier.diagnostics().total_steps == 3

        verifier.reset_diagnostics()
        diag = verifier.diagnostics()
        assert diag.total_steps == 0
        assert diag.steps_accepted == 0
        assert diag.fallback_count == 0


# ==================================================================
# Cross-class integration test
# ==================================================================

class TestCrossClassIntegration:
    """Test that multiple C++ classes work together from Python."""

    def test_full_pipeline(self, a11):
        """Simulate a mini inference pipeline using all three bound classes."""
        # 1. KV cache setup
        kv_cfg = a11.KVCacheConfig()
        kv_cfg.num_layers = 4
        kv_cfg.head_dim = 32
        kv_cfg.max_sequences = 2

        kv_cache = a11.PagedKVCacheManager(kv_cfg)
        kv_cache.allocate_sequence(0)

        # 2. Speculative verification
        spec_cfg = a11.SpecConfig()
        spec_cfg.log_diagnostics = False

        verifier = a11.SpeculativeTreeVerifier(spec_cfg)
        tree = verifier.build_draft_tree(50)
        assert tree.token_id == 50

        # 3. Verify some tokens
        draft_tokens = [51, 52, 53]
        draft_probs = [-0.5, -0.6, -0.7]
        target_probs = [-0.6, -0.7, -1.0]

        accepted_len, _ = verifier.verify_draft_tokens(
            draft_tokens, draft_probs, target_probs)
        assert 0 <= accepted_len <= len(draft_tokens)

        # Cleanup (implicit via Python GC + pybind11 RAII)
        del kv_cache, verifier
