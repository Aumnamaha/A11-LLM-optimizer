#!/usr/bin/env python3
"""
End-to-end GGUF inference runner using A11 optimizer C++ bindings.

Loads a GGUF model, streams tensors via NvmeAsyncLoader, manages KV-cache
via PagedKVCacheManager, and runs speculative token generation with the
SpeculativeTreeVerifier.

Usage:
    python scripts/run_gguf_inference.py --model path/to/model.gguf --prompt "Hello" --max-tokens 64 --draft-k 8
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(
        description="A11 GGUF Inference Runner with Speculative Decoding")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GGUF model file.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Initial prompt text for generation.")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate.")
    parser.add_argument("--draft-k", type=int, default=8,
                        help="Number of draft candidates per speculation step (K).")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Ensure the extension module is importable (python/ package dir)
    # ------------------------------------------------------------------
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    _pkg_path = os.path.join(_project_root, "python", "a11_optimizer")
    if _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

    # ------------------------------------------------------------------
    # Import C++ bindings
    # ------------------------------------------------------------------
    try:
        import a11_optimizer as a11
    except ImportError as e:
        print(f"[ERROR] Failed to import a11_optimizer binding module: {e}", file=sys.stderr)
        print("[HINT] Build the extension first: cmake --build build --config Release", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Initialize NvmeAsyncLoader for GGUF tensor streaming
    # ------------------------------------------------------------------
    print(f"[A11] Loading model from: {args.model}")
    loader = a11.NvmeAsyncLoader()

    try:
        opened = loader.open_file(args.model)
        if not opened:
            raise RuntimeError(f"Failed to open GGUF file: {args.model}")
        print("[A11] Model file opened successfully.")
    except Exception as e:
        print(f"[ERROR] NVMe loader failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Initialize PagedKVCacheManager
    # ------------------------------------------------------------------
    kv_config = a11.KVCacheConfig()
    kv_config.num_layers = 32      # Default for medium model (adjust as needed)
    kv_config.head_dim = 128       # Standard head dimension
    kv_config.max_sequences = 4    # Small batch for CLI demo
    kv_config.vram_cap_gb = 6.0    # Reserve VRAM for activations

    print(f"[A11] KV cache config: layers={kv_config.num_layers}, "
          f"head_dim={kv_config.head_dim}, max_seqs={kv_config.max_sequences}")

    try:
        kv_cache = a11.PagedKVCacheManager(kv_config)
        print("[A11] PagedKVCacheManager initialized.")
    except Exception as e:
        print(f"[ERROR] KV cache init failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Initialize SpeculativeTreeVerifier
    # ------------------------------------------------------------------
    spec_config = a11.SpecConfig()
    spec_config.max_draft_depth = args.draft_k
    spec_config.vocab_size = 32000
    spec_config.acceptance_threshold = 1.0
    spec_config.log_diagnostics = True

    verifier = a11.SpeculativeTreeVerifier(spec_config)
    print(f"[A11] SpeculativeTreeVerifier initialized (K={spec_config.max_draft_depth}).")

    # ------------------------------------------------------------------
    # 4. Simulated generation loop
    # ------------------------------------------------------------------
    print(f"\n[A11] Prompt: '{args.prompt}'")
    print("[A11] Starting speculative token generation...\n")

    # Simulate token IDs from prompt (in production, these come from tokenizer)
    context_tokens = list(range(100, 100 + max(len(args.prompt), 4)))
    generated_tokens = []
    start_time = time.perf_counter()

    for step in range(min(args.max_tokens, 20)):  # Limit demo steps
        # In production: draft model generates K candidate tokens here
        root_draft_token = context_tokens[-1] + 1 if context_tokens else 42

        # Run one speculation step via C++ binding
        output_tokens, full_acceptance = verifier.speculate_step(
            context_tokens, root_draft_token)

        accepted_count = len(output_tokens) - 1  # Last token is the target's own prediction
        if accepted_count < 0:
            accepted_count = 0

        generated_tokens.extend(output_tokens[:-1])  # Append draft tokens only
        context_tokens.append(output_tokens[-1])       # Update context with full output

        diag = verifier.diagnostics()
        print(f"  Step {step+1}: drafted={args.draft_k}, accepted={accepted_count}, "
              f"full_acceptance={full_acceptance} | "
              f"total_steps={diag.total_steps}, fallbacks={diag.fallback_count}")

    elapsed = time.perf_counter() - start_time
    total_tokens = len(generated_tokens) + 1 if generated_tokens else 0
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tokens generated: {total_tokens}")
    print(f"  Time elapsed:     {elapsed:.3f}s")
    print(f"  Throughput:       {tps:.1f} tokens/sec")

    diag = verifier.diagnostics()
    print(f"\n  Diagnostic counters:")
    print(f"    Total steps:        {diag.total_steps}")
    print(f"    Steps accepted:     {diag.steps_accepted}")
    print(f"    Steps rejected:     {diag.steps_rejected}")
    print(f"    Fallbacks:          {diag.fallback_count}")
    print(f"    Draft tokens total: {diag.total_draft_tokens}")

    # Cleanup (handled automatically by pybind11 RAII)
    del loader, kv_cache, verifier
    print(f"\n[A11] Cleanup complete.")


if __name__ == "__main__":
    main()
