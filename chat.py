import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import time
from engine.pipeline import A11Pipeline
from engine.context_manager import WorkspaceContextManager

def launch_chat():
    MODEL_DIR = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    TARGET_WORKSPACE = "/home/knk/Projects/A11-LLM-optimizer"

    if not os.path.exists(MODEL_DIR):
        print(f"[Error] Model path missing: {MODEL_DIR}")
        sys.exit(1)

    print("==================================================================")
    print("      A11-LLM-Optimizer Interactive Codebase Chatbot             ")
    print("==================================================================")
    print(f"  • Model Target: Qwen 3.8 Flash Next (Multi-Shard MoE)")
    print(f"  • Target Workspace: {TARGET_WORKSPACE}")
    print("  • Type 'exit' or 'quit' to end session.")
    print("==================================================================\n")

    ctx_mgr = WorkspaceContextManager(TARGET_WORKSPACE)
    pipeline = A11Pipeline(MODEL_DIR, vram_budget_gb=4.0, ram_budget_gb=16.0)

    try:
        while True:
            try:
                user_input = input("\n[You]: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("\n[Chatbot] Closing session...")
                break

            full_prompt = ctx_mgr.build_workspace_prompt(user_input)
            
            print("\n[A11-Assistant]: ", end="", flush=True)
            start_t = time.time()
            token_count = 0
            
            for token in pipeline.stream_inference(full_prompt, max_new_tokens=200):
                print(token, end="", flush=True)
                token_count += 1
                
            elapsed = time.time() - start_t
            tps = token_count / max(elapsed, 0.001)
            stats = pipeline.get_cache_stats()
            
            print(f"\n\n  ┌─ [Performance Metrics]")
            print(f"  ├─ Tokens Generated: {token_count} | Time: {elapsed:.3f}s | Speed: {tps:.1f} tok/s")
            print(f"  └─ Cache Stats: Hits: {stats['hits']} | Misses: {stats['misses']} | Hit Rate: {stats['hit_rate_pct']}%")

    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    launch_chat()
