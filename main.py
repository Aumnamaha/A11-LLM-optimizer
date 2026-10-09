import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
from engine.pipeline import A11Pipeline

def inspect_directory(dir_path: str) -> str:
    """Scans directory contents to build context for the model prompt."""
    if not os.path.exists(dir_path):
        return f"Directory '{dir_path}' does not exist."

    file_list = []
    for root, dirs, files in os.walk(dir_path):
        rel_root = os.path.relpath(root, dir_path)
        prefix = "" if rel_root == "." else f"{rel_root}/"
        for f in files:
            file_path = os.path.join(root, f)
            size_kb = os.path.getsize(file_path) / 1024
            file_list.append(f" - {prefix}{f} ({size_kb:.1f} KB)")

    if not file_list:
        return f"Directory '{dir_path}' is empty."

    formatted_tree = "\n".join(file_list[:15]) # Limit top entries for token budget
    return formatted_tree

def main():
    MODEL_DIR = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    TARGET_DIR = "/home/knk/Projects/A11-LLM-optimizer"

    print("==========================================================")
    print("      A11-LLM-Optimizer Directory Analysis Test           ")
    print("==========================================================")
    
    # 1. Scan target directory
    print(f"\n[Directory Inspector] Scanning target: '{TARGET_DIR}'...")
    dir_summary = inspect_directory(TARGET_DIR)
    print(f"Directory Structure Detected:\n{dir_summary}")

    # 2. Build prompt for model execution
    prompt = (
        f"The user wants an analysis of the 'Student_Feedback' project directory.\n"
        f"Directory contents:\n{dir_summary}\n\n"
        f"Explain the purpose, structure, and key components of this directory."
    )

    # 3. Initialize Engine & Execute
    pipeline = A11Pipeline(MODEL_DIR, vram_budget_gb=4.0, ram_budget_gb=16.0)
    
    try:
        print("\n--- [Executing Model Inference Step] ---")
        pipeline.run_inference_step(prompt)
    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    main()
