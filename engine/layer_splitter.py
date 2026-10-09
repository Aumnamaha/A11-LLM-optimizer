from typing import Any, List, Optional
from engine.loader import SplitGGUFLoader


class LayerMemorySplitter:
    """
    Evaluates hardware memory budgets and assigns model tensors 
    to GPU VRAM, System RAM, or NVMe SSD (mmap).
    """
    def __init__(
        self,
        loader: SplitGGUFLoader,
        vram_budget_gb: float = 4.0,
        ram_budget_gb: float = 16.0,
        max_hot_experts: Optional[int] = None,
    ):
        self.loader = loader
        self.vram_budget_gb = max(0.0, float(vram_budget_gb))
        self.ram_budget_gb = max(0.0, float(ram_budget_gb))
        self.max_hot_experts = max_hot_experts
        
        self.vram_tensors: List[str] = []
        self.ram_tensors: List[str] = []
        self.ssd_tensors: List[str] = []

        self.vram_allocated_bytes: int = 0
        self.ram_allocated_bytes: int = 0
        self.ssd_allocated_bytes: int = 0
        
        self._calculate_allocations()

    def _calculate_allocations(self):
        vram_limit_bytes = int(self.vram_budget_gb * (1024 ** 3))
        ram_limit_bytes = int(self.ram_budget_gb * (1024 ** 3))

        print(f"\n[Layer Splitter] Allocating tensors for Hardware Profile:")
        print(f"  • VRAM Budget: {self.vram_budget_gb:.1f} GB | RAM Budget: {self.ram_budget_gb:.1f} GB")

        for t_name, t_info in self.loader.tensors.items():
            t_lower = t_name.lower()
            t_bytes = t_info.get("n_bytes", 0)

            is_core = any(k in t_lower for k in ["attn", "norm", "router", "gate_inp", "output", "token_embd"])
            is_expert = any(k in t_lower for k in ["exps", "ffn"])

            # 1. Essential Attention, Router, & Norm heads stay locked in VRAM (or fallback to RAM)
            if is_core:
                if vram_limit_bytes > 0 and (self.vram_allocated_bytes + t_bytes <= vram_limit_bytes):
                    self.vram_tensors.append(t_name)
                    self.vram_allocated_bytes += t_bytes
                else:
                    # Fallback core tensors to System RAM if VRAM is exhausted or 0
                    self.ram_tensors.append(t_name)
                    self.ram_allocated_bytes += t_bytes

            # 2. Hot MoE Experts cached in System RAM
            elif is_expert:
                hot_expert_limit_reached = (
                    self.max_hot_experts is not None and len(self.ram_tensors) >= self.max_hot_experts
                )
                if not hot_expert_limit_reached and (self.ram_allocated_bytes + t_bytes <= ram_limit_bytes):
                    self.ram_tensors.append(t_name)
                    self.ram_allocated_bytes += t_bytes
                else:
                    self.ssd_tensors.append(t_name)
                    self.ssd_allocated_bytes += t_bytes

            # 3. N-Gram Lookup tables & remaining Cold MoE Experts stay on NVMe SSD
            else:
                self.ssd_tensors.append(t_name)
                self.ssd_allocated_bytes += t_bytes

        vram_gb = self.vram_allocated_bytes / (1024 ** 3)
        ram_gb = self.ram_allocated_bytes / (1024 ** 3)
        ssd_gb = self.ssd_allocated_bytes / (1024 ** 3)

        print(f"  • VRAM Tensors (Core Heads/Routers): {len(self.vram_tensors)} ({vram_gb:.2f} GB)")
        print(f"  • System RAM Tensors (Hot Experts): {len(self.ram_tensors)} ({ram_gb:.2f} GB)")
        print(f"  • NVMe SSD Tensors (N-Gram / Cold Experts): {len(self.ssd_tensors)} ({ssd_gb:.2f} GB)")


if __name__ == "__main__":
    import os
    target_dir = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    if os.path.exists(target_dir):
        loader = SplitGGUFLoader(target_dir)
        splitter = LayerMemorySplitter(loader, vram_budget_gb=4.0, ram_budget_gb=16.0)
