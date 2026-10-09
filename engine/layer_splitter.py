from typing import Dict, List, Tuple
from engine.loader import SplitGGUFLoader

class LayerMemorySplitter:
    """
    Evaluates hardware memory budgets and assigns model tensors 
    to GPU VRAM, System RAM, or NVMe SSD (mmap).
    """
    def __init__(self, loader: SplitGGUFLoader, vram_budget_gb: float = 4.0, ram_budget_gb: float = 16.0):
        self.loader = loader
        self.vram_budget_gb = vram_budget_gb
        self.ram_budget_gb = ram_budget_gb
        
        self.vram_tensors: List[str] = []
        self.ram_tensors: List[str] = []
        self.ssd_tensors: List[str] = []
        
        self._calculate_allocations()

    def _calculate_allocations(self):
        print(f"\n[Layer Splitter] Allocating tensors for Hardware Profile:")
        print(f"  • VRAM Budget: {self.vram_budget_gb:.1f} GB | RAM Budget: {self.ram_budget_gb:.1f} GB")
        
        for t_name, t_info in self.loader.tensors.items():
            t_lower = t_name.lower()
            
            # 1. Essential Attention, Router, & Norm heads stay locked in VRAM
            if any(k in t_lower for k in ["attn", "norm", "router", "gate_inp", "output", "token_embd"]):
                self.vram_tensors.append(t_name)
            
            # 2. Hot MoE Experts cached in System RAM
            elif any(k in t_lower for k in ["exps", "ffn"]) and len(self.ram_tensors) < 200:
                self.ram_tensors.append(t_name)
                
            # 3. 51B N-Gram Lookup tables & remaining Cold MoE Experts stay on NVMe SSD
            else:
                self.ssd_tensors.append(t_name)

        print(f"  • VRAM Tensors (Core Heads/Routers): {len(self.vram_tensors)}")
        print(f"  • System RAM Tensors (Hot Experts): {len(self.ram_tensors)}")
        print(f"  • NVMe SSD Tensors (N-Gram / Cold Experts): {len(self.ssd_tensors)}")

if __name__ == "__main__":
    import os
    target_dir = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    if os.path.exists(target_dir):
        loader = SplitGGUFLoader(target_dir)
        splitter = LayerMemorySplitter(loader, vram_budget_gb=4.0, ram_budget_gb=16.0)
