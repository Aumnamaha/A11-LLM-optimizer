import base64
import json
import math
import os
import re
import sys
import urllib.request
from typing import Dict, Generator, Optional, Tuple

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.hardware import (
    analyze_model_requirements,
    detect_hardware,
    evaluate_compatibility,
    print_hardware_and_compatibility_banner,
)
from engine.kv_cache import KVCacheManager
from engine.layer_splitter import LayerMemorySplitter
from engine.loader import SplitGGUFLoader
from engine.mmap_loader import DirectSSDMmapManager
from engine.pager import MemoryPager, ModelPageTable
from engine.prefetch import AsyncPrefetcher
from engine.router import MoERouter
from engine.sampler import TokenSampler
from engine.tokenizer import QwenTokenizerManager
from engine.weights import WeightStore, plan_ram_budgets


class A11Pipeline:
    """End-to-end inference pipeline executing actual model weights with MoE routing,

    direct SSD mmap streaming, byte-budgeted RAM caching, and asynchronous prefetching.
    """

    def __init__(
        self,
        model_path: str,
        vram_budget_gb: Optional[float] = None,
        ram_budget_gb: Optional[float] = None,
        max_layers: Optional[int] = None,
    ):
        print("==========================================================")
        print("  A11-LLM-Optimizer Real Weight Execution Engine          ")
        print("==========================================================")

        self.loader = SplitGGUFLoader(model_path)

        # Dynamic Hardware Detection & Compatibility Evaluation
        self.hardware_profile = detect_hardware()
        self.model_requirements = analyze_model_requirements(self.loader)
        self.compatibility_report = evaluate_compatibility(
            self.hardware_profile,
            self.model_requirements,
            requested_vram_gb=vram_budget_gb,
            requested_ram_gb=ram_budget_gb,
        )
        print_hardware_and_compatibility_banner(
            self.hardware_profile,
            self.model_requirements,
            self.compatibility_report,
        )

        effective_vram = self.compatibility_report.effective_vram_budget_gb
        effective_ram = self.compatibility_report.effective_ram_budget_gb

        self.splitter = LayerMemorySplitter(
            self.loader,
            vram_budget_gb=effective_vram,
            ram_budget_gb=effective_ram,
            max_hot_experts=self.compatibility_report.max_hot_experts,
        )
        self.mmap_mgr = DirectSSDMmapManager(self.loader)
        self.tokenizer = QwenTokenizerManager(model_path)

        # Model architecture configuration
        vocab_size = (
            len(self.loader.metadata.get("tokenizer.ggml.tokens", []))
            or getattr(self.tokenizer, "vocab_size", 151936)
        )
        self.config = ModelConfig.from_metadata(self.loader.metadata, vocab_size=vocab_size)
        self.num_layers = min(self.config.n_layer, max_layers) if max_layers is not None else self.config.n_layer

        # Compute device - use GPU if VRAM budget > 0 AND CUDA/ROCm is actually available
        cuda_available = False
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                cuda_available = True
            # Also check for ROCm
            elif getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
                cuda_available = True
        except Exception:
            pass
        self.device = torch.device("cuda" if (effective_vram > 0 and cuda_available) else "cpu")
        
        # Use BF16 for compute and resident core (2x memory savings vs FP32)
        self.compute_dtype = torch.bfloat16

        # Byte-budgeted weight store and RAM cache.
        # The resident core is held dequantized in BF16 by WeightStore, so the
        # expert cache is sized from what is *left* of the RAM budget. Counting
        # the resident footprint against the budget is what prevents the
        # resident + expert caches from oversubscribing host RAM (OOM).
        ram_budget_bytes = int(effective_ram * (1024 ** 3))
        resident_bf16_bytes, expert_cache_bytes = plan_ram_budgets(self.loader, ram_budget_bytes, self.compute_dtype)

        # Page table & slot-based pager for expert weights.
        # Slot count derived from ACTUAL RAM budget minus resident footprint.
        # Each expert tensor (gate/up/down) maps to one 2 MiB page (pages_per_expert=1)
        # and dequantizes to ~3 MiB BF16 (half of FP32).
        dequantized_expert_bytes = 3 * 1024 * 1024  # 3 MiB per expert tensor [out, in] in BF16
        pager_budget_bytes = max(0, expert_cache_bytes)  # already computed: effective_ram - resident_bf16
        n_slots_ram = max(1, pager_budget_bytes // dequantized_expert_bytes)

        # VRAM slot calculation: (VRAM_Budget - KV_Cache - Resident_Core) / Page_Size
        # Only allocate VRAM slots if GPU compute is available
        if self.device.type == "cuda":
            kv_cache_estimate_gb = 1.5  # Conservative estimate for generation context
            vram_budget_for_pager_gb = max(0.0, effective_vram - kv_cache_estimate_gb - 0.5)  # 0.5GB buffer
            vram_page_bytes = dequantized_expert_bytes  # same 3 MiB per expert page in BF16
            n_slots_vram = max(1, int(vram_budget_for_pager_gb * (1024 ** 3) // vram_page_bytes))
        else:
            n_slots_vram = 0
            kv_cache_estimate_gb = 0.0
            vram_budget_for_pager_gb = 0.0

        page_table = ModelPageTable("reports/page_table.jsonl", self.loader)
        pager = MemoryPager(
            page_table, 
            self.loader, 
            self.mmap_mgr, 
            n_slots_ram=n_slots_ram,
            n_slots_vram=n_slots_vram,
            vram_device=self.device,
            compute_dtype=self.compute_dtype,
        )

        self.weights = WeightStore(
            self.loader,
            self.mmap_mgr,
            pager=pager,
            n_experts=self.config.n_expert,
            resident_dtype=self.compute_dtype,
        )
        # Legacy HotExpertRAMCache removed; pager is the sole expert cache.
        prefetch_budget = min(512 * 1024 * 1024, max(1024 * 1024, expert_cache_bytes // 4))
        
        # New async prefetcher that works with the pager
        self.prefetcher = AsyncPrefetcher(pager, num_workers=1)

        # KV Cache Manager with paged storage
        kv_page_size = 256  # tokens per page
        kv_max_pages = max(1, int((kv_cache_estimate_gb * (1024 ** 3)) // 
                                   (self.num_layers * kv_page_size * self.config.n_head_kv * self.config.head_dim * 2 * 2)))
        self.kv_cache_manager = KVCacheManager(
            n_layers=self.num_layers,
            n_head_kv=self.config.n_head_kv,
            head_dim=self.config.head_dim,
            page_size=kv_page_size,
            max_pages=kv_max_pages,
            device=self.device,
            dtype=self.compute_dtype,
        )

        self.router = MoERouter(num_experts=self.config.n_expert, top_k=self.config.n_expert_used)
        # Register prefetch callback: router dispatches page IDs immediately after Top-K
        self.router.set_prefetch_callback(self._make_prefetch_callback())
        self.sampler = TokenSampler(temperature=0.7, top_k=40, top_p=0.9)

        # Persistent Workspace Memory
        self.active_target_dir = None
        self.active_github_user = None
        self.active_github_repo = None
        self.last_web_data = None

        print(f"[A11-Pipeline] Engine Active | Direct SSD mmap active.")
        print(
            f"[A11-Pipeline] Allocation: {self.splitter.vram_budget_gb:.1f}GB VRAM / "
            f"{self.splitter.ram_budget_gb:.1f}GB System RAM Hot Cache "
            f"({self.num_layers}/{self.config.n_layer} active layers)."
        )
        print(
            f"[A11-Pipeline] RAM split: resident core {resident_bf16_bytes / (1024 ** 3):.2f}GB (bf16) "
            f"+ expert cache {expert_cache_bytes / (1024 ** 3):.2f}GB "
            f"= {ram_budget_bytes / (1024 ** 3):.2f}GB budget."
        )
        print(
            f"[A11-Pipeline] Pager: {n_slots_ram} RAM slots × ~3MiB = {n_slots_ram * 3 / 1024:.2f}GiB "
            f"(from {effective_ram:.1f}GiB RAM − {resident_bf16_bytes / (1024 ** 3):.2f}GiB resident)."
        )
        print(
            f"[A11-Pipeline] VRAM Pager: {n_slots_vram} VRAM slots × ~3MiB = {n_slots_vram * 3 / 1024:.2f}GiB "
            f"(from {effective_vram:.1f}GiB VRAM − {kv_cache_estimate_gb:.1f}GiB KV cache est.)."
        ) if n_slots_vram > 0 else print(
            f"[A11-Pipeline] VRAM Pager: DISABLED (no CUDA/ROCm GPU available)"
        )
        print(f"[A11-Pipeline] Compute Device: {self.device} | Compute Dtype: {self.compute_dtype}")

    def _make_prefetch_callback(self):
        """Create a callback that dispatches page IDs to the prefetcher."""
        def callback(top_k_indices: torch.Tensor):
            # top_k_indices shape: [batch, top_k]
            indices = top_k_indices[0].tolist()  # batch=1
            layer_idx = getattr(self, '_current_layer_idx', 0)
            
            # Generate page IDs for all 3 expert tensors (gate, up, down)
            # for each selected expert
            page_ids = []
            for idx in indices:
                for kind in ("gate", "up", "down"):
                    tensor_tag = kind.upper()
                    page_id = f"E{idx:03d}:L{layer_idx:02d}:T{tensor_tag}:P000"
                    page_ids.append(page_id)
            
            # Also prefetch next layer's experts (temporal locality)
            if layer_idx + 1 < self.num_layers:
                next_layer = layer_idx + 1
                for idx in indices:
                    for kind in ("gate", "up", "down"):
                        tensor_tag = kind.upper()
                        page_id = f"E{idx:03d}:L{next_layer:02d}:T{tensor_tag}:P000"
                        page_ids.append(page_id)
            
            if page_ids:
                self.prefetcher.request_prefetch_batch(page_ids)
        return callback

    def _clean_github_slug(self, raw_str: str) -> str:
        if not raw_str:
            return ""
        clean = raw_str.replace("https:", "").replace("http:", "")
        clean = re.sub(r'[\(\)\[\]\'"`]', '', clean)
        clean = re.split(r'[:/]', clean)[0]
        return clean.strip()

    def _fetch_github_repo_contents(self, user: str, repo: str) -> str:
        api_url = f"https://api.github.com/repos/{user}/{repo}/contents"
        try:
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                items = json.loads(response.read().decode('utf-8'))
                files = [item['name'] for item in items if isinstance(item, dict) and 'name' in item]
                return f"GitHub Repository '{user}/{repo}' files: {', '.join(files)}"
        except Exception as e:
            return f"Failed to fetch repo '{user}/{repo}': {e}"

    def _fetch_github_file_via_api(self, user: str, repo: str, file_path: str) -> str:
        api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{file_path}"
        try:
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode('utf-8'))
                if "content" in payload:
                    raw_bytes = base64.b64decode(payload["content"])
                    return raw_bytes.decode('utf-8', errors='ignore')[:3000]
        except Exception as e:
            return f"Error fetching '{file_path}': {e}"
        return f"File '{file_path}' not found in '{user}/{repo}'."

    def _read_local_file_preview(self, dir_path: str, filename: str) -> str:
        target_path = os.path.join(dir_path, filename)
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [line.rstrip() for line in f.readlines()[:20]]
                    return "\n".join(lines)
            except Exception as e:
                return f"Error reading file '{filename}': {e}"
        return f"File '{filename}' not found in target directory '{dir_path}'."

    def _read_readme_content(self, dir_path: str) -> str:
        if not os.path.exists(dir_path):
            return ""
        for name in ["README.md", "readme.md", "README.txt", "README"]:
            target_file = os.path.join(dir_path, name)
            if os.path.exists(target_file):
                try:
                    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read(4000)
                except Exception:
                    pass
        return ""

    def _resolve_prompt_context(self, prompt: str) -> str:
        """Inspect prompt for tool invocations (workspace inspection, github previews)

        and prepend contextual snippets to the prompt text.
        """
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        user_line = ""
        for line in reversed(lines):
            if not line.startswith("<|im_") and not line.startswith("Workspace") and not line.startswith("Codebase") and not line.startswith("Files:"):
                user_line = line.replace("<|im_end|>", "").strip()
                break
        if not user_line:
            return prompt

        user_lower = user_line.lower()

        # Tool 1: GitHub URL Fetch
        url_match = re.search(r'github\.com/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)', user_line)
        if url_match:
            self.active_github_user = self._clean_github_slug(url_match.group(1))
            self.active_github_repo = self._clean_github_slug(url_match.group(2))
            self.last_web_data = self._fetch_github_repo_contents(self.active_github_user, self.active_github_repo)
            return f"Context:\n{self.last_web_data}\n\nUser: {prompt}"

        # Tool 2: Local file preview
        if any(kw in user_lower for kw in ["preview", "read", "show", "cat", "explain"]) and any(ext in user_lower for ext in [".py", ".md", ".txt", ".sql", ".json", ".sh"]):
            file_match = re.search(r'([a-zA-Z0-9_\-\.]+\.(py|md|txt|sql|json|sh))', user_line)
            target_file = file_match.group(1) if file_match else "app.py"
            if self.active_target_dir:
                preview = self._read_local_file_preview(self.active_target_dir, target_file)
                return f"Context from {target_file}:\n{preview}\n\nUser: {prompt}"

        return prompt

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(self, q: torch.Tensor, k: torch.Tensor, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = q.shape[-1]
        inv_freq = 1.0 / (self.config.rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.tensor([pos], dtype=torch.float32, device=q.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)
        return q_out, k_out

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to compute device and dtype if needed."""
        if tensor.device != self.device or tensor.dtype != self.compute_dtype:
            return tensor.to(self.device, dtype=self.compute_dtype, non_blocking=True)
        return tensor

    def forward_layer(
        self,
        layer_idx: int,
        h: torch.Tensor,
        pos: int,
    ) -> torch.Tensor:
        """Executes a single transformer layer using actual model weights and MoE routing."""
        cfg = self.config

        # 1. Attention Pre-norm
        attn_norm_w = self._to_device(self.weights.get_torch(f"blk.{layer_idx}.attn_norm.weight"))
        h_norm = self._rms_norm(h, attn_norm_w, cfg.rms_eps)

        # 2. Q, K, V projections
        def project(proj_name: str, heads: int) -> torch.Tensor:
            proj = torch.matmul(h_norm, self._to_device(self.weights.get_torch(f"blk.{layer_idx}.attn_{proj_name}.weight")).t())
            bias_name = f"blk.{layer_idx}.attn_{proj_name}.bias"
            if bias_name in self.loader.tensors:
                proj = proj + self._to_device(self.weights.get_torch(bias_name))
            return proj.view(1, heads, cfg.head_dim)

        q = project("q", cfg.n_head)
        k = project("k", cfg.n_head_kv)
        v = project("v", cfg.n_head_kv)

        # Optional Q/K head normalization
        q_norm_key = f"blk.{layer_idx}.attn_q_norm.weight"
        if q_norm_key in self.loader.tensors:
            q = self._rms_norm(q, self._to_device(self.weights.get_torch(q_norm_key)), cfg.rms_eps)
        k_norm_key = f"blk.{layer_idx}.attn_k_norm.weight"
        if k_norm_key in self.loader.tensors:
            k = self._rms_norm(k, self._to_device(self.weights.get_torch(k_norm_key)), cfg.rms_eps)

        # Rotary position embeddings
        q, k = self._apply_rotary_emb(q, k, pos)

        # KV Cache via paged manager
        # pos is the absolute position in the sequence
        self.kv_cache_manager.append(layer_idx, k, v)
        k_cat, v_cat = self.kv_cache_manager.get_kv(layer_idx, 0, pos + 1)

        # Grouped Query Attention (GQA)
        rep = cfg.n_head // cfg.n_head_kv
        k_expanded = k_cat.repeat_interleave(rep, dim=1)  # [seq_len, n_head, head_dim]
        v_expanded = v_cat.repeat_interleave(rep, dim=1)  # [seq_len, n_head, head_dim]

        # Attention scores - compute in float32 for einsum compatibility
        q_f32 = q.float()
        k_expanded_f32 = k_expanded.float()
        scale = torch.tensor(1.0 / math.sqrt(cfg.head_dim), dtype=torch.float32, device=q.device)
        scores = torch.einsum("bhd,shd->bhs", q_f32, k_expanded_f32) * scale
        probs = F.softmax(scores, dim=-1).to(self.compute_dtype)
        attn_out = torch.einsum("bhs,shd->bhd", probs, v_expanded).reshape(1, -1)
        attn_proj = torch.matmul(attn_out, self._to_device(self.weights.get_torch(f"blk.{layer_idx}.attn_output.weight")).t())

        h = h + attn_proj

        # 3. FFN Pre-norm
        ffn_norm_w = self._to_device(self.weights.get_torch(f"blk.{layer_idx}.ffn_norm.weight"))
        h_ffn_norm = self._rms_norm(h, ffn_norm_w, cfg.rms_eps)

        router_key = f"blk.{layer_idx}.ffn_gate_inp.weight"
        if router_key in self.loader.tensors:
            # MoE: route with the checkpoint's learned gate and execute selected experts.
            router_w = self._to_device(self.weights.get_torch(router_key))
            # Track current layer for prefetch callback
            self._current_layer_idx = layer_idx
            active_indices, active_weights = self.router.select_experts(h_ffn_norm, router_w.t())
            # Callback already dispatched page IDs for current + next layer
            # No need for legacy prefetcher call

            ffn_out = torch.zeros_like(h)
            for w, exp_idx in zip(active_weights[0], active_indices[0]):
                idx = int(exp_idx)
                gate = self.weights.get_expert(layer_idx, "gate", idx)  # already on GPU from pager
                up = self.weights.get_expert(layer_idx, "up", idx)
                down = self.weights.get_expert(layer_idx, "down", idx)

                act = F.silu(torch.matmul(h_ffn_norm, gate.t())) * torch.matmul(h_ffn_norm, up.t())
                ffn_out += w * torch.matmul(act, down.t())
        else:
            # Dense SwiGLU FFN (for example Qwen2.5 7B): no router or expert tensors.
            gate_key = f"blk.{layer_idx}.ffn_gate.weight"
            up_key = f"blk.{layer_idx}.ffn_up.weight"
            down_key = f"blk.{layer_idx}.ffn_down.weight"
            missing = [key for key in (gate_key, up_key, down_key) if key not in self.loader.tensors]
            if missing:
                raise KeyError(
                    f"Layer {layer_idx} has no MoE router ({router_key}) and is missing "
                    f"dense FFN tensors: {', '.join(missing)}"
                )
            gate = self._to_device(self.weights.get_torch(gate_key))
            up = self._to_device(self.weights.get_torch(up_key))
            down = self._to_device(self.weights.get_torch(down_key))
            act = F.silu(torch.matmul(h_ffn_norm, gate.t())) * torch.matmul(h_ffn_norm, up.t())
            ffn_out = torch.matmul(act, down.t())

        h = h + ffn_out
        return h

    def forward_token(
        self,
        token_id: int,
        pos: int,
    ) -> torch.Tensor:
        """Executes a single token forward pass through the transformer backbone.

        Wrapped in ``torch.inference_mode()`` so inference never builds or retains
        an autograd graph (weights are ``requires_grad=False`` today, but this
        keeps a stray grad-requiring tensor from turning into a RAM leak).
        """
        with torch.inference_mode():
            embed_table = self._to_device(self.weights.get_torch("token_embd.weight"))
            h = embed_table[token_id].unsqueeze(0)  # [1, n_embd]
            h = self._to_device(h)

            for layer_idx in range(self.num_layers):
                h = self.forward_layer(layer_idx, h, pos)

            output_norm_w = self._to_device(self.weights.get_torch("output_norm.weight"))
            h_norm = self._rms_norm(h, output_norm_w, self.config.rms_eps)
            output_w = self._to_device(self.weights.get_torch("output.weight"))
            logits = torch.matmul(h_norm, output_w.t())  # [1, vocab_size]
            return logits.squeeze(0)

    def stream_inference(self, prompt: str, max_new_tokens: int = 250) -> Generator[str, None, None]:
        """Streams autoregressively generated tokens evaluated on actual model weights."""
        resolved_prompt = self._resolve_prompt_context(prompt)
        prompt_tokens = self.tokenizer.encode(resolved_prompt)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.encode("Hello")[0]]

        # Clear KV cache at start of new sequence
        self.kv_cache_manager.clear()

        # 1. Prefill prompt tokens
        logits = None
        for pos, tok in enumerate(prompt_tokens):
            logits = self.forward_token(tok, pos)

        # 2. Autoregressive token generation
        curr_pos = len(prompt_tokens)
        for _ in range(max_new_tokens):
            if logits is None:
                break
            next_token_id = self.sampler.sample_next_token(logits)
            if next_token_id == self.config.eos_token_id:
                break

            token_str = self.tokenizer.decode([next_token_id])
            yield token_str

            logits = self.forward_token(next_token_id, curr_pos)
            curr_pos += 1

    def run_inference_step(self, prompt: str, max_new_tokens: int = 250) -> str:
        tokens = []
        for token in self.stream_inference(prompt, max_new_tokens=max_new_tokens):
            print(token, end="", flush=True)
            tokens.append(token)
        print()
        return "".join(tokens)

    def get_cache_stats(self) -> dict:
        pager_stats = {}
        if hasattr(self.weights, "pager") and self.weights.pager is not None:
            pager_stats = self.weights.pager.stats
        prefetch_stats = {}
        if hasattr(self, "prefetcher") and self.prefetcher is not None:
            prefetch_stats = self.prefetcher.get_stats()
        return {
            "cached_pages": int(pager_stats.get("ram_hits", 0) + pager_stats.get("ssd_page_faults", 0)),
            "pager_ram_hits": int(pager_stats.get("ram_hits", 0)),
            "pager_ssd_faults": int(pager_stats.get("ssd_page_faults", 0)),
            "pager_ram_evictions": int(pager_stats.get("ram_evictions", 0)),
            "pager_vram_hits": int(pager_stats.get("vram_hits", 0)),
            "pager_vram_evictions": int(pager_stats.get("vram_evictions", 0)),
            "pager_pcie_transfers": int(pager_stats.get("pcie_transfers", 0)),
            "pager_decomp_time_ms": round(pager_stats.get("decompression_time_ms", 0.0), 2),
            "prefetch_queue_size": prefetch_stats.get("queue_size", 0),
            "prefetch_pending": prefetch_stats.get("pending_count", 0),
            "prefetch_workers": prefetch_stats.get("workers", 0),
        }

    def shutdown(self):
        self.prefetcher.stop()
        self.mmap_mgr.close()


if __name__ == "__main__":
    MODEL_DIR = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    if os.path.exists(MODEL_DIR):
        pipeline = A11Pipeline(MODEL_DIR, max_layers=2)
        try:
            sample_out = pipeline.run_inference_step("Hello", max_new_tokens=3)
            print(f"\nGenerated: {sample_out}")
        finally:
            pipeline.shutdown()
