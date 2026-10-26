import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MoERouter:
    """
    Evaluates router/gating weights (ffn_gate_inp) per token 
    to determine the Top-K active experts for execution.

    Supports optional early logit dispatch for asynchronous prefetching.
    """

    def __init__(self, num_experts: int = 512, top_k: int = 8):
        self.num_experts = num_experts
        self.top_k = top_k
        self._prefetch_callback = None

    def set_prefetch_callback(self, callback):
        """Register a callback to be invoked with (layer, kind, expert_indices) 
        immediately after Top-K selection."""
        self._prefetch_callback = callback

    def select_experts(self, hidden_states: torch.Tensor, router_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates gating logits and returns selected expert indices & routing weights.
        """
        # hidden_states: [batch, hidden_dim]
        # router_weights: [hidden_dim, num_experts]
        logits = torch.matmul(hidden_states, router_weights)
        routing_weights = F.softmax(logits, dim=-1)
        
        # Extract Top-K active experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Early dispatch: invoke prefetch callback with selected experts
        # The callback receives (layer, kind, expert_indices) and should
        # enqueue the corresponding page IDs.
        if self._prefetch_callback is not None:
            self._prefetch_callback(top_k_indices)
        
        return top_k_indices, top_k_weights


if __name__ == "__main__":
    router = MoERouter(num_experts=512, top_k=8)
    dummy_hidden = torch.randn(1, 4096)
    dummy_router_w = torch.randn(4096, 512)
    
    indices, weights = router.select_experts(dummy_hidden, dummy_router_w)
    print(f"[Router Test] Selected Top-8 Experts: {indices[0].tolist()}")