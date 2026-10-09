import torch
import torch.nn.functional as F

class TokenSampler:
    """
    Applies Temperature scaling, Top-K filtering, and Top-P (Nucleus) 
    sampling to raw model logits to produce coherent token predictions.
    """
    def __init__(self, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9):
        self.temperature = max(temperature, 1e-5)
        self.top_k = top_k
        self.top_p = top_p

    def sample_next_token(self, logits: torch.Tensor) -> int:
        # Apply Temperature
        scaled_logits = logits / self.temperature
        
        # Apply Top-K Filtering
        if self.top_k > 0:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, self.top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float('-inf')

        # Apply Top-P (Nucleus) Filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = float('-inf')

        # Softmax to probabilities & sample
        probs = F.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

if __name__ == "__main__":
    sampler = TokenSampler(temperature=0.7, top_k=40, top_p=0.9)
    dummy_logits = torch.randn(32000)
    # Bias specific token index to verify sampling priority
    dummy_logits[3838] += 10.0 
    sampled_id = sampler.sample_next_token(dummy_logits)
    print(f"[Sampler Test] Sampled Token ID: {sampled_id}")
