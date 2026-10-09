import os
from typing import List
from transformers import AutoTokenizer

class QwenTokenizerManager:
    """
    Tokenizer interface for Qwen series models. Saves vocabulary locally
    to prevent remote network lookups during offline execution.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cache_dir = os.path.join(model_dir, "tokenizer_cache")
        self.tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        print(f"\n[Tokenizer Manager] Loading Qwen Tokenizer Vocabulary...")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            # Try loading from local disk cache first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cache_dir, 
                trust_remote_code=True,
                local_files_only=os.path.exists(os.path.join(self.cache_dir, "tokenizer.json"))
            )
            print("  • Loaded tokenizer from local cache directory.")
        except Exception:
            try:
                # Fetch online once and save locally
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-7B-Instruct", 
                    trust_remote_code=True
                )
                self.tokenizer.save_pretrained(self.cache_dir)
                print(f"  • Tokenizer downloaded and cached locally at '{self.cache_dir}'.")
            except Exception as e:
                print(f"  [Warning] Tokenizer initialization fallback: {e}")
                self.tokenizer = None

    def encode(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [ord(c) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        return "".join([chr(tid) if tid < 256 else "?" for tid in token_ids])

if __name__ == "__main__":
    target_dir = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    tok_mgr = QwenTokenizerManager(target_dir)
    
    sample_text = "What caused World War 2?"
    tokens = tok_mgr.encode(sample_text)
    decoded = tok_mgr.decode(tokens)
    
    print(f"[Tokenizer Test]")
    print(f"  • Input Text: '{sample_text}'")
    print(f"  • Encoded Tokens ({len(tokens)}): {tokens[:8]}...")
    print(f"  • Decoded Output: '{decoded}'")
