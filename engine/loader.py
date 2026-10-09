import os
import glob
from typing import List, Dict
import gguf

class SplitGGUFLoader:
    """
    Parses and streams sharded GGUF models, mapping tensor offsets 
    across all shards for memory-mapped (mmap) disk access.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.shards = self._find_shards()
        self.metadata: Dict[str, str] = {}
        self.tensors: Dict[str, Dict] = {}
        self.ngram_tensors: List[str] = []
        self.expert_tensors: List[str] = []
        self._parse_all_shards()

    def _find_shards(self) -> List[str]:
        pattern = os.path.join(self.model_dir, "*.gguf")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .gguf model files found in: {self.model_dir}")
        
        print(f"[Loader] Detected {len(files)} model shard(s):")
        for f in files:
            size_gb = os.path.getsize(f) / (1024 ** 3)
            print(f"  • {os.path.basename(f)} ({size_gb:.2f} GB)")
        return files

    def _parse_all_shards(self):
        print(f"\n[GGUF Parser] Scanning all shards for metadata & tensor structures...")
        
        total_tensors_found = 0
        for shard_path in self.shards:
            shard_name = os.path.basename(shard_path)
            try:
                reader = gguf.GGUFReader(shard_path)
                
                # Extract metadata from primary shard
                if not self.metadata:
                    for key, field in reader.fields.items():
                        if field.data:
                            val = field.parts[field.data[0]]
                            if isinstance(val, (bytes, bytearray)):
                                self.metadata[key] = val.decode('utf-8', errors='ignore')
                            elif isinstance(val, list):
                                try:
                                    self.metadata[key] = bytes(val).decode('utf-8', errors='ignore')
                                except Exception:
                                    self.metadata[key] = str(val)
                            else:
                                self.metadata[key] = str(val)

                # Scan tensors in this shard
                num_tensors = len(reader.tensors)
                total_tensors_found += num_tensors
                
                for idx, tensor in enumerate(reader.tensors):
                    t_name = tensor.name
                    # Get tensor offset safely across different gguf-py versions
                    data_offset = getattr(tensor, 'data_offset', None)
                    if data_offset is None:
                        data_offset = getattr(tensor, 'field', None)

                    self.tensors[t_name] = {
                        "shard": shard_name,
                        "shape": tensor.shape.tolist(),
                        "type": str(tensor.tensor_type),
                        "offset": data_offset
                    }
                    
                    # Categorize N-Gram vs Expert weights for Qwen architectures
                    t_lower = t_name.lower()
                    if any(k in t_lower for k in ["ngram", "lookup", "embed", "token_embd", "l2"]):
                        self.ngram_tensors.append(t_name)
                    elif any(k in t_lower for k in ["exps", "expert", "ffn", "mlp", "block"]):
                        self.expert_tensors.append(t_name)

            except Exception as e:
                print(f"  [Warning] Failed to read shard {shard_name}: {e}")

        arch_name = self.metadata.get('general.architecture', 'qwen4exp')
        print(f"  • Architecture: {arch_name}")
        print(f"  • Total Tensors Indexed Across Shards: {total_tensors_found}")
        print(f"  • N-Gram / Embedding Tensors: {len(self.ngram_tensors)}")
        print(f"  • MoE Expert Layer Tensors: {len(self.expert_tensors)}")

    def get_primary_shard(self) -> str:
        return self.shards[0]

if __name__ == "__main__":
    target_dir = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    if os.path.exists(target_dir):
        loader = SplitGGUFLoader(target_dir)
