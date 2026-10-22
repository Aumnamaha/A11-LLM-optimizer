"""Tests for the Phase 5/7 MemoryPager (no model files required)."""
import pytest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

from engine.pager import ModelPageTable, PageTableEntry, RAMSlotPool, VRAMSlotPool, MemoryPager


class _FakeCUDATensor(torch.Tensor):
    """Fake tensor that mimics CUDA behavior without requiring actual CUDA."""
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return torch.Tensor.__new__(cls, *args, **kwargs)
    
    @property
    def device(self):
        return torch.device("cuda")


def _make_fake_cuda_tensor(*shape):
    """Create a tensor that reports as CUDA without requiring actual CUDA."""
    t = torch.randn(*shape)
    # Monkey-patch the device property
    t._mock_cuda = True
    orig_device = type(t).device
    def mock_device(self):
        return torch.device("cuda")
    type(t).device = property(mock_device)
    return t


class FakeLoader:
    def __init__(self):
        self.tensors = {
            "blk.0.ffn_gate_exps.weight": {
                "shape": [2048, 768, 128],
                "type": "12",  # Q4_K
                "offset": 1000,
                "shard": "test.gguf",
            },
            "blk.0.ffn_up_exps.weight": {
                "shape": [2048, 768, 128],
                "type": "12",
                "offset": 2000,
                "shard": "test.gguf",
            },
            "blk.0.ffn_down_exps.weight": {
                "shape": [768, 2048, 128],
                "type": "14",  # Q6_K
                "offset": 3000,
                "shard": "test.gguf",
            },
        }


class FakeMMapManager:
    def __init__(self):
        self._mmap = b"\x00" * 1000000  # 1MB fake mmap

    def get_mmap(self, shard_key):
        return self._mmap


class FakePageTable(ModelPageTable):
    def __init__(self):
        self.entries = {}
        self.by_tensor_expert_page = {}
        # Add a single-page expert entry
        e = PageTableEntry(
            id="E000:L00:TGATE:P000",
            layer=0, expert=0, tensor="gate", page=0,
            pages_per_expert=1,
            byte_offset=1000, byte_length=1000, shard="test.gguf",
        )
        self.entries[e.id] = e
        self.by_tensor_expert_page[("blk.0.ffn_gate_exps.weight", 0, 0)] = e.id


def test_ram_slot_pool_basic():
    pool = RAMSlotPool(3)
    assert pool.free_slots == 3
    s1 = pool.acquire()
    assert pool.free_slots == 2
    s2 = pool.acquire()
    assert pool.free_slots == 1
    pool.put(s1, b"data1")
    pool.put(s2, b"data2")
    assert pool.get(s1) == b"data1"
    assert pool.get(s2) == b"data2"
    pool.release(s1)
    assert pool.free_slots == 2


def test_ram_slot_pool_eviction():
    pool = RAMSlotPool(2)
    s1 = pool.acquire()
    pool.put(s1, b"data1")
    s2 = pool.acquire()
    pool.put(s2, b"data2")
    assert pool.free_slots == 0
    # next acquire should evict LRU (s1)
    s3 = pool.acquire()
    assert pool.evictions == 1
    assert s3 == s1  # evicted slot reused
    assert pool.get(s1) is None  # old data gone
    pool.put(s3, b"data3")
    assert pool.get(s3) == b"data3"


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
def test_vram_slot_pool_basic(mock_device_count, mock_cuda_avail):
    """Test VRAM slot pool with mocked CUDA device."""
    # Create a mock device object
    mock_device = MagicMock()
    mock_device.type = "cuda"
    mock_device.index = 0
    
    pool = VRAMSlotPool(2, mock_device)
    assert pool.free_slots == 2
    
    # Proper API: acquire slot, then put tensor
    s1 = pool.acquire()
    pool.put(s1, torch.randn(10, 10))
    s2 = pool.acquire()
    pool.put(s2, torch.randn(10, 10))
    assert pool.get(s1) is not None
    assert pool.get(s2) is not None
    assert pool.free_slots == 0


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
def test_vram_slot_pool_eviction(mock_device_count, mock_cuda_avail):
    """Test VRAM eviction drops GPU tensor (source of truth is RAM)."""
    mock_device = MagicMock()
    mock_device.type = "cuda"
    mock_device.index = 0
    
    pool = VRAMSlotPool(2, mock_device)
    s1 = pool.acquire()
    pool.put(s1, torch.randn(10, 10))
    s2 = pool.acquire()
    pool.put(s2, torch.randn(10, 10))
    assert pool.free_slots == 0
    # Evict LRU (slot s1)
    evicted_slot = pool.acquire()  # should evict s1
    assert evicted_slot == s1
    assert pool.evictions == 1
    assert pool.get(s1) is None  # evicted tensor gone
    # Reuse slot
    t3 = torch.randn(10, 10)
    pool.put(evicted_slot, t3)
    assert pool.get(evicted_slot) is t3


def test_model_page_table_from_dict():
    pt = FakePageTable()
    entry = pt.get("E000:L00:TGATE:P000")
    assert entry is not None
    assert entry.layer == 0
    assert entry.expert == 0
    assert entry.tensor == "gate"
    assert entry.pages_per_expert == 1
    assert pt.get("nonexistent") is None


@patch("engine.pager.dequantize")
def test_memory_pager_ram_only(mock_dequantize):
    """Test pager with RAM tier only (n_slots_vram=0)."""
    mock_dequantize.return_value = np.arange(768 * 2048, dtype=np.float32)

    loader = FakeLoader()
    mmap_mgr = FakeMMapManager()
    page_table = FakePageTable()
    pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=10, n_slots_vram=0)

    t = pager.get_page("E000:L00:TGATE:P000")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (768, 2048)
    assert t.device.type == "cpu"
    assert pager.stats["ssd_page_faults"] == 1
    assert pager.stats["ram_hits"] == 0

    # second access = RAM hit
    t2 = pager.get_page("E000:L00:TGATE:P000")
    assert t2 is t
    assert pager.stats["ram_hits"] == 1
    assert pager.stats["ssd_page_faults"] == 1


def test_memory_pager_ram_eviction():
    import numpy as np
    from engine.pager import MemoryPager

    loader = FakeLoader()
    mmap_mgr = FakeMMapManager()
    page_table = FakePageTable()
    e2 = PageTableEntry(
        id="E001:L00:TGATE:P000",
        layer=0, expert=1, tensor="gate", page=0,
        pages_per_expert=1,
        byte_offset=2000, byte_length=1000, shard="test.gguf",
    )
    page_table.entries[e2.id] = e2
    page_table.by_tensor_expert_page[("blk.0.ffn_gate_exps.weight", 1, 0)] = e2.id

    pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=1, n_slots_vram=0)

    with patch("engine.pager.dequantize") as mock_dq:
        mock_dq.return_value = np.arange(768 * 2048, dtype=np.float32)
        t1 = pager.get_page("E000:L00:TGATE:P000")
        assert pager.stats["ssd_page_faults"] == 1
        t2 = pager.get_page("E001:L00:TGATE:P000")
        assert pager.stats["ssd_page_faults"] == 2
        assert pager.stats["ram_evictions"] == 1
        assert "E000:L00:TGATE:P000" not in pager._entry_to_ram_slot


@patch("engine.pager.dequantize")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3))
def test_memory_pager_vram_transfer(mock_mem_get_info, mock_device_count, mock_cuda_avail, mock_dequantize):
    """Test full 3-tier hierarchy: SSD → RAM → VRAM transfer."""
    mock_dequantize.return_value = np.arange(768 * 2048, dtype=np.float32)

    loader = FakeLoader()
    mmap_mgr = FakeMMapManager()
    page_table = FakePageTable()

    # Mock device
    mock_device = MagicMock()
    mock_device.type = "cuda"
    mock_device.index = 0

    # Mock tensor.to() to simulate GPU transfer
    original_to = torch.Tensor.to
    def mock_to(self, device, **kwargs):
        if getattr(device, 'type', None) == "cuda":
            # Return a new tensor with mocked cuda device
            # Handle bfloat16 by converting to float32 first
            if self.dtype == torch.bfloat16:
                new_t = torch.tensor(self.to(torch.float32).numpy())
            else:
                new_t = torch.tensor(self.numpy())
            new_t._mock_cuda = True
            type(new_t).device = property(lambda self: mock_device)
            return new_t
        return original_to(self, device, **kwargs)

    with patch.object(torch.Tensor, "to", mock_to):
        pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=10, n_slots_vram=5, vram_device=mock_device)
        
        assert pager._vram_enabled is True
        assert pager.vram_pool is not None
        assert pager.vram_pool.n_slots == 5

        # First access: SSD → RAM → VRAM
        t = pager.get_page("E000:L00:TGATE:P000")
        assert t.device.type == "cuda"
        assert pager.stats["ssd_page_faults"] == 1
        assert pager.stats["ram_hits"] == 0
        assert pager.stats["vram_hits"] == 0
        assert pager.stats["pcie_transfers"] == 1

        # Second access: VRAM hit
        t2 = pager.get_page("E000:L00:TGATE:P000")
        assert t2.device.type == "cuda"
        assert pager.stats["vram_hits"] == 1
        assert pager.stats["pcie_transfers"] == 1  # no additional transfer


@patch("engine.pager.dequantize")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3))
def test_memory_pager_vram_eviction(mock_mem_get_info, mock_device_count, mock_cuda_avail, mock_dequantize):
    """Test VRAM eviction when VRAM pool is full."""
    mock_dequantize.return_value = np.arange(768 * 2048, dtype=np.float32)

    loader = FakeLoader()
    mmap_mgr = FakeMMapManager()
    page_table = FakePageTable()

    # Add second page
    e2 = PageTableEntry(
        id="E001:L00:TGATE:P000",
        layer=0, expert=1, tensor="gate", page=0,
        pages_per_expert=1,
        byte_offset=2000, byte_length=1000, shard="test.gguf",
    )
    page_table.entries[e2.id] = e2
    page_table.by_tensor_expert_page[("blk.0.ffn_gate_exps.weight", 1, 0)] = e2.id

    mock_device = MagicMock()
    mock_device.type = "cuda"
    mock_device.index = 0

    def mock_to_vram(self, device, **kwargs):
        if getattr(device, 'type', None) == "cuda":
            if self.dtype == torch.bfloat16:
                new_t = torch.tensor(self.to(torch.float32).numpy())
            else:
                new_t = torch.tensor(self.numpy())
            new_t._mock_cuda = True
            type(new_t).device = property(lambda self: device)
            return new_t
        return self

    with patch.object(torch.Tensor, "to", mock_to_vram):
        pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=10, n_slots_vram=1, vram_device=mock_device)

        # Fill VRAM with first page
        t1 = pager.get_page("E000:L00:TGATE:P000")
        assert pager.stats["vram_evictions"] == 0
        assert pager.stats["pcie_transfers"] == 1

        # Second page triggers VRAM eviction
        t2 = pager.get_page("E001:L00:TGATE:P000")
        assert pager.stats["vram_evictions"] == 1
        assert pager.stats["pcie_transfers"] == 2

        # First page still in RAM tier (source of truth)
        assert "E000:L00:TGATE:P000" in pager._entry_to_ram_slot


def test_memory_pager_stats():
    import numpy as np
    loader = FakeLoader()
    mmap_mgr = FakeMMapManager()
    page_table = FakePageTable()
    pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=10, n_slots_vram=0)

    with patch("engine.pager.dequantize") as mock_dq:
        mock_dq.return_value = np.arange(768 * 2048, dtype=np.float32)
        _ = pager.get_page("E000:L00:TGATE:P000")
        stats = pager.stats
        assert "ssd_page_faults" in stats
        assert "ram_hits" in stats
        assert "ram_evictions" in stats
        assert "vram_hits" in stats
        assert "vram_evictions" in stats
        assert "pcie_transfers" in stats
        assert "decompression_time_ms" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])