import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Minimum supported hardware envelope for the paged-MoE runtime.
#
# These are *requirements*, not hard-coded budgets. Cache tiers are auto-sized
# from the detected hardware (see evaluate_compatibility); this envelope only
# defines the floor the runtime is designed to operate within, so a machine at
# or above it is supported and anything below it is flagged as constrained.
# ---------------------------------------------------------------------------
MIN_VRAM_GB = 2.0
MIN_RAM_GB = 8.0


@dataclass
class HardwareProfile:
    """Snapshot of host system memory and GPU hardware capabilities."""
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    ram_utilization_pct: float
    gpu_detected: bool
    gpu_name: str
    gpu_backend: str  # "cuda", "rocm", "drm_sysfs", "nvidia_smi", "mps", "cpu"
    total_vram_gb: float
    free_vram_gb: float
    has_discrete_gpu: bool
    cpu_count: int
    platform_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelHardwareRequirements:
    """Estimated memory footprints for the loaded model architecture."""
    model_name: str
    total_model_size_gb: float
    core_heads_size_gb: float  # Attn, Norm, Router, Gate, Embeddings, Output
    moe_experts_size_gb: float
    ngram_size_gb: float
    min_ram_required_gb: float
    recommended_ram_gb: float
    recommended_vram_gb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareCompatibilityReport:
    """Hardware compatibility analysis and final assigned memory budgets."""
    is_compatible: bool
    status: str  # "OPTIMAL", "COMPATIBLE", "CONSTRAINED", "INSUFFICIENT_MEMORY"
    effective_vram_budget_gb: float
    effective_ram_budget_gb: float
    max_hot_experts: int
    offload_strategy: str
    messages: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_system_ram() -> Tuple[float, float, float, float]:
    """
    Detects (total_ram_gb, available_ram_gb, used_ram_gb, utilization_pct).
    Prefers psutil, with fallbacks for /proc/meminfo and platform-specific commands.
    """
    # 1. psutil
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        avail_gb = vm.available / (1024 ** 3)
        used_gb = vm.used / (1024 ** 3)
        pct = vm.percent
        return round(total_gb, 2), round(avail_gb, 2), round(used_gb, 2), round(pct, 1)
    except Exception:
        pass

    # 2. Linux /proc/meminfo fallback
    if os.path.exists("/proc/meminfo"):
        try:
            mem_info: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip().split()[0]
                        if val_str.isdigit():
                            mem_info[key] = int(val_str)  # kB
            if "MemTotal" in mem_info:
                total_kb = mem_info["MemTotal"]
                avail_kb = mem_info.get("MemAvailable", mem_info.get("MemFree", total_kb // 2))
                total_gb = (total_kb * 1024) / (1024 ** 3)
                avail_gb = (avail_kb * 1024) / (1024 ** 3)
                used_gb = max(0.0, total_gb - avail_gb)
                pct = (used_gb / max(total_gb, 0.001)) * 100
                return round(total_gb, 2), round(avail_gb, 2), round(used_gb, 2), round(pct, 1)
        except Exception:
            pass

    # 3. macOS sysctl fallback
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            total_bytes = int(out)
            total_gb = total_bytes / (1024 ** 3)
            avail_gb = total_gb * 0.5  # estimate
            return round(total_gb, 2), round(avail_gb, 2), round(total_gb - avail_gb, 2), 50.0
        except Exception:
            pass

    # Fallback default
    return 16.0, 8.0, 8.0, 50.0


def _query_drm_gpu_devices() -> List[Dict[str, Any]]:
    """
    Scans Linux DRM subsystem (/sys/class/drm/card*) for AMD/Intel/NVIDIA GPU devices
    and their dedicated VRAM totals and usage.
    """
    devices: List[Dict[str, Any]] = []
    card_dirs = sorted(glob.glob("/sys/class/drm/card[0-9]*"))
    for card_dir in card_dirs:
        # Avoid sub-connectors like card1-HDMI-A-1
        if not os.path.isdir(os.path.join(card_dir, "device")):
            continue

        tot_file = os.path.join(card_dir, "device/mem_info_vram_total")
        used_file = os.path.join(card_dir, "device/mem_info_vram_used")

        if not os.path.exists(tot_file):
            continue

        try:
            with open(tot_file, "r", encoding="utf-8") as f:
                tot_bytes = int(f.read().strip())
        except Exception:
            continue

        used_bytes = 0
        if os.path.exists(used_file):
            try:
                with open(used_file, "r", encoding="utf-8") as f:
                    used_bytes = int(f.read().strip())
            except Exception:
                used_bytes = 0

        free_bytes = max(0, tot_bytes - used_bytes)
        card_name = os.path.basename(card_dir)

        # Retrieve GPU descriptive name
        gpu_name = f"GPU ({card_name})"
        uevent_path = os.path.join(card_dir, "device/uevent")
        slot_name = None
        if os.path.exists(uevent_path):
            try:
                with open(uevent_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("PCI_SLOT_NAME="):
                            slot_name = line.strip().split("=")[1]
                            break
            except Exception:
                pass

        if slot_name and shutil.which("lspci"):
            try:
                lspci_out = subprocess.check_output(["lspci", "-s", slot_name], text=True, timeout=2).strip()
                if lspci_out:
                    # Strip leading PCI address & controller class prefix
                    clean_name = re.sub(r"^[0-9a-fA-F:\.]+\s+[^:]+:\s*", "", lspci_out)
                    gpu_name = clean_name.strip()
            except Exception:
                pass

        devices.append({
            "name": gpu_name,
            "total_bytes": tot_bytes,
            "free_bytes": free_bytes,
            "total_gb": round(tot_bytes / (1024 ** 3), 2),
            "free_gb": round(free_bytes / (1024 ** 3), 2),
            "card": card_name
        })

    return devices


def _query_nvidia_smi() -> List[Dict[str, Any]]:
    """Queries nvidia-smi if installed for GPU name and VRAM."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, text=True, timeout=3).strip()
        devices = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                tot_mb = float(parts[1])
                free_mb = float(parts[2])
                devices.append({
                    "name": name,
                    "total_gb": round(tot_mb / 1024.0, 2),
                    "free_gb": round(free_mb / 1024.0, 2),
                    "total_bytes": int(tot_mb * 1024 * 1024),
                    "free_bytes": int(free_mb * 1024 * 1024),
                })
        return devices
    except Exception:
        return []


def detect_hardware() -> HardwareProfile:
    """
    Performs full system hardware inspection across CPU, System RAM, and GPU VRAM.
    Inspects PyTorch CUDA/ROCm, Linux DRM sysfs (AMD/Intel), nvidia-smi, and macOS MPS.
    """
    total_ram, avail_ram, used_ram, ram_pct = _detect_system_ram()
    cpu_count = os.cpu_count() or 4
    platform_name = f"{platform.system()} ({platform.machine()})"

    gpu_detected = False
    gpu_name = "None (CPU Execution)"
    gpu_backend = "cpu"
    total_vram = 0.0
    free_vram = 0.0

    # 1. PyTorch CUDA / ROCm Check
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_detected = True
            is_rocm = getattr(torch.version, "hip", None) is not None
            gpu_backend = "rocm" if is_rocm else "cuda"
            # Pick device with greatest VRAM
            best_dev = 0
            best_tot = 0
            for i in range(torch.cuda.device_count()):
                try:
                    f_b, t_b = torch.cuda.mem_get_info(i)
                    if t_b > best_tot:
                        best_tot = t_b
                        best_dev = i
                except Exception:
                    pass
            gpu_name = torch.cuda.get_device_name(best_dev)
            f_b, t_b = torch.cuda.mem_get_info(best_dev)
            total_vram = round(t_b / (1024 ** 3), 2)
            free_vram = round(f_b / (1024 ** 3), 2)
    except Exception:
        pass

    # 2. Linux DRM sysfs Check (for AMD/Intel GPUs on Linux when PyTorch CPU is loaded)
    if not gpu_detected and platform.system() == "Linux":
        drm_devs = _query_drm_gpu_devices()
        if drm_devs:
            # Pick card with highest total VRAM (prefers discrete over APU/iGPU)
            best_dev = max(drm_devs, key=lambda d: d["total_bytes"])
            if best_dev["total_gb"] >= 0.5:  # At least 512MB VRAM
                gpu_detected = True
                gpu_name = best_dev["name"]
                gpu_backend = "drm_sysfs"
                total_vram = best_dev["total_gb"]
                free_vram = best_dev["free_gb"]

    # 3. nvidia-smi Fallback
    if not gpu_detected:
        nv_devs = _query_nvidia_smi()
        if nv_devs:
            best_dev = max(nv_devs, key=lambda d: d["total_gb"])
            gpu_detected = True
            gpu_name = best_dev["name"]
            gpu_backend = "nvidia_smi"
            total_vram = best_dev["total_gb"]
            free_vram = best_dev["free_gb"]

    # 4. macOS MPS Unified Memory Check
    if not gpu_detected and platform.system() == "Darwin":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_detected = True
                gpu_name = "Apple Silicon GPU (Unified Memory)"
                gpu_backend = "mps"
                # Unified memory allows up to ~70% of available RAM as VRAM
                total_vram = round(total_ram * 0.75, 2)
                free_vram = round(avail_ram * 0.70, 2)
        except Exception:
            pass

    has_discrete = gpu_detected and total_vram >= 1.0

    return HardwareProfile(
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        used_ram_gb=used_ram,
        ram_utilization_pct=ram_pct,
        gpu_detected=gpu_detected,
        gpu_name=gpu_name,
        gpu_backend=gpu_backend,
        total_vram_gb=total_vram,
        free_vram_gb=free_vram,
        has_discrete_gpu=has_discrete,
        cpu_count=cpu_count,
        platform_name=platform_name,
    )


def analyze_model_requirements(loader: Any) -> ModelHardwareRequirements:
    """
    Computes required byte allocations for core heads, MoE experts, and N-gram tables
    based on the indexed GGUF metadata.
    """
    total_bytes = 0
    core_bytes = 0
    moe_bytes = 0
    ngram_bytes = 0

    for t_name, t_info in getattr(loader, "tensors", {}).items():
        t_lower = t_name.lower()
        t_bytes = t_info.get("n_bytes", 0)
        total_bytes += t_bytes

        if any(k in t_lower for k in ["attn", "norm", "router", "gate_inp", "output", "token_embd"]):
            core_bytes += t_bytes
        elif any(k in t_lower for k in ["exps", "ffn"]):
            moe_bytes += t_bytes
        elif any(k in t_lower for k in ["ngram", "lookup"]):
            ngram_bytes += t_bytes

    total_gb = round(total_bytes / (1024 ** 3), 2)
    core_gb = round(core_bytes / (1024 ** 3), 2)
    moe_gb = round(moe_bytes / (1024 ** 3), 2)
    ngram_gb = round(ngram_bytes / (1024 ** 3), 2)

    # Calculate model name from metadata
    metadata = getattr(loader, "metadata", {})
    model_name = metadata.get("general.name", "Qwen MoE Architecture")

    # Minimum system requirements: core heads + KV cache buffer + runtime
    # overhead. Recommendations are floored by the runtime's minimum supported
    # envelope (derived from the model, not pinned to any one machine).
    min_ram = max(2.0, round(core_gb + 1.5, 2))
    rec_ram = max(MIN_RAM_GB, round(core_gb + 4.5, 2))
    rec_vram = max(MIN_VRAM_GB, round(core_gb + 1.0, 2)) if core_gb > 0 else 0.0

    return ModelHardwareRequirements(
        model_name=model_name,
        total_model_size_gb=total_gb,
        core_heads_size_gb=core_gb,
        moe_experts_size_gb=moe_gb,
        ngram_size_gb=ngram_gb,
        min_ram_required_gb=min_ram,
        recommended_ram_gb=rec_ram,
        recommended_vram_gb=rec_vram,
    )


def evaluate_compatibility(
    profile: HardwareProfile,
    model_req: ModelHardwareRequirements,
    requested_vram_gb: Optional[float] = None,
    requested_ram_gb: Optional[float] = None,
) -> HardwareCompatibilityReport:
    """
    Evaluates hardware compatibility against model requirements, clamps requested
    budgets safely to available hardware limits, and establishes optimal MoE caching tiers.
    """
    warnings: List[str] = []
    messages: List[str] = []

    # 1. Resolve VRAM Budget
    if requested_vram_gb is None or requested_vram_gb < 0:
        if profile.has_discrete_gpu:
            # Allocate safe fraction of available VRAM
            safe_vram = round(min(profile.free_vram_gb * 0.88, profile.total_vram_gb * 0.85), 2)
            # Ensure safe VRAM covers core heads if possible
            if profile.free_vram_gb >= model_req.core_heads_size_gb:
                effective_vram = max(model_req.core_heads_size_gb, safe_vram)
                effective_vram = min(effective_vram, profile.free_vram_gb)
            else:
                effective_vram = safe_vram
        else:
            effective_vram = 0.0
    else:
        # User specified explicit budget: validate and clamp to physical hardware
        if profile.total_vram_gb == 0.0 and requested_vram_gb > 0.0:
            warnings.append(
                f"Requested VRAM budget ({requested_vram_gb:.1f} GB) cannot be fulfilled: "
                f"no dedicated GPU detected. Falling back to 0.0 GB VRAM (CPU/RAM mode)."
            )
            effective_vram = 0.0
        elif requested_vram_gb > profile.total_vram_gb:
            safe_vram = round(min(profile.free_vram_gb, profile.total_vram_gb * 0.90), 2)
            warnings.append(
                f"Requested VRAM budget ({requested_vram_gb:.1f} GB) exceeds physical VRAM "
                f"({profile.total_vram_gb:.2f} GB). Clamped to {safe_vram:.2f} GB to prevent driver crashes."
            )
            effective_vram = safe_vram
        elif requested_vram_gb > profile.free_vram_gb and profile.free_vram_gb > 0:
            safe_vram = round(profile.free_vram_gb * 0.90, 2)
            warnings.append(
                f"Requested VRAM budget ({requested_vram_gb:.1f} GB) exceeds currently free VRAM "
                f"({profile.free_vram_gb:.2f} GB). Clamped to {safe_vram:.2f} GB."
            )
            effective_vram = safe_vram
        else:
            effective_vram = float(requested_vram_gb)

    # 2. Resolve RAM Budget
    if requested_ram_gb is None or requested_ram_gb < 0:
        # Auto budget: aim for the minimum supported cache and grow with free
        # RAM, but never overcommit currently-available memory (0.85 cap).
        safe_ram = round(min(profile.available_ram_gb * 0.75, profile.total_ram_gb * 0.60), 2)
        ram_cap = round(profile.available_ram_gb * 0.85, 2)
        effective_ram = (
            max(2.0, min(max(MIN_RAM_GB, safe_ram), ram_cap))
            if ram_cap > 0
            else max(2.0, safe_ram)
        )
    else:
        # User specified explicit budget: validate and clamp to physical hardware
        if requested_ram_gb > profile.total_ram_gb:
            safe_ram = round(max(2.0, profile.total_ram_gb * 0.70), 2)
            warnings.append(
                f"Requested RAM budget ({requested_ram_gb:.1f} GB) exceeds total physical RAM "
                f"({profile.total_ram_gb:.2f} GB). Clamped to safe budget of {safe_ram:.2f} GB to prevent system freeze."
            )
            effective_ram = safe_ram
        elif requested_ram_gb > profile.available_ram_gb:
            safe_ram = round(max(2.0, profile.available_ram_gb * 0.85), 2)
            warnings.append(
                f"Requested RAM budget ({requested_ram_gb:.1f} GB) exceeds currently available RAM "
                f"({profile.available_ram_gb:.2f} GB). Clamped to {safe_ram:.2f} GB to avoid OS paging/thrashing."
            )
            effective_ram = safe_ram
        else:
            effective_ram = float(requested_ram_gb)

    # 3. Assess Compatibility Level
    is_compatible = True

    # Minimum supported envelope (requirements, not budgets).
    below_ram_floor = profile.total_ram_gb < MIN_RAM_GB
    below_vram_floor = profile.has_discrete_gpu and profile.total_vram_gb < MIN_VRAM_GB
    if below_ram_floor:
        warnings.append(
            f"Total system RAM ({profile.total_ram_gb:.2f} GB) is below the runtime's minimum "
            f"supported {MIN_RAM_GB:.1f} GB; expert caching will be limited."
        )
    if below_vram_floor:
        warnings.append(
            f"GPU VRAM ({profile.total_vram_gb:.2f} GB) is below the runtime's minimum "
            f"supported {MIN_VRAM_GB:.1f} GB; running with a reduced or disabled VRAM expert cache."
        )

    if profile.available_ram_gb < model_req.min_ram_required_gb:
        status = "INSUFFICIENT_MEMORY"
        is_compatible = False
        warnings.append(
            f"Available RAM ({profile.available_ram_gb:.2f} GB) is below the minimum required "
            f"({model_req.min_ram_required_gb:.2f} GB). Close other applications to free memory."
        )
    elif profile.available_ram_gb < model_req.recommended_ram_gb and not profile.has_discrete_gpu:
        status = "CONSTRAINED"
        messages.append("Operating in low-memory CPU offload mode. Token generation speed may be restricted.")
    elif profile.has_discrete_gpu and effective_vram >= model_req.core_heads_size_gb:
        status = "OPTIMAL" if profile.total_ram_gb >= 16.0 else "COMPATIBLE"
        messages.append("GPU acceleration active: core attention, router, and norm layers locked in VRAM.")
    else:
        status = "COMPATIBLE"
        messages.append("Direct SSD offloading active with System RAM caching.")

    # Below the supported envelope but still functional: mark as constrained.
    if status == "COMPATIBLE" and (below_ram_floor or below_vram_floor):
        status = "CONSTRAINED"

    # 4. Compute Dynamic Hot Experts Capacity
    # Dynamically scale expert cache based on allocated RAM budget
    max_hot_experts = max(32, min(512, int(effective_ram * 28)))

    # 5. Determine Offload Strategy Description
    if effective_vram >= model_req.core_heads_size_gb:
        offload_strategy = (
            f"Hybrid Acceleration: Core Heads in {effective_vram:.1f}GB VRAM | "
            f"Hot Experts in {effective_ram:.1f}GB RAM Cache | Cold Experts on NVMe SSD (mmap)"
        )
    else:
        offload_strategy = (
            f"CPU/RAM Offload: Core Heads & Hot Experts in {effective_ram:.1f}GB RAM | "
            f"Cold Experts on NVMe SSD (mmap)"
        )

    return HardwareCompatibilityReport(
        is_compatible=is_compatible,
        status=status,
        effective_vram_budget_gb=round(effective_vram, 2),
        effective_ram_budget_gb=round(effective_ram, 2),
        max_hot_experts=max_hot_experts,
        offload_strategy=offload_strategy,
        messages=messages,
        warnings=warnings,
    )


def print_hardware_and_compatibility_banner(
    profile: HardwareProfile,
    model_req: ModelHardwareRequirements,
    report: HardwareCompatibilityReport,
):
    """Prints a structured, readable summary of hardware detection and compatibility."""
    print("\n[Hardware Detection]")
    print(f"  • Platform: {profile.platform_name} | {profile.cpu_count} CPU cores")
    print(f"  • System RAM: {profile.total_ram_gb:.2f} GB Total | {profile.available_ram_gb:.2f} GB Available ({profile.ram_utilization_pct:.1f}% used)")
    if profile.gpu_detected:
        print(f"  • GPU Device: {profile.gpu_name} (Backend: {profile.gpu_backend})")
        print(f"  • VRAM: {profile.total_vram_gb:.2f} GB Total | {profile.free_vram_gb:.2f} GB Free")
    else:
        print("  • GPU Device: No Discrete GPU Detected (CPU / Unified Host Mode)")

    print(f"\n[Compatibility Assessment]")
    status_icon = "✓" if report.is_compatible else "✗"
    print(f"  • Status: [{status_icon} {report.status}] - Model Size: {model_req.total_model_size_gb:.2f} GB")
    print(f"  • Required Base Parameters: {model_req.core_heads_size_gb:.2f} GB | MoE Experts: {model_req.moe_experts_size_gb:.2f} GB")
    print(f"  • Architecture Strategy: {report.offload_strategy}")
    print(f"  • Target Budgets: {report.effective_vram_budget_gb:.1f} GB VRAM | {report.effective_ram_budget_gb:.1f} GB System RAM Hot Cache (~{report.max_hot_experts} Experts)")
    print(f"  • Minimum Supported Envelope: {MIN_VRAM_GB:.0f} GB VRAM / {MIN_RAM_GB:.0f} GB RAM")

    if report.warnings:
        print("  • Notices / Adaptations:")
        for w in report.warnings:
            print(f"    - {w}")
