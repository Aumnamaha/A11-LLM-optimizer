#!/usr/bin/env python3
"""Phase 0 baseline harness for the A11 Paged-MoE project.

Runs the *unmodified* model through llama.cpp (``llama-bench``) across a matrix
of memory-placement and I/O configurations, while sampling GPU VRAM, process
RSS and NVMe read volume. It emits machine-readable artifacts under ``baseline/``:

    baseline/benchmark.json   raw + parsed llama-bench results per config
    baseline/memory.json      peak VRAM / RSS / SSD bytes read per config
    baseline/devices.json     hardware + llama.cpp backend inventory
    baseline/SUMMARY.md       human-readable comparison table

This deliberately does NOT touch the custom runtime. It establishes the
"what happens if we just run the model" reference that later phases compare
against (see phases.txt, Phase 0).

Usage:
    python tools/benchmark/llama_baseline.py --model models/Qwen3-30B-A3B
    python tools/benchmark/llama_baseline.py --model models/qwen2.5-7b-instruct-q4_k_m.gguf --label control-7b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is optional
    psutil = None

DISKSTATS = "/proc/diskstats"
VRAM_GLOB = "/sys/class/drm/card*/device/mem_info_vram_used"
VRAM_TOTAL_GLOB = "/sys/class/drm/card*/device/mem_info_vram_total"

# Placement / I/O configurations exercised for an MoE model. Each entry is a
# llama-bench argument fragment. GPU index 1 == the discrete RX 6500 (Vulkan1).
DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {"name": "cpu_mmap", "args": ["-ngl", "0"]},
    {"name": "cpu_dio", "args": ["-ngl", "0", "-lm", "dio"]},
    # -ncmoe 999 keeps every layer's MoE experts on the CPU (999 >= n_layer).
    {"name": "vulkan_attn_cpu_moe", "args": ["-ngl", "99", "-mg", "1", "-ncmoe", "999"]},
    {"name": "vulkan_attn_cpu_moe_dio", "args": ["-ngl", "99", "-mg", "1", "-ncmoe", "999", "-lm", "dio"]},
    {"name": "vulkan_all", "args": ["-ngl", "99", "-mg", "1"]},
]


def which_or_die(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required executable not found on PATH: {name}")
    return path


def run(cmd: List[str], timeout: float) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "wall_seconds": time.perf_counter() - start,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "wall_seconds": time.perf_counter() - start,
        }


def parse_json_array(text: str) -> Optional[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def list_devices(llama_cli: str) -> Dict[str, Any]:
    """Parse ``llama-cli --list-devices`` into a structured inventory."""
    res = run([llama_cli, "--list-devices"], timeout=60)
    combined = res["stdout"] + "\n" + res["stderr"]
    devices = []
    for line in combined.splitlines():
        line = line.strip()
        if ":" not in line or line.lower().startswith("available"):
            continue
        name, _, rest = line.partition(":")
        if not name.strip().lower().startswith(("vulkan", "cuda", "hip", "metal", "cpu")):
            continue
        devices.append({"index": name.strip(), "detail": rest.strip()})
    return {
        "raw": combined.strip(),
        "devices": devices,
        "vram_total_bytes": _vram_totals(),
    }


def _vram_totals() -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for path in glob.glob(VRAM_TOTAL_GLOB):
        card = path.split("/")[4]
        try:
            with open(path) as fh:
                totals[card] = int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return totals


def _vram_used() -> Dict[str, int]:
    used: Dict[str, int] = {}
    for path in glob.glob(VRAM_GLOB):
        card = path.split("/")[4]
        try:
            with open(path) as fh:
                used[card] = int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return used


def _disk_read_sectors(device: str) -> Optional[int]:
    try:
        with open(DISKSTATS) as fh:
            for line in fh:
                parts = line.split()
                # major minor name ... sectors_read is field index 5
                if len(parts) > 5 and parts[2] == device:
                    return int(parts[5])
    except (OSError, ValueError):
        return None
    return None


def _proc_tree_rss(pid: int) -> int:
    if psutil is None:
        return 0
    try:
        parent = psutil.Process(pid)
        procs = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue
    return total


class ResourceSampler(threading.Thread):
    """Polls VRAM / RSS / NVMe counters while a subprocess runs.

    Optionally enforces a system-RAM floor: if available RAM drops below
    ``guard_bytes`` the child is killed before the host OOM fail-safe can fire.
    """

    def __init__(self, pid: int, disk: str, interval: float = 0.25,
                 guard_bytes: Optional[int] = None):
        super().__init__(daemon=True)
        self.pid = pid
        self.disk = disk
        self.interval = interval
        self.guard_bytes = guard_bytes
        self._stop = threading.Event()
        self.peak_vram: Dict[str, int] = {}
        self.peak_rss = 0
        self.min_available: Optional[int] = None
        self.guard_tripped = False
        self.start_sectors = _disk_read_sectors(disk)
        self.samples = 0

    def _check_guard(self) -> None:
        if psutil is None or not self.guard_bytes:
            return
        available = psutil.virtual_memory().available
        self.min_available = available if self.min_available is None else min(self.min_available, available)
        if available < self.guard_bytes and not self.guard_tripped:
            # Stop the child ourselves so the host OOM fail-safe never fires.
            self.guard_tripped = True
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass

    def run(self) -> None:
        while not self._stop.is_set():
            self.samples += 1
            for card, used in _vram_used().items():
                self.peak_vram[card] = max(self.peak_vram.get(card, 0), used)
            self.peak_rss = max(self.peak_rss, _proc_tree_rss(self.pid))
            self._check_guard()
            self._stop.wait(self.interval)

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        self.join(timeout=5)
        end_sectors = _disk_read_sectors(self.disk)
        ssd_bytes = None
        if self.start_sectors is not None and end_sectors is not None:
            ssd_bytes = max(0, end_sectors - self.start_sectors) * 512
        return {
            "peak_vram_bytes": self.peak_vram,
            "peak_rss_bytes": self.peak_rss,
            "min_available_bytes": self.min_available,
            "guard_tripped": self.guard_tripped,
            "ssd_read_bytes": ssd_bytes,
            "samples": self.samples,
        }


def run_config(
    llama_bench: str,
    model: Path,
    config: Dict[str, Any],
    prompt_tokens: int,
    gen_tokens: int,
    reps: int,
    threads: int,
    timeout: float,
    disk: str,
    guard_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    cmd = [
        llama_bench,
        "-m", str(model),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(reps),
        "-t", str(threads),
        "-o", "json",
        *config["args"],
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = ResourceSampler(proc.pid, disk, guard_bytes=guard_bytes)
    sampler.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode: Optional[int] = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        returncode = None
        timed_out = True
    resources = sampler.stop()

    record: Dict[str, Any] = {
        "name": config["name"],
        "args": config["args"],
        "command": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "results": parse_json_array(stdout),
        "stderr_tail": "\n".join(stderr.strip().splitlines()[-12:]),
        "resources": resources,
    }
    return record


def summarize(records: List[Dict[str, Any]]) -> str:
    lines = [
        "| config | status | pp tok/s | tg tok/s | peak RSS (GiB) | min free RAM (GiB) | "
        "peak VRAM RX6500 (MiB) | SSD read (GiB) | wall (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        pp = tg = None
        for entry in rec.get("results") or []:
            if entry.get("n_prompt", 0):
                pp = entry.get("avg_ts")
            elif entry.get("n_gen", 0):
                tg = entry.get("avg_ts")
        res = rec["resources"]
        vram = res["peak_vram_bytes"].get("card1", 0) / (1024 ** 2)
        rss = res["peak_rss_bytes"] / (1024 ** 3)
        min_free = res.get("min_available_bytes")
        min_free_s = f"{min_free / 1024 ** 3:.2f}" if min_free is not None else "-"
        ssd = res["ssd_read_bytes"]
        ssd_gib = f"{ssd / 1024 ** 3:.2f}" if ssd is not None else "n/a"
        if rec["timed_out"]:
            status = "timeout"
        elif res.get("guard_tripped"):
            status = "mem-guard"
        elif rec["returncode"] == 0:
            status = "ok"
        else:
            status = f"rc={rec['returncode']}"
        pp_s = f"{pp:.2f}" if pp is not None else "-"
        tg_s = f"{tg:.2f}" if tg is not None else "-"
        lines.append(
            f"| {rec['name']} | {status} | {pp_s} | {tg_s} | {rss:.2f} | {min_free_s} | "
            f"{vram:.0f} | {ssd_gib} | {rec['wall_seconds']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="GGUF file or directory containing one")
    parser.add_argument("--label", help="label for this baseline (defaults to model stem)")
    parser.add_argument("--baseline-dir", default="baseline", help="output directory")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 6)
    parser.add_argument("--timeout", type=float, default=900.0, help="per-config timeout (s)")
    parser.add_argument("--disk", default="nvme0n1", help="block device to read SSD counters from")
    parser.add_argument("--configs", help="JSON file with a list of {name,args} configs")
    parser.add_argument("--force", action="store_true",
                        help="re-run configs already present in benchmark.json")
    parser.add_argument("--only", help="comma-separated config names to (re)run; others are kept")
    parser.add_argument("--mem-floor-gib", type=float, default=1.5,
                        help="abort a run if free system RAM drops below this (0 disables)")
    args = parser.parse_args()
    guard_bytes = int(args.mem_floor_gib * 1024 ** 3) if args.mem_floor_gib > 0 else None

    llama_bench = which_or_die("llama-bench")
    llama_cli = which_or_die("llama-cli")

    model = Path(args.model).expanduser().resolve()
    if model.is_dir():
        shards = sorted(model.glob("*.gguf"))
        if not shards:
            raise SystemExit(f"no .gguf in {model}")
        model = shards[0]
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = json.loads(Path(args.configs).read_text())

    out_dir = Path(args.baseline_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    version_res = run([llama_bench, "--version"], timeout=60)
    version_text = (version_res["stdout"] + version_res["stderr"]).strip()
    version_lines = [ln for ln in version_text.splitlines() if "version:" in ln]
    inventory = {
        "llama_cpp_version": version_lines[0].strip() if version_lines else (version_text.splitlines()[0] if version_text else "unknown"),
        "label": args.label or model.stem,
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "backends": list_devices(llama_cli),
    }
    (out_dir / "devices.json").write_text(json.dumps(inventory, indent=2) + "\n")

    # Resume: reuse any already-completed configs so partial runs survive restarts.
    existing: Dict[str, Dict[str, Any]] = {}
    bench_path = out_dir / "benchmark.json"
    if bench_path.exists():
        try:
            previous = json.loads(bench_path.read_text())
            existing = {r["name"]: r for r in previous.get("runs", []) if "name" in r}
        except (json.JSONDecodeError, OSError):
            existing = {}

    only = {n.strip() for n in args.only.split(",")} if args.only else None
    if only:
        unknown = only - {c["name"] for c in configs}
        if unknown:
            raise SystemExit(f"--only names not in configs: {sorted(unknown)}")

    results_by_name: Dict[str, Dict[str, Any]] = dict(existing)
    print(f"[baseline] model={model.name} configs={len(configs)} timeout={args.timeout}s "
          f"mem-floor={args.mem_floor_gib}GiB reused={len(existing)}"
          + (f" only={sorted(only)}" if only else ""))
    for config in configs:
        name = config["name"]
        if only is not None and name not in only:
            continue
        if name in existing and not args.force and not only:
            print(f"[baseline] reusing {name} (already in benchmark.json)", flush=True)
            continue
        print(f"[baseline] running {name} ...", flush=True)
        rec = run_config(
            llama_bench, model, config,
            args.prompt_tokens, args.gen_tokens, args.reps,
            args.threads, args.timeout, args.disk, guard_bytes,
        )
        results_by_name[name] = rec
        ordered = [results_by_name[c["name"]] for c in configs if c["name"] in results_by_name]
        (out_dir / "benchmark.json").write_text(
            json.dumps({"inventory": inventory, "runs": ordered}, indent=2) + "\n"
        )
        (out_dir / "memory.json").write_text(
            json.dumps({r["name"]: r["resources"] for r in ordered}, indent=2) + "\n"
        )
        if rec["resources"].get("guard_tripped"):
            status = "mem-guard"
        elif rec["timed_out"]:
            status = "timeout"
        else:
            status = f"rc={rec['returncode']}"
        print(f"[baseline]   -> {status} in {rec['wall_seconds']:.1f}s", flush=True)

    records = [results_by_name[c["name"]] for c in configs if c["name"] in results_by_name]

    summary = [
        f"# Baseline summary — {inventory['label']}",
        "",
        f"- Model: `{model}` ({inventory['model_size_bytes'] / 1024 ** 3:.2f} GiB)",
        f"- Generated: {inventory['timestamp']}",
        f"- llama.cpp: `{inventory['llama_cpp_version']}`",
        f"- Configs: {args.prompt_tokens} prompt tok, {args.gen_tokens} gen tok, {args.reps} reps, {args.threads} threads",
        "",
        "## Results",
        "",
        summarize(records),
        "",
        "## Notes",
        "",
        "- VRAM RX6500 = `/sys/class/drm/card1/device/mem_info_vram_used` peak.",
        "- `pp` = prompt processing throughput; `tg` = token generation throughput.",
        "- `SSD read` is NVMe sectors read during the run (page-in from the GGUF).",
        "- `mem-guard` = run aborted by the harness RAM floor before the host OOM fail-safe fired.",
        "- A raw `rc=-9` means the host OOM killer terminated the process (RAM insufficient).",
        "- This table is the *reference* for the paged-MoE runtime ablations (Phase 19).",
        "",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(summary))
    print(f"[baseline] wrote {out_dir}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
