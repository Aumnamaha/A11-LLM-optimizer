# Build Reproducible Paged-MoE Model Testing Environment

## Objective

Create a clean, reproducible environment for testing the target MoE model before implementing the custom Paged-MoE runtime.

The environment must allow us to measure:

* model size
* total parameters
* active parameters per token
* VRAM usage
* RAM usage
* SSD storage usage
* model loading time
* inference latency
* tokens/sec
* expert routing behavior
* expert reuse/locality
* GPU/CPU utilization

The environment should be independent from the future paging implementation.

---

## Target Model

Use a **7B-class sparse MoE model** suitable for Q4_K_M inference.

Target characteristics:

* ~7B total parameters
* ~3B active parameters/token target
* MoE architecture
* Q4_K_M quantization
* Hugging Face/compatible model format
* Model must be legally downloadable and redistributable only according to its license

Do NOT modify the model architecture yet.

---

## Environment Requirements

Create a reproducible environment with:

### Software

* Linux
* Python 3.11+
* Git
* C/C++ build tools
* CMake
* llama.cpp or equivalent inference backend
* Vulkan/ROCm-compatible GPU backend where applicable
* monitoring utilities

Prefer a virtual environment/isolated environment so the system Python installation is not modified.

---

## Repository Structure

Create:

```text
tests/model_env/
├── README.md
├── requirements.txt
├── setup.sh
├── run_baseline.sh
├── collect_metrics.sh
├── config/
│   └── baseline.yaml
├── results/
│   └── .gitkeep
└── scripts/
    ├── download_model.sh
    ├── benchmark.py
    ├── memory_monitor.py
    └── routing_stats.py
```

Do not commit model weights into Git.

---

## Environment Setup

The setup script must:

1. Create the Python virtual environment.
2. Install required Python dependencies.
3. Verify compiler/build tools.
4. Verify GPU backend.
5. Verify inference backend.
6. Create the required directories.
7. Print detected CPU, GPU, RAM and storage information.
8. Fail clearly if a required dependency is missing.

The script must be safe to run more than once.

---

## Baseline Test

Create a baseline test that loads the model **without any custom paging/cache implementation**.

Record:

```text
Model:
Quantization:
Total parameters:
Active parameters/token:
Model file size:
Load time:
Prompt tokens:
Generated tokens:
Generation time:
Tokens/sec:
Peak VRAM:
Peak RAM:
CPU usage:
GPU usage:
SSD read volume:
```

Use a fixed prompt/test set so future implementations can be compared against the same baseline.

---

## Expert Routing Test

Collect routing statistics from the model.

Measure:

* expert IDs selected per token
* number of unique experts used
* expert selection frequency
* expert reuse distance
* consecutive expert reuse
* routing locality
* top-k expert distribution

Generate a summary such as:

```text
Total tokens:
Unique experts used:
Top 10 most-used experts:
Average expert reuse distance:
Expert locality score:
Routing entropy:
```

This is important because the effectiveness of the future VRAM/RAM cache depends heavily on expert locality.

---

## Memory Measurement

Measure **model/runtime resource usage separately from the rest of the desktop where possible**.

Record:

### VRAM

```text
baseline VRAM
peak VRAM
VRAM used by model
```

### RAM

```text
baseline RAM
peak RAM
model/runtime RAM
```

### SSD

```text
model size
additional runtime storage
read bandwidth during inference
total bytes read
```

Do not report only total desktop RAM usage without identifying the model/runtime contribution.

---

## Benchmark Protocol

Run each benchmark at least 3 times.

Use the same:

* model
* quantization
* context length
* prompt
* generation length
* GPU offload settings
* thread count
* backend

Report:

```text
mean
minimum
maximum
standard deviation
```

---

## Baseline Test Cases

Create these tests:

### Test A — Short prompt

Small input and short generation.

### Test B — Long prompt

Large context with fixed generation length.

### Test C — Repeated workload

Run the same prompt repeatedly to expose expert locality/reuse.

### Test D — Changing workload

Use different prompts to change expert routing.

### Test E — Stress workload

Long generation with maximum practical context.

---

## Output

Generate machine-readable results:

```text
results/
├── baseline.json
├── routing.json
├── memory.json
└── benchmark.json
```

Also create a human-readable:

```text
results/SUMMARY.md
```

with tables comparing all tests.

---

## Important Constraints

Do NOT implement:

* SSD paging
* VRAM expert cache
* RAM expert cache
* predictive prefetching
* custom eviction
* custom model format

Those are separate tasks.

This task is only to create the **trusted baseline environment and measurements** that the Paged-MoE runtime will later be compared against.

---

## Acceptance Criteria

The task is complete when:

* [ ] Environment can be recreated from a clean checkout.
* [ ] Model can be loaded successfully.
* [ ] Baseline inference works.
* [ ] VRAM usage is recorded.
* [ ] RAM usage is recorded.
* [ ] SSD/model size is recorded.
* [ ] Tokens/sec is measured.
* [ ] Expert routing statistics are collected.
* [ ] At least 3 benchmark repetitions are completed.
* [ ] Results are stored as JSON.
* [ ] Human-readable summary is generated.
* [ ] No model weights are committed to Git.
* [ ] README explains the complete setup and benchmark procedure.
* [ ] Results can be reproduced by another team member.

## Deliverables

1. Reproducible environment.
2. Model download/setup procedure.
3. Baseline benchmark scripts.
4. Routing-analysis script.
5. Resource-monitoring scripts.
6. JSON benchmark results.
7. Final `SUMMARY.md`.
8. Documentation in `README.md`.

## Final report

The final report must answer:

1. How large is the model?
2. How many parameters are active per token?
3. How much VRAM does baseline inference use?
4. How much RAM does baseline inference use?
5. How much SSD storage is required?
6. What is baseline tokens/sec?
7. How frequently are experts reused?
8. How strong is expert locality?
9. What is the largest observed bottleneck?
10. What cache size would likely provide the most benefit?

Create a pull request containing all environment files, scripts and benchmark results.
