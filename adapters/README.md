# LoRA Adapters

The local llama.cpp setup expects these adapter files:

| Routed domain | File | llama.cpp adapter ID |
| --- | --- | ---: |
| Python | `python_coder.gguf` | 0 |
| Medical | `medical_expert.gguf` | 1 |
| Creative | `creative_writer.gguf` | 2 |

Place the files in this directory. They must be real LoRA adapters trained for
the selected base model and converted to a GGUF format supported by the
installed llama.cpp build. A `.safetensors` filename alone is not sufficient.
The server launcher loads adapters in this order, and each generation request
sets the chosen ID to scale `1.0` and the other IDs to `0.0`.

This repository does not contain or train these domain adapters. Supply
appropriately licensed adapter weights or train/export them separately. Override
the filenames or directory with `PYTHON_ADAPTER_FILE`, `MEDICAL_ADAPTER_FILE`,
`CREATIVE_ADAPTER_FILE`, or `ADAPTERS_PATH` in `.env`. Adapter weights are
intentionally excluded from Git.