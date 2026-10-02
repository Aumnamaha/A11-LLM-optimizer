import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_llama_server.sh"


def test_launcher_builds_server_arguments_for_configured_assets(tmp_path):
    model_dir = tmp_path / "models"
    adapter_dir = tmp_path / "adapters"
    binary_dir = tmp_path / "bin"
    model_dir.mkdir()
    adapter_dir.mkdir()
    binary_dir.mkdir()
    model = model_dir / "base.gguf"
    adapters = [adapter_dir / name for name in ("python.gguf", "medical.gguf", "creative.gguf")]
    for asset in [model, *adapters]:
        asset.touch()

    capture = tmp_path / "arguments.txt"
    llama_server = binary_dir / "llama-server"
    llama_server.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n')
    llama_server.chmod(0o755)
    environment = {
        **os.environ,
        "ENV_FILE": "/dev/null",
        "MODEL_BASE_PATH": str(model_dir),
        "ADAPTERS_PATH": str(adapter_dir),
        "BASE_MODEL_FILE": model.name,
        "PYTHON_ADAPTER_FILE": adapters[0].name,
        "MEDICAL_ADAPTER_FILE": adapters[1].name,
        "CREATIVE_ADAPTER_FILE": adapters[2].name,
        "LLAMA_CONTEXT_SIZE": "1024",
        "LLAMA_GPU_LAYERS": "12",
        "LLAMA_SERVER_HOST": "127.0.0.1",
        "LLAMA_SERVER_PORT": "8091",
        "CAPTURE_FILE": str(capture),
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
    }

    subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=environment, check=True)

    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--model") + 1] == str(model)
    assert [arguments[index + 1] for index, value in enumerate(arguments) if value == "--lora"] == [
        str(adapter) for adapter in adapters
    ]
    assert arguments[arguments.index("--ctx-size") + 1] == "1024"
    assert arguments[arguments.index("--n-gpu-layers") + 1] == "12"
    assert arguments[arguments.index("--port") + 1] == "8091"


def test_launcher_reports_missing_model_asset(tmp_path):
    environment = {
        **os.environ,
        "ENV_FILE": "/dev/null",
        "MODEL_BASE_PATH": str(tmp_path / "models"),
        "ADAPTERS_PATH": str(tmp_path / "adapters"),
        "PATH": "/usr/bin:/bin",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, env=environment, capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "Required model asset not found" in result.stderr
