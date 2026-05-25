"""Serve pr-reviewer-7b-instruct-awq via vLLM on Modal as OpenAI-compatible HTTP API.

Adapted from Modal's official vLLM example:
https://modal.com/docs/examples/vllm_inference

Scales to zero when idle. Cold start: ~60-90 sec. Cost: ~$0.20/hr when warm.

Run: uv run modal deploy scripts/serve_vllm.py
Test: see curl commands at bottom
"""

import modal

HF_REPO_ID = "Hardik55/pr-reviewer-7b-instruct-awq"

# Modal's canonical vLLM image: nvidia/cuda + add_python + uv_pip_install vllm
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.7.3", "transformers==4.48.3", "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "0",
        }
    )
)

# Two caches: model weights and vLLM JIT artifacts
hf_cache = modal.Volume.from_name("vllm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-jit-cache", create_if_missing=True)

app = modal.App("pr-reviewer-vllm")

VLLM_PORT = 8000
MINUTES = 60


@app.function(
    image=vllm_image,
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        HF_REPO_ID,
        "--served-model-name",
        "pr-reviewer-7b-instruct-awq",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--quantization",
        "awq",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",  # faster cold starts, slightly slower inference
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
