import requests

from a11_llm_optimizer.optimizer import Optimizer


def run_inference(prompt):
    optimizer = Optimizer()

    print(f"\n[1] Evaluating Prompt: {prompt}")
    result = optimizer.optimize(prompt)

    adapter_path = result.get("adapter_path")
    print(f"[2] Mapped Adapter: {adapter_path if adapter_path else 'None (Using Base Model)'}")

    # Extract inference variables from the optimizer's request dictionary
    req_meta = result.get("request", {})

    # Wrap the prompt in ChatML format so the model knows it is an instruction
    base_prompt = req_meta.get("prompt", prompt)
    chatml_prompt = f"<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Build the exact payload llama.cpp requires
    payload = {
        "prompt": chatml_prompt,
        "n_predict": req_meta.get("max_tokens", 512),
        "temperature": req_meta.get("temperature", 0.3),
    }

    # Inject the LoRA path directly into the API request to trigger the hot-swap
    if adapter_path:
        payload["lora"] = [{"path": adapter_path, "scale": req_meta.get("scale", 1.0)}]

    print("[3] Sending payload to local server...")

    try:
        response = requests.post("http://127.0.0.1:8080/completion", json=payload)

        if response.status_code == 200:
            data = response.json()
            content = data.get("content", "").strip()
            timings = data.get("timings", {})

            # --- Architectural Parameter Estimation ---
            # Qwen2.5-3B is a 3.02B parameter dense model
            base_params = 3.02
            # A typical LoRA adapter adds roughly 50 million parameters
            lora_params = 0.05 if adapter_path else 0.00
            total_params = base_params + lora_params

            print("\n--- Output ---")
            print(content)
            print("--------------")

            # Print the evaluation metrics
            print("\n=== Performance Evaluation ===")
            print(
                f"Prompt Evaluated:  {timings.get('prompt_n')} tokens in "
                f"{timings.get('prompt_ms')} ms "
                f"({timings.get('prompt_per_second'):.2f} t/s)"
            )
            print(
                f"Text Generated:    {timings.get('predicted_n')} tokens in "
                f"{timings.get('predicted_ms')} ms "
                f"({timings.get('predicted_per_second'):.2f} t/s)"
            )

            print("\n=== Hardware & Parameter Footprint ===")
            print(f"Total Parameters:      ~{total_params:.2f} Billion")
            print(f"Frozen Base Weights:   ~{base_params:.2f} Billion (Read-only)")
            print(f"Dynamic LoRA Weights:  ~{lora_params:.2f} Billion")
            print(
                f"Active Firing Neurons: ~{total_params:.2f} Billion "
                "(Dense execution - 100% utilized)"
            )

        else:
            print(f"Server returned an error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print(
            "\nError: Could not connect to the local server. "
            "Is llama-server running in the other terminal?"
        )


# Execute the test
run_inference("Write a Python script to scrape a website.")
