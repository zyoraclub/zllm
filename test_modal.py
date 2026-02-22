"""
Test ZLLM on Modal with T4 GPU.
Run: modal run test_modal.py
"""
import modal

app = modal.App("zllm-test")

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "sentencepiece", "huggingface_hub", "numpy")
    .run_commands("git clone https://github.com/zyoraclub/zllm.git /zllm")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def test_zllm():
    import sys
    sys.path.insert(0, "/zllm")
    
    import torch
    from huggingface_hub import hf_hub_download
    import os
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    
    # Download model
    os.makedirs("/zllm/models", exist_ok=True)
    print("\nDownloading Qwen 2.5 Coder 7B...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        local_dir="/zllm/models"
    )
    print("Download complete!")
    
    # Load model with layer streaming
    from transformers import AutoTokenizer
    from zllm.engine.inference import load_engine
    
    print("\n=== Loading ZLLM Engine ===")
    engine = load_engine("/zllm/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf", device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    print(f"\nGPU layers: {len(engine._gpu_layers)}")
    print(f"CPU layers: {len(engine._cpu_layers)}")
    print(f"Hot layers on GPU: {sorted(engine._gpu_layers.keys())}")
    
    # Check memory after loading
    print(f"\nVRAM used: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]) / 1024**3:.2f} GB")
    
    # Test inference
    print("\n=== Testing Inference ===")
    prompt = "Write a Python function to calculate fibonacci numbers:"
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    
    output = engine.generate(tokens, max_new_tokens=150, temperature=0.7)
    result = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\n{result}")
    
    engine.close()
    print("\n=== Test Complete ===")
    return result

@app.local_entrypoint()
def main():
    result = test_zllm.remote()
    print("\n" + "="*50)
    print("RESULT FROM MODAL:")
    print("="*50)
    print(result)
