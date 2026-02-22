# zllm

**Memory-efficient LLM inference for everyone.** Run large language models on limited hardware with zero configuration.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

## ✨ Features

- **🧠 Memory Efficient** - Run 70B models on 4GB RAM with layer streaming
- **⚡ Zero Config** - Auto-detects hardware and optimizes settings
- **🎯 Simple CLI** - Just `zllm run llama3` to start chatting
- **🔑 API Keys** - Built-in API key management for secure access
- **💬 Web UI** - Beautiful chat interface included
- **🚀 Semantic Cache** - Cache similar prompts for 10-100x faster responses
- **🌍 Multi-Hardware** - Works on NVIDIA, AMD, Apple Silicon, and CPU

## 🚀 Quick Start

### Installation

```bash
pip install zllm
```

### Basic Usage

```bash
# Start chatting with a model
zllm run llama3

# Or specify a HuggingFace model
zllm run meta-llama/Llama-3-8B-Instruct

# Start API server
zllm serve --model llama3 --port 8000

# Launch Web UI
zllm ui
```

### Python SDK

```python
from zllm import ZLLM

# Simple usage
llm = ZLLM("meta-llama/Llama-3-8B-Instruct")
response = llm.chat("What is the capital of France?")
print(response)

# Streaming
for token in llm.chat_stream("Tell me a story"):
    print(token, end="")

# With configuration
from zllm import ZLLMConfig

config = ZLLMConfig(
    quantization="int4",
    enable_semantic_cache=True,
    max_new_tokens=4096,
)
llm = ZLLM("meta-llama/Llama-3-70B-Instruct", config=config)
```

## 📖 Commands

| Command | Description |
|---------|-------------|
| `zllm run <model>` | Run a model interactively |
| `zllm serve --model <model>` | Start API server |
| `zllm ui` | Launch Web UI |
| `zllm pull <model>` | Download a model |
| `zllm list` | List downloaded models |
| `zllm search <query>` | Search HuggingFace models |
| `zllm quantize <model>` | Quantize a model |
| `zllm info` | Show system information |

## 🔌 API

zllm provides an OpenAI-compatible API:

```bash
# Start the server
zllm serve --model llama3 --port 8000

# Use with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Works with OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🧠 Memory Efficiency

zllm uses layer streaming to run large models on limited hardware:

| Model | Standard | With zllm (4-bit + streaming) |
|-------|----------|-------------------------------|
| Llama-3-8B | 16GB VRAM | 4GB VRAM |
| Llama-3-70B | 140GB VRAM | 16GB VRAM |
| Mixtral-8x7B | 90GB VRAM | 12GB VRAM |

## ⚡ Semantic Cache

zllm caches responses by meaning, not just exact matches:

```python
# These will return the same cached response:
llm.chat("What is the capital of France?")
llm.chat("Tell me France's capital city")  # Cache hit!
llm.chat("France capital?")  # Cache hit!
```

## 🛠️ Configuration

```python
from zllm import ZLLMConfig

config = ZLLMConfig(
    # Hardware
    device="auto",  # "cuda", "mps", "cpu", "auto"
    
    # Memory management
    enable_layer_streaming=True,
    offload_to_cpu=True,
    
    # Quantization
    quantization="int4",  # "int4", "int8", None
    auto_quantize=True,
    
    # Generation
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    
    # Cache
    enable_cache=True,
    enable_semantic_cache=True,
    semantic_cache_threshold=0.92,
    
    # Server
    host="127.0.0.1",
    port=8000,
)
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the AI community**
