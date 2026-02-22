"""
FastAPI server with OpenAI-compatible API.

Provides:
- /v1/chat/completions - Chat API (OpenAI compatible)
- /v1/completions - Text completion API
- /v1/models - List models
- /health - Health check
- /api/keys - API key management
"""

import asyncio
import secrets
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from zllm import ZLLM, ZLLMConfig
from zllm.hardware.auto_detect import detect_hardware


# ============== Pydantic Models ==============

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "zllm"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class APIKeyCreate(BaseModel):
    name: str
    rate_limit: Optional[int] = 100  # requests per minute
    expires_days: Optional[int] = None
    allowed_models: Optional[List[str]] = None


class APIKey(BaseModel):
    key: str
    name: str
    created_at: str
    last_used: Optional[str] = None
    requests: int = 0
    rate_limit: int = 100
    expires_at: Optional[str] = None
    allowed_models: Optional[List[str]] = None
    active: bool = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: Optional[str]
    gpu_available: bool
    cache_stats: Dict[str, Any]


class BatchingStatsResponse(BaseModel):
    enabled: bool
    running: bool = False
    pending_requests: int = 0
    active_requests: int = 0
    free_slots: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    total_tokens_generated: int = 0
    throughput_tokens_per_second: Optional[float] = None
    avg_queue_time_ms: Optional[float] = None
    avg_generation_time_ms: Optional[float] = None


# ============== Server State ==============

class ServerState:
    """Holds the server state including the loaded model."""
    
    def __init__(self):
        self.llm: Optional[ZLLM] = None
        self.model_id: Optional[str] = None
        self.api_keys: Dict[str, APIKey] = {}
        self.api_keys_file: Path = Path.home() / ".cache" / "zllm" / "api_keys.json"
        self.batching_engine = None  # For continuous batching
        self.enable_batching: bool = True  # Enable by default
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from disk."""
        if self.api_keys_file.exists():
            try:
                with open(self.api_keys_file) as f:
                    data = json.load(f)
                    for key, info in data.items():
                        self.api_keys[key] = APIKey(**info)
            except Exception:
                pass
    
    def _save_api_keys(self):
        """Save API keys to disk."""
        self.api_keys_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.api_keys_file, "w") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.api_keys.items()},
                f,
                indent=2
            )
    
    def create_api_key(self, request: APIKeyCreate) -> APIKey:
        """Create a new API key."""
        key = f"sk-zllm-{secrets.token_hex(24)}"
        
        api_key = APIKey(
            key=key,
            name=request.name,
            created_at=datetime.now().isoformat(),
            rate_limit=request.rate_limit or 100,
            allowed_models=request.allowed_models,
        )
        
        if request.expires_days:
            from datetime import timedelta
            expires = datetime.now() + timedelta(days=request.expires_days)
            api_key.expires_at = expires.isoformat()
        
        self.api_keys[key] = api_key
        self._save_api_keys()
        
        return api_key
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key."""
        if key not in self.api_keys:
            return None
        
        api_key = self.api_keys[key]
        
        if not api_key.active:
            return None
        
        if api_key.expires_at:
            expires = datetime.fromisoformat(api_key.expires_at)
            if datetime.now() > expires:
                return None
        
        # Update last used
        api_key.last_used = datetime.now().isoformat()
        api_key.requests += 1
        self._save_api_keys()
        
        return api_key
    
    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self.api_keys:
            self.api_keys[key].active = False
            self._save_api_keys()
            return True
        return False


state = ServerState()


# ============== Dependencies ==============

async def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> Optional[APIKey]:
    """Verify API key from headers."""
    # Get key from either header
    key = None
    if authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    elif x_api_key:
        key = x_api_key
    
    # If no keys configured, allow all
    if not state.api_keys:
        return None
    
    if not key:
        raise HTTPException(status_code=401, detail="API key required")
    
    api_key = state.validate_api_key(key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key


# ============== Create App ==============

def create_app(
    model_id: Optional[str] = None,
    config: Optional[ZLLMConfig] = None,
    speed_mode: str = "balanced",
) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        model_id: Model to load on startup
        config: Configuration options
        speed_mode: Speed vs memory trade-off (fast/balanced/memory)
    """
    app = FastAPI(
        title="zllm API",
        description="Memory-efficient LLM inference API (OpenAI compatible)",
        version="0.1.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Startup
    @app.on_event("startup")
    async def startup():
        if model_id:
            state.model_id = model_id
            cfg = config or ZLLMConfig(model_id=model_id, speed_mode=speed_mode)
            state.llm = ZLLM(model_id, config=cfg)
            
            # Initialize continuous batching engine
            if state.enable_batching and state.llm.model and state.llm.tokenizer:
                from zllm.core.batching import create_batching_engine
                state.batching_engine = create_batching_engine(
                    model=state.llm.model,
                    tokenizer=state.llm.tokenizer,
                    max_batch_size=8,
                    device=str(state.llm.device),
                )
    
    # Shutdown
    @app.on_event("shutdown")
    async def shutdown():
        if state.batching_engine:
            state.batching_engine.stop()
        if state.llm:
            state.llm.unload()
    
    # ============== Routes ==============
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        hw = detect_hardware()
        
        cache_stats = {}
        if state.llm:
            cache_stats = state.llm.get_cache_stats()
        
        return HealthResponse(
            status="healthy",
            model_loaded=state.llm is not None,
            model_id=state.model_id,
            gpu_available=hw.has_gpu,
            cache_stats=cache_stats,
        )
    
    @app.get("/stats/batching", response_model=BatchingStatsResponse)
    async def batching_stats():
        """Get continuous batching statistics."""
        if not state.batching_engine:
            return BatchingStatsResponse(enabled=False)
        
        stats = state.batching_engine.get_stats()
        return BatchingStatsResponse(
            enabled=True,
            running=stats.get("running", False),
            pending_requests=stats.get("pending_requests", 0),
            active_requests=stats.get("active_requests", 0),
            free_slots=stats.get("free_slots", 0),
            total_requests=stats.get("total_requests", 0),
            completed_requests=stats.get("completed_requests", 0),
            total_tokens_generated=stats.get("total_tokens_generated", 0),
            throughput_tokens_per_second=stats.get("throughput_tokens_per_second"),
            avg_queue_time_ms=stats.get("avg_queue_time_ms"),
            avg_generation_time_ms=stats.get("avg_generation_time_ms"),
        )
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models(api_key: Optional[APIKey] = Depends(verify_api_key)):
        """List available models."""
        models = []
        
        if state.model_id:
            models.append(ModelInfo(
                id=state.model_id,
                created=int(time.time()),
            ))
        
        return ModelsResponse(data=models)
    
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        api_key: Optional[APIKey] = Depends(verify_api_key),
    ):
        """OpenAI-compatible chat completions endpoint."""
        if state.llm is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        # Extract messages
        system_prompt = None
        history = []
        user_message = ""
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
            elif msg.role == "assistant":
                history.append({"role": "assistant", "content": msg.content})
            
            if msg.role == "user" and msg != request.messages[-1]:
                history.append({"role": "user", "content": msg.content})
        
        # Generate completion ID
        completion_id = f"chatcmpl-{secrets.token_hex(12)}"
        
        if request.stream:
            async def stream_response() -> AsyncIterator[str]:
                """Stream SSE response."""
                full_response = []
                
                for token in state.llm.chat_stream(
                    user_message,
                    system_prompt=system_prompt,
                    history=history,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_tokens,
                ):
                    full_response.append(token)
                    
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Final chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response_text = state.llm.chat(
                user_message,
                system_prompt=system_prompt,
                history=history,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_tokens,
            )
            
            return ChatCompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage={
                    "prompt_tokens": 0,  # TODO: Implement token counting
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
    
    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest,
        api_key: Optional[APIKey] = Depends(verify_api_key),
    ):
        """OpenAI-compatible text completions endpoint."""
        if state.llm is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        completion_id = f"cmpl-{secrets.token_hex(12)}"
        
        output = state.llm.generate(
            request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
        )
        
        return CompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
            ],
            usage={
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.tokens_generated,
                "total_tokens": output.total_tokens,
            },
        )
    
    # ============== API Key Management ==============
    
    @app.post("/api/keys", response_model=APIKey)
    async def create_key(request: APIKeyCreate):
        """Create a new API key."""
        return state.create_api_key(request)
    
    @app.get("/api/keys")
    async def list_keys():
        """List all API keys (without showing full key)."""
        keys = []
        for key, info in state.api_keys.items():
            masked = f"{key[:10]}...{key[-4:]}"
            keys.append({
                "key": masked,
                "name": info.name,
                "created_at": info.created_at,
                "last_used": info.last_used,
                "requests": info.requests,
                "active": info.active,
            })
        return {"keys": keys}
    
    @app.delete("/api/keys/{key}")
    async def revoke_key(key: str):
        """Revoke an API key."""
        if state.revoke_api_key(key):
            return {"status": "revoked"}
        raise HTTPException(status_code=404, detail="Key not found")
    
    # ============== Cache Management ==============
    
    @app.get("/api/cache/stats")
    async def cache_stats():
        """Get cache statistics."""
        if state.llm is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return state.llm.get_cache_stats()
    
    @app.post("/api/cache/clear")
    async def clear_cache():
        """Clear the response cache."""
        if state.llm:
            state.llm.clear_cache()
        return {"status": "cleared"}
    
    # ============== System Info ==============
    
    @app.get("/api/system")
    async def system_info():
        """Get system information."""
        hw = detect_hardware()
        
        gpus = []
        for gpu in hw.gpus:
            gpus.append({
                "name": gpu.name,
                "total_memory_gb": gpu.total_memory_gb,
                "free_memory_gb": gpu.free_memory_gb,
                "type": gpu.device_type.value,
            })
        
        return {
            "os": hw.system.os_name,
            "cpu": hw.system.cpu_name,
            "cpu_count": hw.system.cpu_count,
            "ram_total_gb": hw.system.total_ram_gb,
            "ram_available_gb": hw.system.available_ram_gb,
            "gpus": gpus,
            "best_device": hw.best_device.value,
        }
    
    @app.get("/stats/speculative")
    async def speculative_stats():
        """Get speculative decoding statistics."""
        if not state.llm or not state.llm.speculative_decoder:
            return {
                "enabled": False,
                "message": "Speculative decoding not enabled. Start with --speculative flag.",
            }
        
        stats = state.llm.speculative_decoder.get_stats()
        return {
            "enabled": True,
            "acceptance_rate": stats.get("acceptance_rate", 0),
            "speedup_factor": stats.get("speedup_factor", 1.0),
            "accepted_tokens": stats.get("accepted_tokens", 0),
            "rejected_tokens": stats.get("rejected_tokens", 0),
            "draft_forward_passes": stats.get("draft_forward_passes", 0),
            "target_forward_passes": stats.get("target_forward_passes", 0),
            "is_fallback_mode": stats.get("is_fallback_mode", False),
            "draft_model": state.llm.config.draft_model_id,
        }
    
    @app.get("/stats/flash_attention")
    async def flash_attention_stats():
        """Get flash attention configuration."""
        if not state.llm or not state.llm.flash_attention_config:
            return {
                "enabled": False,
            }
        
        return {
            "enabled": True,
            **state.llm.flash_attention_config,
        }
    
    return app


# ============== Run directly ==============

if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else None
    app = create_app(model_id=model)
    uvicorn.run(app, host="127.0.0.1", port=8000)
