"""
Model Hub for discovering and searching models.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from huggingface_hub import HfApi


@dataclass
class HubModel:
    """Model information from HuggingFace Hub."""
    model_id: str
    author: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str]
    library: Optional[str]
    
    @property
    def name(self) -> str:
        return self.model_id.split("/")[-1]
    
    @property
    def is_llm(self) -> bool:
        return self.pipeline_tag in ["text-generation", "conversational"]


class ModelHub:
    """
    Interface to HuggingFace Model Hub for discovering models.
    """
    
    # Popular LLM models that work well with zllm
    RECOMMENDED_MODELS = [
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3-70B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
    ]
    
    # Memory requirements (approximate, in GB for fp16)
    MODEL_SIZES = {
        "7b": 14,
        "8b": 16,
        "13b": 26,
        "34b": 68,
        "70b": 140,
        "72b": 144,
    }
    
    def __init__(self):
        self.api = HfApi()
    
    def search(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> List[HubModel]:
        """
        Search for models on HuggingFace Hub.
        
        Args:
            query: Search query
            limit: Maximum results
            sort: Sort by ("downloads", "likes", "trending")
        
        Returns:
            List of matching models
        """
        models = self.api.list_models(
            search=query,
            task="text-generation",
            sort=sort,
            direction=-1,
            limit=limit,
        )
        
        results = []
        for model in models:
            results.append(HubModel(
                model_id=model.id,
                author=model.author or "",
                downloads=model.downloads or 0,
                likes=model.likes or 0,
                tags=model.tags or [],
                pipeline_tag=model.pipeline_tag,
                library=model.library_name,
            ))
        
        return results
    
    def get_recommended(self, max_memory_gb: float = 16) -> List[str]:
        """
        Get recommended models that fit in available memory.
        
        Args:
            max_memory_gb: Maximum available GPU memory in GB
        
        Returns:
            List of model IDs that should fit
        """
        recommended = []
        
        for model_id in self.RECOMMENDED_MODELS:
            # Estimate size from model name
            name_lower = model_id.lower()
            estimated_size = 0
            
            for size_key, size_gb in self.MODEL_SIZES.items():
                if size_key in name_lower:
                    estimated_size = size_gb
                    break
            
            # Check if fits (with 4-bit quantization, size is ~25%)
            if estimated_size == 0:
                estimated_size = 14  # Default to 7B size
            
            min_memory_full = estimated_size
            min_memory_int4 = estimated_size * 0.25
            
            if min_memory_full <= max_memory_gb or min_memory_int4 <= max_memory_gb:
                recommended.append(model_id)
        
        return recommended
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        try:
            model = self.api.model_info(model_id)
            return {
                "model_id": model.id,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "library": model.library_name,
                "pipeline_tag": model.pipeline_tag,
                "created_at": str(model.created_at) if model.created_at else None,
                "last_modified": str(model.last_modified) if model.last_modified else None,
            }
        except Exception:
            return None
    
    def estimate_memory_requirement(
        self,
        model_id: str,
        quantization: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for a model.
        
        Returns:
            Dict with 'min_gpu', 'min_cpu', 'recommended' in GB
        """
        name_lower = model_id.lower()
        base_size = 14  # Default 7B
        
        for size_key, size_gb in self.MODEL_SIZES.items():
            if size_key in name_lower:
                base_size = size_gb
                break
        
        # Adjust for quantization
        if quantization == "int4":
            multiplier = 0.25
        elif quantization == "int8":
            multiplier = 0.5
        else:
            multiplier = 1.0
        
        model_size = base_size * multiplier
        
        # KV cache overhead (~20% for inference)
        inference_overhead = model_size * 0.2
        
        return {
            "model_size_gb": model_size,
            "min_gpu_gb": model_size + inference_overhead,
            "min_cpu_gb": model_size * 1.5,  # Needs more for CPU inference
            "recommended_gb": (model_size + inference_overhead) * 1.2,
            "with_layer_streaming_gb": model_size * 0.15,  # Only need ~15% with streaming
        }
