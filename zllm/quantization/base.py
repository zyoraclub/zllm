"""
Quantization configuration and utilities.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class QuantizationType(str, Enum):
    """Supported quantization types."""
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"
    NONE = "none"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization type
    quant_type: QuantizationType = QuantizationType.NONE
    
    # Bits for quantization
    bits: int = 4
    
    # Group size for grouped quantization
    group_size: int = 128
    
    # Whether to use double quantization
    double_quant: bool = True
    
    # Compute dtype
    compute_dtype: str = "float16"
    
    # For GPTQ/AWQ
    desc_act: bool = False
    sym: bool = True
    
    # Calibration settings
    calibration_dataset: Optional[str] = None
    calibration_samples: int = 128
    
    def to_bitsandbytes_config(self):
        """Convert to bitsandbytes configuration."""
        try:
            from transformers import BitsAndBytesConfig
            import torch
            
            if self.quant_type == QuantizationType.INT4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=self.double_quant,
                )
            elif self.quant_type == QuantizationType.INT8:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                return None
        except ImportError:
            return None
    
    @classmethod
    def int4(cls) -> "QuantizationConfig":
        """Create a 4-bit quantization config."""
        return cls(quant_type=QuantizationType.INT4, bits=4)
    
    @classmethod
    def int8(cls) -> "QuantizationConfig":
        """Create an 8-bit quantization config."""
        return cls(quant_type=QuantizationType.INT8, bits=8)


class AutoQuantizer:
    """
    Automatically select and apply quantization.
    
    Chooses the best quantization method based on:
    - Available memory
    - Model size
    - Hardware capabilities
    """
    
    @staticmethod
    def get_recommended_config(
        model_size_gb: float,
        available_memory_gb: float,
        has_cuda: bool = True,
    ) -> QuantizationConfig:
        """
        Get recommended quantization config.
        
        Args:
            model_size_gb: Model size in GB (fp16)
            available_memory_gb: Available GPU/CPU memory in GB
            has_cuda: Whether CUDA is available
        """
        # Calculate memory needed with overhead
        memory_with_overhead = model_size_gb * 1.3
        
        if memory_with_overhead <= available_memory_gb:
            # No quantization needed
            return QuantizationConfig(quant_type=QuantizationType.NONE)
        
        # Check if INT8 fits
        int8_size = model_size_gb * 0.55
        if int8_size * 1.3 <= available_memory_gb:
            return QuantizationConfig.int8()
        
        # Use INT4
        return QuantizationConfig.int4()
    
    @staticmethod
    def estimate_quantized_size(
        model_size_gb: float,
        quant_config: QuantizationConfig,
    ) -> float:
        """
        Estimate model size after quantization.
        
        Args:
            model_size_gb: Original model size in GB (fp16)
            quant_config: Quantization configuration
        
        Returns:
            Estimated size in GB
        """
        multipliers = {
            QuantizationType.NONE: 1.0,
            QuantizationType.INT8: 0.5,
            QuantizationType.INT4: 0.25,
            QuantizationType.GPTQ: 0.25,
            QuantizationType.AWQ: 0.25,
        }
        
        multiplier = multipliers.get(quant_config.quant_type, 1.0)
        return model_size_gb * multiplier
