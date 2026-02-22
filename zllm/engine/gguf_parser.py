"""
GGUF File Parser - Pure Python implementation.

Reads GGUF files without any external dependencies.
Based on GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

GGUF Format:
- Header: magic, version, tensor_count, metadata_kv_count
- Metadata: key-value pairs with model info
- Tensor Info: name, shape, type, offset for each tensor
- Tensor Data: raw tensor data (aligned)
"""

import struct
import mmap
from pathlib import Path
from typing import Dict, List, Any, Optional, BinaryIO, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class GGMLType(IntEnum):
    """GGML tensor data types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # Q4_2 = 4  # deprecated
    # Q4_3 = 5  # deprecated
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


# Block sizes for quantized types
GGML_BLOCK_SIZES = {
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.Q2_K: 256,
    GGMLType.Q3_K: 256,
    GGMLType.Q4_K: 256,
    GGMLType.Q5_K: 256,
    GGMLType.Q6_K: 256,
    GGMLType.Q8_K: 256,
}

# Bytes per block for quantized types
GGML_TYPE_SIZES = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.Q4_0: 18,      # 32 values in 18 bytes (2 + 16)
    GGMLType.Q4_1: 20,      # 32 values in 20 bytes (2 + 2 + 16)
    GGMLType.Q5_0: 22,      # 32 values in 22 bytes
    GGMLType.Q5_1: 24,      # 32 values in 24 bytes
    GGMLType.Q8_0: 34,      # 32 values in 34 bytes (2 + 32)
    GGMLType.Q8_1: 36,      # 32 values in 36 bytes
    GGMLType.Q2_K: 84,      # 256 values
    GGMLType.Q3_K: 110,     # 256 values
    GGMLType.Q4_K: 144,     # 256 values
    GGMLType.Q5_K: 176,     # 256 values
    GGMLType.Q6_K: 210,     # 256 values
    GGMLType.Q8_K: 292,     # 256 values
    GGMLType.BF16: 2,
    GGMLType.I8: 1,
    GGMLType.I16: 2,
    GGMLType.I32: 4,
    GGMLType.I64: 8,
    GGMLType.F64: 8,
}


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class GGUFTensor:
    """Information about a tensor in the GGUF file."""
    name: str
    shape: Tuple[int, ...]
    dtype: GGMLType
    offset: int  # Offset from start of tensor data section
    n_elements: int
    n_bytes: int
    
    @property
    def is_quantized(self) -> bool:
        return self.dtype not in (GGMLType.F32, GGMLType.F16, GGMLType.BF16)
    
    @property
    def quantization_name(self) -> str:
        return self.dtype.name


@dataclass
class GGUFMetadata:
    """Parsed GGUF file metadata."""
    version: int
    tensor_count: int
    metadata_kv_count: int
    
    # Common metadata fields
    architecture: str = ""
    name: str = ""
    quantization_version: int = 0
    
    # Model architecture params
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0  # Number of layers
    feed_forward_length: int = 0
    attention_head_count: int = 0
    attention_head_count_kv: int = 0
    vocab_size: int = 0
    
    # Rope params
    rope_dimension_count: int = 0
    rope_freq_base: float = 10000.0
    
    # Tokenizer
    tokenizer_model: str = ""
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    
    # All raw metadata
    raw: Dict[str, Any] = field(default_factory=dict)


class GGUFParser:
    """
    Parse GGUF files - our own implementation.
    
    Example:
        parser = GGUFParser("model.gguf")
        metadata = parser.metadata
        
        # Get tensor info
        for tensor in parser.tensors.values():
            print(f"{tensor.name}: {tensor.shape} ({tensor.quantization_name})")
        
        # Load a specific tensor
        data = parser.load_tensor("model.layers.0.attention.wq.weight")
    """
    
    MAGIC = b"GGUF"
    ALIGNMENT = 32  # Data alignment in bytes
    
    def __init__(self, path: str):
        """
        Load a GGUF file.
        
        Args:
            path: Path to .gguf file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {path}")
        
        self.file_size = self.path.stat().st_size
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        
        # Parsed data
        self.metadata: GGUFMetadata = None
        self.tensors: Dict[str, GGUFTensor] = {}
        self._tensor_data_offset: int = 0
        
        # Parse the file
        self._parse()
    
    def _parse(self) -> None:
        """Parse the GGUF file header and metadata."""
        with open(self.path, "rb") as f:
            # Read magic
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(f"Invalid GGUF magic: {magic}, expected {self.MAGIC}")
            
            # Read version
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")
            
            # Read counts
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
            
            # Initialize metadata
            self.metadata = GGUFMetadata(
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
            )
            
            # Parse metadata key-value pairs
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                value = self._read_value(f)
                self.metadata.raw[key] = value
                self._set_metadata_field(key, value)
            
            # Parse tensor info
            for _ in range(tensor_count):
                tensor = self._read_tensor_info(f)
                self.tensors[tensor.name] = tensor
            
            # Calculate tensor data offset (aligned)
            current_pos = f.tell()
            self._tensor_data_offset = (current_pos + self.ALIGNMENT - 1) // self.ALIGNMENT * self.ALIGNMENT
    
    def _read_string(self, f: BinaryIO) -> str:
        """Read a length-prefixed string."""
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")
    
    def _read_value(self, f: BinaryIO) -> Any:
        """Read a typed value."""
        value_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
        
        if value_type == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(f)
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def _read_array(self, f: BinaryIO) -> List[Any]:
        """Read an array value."""
        element_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
        length = struct.unpack("<Q", f.read(8))[0]
        
        result = []
        for _ in range(length):
            if element_type == GGUFValueType.STRING:
                result.append(self._read_string(f))
            elif element_type == GGUFValueType.UINT32:
                result.append(struct.unpack("<I", f.read(4))[0])
            elif element_type == GGUFValueType.INT32:
                result.append(struct.unpack("<i", f.read(4))[0])
            elif element_type == GGUFValueType.FLOAT32:
                result.append(struct.unpack("<f", f.read(4))[0])
            elif element_type == GGUFValueType.UINT64:
                result.append(struct.unpack("<Q", f.read(8))[0])
            elif element_type == GGUFValueType.INT64:
                result.append(struct.unpack("<q", f.read(8))[0])
            else:
                # For other types, read as raw bytes
                result.append(self._read_value_by_type(f, element_type))
        
        return result
    
    def _read_value_by_type(self, f: BinaryIO, vtype: GGUFValueType) -> Any:
        """Read a value of a specific type."""
        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack("<?", f.read(1))[0]
        else:
            raise ValueError(f"Unsupported array element type: {vtype}")
    
    def _read_tensor_info(self, f: BinaryIO) -> GGUFTensor:
        """Read tensor info (not the data itself)."""
        name = self._read_string(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack("<Q", f.read(8))[0])
        shape = tuple(shape)
        
        dtype = GGMLType(struct.unpack("<I", f.read(4))[0])
        offset = struct.unpack("<Q", f.read(8))[0]
        
        # Calculate number of elements and bytes
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        
        # Calculate bytes based on type
        if dtype in GGML_TYPE_SIZES:
            if dtype in GGML_BLOCK_SIZES:
                block_size = GGML_BLOCK_SIZES[dtype]
                type_size = GGML_TYPE_SIZES[dtype]
                n_blocks = (n_elements + block_size - 1) // block_size
                n_bytes = n_blocks * type_size
            else:
                n_bytes = n_elements * GGML_TYPE_SIZES[dtype]
        else:
            n_bytes = 0  # Unknown
        
        return GGUFTensor(
            name=name,
            shape=shape,
            dtype=dtype,
            offset=offset,
            n_elements=n_elements,
            n_bytes=n_bytes,
        )
    
    def _set_metadata_field(self, key: str, value: Any) -> None:
        """Set a metadata field from a key-value pair."""
        # Map common GGUF keys to our metadata fields
        key_mapping = {
            "general.architecture": "architecture",
            "general.name": "name",
            "general.quantization_version": "quantization_version",
            ".context_length": "context_length",
            ".embedding_length": "embedding_length",
            ".block_count": "block_count",
            ".feed_forward_length": "feed_forward_length",
            ".attention.head_count": "attention_head_count",
            ".attention.head_count_kv": "attention_head_count_kv",
            ".vocab_size": "vocab_size",
            ".rope.dimension_count": "rope_dimension_count",
            ".rope.freq_base": "rope_freq_base",
            "tokenizer.ggml.model": "tokenizer_model",
            "tokenizer.ggml.bos_token_id": "bos_token_id",
            "tokenizer.ggml.eos_token_id": "eos_token_id",
            "tokenizer.ggml.padding_token_id": "pad_token_id",
        }
        
        # Check for direct match
        if key in key_mapping:
            setattr(self.metadata, key_mapping[key], value)
            return
        
        # Check for architecture-prefixed keys (e.g., "llama.context_length")
        for suffix, field in key_mapping.items():
            if suffix.startswith(".") and key.endswith(suffix):
                setattr(self.metadata, field, value)
                return
    
    def _open_mmap(self) -> None:
        """Open file with memory mapping for efficient tensor loading."""
        if self._mmap is None:
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
    
    def _close_mmap(self) -> None:
        """Close memory mapping."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def load_tensor_raw(self, name: str) -> bytes:
        """
        Load raw tensor data (still quantized).
        
        Args:
            name: Tensor name
        
        Returns:
            Raw bytes of tensor data
        """
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")
        
        tensor = self.tensors[name]
        
        self._open_mmap()
        
        start = self._tensor_data_offset + tensor.offset
        end = start + tensor.n_bytes
        
        return bytes(self._mmap[start:end])
    
    def load_tensor(self, name: str, device: str = "cpu") -> "torch.Tensor":
        """
        Load and dequantize a tensor.
        
        Args:
            name: Tensor name
            device: Target device
        
        Returns:
            Dequantized tensor as torch.Tensor
        """
        import torch
        from .quantization import dequantize_tensor
        
        tensor_info = self.tensors[name]
        raw_data = self.load_tensor_raw(name)
        
        # Dequantize
        tensor = dequantize_tensor(raw_data, tensor_info.dtype, tensor_info.shape)
        
        return tensor.to(device)
    
    def iter_tensors(self, pattern: Optional[str] = None):
        """
        Iterate over tensors, optionally filtering by name pattern.
        
        Args:
            pattern: Optional regex pattern to filter tensor names
        
        Yields:
            (name, GGUFTensor) tuples
        """
        import re
        
        for name, tensor in self.tensors.items():
            if pattern is None or re.search(pattern, name):
                yield name, tensor
    
    def get_layer_tensors(self, layer_idx: int) -> Dict[str, GGUFTensor]:
        """
        Get all tensors for a specific layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Dict of tensor name -> GGUFTensor for that layer
        """
        pattern = f"layers.{layer_idx}."
        return {
            name: tensor
            for name, tensor in self.tensors.items()
            if pattern in name
        }
    
    def __repr__(self) -> str:
        return (
            f"GGUFParser({self.path.name}, "
            f"v{self.metadata.version}, "
            f"{self.metadata.tensor_count} tensors, "
            f"{self.metadata.architecture})"
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._close_mmap()
    
    def close(self):
        """Close the parser and release resources."""
        self._close_mmap()


def inspect_gguf(path: str) -> None:
    """
    Print detailed information about a GGUF file.
    
    Useful for debugging and understanding model structure.
    """
    parser = GGUFParser(path)
    
    print(f"GGUF File: {parser.path.name}")
    print(f"Size: {parser.file_size / (1024**3):.2f} GB")
    print(f"Version: {parser.metadata.version}")
    print(f"Tensor Count: {parser.metadata.tensor_count}")
    print()
    
    print("=== Model Info ===")
    print(f"Architecture: {parser.metadata.architecture}")
    print(f"Name: {parser.metadata.name}")
    print(f"Layers: {parser.metadata.block_count}")
    print(f"Context Length: {parser.metadata.context_length}")
    print(f"Embedding Size: {parser.metadata.embedding_length}")
    print(f"Vocab Size: {parser.metadata.vocab_size}")
    print(f"Attention Heads: {parser.metadata.attention_head_count}")
    print(f"KV Heads: {parser.metadata.attention_head_count_kv}")
    print()
    
    print("=== Quantization ===")
    quant_types = {}
    for tensor in parser.tensors.values():
        qtype = tensor.dtype.name
        if qtype not in quant_types:
            quant_types[qtype] = 0
        quant_types[qtype] += 1
    
    for qtype, count in sorted(quant_types.items()):
        print(f"  {qtype}: {count} tensors")
    
    print()
    print("=== Sample Tensors ===")
    for i, (name, tensor) in enumerate(parser.tensors.items()):
        if i >= 10:
            print(f"  ... and {len(parser.tensors) - 10} more")
            break
        print(f"  {name}: {tensor.shape} ({tensor.dtype.name})")
    
    parser.close()
