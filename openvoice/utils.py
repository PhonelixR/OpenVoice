import re
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union


def get_hparams_from_file(config_path: str) -> 'HParams':
    """Load hyperparameters from JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    return HParams(**config)


class HParams:
    """Hyperparameters container with dictionary-like and attribute-like access."""
    
    def init(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            setattr(self, k, v)
    
    def keys(self):
        return self.dict.keys()
    
    def items(self):
        return self.dict.items()
    
    def values(self):
        return self.dict.values()
    
    def len(self):
        return len(self.dict)
    
    def getitem(self, key):
        return getattr(self, key)
    
    def setitem(self, key, value):
        setattr(self, key, value)
    
    def contains(self, key):
        return key in self.dict
    
    def repr(self):
        return f"HParams({self.dict})"
    
    def get(self, key, default=None):
        """Dictionary-style get with default."""
        return getattr(self, key, default)
    
    def to_dict(self):
        """Convert to nested dictionary."""
        result = {}
        for k, v in self.dict.items():
            if isinstance(v, HParams):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def string_to_bits(string: str, pad_len: int = 8) -> np.ndarray:
    """Convert string to 8-bit binary array with padding."""
    ascii_values = [ord(char) for char in string]
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]
    
    numpy_array = np.array(bit_arrays, dtype=np.uint8)
    
    # Handle padding
    if len(numpy_array) < pad_len:
        padded = np.zeros((pad_len, 8), dtype=np.uint8)
        padded[:len(numpy_array)] = numpy_array
        # Fill padding with zeros (or space character bits: 00100000)
        return padded
    else:
        return numpy_array[:pad_len]


def bits_to_string(bits_array: np.ndarray) -> str:
    """Convert 8-bit binary array back to string."""
    # Flatten if needed and reshape to (n, 8)
    if bits_array.ndim == 1:
        if len(bits_array) % 8 != 0:
            # Pad to multiple of 8
            bits_array = np.pad(bits_array, (0, 8 - len(bits_array) % 8))
        bits_array = bits_array.reshape(-1, 8)
    
    # Convert to string
    chars = []
    for row in bits_array:
        # Convert bits to integer
        byte_val = 0
        for i in range(8):
            if i < len(row) and row[i]:
                byte_val |= (1 << (7 - i))
        
        # Skip null bytes at the end
        if byte_val == 0:
            continue
        
        chars.append(chr(byte_val))
    
    return ''.join(chars).rstrip('\x00')


def split_sentence(text: str, min_len: int = 10, language_str: str = '[EN]') -> List[str]:
    """Split text into sentences based on language."""
    if language_str in ['EN', 'en', 'English']:
        return split_sentences_latin(text, min_len=min_len)
    else:
        return split_sentences_zh(text, min_len=min_len)


def split_sentences_latin(text: str, min_len: int = 10) -> List[str]:
    """Split Latin text into sentences."""
    # Normalize punctuation
    text = re.sub(r'[。！？；]', '.', text)
    text = re.sub(r'[，]', ',', text)
    text = re.sub(r'[“”「」]', '"', text)
    text = re.sub(r'[‘’『』]', "'", text)
    text = re.sub(r'[\<\>\(\)\[\]\"\«\»]+', "", text)
    text = re.sub(r'[\n\t ]+', ' ', text)
    text = re.sub(r'([,.!?;])', r'\1 $#!', text)
    
    # Split
    sentences = [s.strip() for s in text.split('$#!') if s.strip()]
    
    # Group sentences by length
    return _group_sentences_by_length(sentences, min_len, is_latin=True)


def split_sentences_zh(text: str, min_len: int = 10) -> List[str]:
    """Split Chinese text into sentences."""
    text = re.sub(r'[。！？；]', '.', text)
    text = re.sub(r'[，]', ',', text)
    text = re.sub(r'[\n\t ]+', ' ', text)
    text = re.sub(r'([,.!?;])', r'\1 $#!', text)
    
    sentences = [s.strip() for s in text.split('$#!') if s.strip()]
    
    # Group sentences by length
    return _group_sentences_by_length(sentences, min_len, is_latin=False)


def _group_sentences_by_length(sentences: List[str], min_len: int, is_latin: bool = True) -> List[str]:
    """Group sentences to meet minimum length requirement."""
    if not sentences:
        return []
    
    grouped = []
    current_group = []
    current_length = 0
    
    for sent in sentences:
        current_group.append(sent)
        # Count words for Latin, characters for Chinese
        current_length += len(sent.split()) if is_latin else len(sent)
        
        if current_length >= min_len:
            grouped.append(' '.join(current_group))
            current_group = []
            current_length = 0
    
    # Handle last group
    if current_group:
        if grouped and current_length < 2:  # Very short last group
            grouped[-1] = grouped[-1] + ' ' + ' '.join(current_group)
        else:
            grouped.append(' '.join(current_group))
    
    return grouped


# Backward compatibility
merge_short_sentences_latin = lambda sens: _group_sentences_by_length(sens, 2, True)
merge_short_sentences_zh = lambda sens: _group_sentences_by_length(sens, 2, False)
