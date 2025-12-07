""" from https://github.com/keithito/tacotron """
import re
from typing import List, Tuple, Dict, Optional
from openvoice.text import cleaners
from openvoice.text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text: str, symbols: List[str], cleaner_names: List[str]) -> List[int]:
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
        symbols: list of symbols to use
        cleaner_names: names of the cleaner functions to run the text through
    Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    clean_text = _clean_text(text, cleaner_names)
    print(clean_text)
    print(f" length:{len(clean_text)}")
    
    for symbol in clean_text:
        if symbol not in symbol_to_id:
            continue
        symbol_id = symbol_to_id[symbol]
        sequence.append(symbol_id)
    
    print(f" length:{len(sequence)}")
    return sequence


def cleaned_text_to_sequence(cleaned_text: str, symbols: List[str]) -> List[int]:
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        cleaned_text: already cleaned text to convert to a sequence
        symbols: list of symbols to use
    Returns:
        List of integers corresponding to the symbols in the text
    '''
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id]
    return sequence


# Import condicional para compatibilidad
try:
    from openvoice.text.symbols import language_tone_start_map
    HAS_VITS2 = True
except ImportError:
    HAS_VITS2 = False
    language_tone_start_map = {}


def cleaned_text_to_sequence_vits2(
    cleaned_text: str, 
    tones: List[int], 
    language: str, 
    symbols: List[str], 
    languages: List[str]
) -> Tuple[List[int], List[int], List[int]]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        cleaned_text: already cleaned text to convert to a sequence
        tones: list of tones for each symbol
        language: language identifier
        symbols: list of symbols to use
        languages: list of available languages
    Returns:
        tuple: (phones, tones, lang_ids) where phones are symbol IDs, 
               tones are tone IDs, lang_ids are language IDs
    """
    if not HAS_VITS2:
        raise ImportError("VITS2 support not available. Install required dependencies.")
    
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    language_id_map = {s: i for i, s in enumerate(languages)}
    
    # Validar entradas
    if language not in language_id_map:
        raise ValueError(f"Language '{language}' not in available languages: {list(language_id_map.keys())}")
    
    if language not in language_tone_start_map:
        raise ValueError(f"Language '{language}' not in tone start map")
    
    # Convertir símbolos a IDs
    phones = []
    for symbol in cleaned_text:
        if symbol in symbol_to_id:
            phones.append(symbol_to_id[symbol])
        else:
            # Manejar símbolos desconocidos (opcional: usar un símbolo de reemplazo)
            print(f"Warning: Symbol '{symbol}' not in vocabulary")
            continue
    
    # Ajustar tonos
    tone_start = language_tone_start_map[language]
    adjusted_tones = [i + tone_start for i in tones]
    
    # Asegurar que las longitudes coincidan
    if len(phones) != len(adjusted_tones):
        print(f"Warning: Mismatch in lengths - phones: {len(phones)}, tones: {len(adjusted_tones)}")
        # Recortar al mínimo
        min_len = min(len(phones), len(adjusted_tones))
        phones = phones[:min_len]
        adjusted_tones = adjusted_tones[:min_len]
    
    lang_id = language_id_map[language]
    lang_ids = [lang_id] * len(phones)
    
    return phones, adjusted_tones, lang_ids


def sequence_to_text(sequence: List[int]) -> str:
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
        else:
            print(f"Warning: Symbol ID {symbol_id} not in vocabulary")
    return result


def _clean_text(text: str, cleaner_names: List[str]) -> str:
    """Apply a series of cleaners to text"""
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if cleaner is None:
            raise ValueError(f'Unknown cleaner: {name}')
        text = cleaner(text)
    return text


def validate_text(text: str, symbols: List[str]) -> Tuple[bool, List[str]]:
    """Validate if text contains only supported symbols"""
    supported_symbols = set(symbols)
    text_symbols = set(text)
    unsupported = list(text_symbols - supported_symbols)
    return len(unsupported) == 0, unsupported


def get_symbol_frequency(sequence: List[int]) -> Dict[str, int]:
    """Get frequency of each symbol in a sequence"""
    freq = {}
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            symbol = _id_to_symbol[symbol_id]
            freq[symbol] = freq.get(symbol, 0) + 1
    return freq


# Funciones de utilidad adicionales
def split_long_sequence(sequence: List[int], max_length: int = 200) -> List[List[int]]:
    """Split a long sequence into chunks"""
    chunks = []
    for i in range(0, len(sequence), max_length):
        chunks.append(sequence[i:i + max_length])
    return chunks


def remove_unknown_symbols(text: str, symbols: List[str]) -> str:
    """Remove symbols not in the vocabulary"""
    symbol_set = set(symbols)
    return ''.join([c for c in text if c in symbol_set])
