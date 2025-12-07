import re
from openvoice.text.english import english_to_lazy_ipa, english_to_ipa2, english_to_lazy_ipa2
from openvoice.text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo, chinese_to_romaji, chinese_to_lazy_ipa, chinese_to_ipa, chinese_to_ipa2

# Añadir importaciones para japonés y coreano si existen
try:
    from openvoice.text.japanese import japanese_to_ipa2
    HAS_JAPANESE = True
except ImportError:
    HAS_JAPANESE = False
    def japanese_to_ipa2(text):
        raise ImportError("Japanese support not available. Install Japanese dependencies.")

try:
    from openvoice.text.korean import korean_to_ipa
    HAS_KOREAN = True
except ImportError:
    HAS_KOREAN = False
    def korean_to_ipa(text):
        raise ImportError("Korean support not available. Install Korean dependencies.")

def cjke_cleaners2(text):
    """Cleaner for CJKE (Chinese, Japanese, Korean, English) text"""
    # TODAS LAS REGEX CORREGIDAS A RAW STRINGS
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: (japanese_to_ipa2(x.group(1)) if HAS_JAPANESE 
                            else x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: (korean_to_ipa(x.group(1)) if HAS_KOREAN 
                            else x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text

# Funciones de limpieza adicionales comunes en TTS
def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = text.lower()
    # Convertir números a palabras
    text = re.sub(r'(\d+)', lambda x: num2words(int(x.group(1))), text)
    
    # CORREGIDO: Todas las abreviaciones como raw strings
    abbreviations = {
        r"mr\.": "mister",
        r"mrs\.": "misses",
        r"dr\.": "doctor",
        r"st\.": "saint",
        r"co\.": "company",
        r"jr\.": "junior",
        r"sr\.": "senior",
        r"etc\.": "et cetera",
    }
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Función auxiliar para convertir números a palabras
try:
    from num2words import num2words
except ImportError:
    # Fallback simple si num2words no está disponible
    def num2words(num):
        return str(num)

def chinese_cleaners(text):
    """Pipeline for Chinese text cleaning."""
    # Asegurar que el texto esté en formato Unicode
    text = text.strip()
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text

def multilingual_cleaners(text):
    """Cleaner for multilingual text that detects language tags."""
    # Primero, normalizar espacios
    text = re.sub(r'\s+', ' ', text)

    # Buscar y procesar bloques de idioma
    def process_language_block(match):
        lang = match.group(1)
        content = match.group(2)

        if lang == 'ZH':
            return chinese_to_ipa(content) + ' '
        elif lang == 'EN':
            return english_to_ipa2(content) + ' '
        elif lang == 'JA':
            if HAS_JAPANESE:
                return japanese_to_ipa2(content) + ' '
            else:
                print(f"Warning: Japanese support not installed. Keeping text as-is: {content}")
                return content + ' '
        elif lang == 'KO':
            if HAS_KOREAN:
                return korean_to_ipa(content) + ' '
            else:
                print(f"Warning: Korean support not installed. Keeping text as-is: {content}")
                return content + ' '
        else:
            print(f"Warning: Unknown language tag [{lang}]. Keeping text as-is: {content}")
            return content + ' '

    # Procesar bloques con etiquetas de idioma
    text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', process_language_block, text)

    # Asegurar que termina con un punto si no tiene puntuación final
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)

    return text.strip()

# Funciones adicionales útiles para TTS
def remove_extra_punctuation(text):
    """Remove excessive punctuation marks."""
    # Reemplazar múltiples signos de exclamación/interrogación por uno solo
    text = re.sub(r'[!?]{2,}', lambda m: m.group(0)[0], text)
    # Reemplazar múltiples puntos suspensivos por uno solo
    text = re.sub(r'\.{3,}', '...', text)
    # Reemplazar múltiples comas por una sola
    text = re.sub(r',{2,}', ',', text)
    return text

def normalize_whitespace(text):
    """Normalize all whitespace characters to single spaces."""
    # Reemplazar cualquier combinación de espacios, tabs, newlines por un solo espacio
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio y final
    return text.strip()

def clean_quotes(text):
    """Normalize different types of quotes to standard ones."""
    # Comillas dobles
    text = re.sub(r'[«»"＂]', '"', text)
    # Comillas simples
    text = re.sub(r'[『』\'＇]', "'", text)
    return text

# Funciones de limpieza combinadas
def full_cleaners(text):
    """Apply all cleaning steps in sequence."""
    text = normalize_whitespace(text)
    text = clean_quotes(text)
    text = remove_extra_punctuation(text)
    return text

# Función para detectar idioma automáticamente
def detect_language(text):
    """Simple language detection based on character ranges."""
    # Contar caracteres de cada idioma
    import string
    
    # Caracteres chinos/japoneses
    cjk_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    # Caracteres coreanos
    korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7a3'])
    # Caracteres latinos
    latin_chars = len([c for c in text if c in string.ascii_letters])
    
    if cjk_chars > latin_chars and cjk_chars > korean_chars:
        return 'ZH'
    elif korean_chars > latin_chars and korean_chars > cjk_chars:
        return 'KO'
    else:
        return 'EN'

# Registro de cleaners disponibles para importación dinámica
AVAILABLE_CLEANERS = {
    'cjke_cleaners2': cjke_cleaners2,
    'basic_cleaners': basic_cleaners,
    'transliteration_cleaners': transliteration_cleaners,
    'english_cleaners': english_cleaners,
    'chinese_cleaners': chinese_cleaners,
    'multilingual_cleaners': multilingual_cleaners,
    'full_cleaners': full_cleaners,
}

def get_cleaner(name):
    """Get a cleaner function by name."""
    if name not in AVAILABLE_CLEANERS:
        raise ValueError(f"Unknown cleaner: {name}. Available: {list(AVAILABLE_CLEANERS.keys())}")
    return AVAILABLE_CLEANERS[name]
