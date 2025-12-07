import re
from openvoice.text.english import english_to_lazy_ipa, english_to_ipa2, english_to_lazy_ipa2
from openvoice.text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo, chinese_to_romaji, chinese_to_lazy_ipa, chinese_to_ipa, chinese_to_ipa2

# Añadir importaciones para japonés y coreano si existen
try:
    from openvoice.text.japanese import japanese_to_ipa2
except ImportError:
    # Si no existen los módulos de japonés, crear función dummy
    def japanese_to_ipa2(text):
        raise ImportError("Japanese support not available")
    
try:
    from openvoice.text.korean import korean_to_ipa
except ImportError:
    # Si no existen los módulos de coreano, crear función dummy
    def korean_to_ipa(text):
        raise ImportError("Korean support not available")

def cjke_cleaners2(text):
    """Cleaner for CJKE (Chinese, Japanese, Korean, English) text"""
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1)) + ' ', text)
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
    # Convertir números a palabras (esto es un ejemplo simple)
    text = re.sub(r'(\d+)', lambda x: num2words(int(x.group(1))), text)
    # Expandir abreviaciones comunes
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

# Función auxiliar para convertir números a palabras (requiere num2words)
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
            try:
                return japanese_to_ipa2(content) + ' '
            except:
                return content + ' '
        elif lang == 'KO':
            try:
                return korean_to_ipa(content) + ' '
            except:
                return content + ' '
        else:
            return content + ' '
    
    # Procesar bloques con etiquetas de idioma
    text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', process_language_block, text)
    
    # Asegurar que termina con un punto si no tiene puntuación final
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    
    return text.strip()
