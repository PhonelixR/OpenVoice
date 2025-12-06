""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterat (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
import inflect
from unidecode import unidecode

# Intentar importar eng_to_ipa, con fallback alternativo
try:
    import eng_to_ipa as ipa
    ENG_TO_IPA_AVAILABLE = True
except ImportError:
    ENG_TO_IPA_AVAILABLE = False
    # Crear un sustituto simple para eng_to_ipa
    class SimpleIPAConverter:
        @staticmethod
        def convert(text):
            # Un sustituto muy básico que simplemente devuelve el texto
            # En un caso real, necesitarías una implementación mejor
            return text
    ipa = SimpleIPAConverter()

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile(r'\b%s\.' % re.escape(x[0]), re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


# List of (ipa, lazy ipa) pairs:
_lazy_ipa = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('r', 'ɹ'),
    ('æ', 'e'),
    ('ɑ', 'a'),
    ('ɔ', 'o'),
    ('ð', 'z'),
    ('θ', 's'),
    ('ɛ', 'e'),
    ('ɪ', 'i'),
    ('ʊ', 'u'),
    ('ʒ', 'ʥ'),
    ('ʤ', 'ʥ'),
    ('ˈ', '↓'),
]]

# List of (ipa, lazy ipa2) pairs:
_lazy_ipa2 = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('r', 'ɹ'),
    ('ð', 'z'),
    ('θ', 's'),
    ('ʒ', 'ʑ'),
    ('ʤ', 'dʑ'),
    ('ˈ', '↓'),
]]

# List of (ipa, ipa2) pairs
_ipa_to_ipa2 = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('r', 'ɹ'),
    ('ʤ', 'dʒ'),
    ('ʧ', 'tʃ')
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return 'two thousand'
        elif 2000 < num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def mark_dark_l(text):
    return re.sub(r'l([^aeiouæɑɔəɛɪʊ ]*(?: |$))', lambda x: 'ɫ'+x.group(1), text)


def english_to_ipa(text):
    """Convert English text to IPA notation."""
    text = unidecode(text).lower()
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    
    if ENG_TO_IPA_AVAILABLE:
        try:
            phonemes = ipa.convert(text)
        except Exception as e:
            print(f"Warning: eng_to_ipa conversion failed: {e}")
            phonemes = text  # Fallback to plain text
    else:
        print("Warning: eng_to_ipa not available, using fallback")
        phonemes = text  # Fallback to plain text
    
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_to_lazy_ipa(text):
    """Convert English text to lazy IPA notation."""
    text = english_to_ipa(text)
    for regex, replacement in _lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


def english_to_ipa2(text):
    """Convert English text to IPA2 notation."""
    text = english_to_ipa(text)
    text = mark_dark_l(text)
    for regex, replacement in _ipa_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text.replace('...', '…')


def english_to_lazy_ipa2(text):
    """Convert English text to lazy IPA2 notation."""
    text = english_to_ipa(text)
    for regex, replacement in _lazy_ipa2:
        text = re.sub(regex, replacement, text)
    return text


# Funciones de limpieza adicionales para compatibilidad
def english_cleaners(text):
    """Clean English text for TTS processing."""
    text = english_to_ipa2(text)
    return text


def english_cleaners2(text):
    """Alternative English cleaner using lazy IPA2."""
    text = english_to_lazy_ipa2(text)
    return text


# Función para verificar la disponibilidad de las dependencias
def check_dependencies():
    """Check if all dependencies are available."""
    dependencies = {
        'inflect': 'ok',
        'unidecode': 'ok',
        'eng_to_ipa': 'available' if ENG_TO_IPA_AVAILABLE else 'missing (using fallback)'
    }
    return dependencies
