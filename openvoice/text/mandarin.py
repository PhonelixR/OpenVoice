import os
import sys
import re
from pypinyin import lazy_pinyin, BOPOMOFO
import jieba
import cn2an
import logging

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile(re.escape(x[0]), re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'),
    ('b', 'ㄅㄧˋ'),
    ('c', 'ㄙㄧˉ'),
    ('d', 'ㄉㄧˋ'),
    ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'),
    ('g', 'ㄐㄧˋ'),
    ('h', 'ㄝˇㄑㄩˋ'),
    ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'),
    ('k', 'ㄎㄟˋ'),
    ('l', 'ㄝˊㄛˋ'),
    ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'),
    ('o', 'ㄡˉ'),
    ('p', 'ㄆㄧˉ'),
    ('q', 'ㄎㄧㄡˉ'),
    ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'),
    ('t', 'ㄊㄧˋ'),
    ('u', 'ㄧㄡˉ'),
    ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
    ('x', 'ㄝˉㄎㄨˋㄙˋ'),
    ('y', 'ㄨㄞˋ'),
    ('z', 'ㄗㄟˋ')
]]

# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'),
    ('ㄐ', 'ʧ⁼'),
    ('ㄑ', 'ʧʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ʦ`⁼'),
    ('ㄔ', 'ʦ`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ʦ⁼'),
    ('ㄘ', 'ʦʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'e'),
    ('ㄞ', 'ai'),
    ('ㄟ', 'ei'),
    ('ㄠ', 'au'),
    ('ㄡ', 'ou'),
    ('ㄧㄢ', 'yeNN'),
    ('ㄢ', 'aNN'),
    ('ㄧㄣ', 'iNN'),
    ('ㄣ', 'əNN'),
    ('ㄤ', 'aNg'),
    ('ㄧㄥ', 'iNg'),
    ('ㄨㄥ', 'uNg'),
    ('ㄩㄥ', 'yuNg'),
    ('ㄥ', 'əNg'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

# List of (romaji, ipa) pairs:
_romaji_to_ipa = [(re.compile(re.escape(x[0]), re.IGNORECASE), x[1]) for x in [
    ('ʃy', 'ʃ'),
    ('ʧʰy', 'ʧʰ'),
    ('ʧ⁼y', 'ʧ⁼'),
    ('NN', 'n'),
    ('Ng', 'ŋ'),
    ('y', 'j'),
    ('h', 'x')
]]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'x'),
    ('ㄐ', 'tʃ⁼'),
    ('ㄑ', 'tʃʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ts`⁼'),
    ('ㄔ', 'ts`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ts⁼'),
    ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'eɪ'),
    ('ㄠ', 'ɑʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'ɥæn'),
    ('ㄢ', 'an'),
    ('ㄧㄣ', 'in'),
    ('ㄩㄣ', 'ɥn'),
    ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'),
    ('ㄧㄥ', 'iŋ'),
    ('ㄨㄥ', 'ʊŋ'),
    ('ㄩㄥ', 'jʊŋ'),
    ('ㄥ', 'əŋ'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

# List of (bopomofo, ipa2) pairs:
_bopomofo_to_ipa2 = [(re.compile(re.escape(x[0])), x[1]) for x in [
    ('ㄅㄛ', 'pwo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'),
    ('ㄐ', 'tɕ'),
    ('ㄑ', 'tɕʰ'),
    ('ㄒ', 'ɕ'),
    ('ㄓ', 'tʂ'),
    ('ㄔ', 'tʂʰ'),
    ('ㄕ', 'ʂ'),
    ('ㄖ', 'ɻ'),
    ('ㄗ', 'ts'),
    ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ɤ'),
    ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'eɪ'),
    ('ㄠ', 'ɑʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'yæn'),
    ('ㄢ', 'an'),
    ('ㄧㄣ', 'in'),
    ('ㄩㄣ', 'yn'),
    ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'),
    ('ㄧㄥ', 'iŋ'),
    ('ㄨㄥ', 'ʊŋ'),
    ('ㄩㄥ', 'jʊŋ'),
    ('ㄥ', 'ɤŋ'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'y'),
    ('ˉ', '˥'),
    ('ˊ', '˧˥'),
    ('ˇ', '˨˩˦'),
    ('ˋ', '˥˩'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]


def number_to_chinese(text):
    """Convert numbers in text to Chinese characters."""
    try:
        numbers = re.findall(r'\d+(?:\.?\d+)?', text)
        for number in numbers:
            text = text.replace(number, cn2an.an2cn(number), 1)
        return text
    except Exception as e:
        logger.warning(f"Error in number_to_chinese: {e}")
        return text


def chinese_to_bopomofo(text):
    """Convert Chinese text to Bopomofo (Zhuyin) notation."""
    try:
        text = text.replace('、', '，').replace('；', '，').replace('：', '，')
        words = jieba.lcut(text, cut_all=False)
        text = ''
        for word in words:
            if not re.search('[\u4e00-\u9fff]', word):
                text += word
                continue
            bopomofos = lazy_pinyin(word, BOPOMOFO)
            for i in range(len(bopomofos)):
                bopomofos[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopomofos[i])
            if text != '':
                text += ' '
            text += ''.join(bopomofos)
        return text
    except Exception as e:
        logger.warning(f"Error in chinese_to_bopomofo: {e}")
        return text


def latin_to_bopomofo(text):
    """Convert Latin characters to Bopomofo."""
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_romaji(text):
    """Convert Bopomofo to Romaji."""
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_ipa(text):
    """Convert Bopomofo to IPA."""
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_ipa2(text):
    """Convert Bopomofo to IPA2."""
    for regex, replacement in _bopomofo_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text


def chinese_to_romaji(text):
    """Convert Chinese text to Romaji."""
    try:
        text = number_to_chinese(text)
        text = chinese_to_bopomofo(text)
        text = latin_to_bopomofo(text)
        text = bopomofo_to_romaji(text)
        text = re.sub('i([aoe])', r'y\1', text)
        text = re.sub('u([aoəe])', r'w\1', text)
        text = re.sub(r'([ʦsɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                      r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
        text = re.sub(r'([ʦs][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
        return text
    except Exception as e:
        logger.warning(f"Error in chinese_to_romaji: {e}")
        return text


def chinese_to_lazy_ipa(text):
    """Convert Chinese text to lazy IPA."""
    try:
        text = chinese_to_romaji(text)
        for regex, replacement in _romaji_to_ipa:
            text = re.sub(regex, replacement, text)
        return text
    except Exception as e:
        logger.warning(f"Error in chinese_to_lazy_ipa: {e}")
        return text


def chinese_to_ipa(text):
    """Convert Chinese text to IPA."""
    try:
        text = number_to_chinese(text)
        text = chinese_to_bopomofo(text)
        text = latin_to_bopomofo(text)
        text = bopomofo_to_ipa(text)
        text = re.sub(r'i([aoe])', r'j\1', text)
        text = re.sub(r'u([aoəe])', r'w\1', text)
        text = re.sub(r'([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                      r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
        text = re.sub(r'([s][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
        return text
    except Exception as e:
        logger.warning(f"Error in chinese_to_ipa: {e}")
        return text


def chinese_to_ipa2(text):
    """Convert Chinese text to IPA2."""
    try:
        text = number_to_chinese(text)
        text = chinese_to_bopomofo(text)
        text = latin_to_bopomofo(text)
        text = bopomofo_to_ipa2(text)
        text = re.sub(r'i([aoe])', r'j\1', text)
        text = re.sub(r'u([aoəe])', r'w\1', text)
        text = re.sub(r'([ʂɹ]ʰ?)([˩˨˧˦˥ ]+|$)', r'\1ʅ\2', text)
        text = re.sub(r'(sʰ?)([˩˨˧˦˥ ]+|$)', r'\1ɿ\2', text)
        return text
    except Exception as e:
        logger.warning(f"Error in chinese_to_ipa2: {e}")
        return text


# Funciones de limpieza adicionales
def chinese_cleaners(text):
    """Clean Chinese text for TTS processing."""
    return chinese_to_ipa(text)


def chinese_cleaners2(text):
    """Clean Chinese text using IPA2 for TTS processing."""
    return chinese_to_ipa2(text)


# Verificar dependencias
def check_dependencies():
    """Check if all dependencies for Chinese processing are available."""
    dependencies = {
        'pypinyin': 'ok',
        'jieba': 'ok',
        'cn2an': 'ok'
    }
    return dependencies
