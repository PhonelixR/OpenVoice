'''
Defines the set of symbols used in text input to the model.
'''

# japanese_cleaners
# _pad        = '_'
# _punctuation = ',.!?-'
# _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ '


'''# japanese_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
'''


'''# korean_cleaners
_pad        = '_'
_punctuation = ',.!?…~'
_letters = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ '
'''

'''# chinese_cleaners
_pad        = '_'
_punctuation = '，。！？—…'
_letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
'''

# # zh_ja_mixture_cleaners
# _pad        = '_'
# _punctuation = ',.!?-~…'
# _letters = 'AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑ '


'''# sanskrit_cleaners
_pad        = '_'
_punctuation = '।'
_letters = 'ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऽािीुूृॄेैोौ्ॠॢ '
'''

'''# cjks_cleaners
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzʃʧʥʦɯɹəɥçɸɾβŋɦː⁼ʰ`^#*=→↓↑ '
'''

'''# thai_cleaners
_pad        = '_'
_punctuation = '.!? '
_letters = 'กขฃคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์'
'''

# cjke_cleaners2 - activado para chino, japonés, coreano e inglés
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '


'''# shanghainese_cleaners
_pad        = '_'
_punctuation = ',.!?…'
_letters = 'abdfghiklmnopstuvyzøŋȵɑɔɕəɤɦɪɿʑʔʰ̩̃ᴀᴇ15678 '
'''

'''# chinese_dialect_cleaners
_pad        = '_'
_punctuation = ',.!?~…─'
_letters = '#Nabdefghijklmnoprstuvwxyzæçøŋœȵɐɑɒɓɔɕɗɘəɚɛɜɣɤɦɪɭɯɵɷɸɻɾɿʂʅʊʋʌʏʑʔʦʮʰʷˀː˥˦˧˨˩̥̩̃̚ᴀᴇ↑↓∅ⱼ '
'''

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters)

# Special symbol ids
SPACE_ID = symbols.index(" ")

num_ja_tones = 1
num_kr_tones = 1
num_zh_tones = 6
num_en_tones = 4

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    'KR': num_zh_tones + num_ja_tones + num_en_tones,
}

# Validación de símbolos
def validate_symbols():
    """Validate that all symbols are single characters."""
    issues = []
    for i, symbol in enumerate(symbols):
        if len(symbol) != 1:
            issues.append(f"Symbol at index {i} is not a single character: '{symbol}'")
    
    # Verificar duplicados
    duplicates = {sym for sym in symbols if symbols.count(sym) > 1}
    if duplicates:
        issues.append(f"Duplicate symbols found: {duplicates}")
    
    return issues

# Información de exportación
def get_symbol_info():
    """Get information about the symbols."""
    return {
        "total_symbols": len(symbols),
        "pad_symbol": _pad,
        "punctuation_count": len(_punctuation),
        "letter_count": len(_letters),
        "space_index": SPACE_ID,
        "language_tones": {
            "ZH": num_zh_tones,
            "JP": num_ja_tones,
            "EN": num_en_tones,
            "KR": num_kr_tones
        }
    }
