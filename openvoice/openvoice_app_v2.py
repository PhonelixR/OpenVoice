# openvoice_app_v2.py - Versión con 3 opciones: V1, V2 (Legacy), V2 (MeloTTS) con control de velocidad (Y puerto automático)
import os
import torch
import argparse
import gradio as gr
import langid
import sys
import socket
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Configurar para que todo se muestre inmediatamente
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
parser.add_argument("--port", type=int, default=7860, help="puerto para el servidor")
args = parser.parse_args()

print("="*60)
print("🚀 INICIANDO OPENVOICE CON 3 MOTORES TTS")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Dispositivo: {device}")
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Directorio de salida: {output_dir}")

# ========== CARGAR MODELOS V1 ==========
print("\n📦 CARGANDO MODELOS V1...")
v1_en_ckpt = 'checkpoints/base_speakers/EN'
v1_zh_ckpt = 'checkpoints/base_speakers/ZH'
v1_ckpt_converter = 'checkpoints/converter'

# Modelos base V1
print("  → Cargando modelo base EN...")
v1_en_base_tts = BaseSpeakerTTS(f'{v1_en_ckpt}/config.json', device=device)
v1_en_base_tts.load_ckpt(f'{v1_en_ckpt}/checkpoint.pth')
print("  → Cargando modelo base ZH...")
v1_zh_base_tts = BaseSpeakerTTS(f'{v1_zh_ckpt}/config.json', device=device)
v1_zh_base_tts.load_ckpt(f'{v1_zh_ckpt}/checkpoint.pth')

# Convertidor V1
print("  → Cargando convertidor V1...")
v1_tone_converter = ToneColorConverter(f'{v1_ckpt_converter}/config.json', device=device)
v1_tone_converter.load_ckpt(f'{v1_ckpt_converter}/checkpoint.pth')

# Embeddings V1
print("  → Cargando embeddings V1...")
v1_en_default_se = torch.load(f'{v1_en_ckpt}/en_default_se.pth').to(device)
v1_en_style_se = torch.load(f'{v1_en_ckpt}/en_style_se.pth').to(device)
v1_zh_source_se = torch.load(f'{v1_zh_ckpt}/zh_default_se.pth').to(device)

print("✅ MODELOS V1 CARGADOS")

# ========== CARGAR MODELOS V2 ==========
print("\n📦 CARGANDO MODELOS V2...")
v2_ckpt_converter = 'checkpoints_v2/converter'

# Convertidor V2
print("  → Cargando convertidor V2...")
v2_tone_converter = ToneColorConverter(f'{v2_ckpt_converter}/config.json', device=device)
v2_tone_converter.load_ckpt(f'{v2_ckpt_converter}/checkpoint.pth')

# Cargar TODOS los embeddings de V2
v2_ses_path = 'checkpoints_v2/base_speakers/ses'
v2_ses_embeddings = {}
if os.path.exists(v2_ses_path):
    print(f"  → Buscando embeddings en {v2_ses_path}")
    for file in os.listdir(v2_ses_path):
        if file.endswith('.pth'):
            name = file[:-4]  # quitar .pth
            try:
                v2_ses_embeddings[name] = torch.load(
                    os.path.join(v2_ses_path, file), 
                    map_location=device
                )
                print(f"    ✓ {name}")
            except Exception as e:
                print(f"    ✗ {file}: {e}")
else:
    print(f"  ⚠️ No existe el directorio {v2_ses_path}")

print(f"✅ MODELOS V2 CARGADOS (Embeddings: {len(v2_ses_embeddings)})")

# ========== CARGAR MODELO MELOTTS ==========
print("\n📦 INTENTANDO CARGAR MELOTTS...")
melo_models = {}
melo_speakers_cache = {}

try:
    from melo.api import TTS
    print("  → Importando MeloTTS...")
    
    supported_languages = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR']
    
    for lang in supported_languages:
        try:
            print(f"  → Cargando modelo {lang}...")
            model = TTS(language=lang, device=device)
            melo_models[lang] = model
            
            speaker_ids_dict = model.hps.data.spk2id
            
            melo_speakers_cache[lang] = {
                'model': model,
                'speaker_ids': speaker_ids_dict,
                'available_speakers': list(speaker_ids_dict.keys())
            }
            
            print(f"    ✓ Modelo {lang} cargado ({len(speaker_ids_dict)} speakers)")
                
        except Exception as e:
            print(f"    ✗ Error cargando modelo {lang}: {e}")
    
    print("✅ MeloTTS cargado exitosamente")
    
except ImportError as e:
    print(f"  ⚠️ No se pudo importar MeloTTS: {e}")
    print("  ℹ️ Instala MeloTTS con: pip install git+https://github.com/myshell-ai/MeloTTS.git")
except Exception as e:
    print(f"  ⚠️ Error cargando MeloTTS: {e}")

# Idiomas soportados
v1_supported_languages = ['zh', 'en']
v1_styles = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']

# Estilos V2
v2_styles = list(v2_ses_embeddings.keys())
if not v2_styles:
    v2_styles = ['en-default', 'en-us', 'zh']

# Mapear estilos V2 a idiomas sugeridos
v2_style_to_language = {
    'en-default': 'Inglés (predeterminado)',
    'en-us': 'Inglés (EEUU)',
    'en-au': 'Inglés (Australia)',
    'en-br': 'Inglés (Reino Unido)',
    'en-india': 'Inglés (India)',
    'en-newest': 'Inglés (nuevo)',
    'es': 'Español',
    'fr': 'Francés',
    'jp': 'Japonés',
    'kr': 'Coreano',
    'zh': 'Chino'
}

# Mapeo de estilos V2 a configuraciones de MeloTTS
v2_style_to_melo_config = {
    'es': {'language': 'ES', 'speaker_name': 'ES'},
    'fr': {'language': 'FR', 'speaker_name': 'FR'},
    'zh': {'language': 'ZH', 'speaker_name': 'ZH'},
    'jp': {'language': 'JP', 'speaker_name': 'JP'},
    'kr': {'language': 'KR', 'speaker_name': 'KR'},
    'en-default': {'language': 'EN', 'speaker_name': 'EN'},
    'en-us': {'language': 'EN', 'speaker_name': 'EN-US'},
    'en-br': {'language': 'EN', 'speaker_name': 'EN-BR'},
    'en-au': {'language': 'EN', 'speaker_name': 'EN-AU'},
    'en-india': {'language': 'EN', 'speaker_name': 'EN_INDIA'},
    'en-newest': {'language': 'EN', 'speaker_name': 'EN_NEWEST'},
}

def predict(version, prompt, style, audio_file_pth, agree, speed=1.0):
    print(f"\n🎯 PREDICCIÓN INICIADA:")
    print(f"   Versión: {version}")
    print(f"   Estilo: {style}")
    print(f"   Texto: {prompt[:50]}...")
    print(f"   Audio: {audio_file_pth}")
    print(f"   Aceptado: {agree}")
    print(f"   Velocidad: {speed}")
    
    text_hint = ''
    
    # Verificar términos
    if not agree:
        text_hint += '[ERROR] Por favor acepta los Términos y Condiciones!\n'
        gr.Warning("Por favor acepta los Términos y Condiciones!")
        return text_hint, None, None
    
    # Verificar que se subió un audio
    if not audio_file_pth:
        text_hint += '[ERROR] Por favor sube un archivo de audio de referencia\n'
        gr.Warning("Por favor sube un archivo de audio de referencia")
        return text_hint, None, None

    # Detectar idioma (solo para información)
    language_predicted = langid.classify(prompt)[0].strip()
    print(f"🔤 Idioma detectado: {language_predicted}")
    
    # ========== VERSIÓN 1 ==========
    if version == "V1":
        print("📝 Usando V1 (OpenVoice original)")
        if language_predicted not in v1_supported_languages:
            text_hint += f"[ERROR] Idioma {language_predicted} no soportado en V1. Soporta: {v1_supported_languages}\n"
            gr.Warning(f"Idioma {language_predicted} no soportado en V1")
            return text_hint, None, None

        if language_predicted == "zh":
            tts_model = v1_zh_base_tts
            source_se = v1_zh_source_se
            language = 'Chinese'
            if style != 'default':
                text_hint += "[ERROR] Chino solo soporta estilo 'default'\n"
                gr.Warning("Chino solo soporta estilo 'default'")
                return text_hint, None, None
        else:
            tts_model = v1_en_base_tts
            language = 'English'
            if style == 'default':
                source_se = v1_en_default_se
            else:
                source_se = v1_en_style_se
        
        if language == 'English' and style not in v1_styles:
            text_hint += f"[ERROR] Estilo {style} no soportado para inglés V1\n"
            gr.Warning(f"Estilo {style} no soportado para inglés V1")
            return text_hint, None, None
        
        converter = v1_tone_converter
    
    # ========== VERSIÓN 2 (LEGACY - TTS integrado de V1) ==========
    elif version == "V2 (Legacy TTS)":
        print("📝 Usando V2 con TTS integrado de OpenVoice V1 (Legacy)")
        tts_model = v1_en_base_tts
        language = 'English'
        
        if style not in v2_ses_embeddings:
            text_hint += f"[ERROR] Estilo '{style}' no encontrado en embeddings V2\n"
            gr.Warning(f"Estilo '{style}' no encontrado en embeddings V2")
            return text_hint, None, None
        
        style_language = v2_style_to_language.get(style, style)
        text_hint += f"Usando acento/estilo: {style_language} (con TTS Legacy)\n"
        
        source_se = v2_ses_embeddings[style]
        converter = v2_tone_converter
    
    # ========== VERSIÓN 2 (MELOTTS - RECOMENDADO) ==========
    else:
        print("📝 Usando V2 con MeloTTS (Recomendado)")
        
        if not melo_models:
            text_hint += "[ERROR] MeloTTS no está disponible\n"
            gr.Warning("MeloTTS no está disponible")
            return text_hint, None, None
        
        if style not in v2_ses_embeddings:
            text_hint += f"[ERROR] Estilo '{style}' no encontrado en embeddings V2\n"
            gr.Warning(f"Estilo '{style}' no encontrado en embeddings V2")
            return text_hint, None, None
        
        if style not in v2_style_to_melo_config:
            if style.startswith('en-'):
                melo_lang = 'EN'
                melo_speaker = style.upper().replace('-', '_')
            elif style in ['es', 'fr', 'zh', 'jp', 'kr']:
                melo_lang = style.upper()
                melo_speaker = style.upper()
            else:
                text_hint += f"[ERROR] Estilo '{style}' no compatible con MeloTTS\n"
                gr.Warning(f"Estilo '{style}' no compatible con MeloTTS")
                return text_hint, None, None
            
            v2_style_to_melo_config[style] = {
                'language': melo_lang,
                'speaker_name': melo_speaker
            }
            print(f"⚠️  Mapeo automático creado: {style} -> {melo_lang}/{melo_speaker}")
        
        melo_config = v2_style_to_melo_config[style]
        style_language = v2_style_to_language.get(style, style)
        text_hint += f"Usando acento/estilo: {style_language} (con MeloTTS, velocidad: {speed})\n"
        
        if melo_config['language'] not in melo_models:
            available_langs = list(melo_models.keys())
            text_hint += f"[ERROR] Idioma {melo_config['language']} no disponible\n"
            gr.Warning(f"Idioma {melo_config['language']} no disponible")
            return text_hint, None, None
        
        melo_model = melo_models[melo_config['language']]
        melo_speakers_info = melo_speakers_cache[melo_config['language']]
        speaker_ids_dict = melo_speakers_info['speaker_ids']
        
        target_speaker_name = None
        target_speaker_id = None
        
        possible_names = [
            melo_config['speaker_name'],
            melo_config['speaker_name'].replace('_', '-'),
            melo_config['speaker_name'].replace('-', '_'),
            melo_config['speaker_name'].lower(),
            melo_config['speaker_name'].upper(),
            style.replace('-', '_').upper(),
            style.upper().replace('-', '_'),
        ]
        
        for test_name in possible_names:
            if test_name in speaker_ids_dict:
                target_speaker_name = test_name
                target_speaker_id = speaker_ids_dict[test_name]
                break
        
        if target_speaker_name is None:
            available_speakers = melo_speakers_info['available_speakers']
            text_hint += f"[ERROR] Speaker '{melo_config['speaker_name']}' no encontrado\n"
            gr.Warning(f"Speaker '{melo_config['speaker_name']}' no encontrado")
            return text_hint, None, None
        
        print(f"  → Speaker encontrado: {target_speaker_name} (ID: {target_speaker_id})")
        
        source_se = v2_ses_embeddings[style]
        converter = v2_tone_converter

    # Validar longitud del texto
    if len(prompt) < 2:
        text_hint += "[ERROR] Por favor escribe un texto más largo\n"
        gr.Warning("Por favor escribe un texto más largo")
        return text_hint, None, None
        
    if len(prompt) > 500:
        text_hint += "[ERROR] Texto limitado a 500 caracteres\n"
        gr.Warning("Texto limitado a 500 caracteres")
        return text_hint, None, None

    # Procesar audio de referencia
    print("🎤 Extrayendo características de voz...")
    try:
        target_se, audio_name = se_extractor.get_se(
            audio_file_pth, 
            converter, 
            target_dir='processed', 
            vad=True
        )
        print("✅ Características extraídas")
    except Exception as e:
        text_hint += f"[ERROR] Error extrayendo características de voz: {str(e)}\n"
        gr.Warning("Error extrayendo características de voz")
        print(f"❌ Error: {e}")
        return text_hint, None, None

    # Generar audio base
    src_path = f'{output_dir}/tmp.wav'
    print(f"🔊 Generando audio base en: {src_path}")
    
    if version == "V1":
        tts_model.tts(prompt, src_path, speaker=style, language=language)
        print("✅ Audio base generado con TTS V1")
    
    elif version == "V2 (Legacy TTS)":
        tts_model.tts(prompt, src_path, speaker='default', language='English')
        print("✅ Audio base generado con TTS Legacy (V1)")
    
    else:
        try:
            print(f"  → Generando con MeloTTS: {melo_config['language']}, speaker: {target_speaker_name} (ID: {target_speaker_id}), velocidad: {speed}")
            
            # ¡ESTA ES LA LLAVE! Según la API que compartiste
            melo_model.tts_to_file(
                text=prompt,
                speaker_id=target_speaker_id,
                output_path=src_path,
                speed=float(speed),  # Convertir a float y usar el valor del slider
                quiet=True
            )
            print("✅ Audio base generado con MeloTTS")
            
        except Exception as e:
            text_hint += f"[ERROR] Error generando audio con MeloTTS: {str(e)}\n"
            gr.Warning("Error generando audio con MeloTTS")
            print(f"❌ Error MeloTTS: {e}")
            
            try:
                print("  → Intentando método posicional...")
                melo_model.tts_to_file(prompt, target_speaker_id, src_path, speed=float(speed), quiet=True)
                print("✅ Audio base generado con MeloTTS (método posicional)")
            except Exception as e2:
                text_hint += f"[ERROR] Método alternativo también falló: {str(e2)}\n"
                print(f"❌ Error alternativo: {e2}")
                return text_hint, None, None

    # Convertir voz
    save_path = f'{output_dir}/output.wav'
    encode_message = "@MyShell"
    
    print("🔄 Convirtiendo voz...")
    converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message
    )
    
    print(f"✅ Audio final guardado en: {save_path}")
    text_hint += f"✅ Audio generado exitosamente usando {version}\n"
    if version != "V1":
        text_hint += f"   Estilo: {style} ({v2_style_to_language.get(style, 'varios acentos')})\n"
        if "MeloTTS" in version:
            text_hint += f"   Motor TTS: MeloTTS ({melo_config['language']}/{target_speaker_name})\n"
            text_hint += f"   Velocidad: {speed}\n"
        elif "Legacy" in version:
            text_hint += "   Motor TTS: OpenVoice V1 (legacy)\n"
    
    return text_hint, save_path, audio_file_pth

print("\n" + "="*60)
print("🎨 CREANDO INTERFAZ GRADIO...")
print("="*60)

with gr.Blocks(title="OpenVoice - 3 Motores TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙 OpenVoice - Tres Opciones de TTS
    ### V1: Original | V2 (Legacy): TTS de V1 | V2 (MeloTTS): Recomendado para V2
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            version = gr.Radio(
                ["V1", "V2 (Legacy TTS)", "V2 (MeloTTS)"],
                label="Versión & Motor TTS",
                value="V1",
                info="V1: Original. V2 (Legacy): Usa TTS de V1. V2 (MeloTTS): Recomendado para V2"
            )
            
            style = gr.Dropdown(
                label="Estilo",
                value="default",
                choices=v1_styles,
                allow_custom_value=False
            )
            
            def update_styles(version):
                if version == "V1":
                    return gr.Dropdown(
                        choices=v1_styles, 
                        value="default",
                        label="Estilo (emociones para inglés/chino)",
                        allow_custom_value=False
                    )
                else:
                    first_style = v2_styles[0] if v2_styles else "en-default"
                    return gr.Dropdown(
                        choices=v2_styles, 
                        value=first_style,
                        label="Estilo/Acento (selecciona idioma para V2)",
                        allow_custom_value=False
                    )
            
            version.change(update_styles, inputs=version, outputs=style)
            
            input_text = gr.Textbox(
                label="Texto a generar (máx 500 caracteres)",
                value="Hola, esta es mi voz clonada usando OpenVoice",
                lines=3
            )
            
            ref_audio = gr.Audio(
                label="Audio de referencia - Sube o graba una muestra corta (3-10 segundos)",
                type="filepath"
            )
            
            agree = gr.Checkbox(
                label="Acepto los términos (licencia cc-by-nc-4.0)",
                value=False
            )
            
            # Barra de velocidad (solo afecta a MeloTTS)
            speed_slider = gr.Slider(
                minimum=0.1,
                maximum=4.0,
                value=1.0,
                step=0.1,
                label="Velocidad de habla (solo para V2 MeloTTS)",
                info="Más bajo = más lento, más alto = más rápido"
            )
            
            with gr.Row():
                generate_btn = gr.Button("Generar Audio", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Estado", interactive=False, lines=6)
            output_audio = gr.Audio(label="Audio Generado", autoplay=True)
            reference_used = gr.Audio(label="Audio de Referencia Usado", interactive=False)
    
    gr.Markdown("""
    ### 📚 Información de Versiones:
    
    **V1 (OpenVoice Original)**:
    - ✅ Soporta inglés (con emociones) y chino (solo estilo default)
    - ❌ Limitado a 2 idiomas
    
    **V2 (Legacy TTS)**:
    - ⚠️ Usa el TTS integrado de OpenVoice V1
    - ✅ Soporta múltiples idiomas/acentos vía selección de estilo
    - ⚠️ Calidad de TTS limitada
    
    **V2 (MeloTTS - Recomendado)**:
    - ✅ Usa MeloTTS como motor TTS (mejor calidad)
    - ✅ Soporta múltiples idiomas/acentos
    - ✅ Control de velocidad (0.1x - 4.0x)
    - ✅ Recomendado para OpenVoice V2
    
    ### 🎯 Estilos V2 disponibles:
    - es - Español
    - fr - Francés  
    - jp - Japonés
    - kr - Coreano
    - zh - Chino
    - Varios acentos ingleses (en-us, en-au, en-br, etc.)
    
    ### 💡 Instalación de MeloTTS (requerido para V2 MeloTTS):
    ```bash
    pip install git+https://github.com/myshell-ai/MeloTTS.git
    python -m unidic download
    ```
    
    ### 🎛️ Control de velocidad:
    - Solo funciona con **V2 (MeloTTS)**
    - Rango: 0.1 (muy lento) a 4.0 (muy rápido)
    - Valor por defecto: 1.0 (velocidad normal)
    - Para V1 y V2 Legacy, se ignora este ajuste
    """)
    
    generate_btn.click(
        fn=predict,
        inputs=[version, input_text, style, ref_audio, agree, speed_slider],
        outputs=[output_text, output_audio, reference_used]
    )

print("\n" + "="*60)
print("🚀 LANZANDO APLICACIÓN GRADIO...")
print("="*60)
print("📢 SI TODO VA BIEN, VERÁS UN ENLACE ABAJO:")
print("="*60)

def find_free_port(start_port=7860, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port

target_port = args.port
attempts = 0
max_attempts = 5

while attempts < max_attempts:
    try:
        print(f"🔧 Intentando con puerto: {target_port}")
        
        demo.launch(
            debug=True,
            share=args.share,
            server_name="0.0.0.0",
            server_port=target_port,
            quiet=False,
            show_error=True,
            show_api=True,
        )
        break
        
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"⚠️  Puerto {target_port} ocupado, buscando puerto libre...")
            target_port = find_free_port(target_port + 1)
            attempts += 1
        else:
            print(f"❌ Error inesperado: {e}")
            raise
    except Exception as e:
        print(f"❌ Error al lanzar la aplicación: {e}")
        break

if attempts >= max_attempts:
    print("❌ No se pudo encontrar un puerto libre después de varios intentos")
    print("💡 Intenta detener otras instancias de Gradio o especifica un puerto diferente con --port")

print("\n" + "="*60)
print("📝 APLICACIÓN FINALIZADA")
print("="*60)
