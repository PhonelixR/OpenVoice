# openvoice_app_v2.py - Versión que SIEMPRE muestra el enlace
import os
import torch
import argparse
import gradio as gr
import langid
import sys
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Configurar para que todo se muestre inmediatamente
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

print("="*60)
print("🚀 INICIANDO OPENVOICE CON DEBUG ACTIVADO")
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

# Idiomas soportados - DIFERENTE PARA V1 Y V2
v1_supported_languages = ['zh', 'en']  # V1 solo chino e inglés
# V2 soporta todos los idiomas de sus embeddings

# Estilos V1
v1_styles = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']

# Estilos V2 (usando los embeddings disponibles)
v2_styles = list(v2_ses_embeddings.keys())
if not v2_styles:
    v2_styles = ['en-default', 'en-us', 'zh']  # fallback

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

def predict(version, prompt, style, audio_file_pth, agree):
    print(f"\n🎯 PREDICCIÓN INICIADA:")
    print(f"   Versión: {version}")
    print(f"   Estilo: {style}")
    print(f"   Texto: {prompt[:50]}...")
    print(f"   Audio: {audio_file_pth}")
    print(f"   Aceptado: {agree}")
    
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
        print("📝 Usando V1")
        # V1 tiene restricciones de idioma
        if language_predicted not in v1_supported_languages:
            text_hint += f"[ERROR] Idioma {language_predicted} no soportado en V1. Soporta: {v1_supported_languages}\n"
            gr.Warning(f"Idioma {language_predicted} no soportado en V1")
            return text_hint, None, None

        # Configurar según idioma
        if language_predicted == "zh":
            tts_model = v1_zh_base_tts
            source_se = v1_zh_source_se
            language = 'Chinese'
            if style != 'default':
                text_hint += "[ERROR] Chino solo soporta estilo 'default'\n"
                gr.Warning("Chino solo soporta estilo 'default'")
                return text_hint, None, None
        else:  # inglés
            tts_model = v1_en_base_tts
            language = 'English'
            if style == 'default':
                source_se = v1_en_default_se
            else:
                source_se = v1_en_style_se
        
        # Verificar estilo válido
        if language == 'English' and style not in v1_styles:
            text_hint += f"[ERROR] Estilo {style} no soportado para inglés V1\n"
            gr.Warning(f"Estilo {style} no soportado para inglés V1")
            return text_hint, None, None
        
        converter = v1_tone_converter
    
    # ========== VERSIÓN 2 ==========
    else:  # V2
        print("📝 Usando V2")
        # Para V2, NO restringimos por idioma detectado
        # El estilo de V2 define el acento/idioma
        tts_model = v1_en_base_tts  # Usamos modelo base inglés
        language = 'English'  # Siempre inglés para el modelo base
        
        # Verificar que el estilo existe en los embeddings de V2
        if style not in v2_ses_embeddings:
            text_hint += f"[ERROR] Estilo '{style}' no encontrado en embeddings V2\n"
            gr.Warning(f"Estilo '{style}' no encontrado en embeddings V2")
            return text_hint, None, None
        
        # Informar al usuario sobre el idioma del estilo seleccionado
        style_language = v2_style_to_language.get(style, style)
        text_hint += f"Usando acento/estilo: {style_language}\n"
        
        # Usar embedding de V2 como source
        source_se = v2_ses_embeddings[style]
        converter = v2_tone_converter

    # Validar longitud del texto
    if len(prompt) < 2:
        text_hint += "[ERROR] Por favor escribe un texto más largo\n"
        gr.Warning("Por favor escribe un texto más largo")
        return text_hint, None, None
        
    if len(prompt) > 500:  # Aumentado a 500 para V2
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
    else:  # V2
        # Para V2, usamos estilo 'default' ya que el embedding ya tiene el acento
        tts_model.tts(prompt, src_path, speaker='default', language='English')
    
    print("✅ Audio base generado")

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
    text_hint += f"✅ Audio generado exitosamente usando OpenVoice {version}\n"
    if version == "V2":
        text_hint += f"   Estilo: {style} ({v2_style_to_language.get(style, 'varios acentos')})\n"
    
    return text_hint, save_path, audio_file_pth

print("\n" + "="*60)
print("🎨 CREANDO INTERFAZ GRADIO...")
print("="*60)

# Interfaz - CON descripción mejorada para V2
with gr.Blocks(title="OpenVoice V1 & V2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙 OpenVoice - Versión Dual (V1 & V2)
    ### V1: Clonación con emociones | V2: Clonación con acentos/idiomas
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            version = gr.Radio(
                ["V1", "V2"],
                label="Versión de OpenVoice",
                value="V1",
                info="V1: Emociones (solo en/zh) | V2: Acentos/idiomas (multilingüe)"
            )
            
            # Inicializar con estilos V1, luego se actualizarán dinámicamente
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
                    # Para V2, usar el primer estilo disponible
                    first_style = v2_styles[0] if v2_styles else "en-default"
                    return gr.Dropdown(
                        choices=v2_styles, 
                        value=first_style,
                        label="Estilo/Acento (selecciona idioma)",
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
            
            with gr.Row():
                generate_btn = gr.Button("Generar Audio", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Estado", interactive=False, lines=6)
            output_audio = gr.Audio(label="Audio Generado", autoplay=True)
            reference_used = gr.Audio(label="Audio de Referencia Usado", interactive=False)
    
    # Información sobre versiones
    gr.Markdown("""
    ### 📚 Información de Versiones:
    - V1: Soporta inglés (con emociones) y chino (solo estilo default)
    - V2: Soporta múltiples idiomas/acentos vía selección de estilo
    
    ### 🎯 Estilos V2 disponibles:
    - es - Español
    - fr - Francés  
    - jp - Japonés
    - kr - Coreano
    - zh - Chino
    - Varios acentos ingleses (en-us, en-au, en-br, etc.)
    
    ### 💡 Consejos:
    1. Sube un audio claro de 3-10 segundos
    2. En V2, el texto puede estar en cualquier idioma - el acento lo determina el estilo
    3. Textos más largos funcionan mejor que muy cortos
    """)
    
    generate_btn.click(
        fn=predict,
        inputs=[version, input_text, style, ref_audio, agree],
        outputs=[output_text, output_audio, reference_used]
    )

print("\n" + "="*60)
print("🚀 LANZANDO APLICACIÓN GRADIO...")
print("="*60)
print("📢 SI TODO VA BIEN, VERÁS UN ENLACE ABAJO:")
print("="*60)

# FORZAR que se muestre el enlace - lanzar con todas las opciones de debug
demo.launch(
    debug=True,           # Modo debug
    share=args.share,
    server_name="0.0.0.0",
    server_port=7860,
    quiet=False,          # No silencioso
    show_error=True,      # Mostrar errores
    show_api=True,        # Mostrar API
)

print("\n" + "="*60)
print("📝 LA APLICACIÓN SE HA LANZADO (o ha fallado silenciosamente)")
print("="*60)
