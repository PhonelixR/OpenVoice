# openvoice_app.py - Versión SIN archivo por defecto
import os
import torch
import argparse
import gradio as gr
import langid
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== CARGAR MODELOS V1 ==========
print("Cargando modelos V1...")
v1_en_ckpt = 'checkpoints/base_speakers/EN'
v1_zh_ckpt = 'checkpoints/base_speakers/ZH'
v1_ckpt_converter = 'checkpoints/converter'

# Modelos base V1
v1_en_base_tts = BaseSpeakerTTS(f'{v1_en_ckpt}/config.json', device=device)
v1_en_base_tts.load_ckpt(f'{v1_en_ckpt}/checkpoint.pth')
v1_zh_base_tts = BaseSpeakerTTS(f'{v1_zh_ckpt}/config.json', device=device)
v1_zh_base_tts.load_ckpt(f'{v1_zh_ckpt}/checkpoint.pth')

# Convertidor V1
v1_tone_converter = ToneColorConverter(f'{v1_ckpt_converter}/config.json', device=device)
v1_tone_converter.load_ckpt(f'{v1_ckpt_converter}/checkpoint.pth')

# Embeddings V1
v1_en_default_se = torch.load(f'{v1_en_ckpt}/en_default_se.pth').to(device)
v1_en_style_se = torch.load(f'{v1_en_ckpt}/en_style_se.pth').to(device)
v1_zh_source_se = torch.load(f'{v1_zh_ckpt}/zh_default_se.pth').to(device)

# ========== CARGAR MODELOS V2 ==========
print("Cargando modelos V2...")
v2_ckpt_converter = 'checkpoints_v2/converter'

# Convertidor V2
v2_tone_converter = ToneColorConverter(f'{v2_ckpt_converter}/config.json', device=device)
v2_tone_converter.load_ckpt(f'{v2_ckpt_converter}/checkpoint.pth')

# Cargar TODOS los embeddings de V2
v2_ses_path = 'checkpoints_v2/base_speakers/ses'
v2_ses_embeddings = {}
if os.path.exists(v2_ses_path):
    for file in os.listdir(v2_ses_path):
        if file.endswith('.pth'):
            name = file[:-4]  # quitar .pth
            try:
                v2_ses_embeddings[name] = torch.load(
                    os.path.join(v2_ses_path, file), 
                    map_location=device
                )
                print(f"  ✓ Embedding V2 cargado: {name}")
            except Exception as e:
                print(f"  ✗ Error cargando {file}: {e}")

print(f"Modelos V1 cargados: ✓")
print(f"Modelos V2 cargados: ✓ (Embeddings: {len(v2_ses_embeddings)})")

# Idiomas soportados
supported_languages = ['zh', 'en']

# Estilos V1
v1_styles = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']

# Estilos V2 (usando los embeddings disponibles)
v2_styles = list(v2_ses_embeddings.keys())
if not v2_styles:
    v2_styles = ['en-default', 'en-us', 'zh']  # fallback

def predict(version, prompt, style, audio_file_pth, agree):
    text_hint = ''
    
    # Verificar términos
    if not agree:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return text_hint, None, None
    
    # Verificar que se subió un audio
    if not audio_file_pth:
        text_hint += '[ERROR] Please upload a reference audio file\n'
        gr.Warning("Please upload a reference audio file")
        return text_hint, None, None

    # Detectar idioma
    language_predicted = langid.classify(prompt)[0].strip()
    print(f"Detected language: {language_predicted}")
    
    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] Language {language_predicted} not supported. Supported: {supported_languages}\n"
        gr.Warning(f"Language {language_predicted} not supported")
        return text_hint, None, None

    # Validar longitud
    if len(prompt) < 2:
        text_hint += "[ERROR] Please give a longer prompt text\n"
        gr.Warning("Please give a longer prompt text")
        return text_hint, None, None
        
    if len(prompt) > 200:
        text_hint += "[ERROR] Text limited to 200 characters for this demo\n"
        gr.Warning("Text limited to 200 characters")
        return text_hint, None, None

    # ========== VERSIÓN 1 ==========
    if version == "V1":
        # Configurar según idioma
        if language_predicted == "zh":
            tts_model = v1_zh_base_tts
            source_se = v1_zh_source_se
            language = 'Chinese'
            if style != 'default':
                text_hint += "[ERROR] Chinese only supports 'default' style\n"
                gr.Warning("Chinese only supports 'default' style")
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
            text_hint += f"[ERROR] Style {style} not supported for English V1\n"
            gr.Warning(f"Style {style} not supported for English V1")
            return text_hint, None, None
        
        converter = v1_tone_converter
    
    # ========== VERSIÓN 2 ==========
    else:  # V2
        # Para V2, siempre usamos el modelo base en inglés de V1
        # ya que V2 no tiene modelo base propio
        tts_model = v1_en_base_tts
        language = 'English'
        
        # Verificar que el estilo existe en los embeddings de V2
        if style not in v2_ses_embeddings:
            text_hint += f"[ERROR] Style {style} not found in V2 embeddings\n"
            gr.Warning(f"Style {style} not found in V2 embeddings")
            return text_hint, None, None
        
        # Usar embedding de V2 como source
        source_se = v2_ses_embeddings[style]
        converter = v2_tone_converter
    
    # Procesar audio de referencia
    try:
        target_se, audio_name = se_extractor.get_se(
            audio_file_pth, 
            converter, 
            target_dir='processed', 
            vad=True
        )
    except Exception as e:
        text_hint += f"[ERROR] Error extracting voice features: {str(e)}\n"
        gr.Warning("Error extracting voice features")
        return text_hint, None, None

    # Generar audio base
    src_path = f'{output_dir}/tmp.wav'
    
    if version == "V1":
        tts_model.tts(prompt, src_path, speaker=style, language=language)
    else:  # V2
        # Para V2, usamos estilo 'default' ya que el embedding ya tiene el acento
        tts_model.tts(prompt, src_path, speaker='default', language=language)

    # Convertir voz
    save_path = f'{output_dir}/output.wav'
    encode_message = "@MyShell"
    
    converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message
    )

    text_hint += f"✅ Audio generated successfully using OpenVoice {version}\n"
    return text_hint, save_path, audio_file_pth

# Interfaz - SIN archivo por defecto
with gr.Blocks(title="OpenVoice V1 & V2") as demo:
    gr.Markdown("""
    # 🎙 OpenVoice - Versión Dual (V1 & V2)
    Clone voices instantly using either OpenVoice V1 or V2.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            version = gr.Radio(
                ["V1", "V2"],
                label="OpenVoice Version",
                value="V1",
                info="V1: Emotion styles | V2: Accent/Language styles"
            )
            
            # Inicializar con estilos V1, luego se actualizarán dinámicamente
            style = gr.Dropdown(
                label="Style",
                value="default",
                choices=v1_styles,
                allow_custom_value=False
            )
            
            def update_styles(version):
                if version == "V1":
                    return gr.Dropdown(
                        choices=v1_styles, 
                        value="default",
                        label="Style (Emotions)",
                        allow_custom_value=False
                    )
                else:
                    # Para V2, usar el primer estilo disponible
                    first_style = v2_styles[0] if v2_styles else "en-default"
                    return gr.Dropdown(
                        choices=v2_styles, 
                        value=first_style,
                        label="Style (Accents/Languages)",
                        allow_custom_value=False
                    )
            
            version.change(update_styles, inputs=version, outputs=style)
            
            input_text = gr.Textbox(
                label="Text Prompt (max 200 characters)",
                value="Hello, this is my cloned voice using OpenVoice",
                lines=3
            )
            
            ref_audio = gr.Audio(
                label="Reference Audio - Upload or record a short sample (3-10 seconds)",
                type="filepath"
                # SIN value por defecto
            )
            
            agree = gr.Checkbox(
                label="I agree to the terms (cc-by-nc-4.0 license)",
                value=False
            )
            
            with gr.Row():
                generate_btn = gr.Button("Generate Audio", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Status", interactive=False, lines=4)
            output_audio = gr.Audio(label="Generated Audio", autoplay=True)
            reference_used = gr.Audio(label="Reference Audio Used", interactive=False)
    
    generate_btn.click(
        fn=predict,
        inputs=[version, input_text, style, ref_audio, agree],
        outputs=[output_text, output_audio, reference_used]
    )

# Lanzar
if name == "main":
    demo.launch(
        debug=False,
        share=args.share,
        server_name="0.0.0.0" if args.share else None,
        server_port=7860,
        theme=gr.themes.Soft()
    )
