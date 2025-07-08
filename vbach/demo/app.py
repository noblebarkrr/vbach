import gc
import os
import sys
import argparse
from pyngrok import ngrok
import gradio as gr
from datetime import datetime
import torch
import librosa
import tempfile
from datetime import datetime
from assets.vbach_translations import VBACH_TRANSLATIONS as TRANSLATIONS

from vbach.modules.model_manager import (
    download_from_url,
    upload_zip_file,
    upload_separate_files,
    delete_model_name
)

# Глобальная переменная для текущего языка
CURRENT_LANG = "ru"

def set_language(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang


def t(key, **kwargs):
    """Функция для получения перевода с подстановкой значений"""
    translation = TRANSLATIONS[CURRENT_LANG].get(key, key)
    if isinstance(translation, dict):
        # Для вложенных словарей (например, stereo_modes)
        return translation
    return translation.format(**kwargs) if kwargs else translation
# ============== END TRANSLATIONS ==============

def vbach_theme(font):
    theme = gr.themes.Base(primary_hue="rose", spacing_size="sm", font=[gr.themes.GoogleFont(font)])
    return theme

RVC_MODELS_DIR = os.path.join(os.getcwd(), "voice_models")
HUBERT_MODEL_PATH = os.path.join(
    os.getcwd(), "vbach", "models", "embedders", "hubert_base.pt"
)

OUTPUT_FORMAT = ["mp3", "wav", "flac", "aiff", "m4a", "aac", "ogg", "opus"]

def process_audio(input_file=None, input_list=None, template="NAME_MODEL_F0METHOD_PITCH", model_name="", index_rate=0, output_format="wav", stereo_mode="mono", method_pitch="rmvpe+", pitch=0, hop_length=128, filter_radius=3, rms=0.25, protect=0.33, f0_max=1100):
    if not input_file and not input_list:
        raise gr.Error(t("error_no_audio"))
    if not model_name:
        raise gr.Error(t("error_no_model"))
    if input_file is not None and isinstance(input_file, str) and input_list == None:
        if not os.path.exists(input_file):
            gr.Warning(t("warning_file_not_found", file=input_file))

        file_name = os.path.basename(input_file)
        namefile = os.path.splitext(file_name)[0]
        time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = template
        output_dir = tempfile.mkdtemp(prefix="converted_voice_")
        output_name = (
            template
            .replace("DATETIME", time_create_file)
            .replace("NAME", namefile)
            .replace("MODEL", model_name)
            .replace("F0METHOD", method_pitch)
            .replace("PITCH", f"{pitch}")
        )
        output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
        try:
            file_name = os.path.basename(input_file)
            namefile = os.path.splitext(file_name)[0]
            cmd = f"python -m vbach.cli.vbach '{input_file}' '{output_dir}' '{model_name}' --template '{output_name}' --pitch {pitch} --method_pitch {method_pitch} --index_rate {index_rate} --filter_radius {filter_radius} --rms {rms} --protect {protect} --hop_length {hop_length} --f0_min 50 --f0_max {f0_max} --output_format {output_format} --stereo_mode '{stereo_mode}' "
            os.system(cmd)
        finally:
            print(t("success_single"))
            return output_path
                        
    if input_file is None and input_list is not None and isinstance(input_list, list):
        output_dir = tempfile.mkdtemp(prefix="converted_voice_")
        output_paths = []
        progress = gr.Progress()
        for i, file in enumerate(input_list):
            total_steps = len(input_list)
            file_name = os.path.basename(file)
            namefile = os.path.splitext(file_name)[0]
            time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress(
                (i+1, total_steps),
                desc=t("processing", namefile=namefile),
                unit=t("files")
            )
            output_name = (
                template
                .replace("DATETIME", time_create_file)
                .replace("NAME", namefile)
                .replace("MODEL", model_name)
                .replace("F0METHOD", method_pitch)
                .replace("PITCH", f"{pitch}")
            )
            output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
            try:
                cmd = f"python -m vbach.cli.vbach '{file}' '{output_dir}' '{model_name}' --template '{output_name}' --pitch {pitch} --method_pitch {method_pitch} --index_rate {index_rate} --filter_radius {filter_radius} --rms {rms} --protect {protect} --hop_length {hop_length} --f0_min 50 --f0_max {f0_max} --output_format {output_format} --stereo_mode '{stereo_mode}' "
                os.system(cmd)

            finally:
                output_paths.append(output_path)
        print(t("success_batch"))
        return output_paths

def create_demo(lang="ru"):
    set_language(lang)
    
    with gr.TabItem(t("inference")):
        with gr.Row():    
            with gr.Column(scale=3) as input_voice_group:
                with gr.Group() as single_voice_file:
                    input_voice = gr.Audio(type="filepath", interactive=True, show_label=False)
                    batch_upload_btn = gr.Button(t("batch_upload"))
                with gr.Group(visible=False) as batch_voice_file:
                    input_voices = gr.Files(type="filepath", interactive=True, show_label=False)
                    single_upload_btn = gr.Button(t("single_upload"))
                with gr.Group() as single_output_group:
                    converted_voice = gr.Audio(label=t("converted_voice"), type="filepath", interactive=False, scale=6)
                with gr.Group(visible=False) as batch_output_group:
                    converted_voices = gr.Files(label=t("converted_voices"), type="filepath", interactive=False, visible=True, scale=6)

            with gr.Column(scale=6):
                with gr.Column():
                    stereo_mode = gr.Dropdown(
                        label=t("audio_processing"), 
                        choices=list(t("stereo_modes").keys()),
                        value="mono", 
                        interactive=True, 
                        filterable=False
                    )
                    output_format = gr.Dropdown(label=t("output_format"), choices=OUTPUT_FORMAT)
                    template = gr.Text(label=t("name_format"), value="NAME_MODEL_F0METHOD_PITCH", interactive=True)
                    convert_btn = gr.Button(t("convert_single"), variant="primary")
                    convert_batch_btn = gr.Button(t("convert_batch"), variant="primary", visible=False)

        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row(equal_height=True):
                        model_name = gr.Dropdown(label=t("model_name"), interactive=True, filterable=False, scale=6)
                        model_update_btn = gr.Button(t("update_button"), variant="primary", scale=3, size="lg")
                        model_update_btn.click(fn=(lambda : gr.update(choices=[d for d in os.listdir(RVC_MODELS_DIR) if os.path.isdir(os.path.join(RVC_MODELS_DIR, d))])), inputs=None, outputs=model_name)                  
                    
                    method_pitch = gr.Dropdown(label=t("pitch_method"), choices=["mangio-crepe", "rmvpe+", "fcpe"], value="rmvpe+", interactive=True, filterable=False)
                    with gr.Row():
                        pitch = gr.Slider(minimum=-48, maximum=48, step=12, value=0, label=t("pitch"), interactive=True)
                        hop_length = gr.Slider(minimum=2, maximum=512, step=1, value=128, label=t("hop_length"), interactive=True, visible=False)
                        f0_max = gr.Slider(minimum=500, maximum=3500, step=1, value=1100, label=t("f0_max"), interactive=True)
        with gr.Accordion(label=t("advanced_settings"), open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    filter_radius = gr.Slider(minimum=0, maximum=7, step=1, value=3, label=t("filter_radius"), interactive=True)
                    index_rate = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label=t("index_rate"), interactive=True)
                    rms = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25, label=t("rms"), interactive=True)
                    protect = gr.Slider(minimum=0, maximum=0.5, step=0.01, value=0.33, label=t("protect"), interactive=True)

              
    with gr.TabItem(t("model_manager")):
        with gr.TabItem(t("download_url")):
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.HTML(f"<center><h3>{t('download_link')}</h3></center>")
                    model_zip_link = gr.Text(label=t("download_link"))
                    with gr.Group():
                        zip_model_name = gr.Text(
                            label=t("model_name"),
                            info=t("unique_name"),
                        )
                        download_btn = gr.Button(t("download_button"), variant="primary")

                    gr.HTML(
                        f"<h3>{t('supported_sites')}: "
                        "<a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, "
                        "<a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, "
                        "<a href='https://drive.google.com/' target='_blank'>Google Drive</a>, "
                        "<a href='https://disk.yandex.ru/' target='_blank'>Яндекс Диск</a>"
                        "</h3>"
                    )

                    dl_output_message = gr.Text(label=t("output_message"), interactive=False)
                    download_btn.click(
                        download_from_url,
                        inputs=[model_zip_link, zip_model_name],
                        outputs=dl_output_message,
                    )

        with gr.Tab(t("download_zip")):
            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(
                        label=t("zip_file"), file_types=[".zip"], file_count="single"
                    )
                with gr.Column(variant="panel"):
                    gr.HTML(t("upload_steps"))
                    with gr.Group():
                        local_model_name = gr.Text(
                            label=t("model_name"),
                            info=t("unique_name"),
                        )
                        model_upload_button = gr.Button(t("download_button"), variant="primary")

                    local_upload_output_message = gr.Text(label=t("output_message"), interactive=False)
                    model_upload_button.click(
                        upload_zip_file,
                        inputs=[zip_file, local_model_name],
                        outputs=local_upload_output_message,
                    )

        with gr.TabItem(t("download_files")):
            with gr.Group():
                with gr.Row():
                    pth_file = gr.File(
                        label=t("pth_file"), file_types=[".pth"], file_count="single"
                    )
                    index_file = gr.File(
                        label=t("index_file"), file_types=[".index"], file_count="single"
                    )
                with gr.Column(variant="panel"):
                    with gr.Group():
                        separate_model_name = gr.Text(
                            label=t("model_name"),
                            info=t("unique_name"),
                        )
                        separate_upload_button = gr.Button(t("download_button"), variant="primary")

                    separate_upload_output_message = gr.Text(
                        label=t("output_message"), interactive=False
                    )
                    separate_upload_button.click(
                        upload_separate_files,
                        inputs=[pth_file, index_file, separate_model_name],
                        outputs=separate_upload_output_message,
                    )

        with gr.TabItem(t("delete_model")):
          with gr.Column(variant="panel"):
            with gr.Group():
              delete_voicemodel_name = gr.Dropdown(
                label=t("model_name"),
                info=t("delete_info"),
                interactive=True,
                filterable=False
              )
              refresh_delete_btn = gr.Button(t("refresh_button"))
              refresh_delete_btn.click(fn=(lambda : gr.update(choices=[d for d in os.listdir(RVC_MODELS_DIR) if os.path.isdir(os.path.join(RVC_MODELS_DIR, d))])), inputs=None, outputs=delete_voicemodel_name)
              delete_model_output_message = gr.Text(
                label=t("output_message"), interactive=False
              )
              delete_model_btn = gr.Button(t("delete_button"))
              delete_model_btn.click(
                fn=delete_model_name,
                inputs=delete_voicemodel_name,
                outputs=delete_model_output_message
              )


        method_pitch.change(fn=lambda x: gr.update(visible=True if x == "mangio-crepe" else False), inputs=method_pitch, outputs=hop_length)
        batch_upload_btn.click(fn=(lambda : (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))), inputs=None, outputs=[single_voice_file, batch_voice_file, single_output_group, batch_output_group, convert_btn, convert_batch_btn])
        single_upload_btn.click(fn=(lambda : (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))), inputs=None, outputs=[batch_voice_file, single_voice_file, batch_output_group, single_output_group, convert_batch_btn, convert_btn])
        convert_btn.click(fn=process_audio, inputs=[input_voice, gr.State(None), template, model_name, index_rate, output_format, stereo_mode, method_pitch, pitch, hop_length, filter_radius, rms, protect, f0_max], outputs=converted_voice)
        convert_batch_btn.click(fn=process_audio, inputs=[gr.State(None), input_voices, template, model_name, index_rate, output_format, stereo_mode, method_pitch, pitch, hop_length, filter_radius, rms, protect, f0_max], outputs=converted_voices)


def parse_args():
    parser = argparse.ArgumentParser(description="Базовый интерфейс для разделения музыки и вокала")
    
    # Основные параметры запуска
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP-адрес (по умолчанию: 0.0.0.0)")
    parser.add_argument("--server_port", type=int, default=7860, help="Порт (по умолчанию: 7860)")
    parser.add_argument("--share", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="Включить отладку")
    parser.add_argument("--ngrok_token", type=str, help="Аутентификация (формат: username:password)")
    # Настройки безопасности
    parser.add_argument("--auth", type=str, help="Аутентификация (формат: username:password)")
    parser.add_argument("--ssl-keyfile", type=str, help="Путь к SSL ключу")
    parser.add_argument("--ssl-certfile", type=str, help="Путь к SSL сертификату")
    
    # Производительность
    parser.add_argument("--max-file-size", type=str, default="10000MB", help="Максимальный лимит загрузки файлов в интерфейс")
    
    # Шрифт в интерфейсе
    parser.add_argument("--google_font", type=str, default="Tektur", help="Шрифт в интерфейсе")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
            
    with gr.Blocks(title=t("app_title"), theme=vbach_theme(args.google_font)) as demo:
        gr.HTML(f"<h1><center> {t('app_title')} </center></h1>")
        create_demo()
    # Запуск Gradio с парсированными аргументами
    if args.ngrok_token:
        ngrok.set_auth_token(args.ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(args.server_port)
        print(f"Публичная ссылка - {tunnel.public_url}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        auth=args.auth.split(":") if args.auth else None,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        max_file_size=args.max_file_size,
        allowed_paths=["/content"]  # Добавляем разрешенные пути
    )