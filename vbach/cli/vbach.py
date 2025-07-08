import gc
import os
import datetime
import gradio as gr
import torch
import librosa
import tempfile
from datetime import datetime
import argparse
from vbach.infer.infer import Config, load_hubert, get_vc, rvc_infer

# Константы

RVC_MODELS_DIR = os.path.join(os.getcwd(), "voice_models")
HUBERT_MODEL_PATH = os.path.join(
    os.getcwd(), "vbach", "models", "embedders", "hubert_base.pt"
)
OUTPUT_FORMAT = ["mp3", "wav", "flac", "aiff", "m4a", "aac", "ogg", "opus"]

audio_extensions = {".mp3", ".wav", ".flac", ".aiff", ".m4a", ".aac", ".ogg", ".opus"}
    

# Важные функции

def load_rvc_model(voice_model):
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    model_files = os.listdir(model_dir)
    rvc_model_path = next(
        (os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None
    )
    rvc_index_path = next(
        (os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None
    )

    if not rvc_model_path:
        raise ValueError(
            f"\033[91mМодели {voice_model} не существует. "
            "Возможно, вы неправильно ввели имя.\033[0m"
        )

    return rvc_model_path, rvc_index_path

def voice_conversion(
    voice_model,
    vocals_path,
    output_path,
    pitch,
    f0_method,
    index_rate,
    filter_radius,
    volume_envelope,
    protect,
    hop_length,
    f0_min,
    f0_max,
    format_output,
    output_bitrate,
    stereo_mode
):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)

    config = Config()
    hubert_model = load_hubert(config.device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(
        config.device, config.is_half, config, rvc_model_path
    )

    rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
        output_path,
        pitch,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        volume_envelope,
        protect,
        hop_length,
        vc,
        hubert_model,
        f0_min,
        f0_max,
        format_output,
        output_bitrate,
        stereo_mode
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

def cli_conversion(input_audios, template="NAME_MODEL_F0METHOD_PITCH", output_dir="output", model_name="", index_rate=0, output_format="wav", stereo_mode="mono", method_pitch="rmvpe+", pitch=0, hop_length=128, filter_radius=3, rms=0.25, protect=0.33, f0_min=50, f0_max=1100):
    if not input_audios:
        raise ValueError(
            "Не удалось найти аудиофайл(ы). "
            "Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not model_name:
        raise ValueError("Выберите модель голоса для преобразования.")
    if not os.path.exists(input_audios):
        raise ValueError(f"Файл {input_audios} не найден.")

    if not os.path.exists(input_audios):
        raise FileNotFoundError(f"Ошибка: '{input_audios}' не существует.")

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_audios):
        # Проверяем, является ли файл аудио
        ext = os.path.splitext(input_audios)[1].lower()
        if ext not in audio_extensions:
            raise ValueError(f"Ошибка: '{input_audios}' не является аудиофайлом (допустимые расширения: {audio_extensions}).")
        print(f"Найден аудиофайл: {input_audios}")

        try:
            file_name = os.path.basename(input_audios)
            namefile = os.path.splitext(file_name)[0]
            time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = template
            output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
            voice_conversion(model_name, input_audios, output_path, pitch, method_pitch, index_rate, filter_radius, rms, protect, hop_length, f0_min, f0_max, output_format, "320k", stereo_mode)
        finally:
            print("Вокал успешно преобразован")  
    
    elif os.path.isdir(input_audios):
        # Ищем аудиофайлы в папке
        audio_files = []
        for file in os.listdir(input_audios):
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                audio_files.append(os.path.join(input_audios, file))

        if not audio_files:
            raise FileNotFoundError(f"Ошибка: в папке '{input_audios}' нет аудиофайлов (допустимые расширения: {audio_extensions}).")

        print(f"Найдены аудиофайлы: {audio_files}")
    
        try:
            output_paths = []
            for file in audio_files:
                file_name = os.path.basename(file)
                namefile = os.path.splitext(file_name)[0]
                time_create_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = (
                    template
                    .replace("DATETIME", time_create_file)
                    .replace("NAME", namefile)
                    .replace("MODEL", model_name)
                    .replace("F0METHOD", method_pitch)
                    .replace("PITCH", f"{pitch}")
                )
                output_path = os.path.join(output_dir, f"{output_name}.{output_format}")
                voice_conversion(model_name, file, output_path, pitch, method_pitch, index_rate, filter_radius, rms, protect, hop_length, 50, 1100, output_format, "320k", stereo_mode)
                output_paths.append(output_path)
        finally:
            print("Вокалы успешно преобразованы")     
    else:
        raise ValueError(f"Ошибка: '{input_audios}' не является ни файлом, ни папкой.")

def setup_args():
    parser = argparse.ArgumentParser(description='Vbach CLI')
    
    # Обязательные аргументы
    parser.add_argument(
        'input_audios',
        type=str,
        help='Путь к аудиофайлу или папке с аудиофайлами для обработки'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Папка для сохранения результатов конвертации'
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Название голосовой модели RVC для преобразования'
    )
    
    # Необязательные аргументы с значениями по умолчанию
    parser.add_argument(
        '--template',
        type=str,
        default="NAME_MODEL_F0METHOD_PITCH",
        help='Шаблон имени выходного файла (доступные замены: DATETIME, NAME, MODEL, F0METHOD, PITCH)'
    )
    parser.add_argument(
        '--index_rate',
        type=float,
        default=0,
        help='Интенсивность использования индексного файла (от 0.0 до 1.0)',
        metavar='[0.0-1.0]'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        default="wav",
        choices=OUTPUT_FORMAT,
        help='Формат выходного аудиофайла'
    )
    parser.add_argument(
        '--stereo_mode',
        type=str,
        default="mono",
        choices=["mono", "left/right", "sim/dif"],
        help='Режим каналов: моно или стерео'
    )
    parser.add_argument(
        '--method_pitch',
        type=str,
        default="rmvpe+",
        help='Метод извлечения pitch (тона)'
    )
    parser.add_argument(
        '--pitch',
        type=int,
        default=0,
        help='Корректировка тона в полутонах'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=128,
        help='Длина hop (в семплах) для обработки'
    )
    parser.add_argument(
        '--filter_radius',
        type=int,
        default=3,
        help='Радиус фильтра для сглаживания'
    )
    parser.add_argument(
        '--rms',
        type=float,
        default=0.25,
        help='Масштабирование огибающей громкости (RMS)'
    )
    parser.add_argument(
        '--protect',
        type=float,
        default=0.33,
        help='Защита для глухих согласных звуков'
    )
    parser.add_argument(
        '--f0_min',
        type=int,
        default=50,
        help='Минимальная частота pitch (F0) в Hz'
    )
    parser.add_argument(
        '--f0_max',
        type=int,
        default=1100,
        help='Максимальная частота pitch (F0) в Hz'
    )
    
    return parser.parse_args()

# Пример использования:
if __name__ == "__main__":
    args = setup_args()
    cli_conversion(
        input_audios=args.input_audios,
        output_dir=args.output_dir,
        model_name=args.model_name,
        template=args.template,
        index_rate=args.index_rate,
        output_format=args.output_format,
        stereo_mode=args.stereo_mode,
        method_pitch=args.method_pitch,
        pitch=args.pitch,
        hop_length=args.hop_length,
        filter_radius=args.filter_radius,
        rms=args.rms,
        protect=args.protect,
        f0_min=args.f0_min,
        f0_max=args.f0_max
    )


