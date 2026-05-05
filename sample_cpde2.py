from gradio_client import Client, handle_file
from pathlib import Path
import shutil
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

API_URL = "http://127.0.0.1:7860/"

REFERENCE_AUDIO = r"C:\Users\81903\Desktop\作成物\irodoriTTS\ffmpeg\reference.wav"
OUTPUT_DIR = Path(r"C:\Users\81903\Desktop\作成物\irodoriTTS\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

# Whisperモデル
# CPUなら "small" か "base"
# GPUなら device="cuda" でもOK
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

client = Client(API_URL)

print("Irodori-TTS モデル読み込み中...")

client.predict(
    checkpoint="Aratako/Irodori-TTS-500M-v2",
    model_device="cuda",
    model_precision="fp32",
    codec_device="cuda",
    codec_precision="fp32",
    api_name="/_load_model"
)

print("モデル読み込み完了")


def record_mic(seconds=5):
    print(f"{seconds}秒録音します。話してください...")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = np.squeeze(audio)

    temp_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    sf.write(temp_path, audio, SAMPLE_RATE)

    return temp_path


def transcribe_audio(audio_path):
    print("文字起こし中...")

    segments, info = whisper_model.transcribe(
        audio_path,
        language="ja",
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    text = text.strip()

    return text


def generate_tts(text):
    print("Irodori-TTSで音声生成中...")
    print("入力テキスト:", text)

    result = client.predict(
        checkpoint="Aratako/Irodori-TTS-500M-v2",
        model_device="cuda",
        model_precision="fp32",
        codec_device="cuda",
        codec_precision="fp32",

        text=text,
        uploaded_audio=handle_file(REFERENCE_AUDIO),

        num_steps=40,
        num_candidates=1,
        seed_raw="1",

        cfg_guidance_mode="independent",
        cfg_scale_text=3,
        cfg_scale_speaker=5,
        cfg_scale_raw="",

        cfg_min_t=0.5,
        cfg_max_t=1,
        context_kv_cache=True,

        truncation_factor_raw="",
        rescale_k_raw="",
        rescale_sigma_raw="",
        speaker_kv_scale_raw="",
        speaker_kv_min_t_raw="0.9",
        speaker_kv_max_layers_raw="",

        api_name="/_run_generation"
    )

    # 0〜31番目に候補が入る設計
    audio_files = result[:32]

    for i, audio_file in enumerate(audio_files, start=1):
        if not audio_file:
            continue

        if isinstance(audio_file, dict):
            audio_path = audio_file.get("value") or audio_file.get("path")
        else:
            audio_path = audio_file

        if not audio_path:
            continue

        out = OUTPUT_DIR / "mic_output.wav"
        shutil.copy(audio_path, out)

        print("保存:", out)
        return out

    return None


def play_audio(audio_path):
    print("再生中...")

    data, sr = sf.read(audio_path, dtype="float32")
    cable_input_device = 5

    print("仮想マイクへ出力:", cable_input_device)
    sd.play(data, sr, device=cable_input_device)
    sd.wait()


while True:
    input_text = input("\nEnterで録音開始 / qで終了: ")

    if input_text.lower() == "q":
        break

    mic_wav = record_mic(RECORD_SECONDS)

    text = transcribe_audio(mic_wav)

    if not text:
        print("文字起こしできませんでした")
        continue

    print("文字起こし結果:")
    print(text)

    output_wav = generate_tts(text)

    if output_wav:
        play_audio(output_wav)
    else:
        print("音声生成に失敗しました")