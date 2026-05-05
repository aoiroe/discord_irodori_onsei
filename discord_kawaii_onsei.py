from gradio_client import Client, handle_file
from pathlib import Path
from faster_whisper import WhisperModel

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import shutil
import queue
import time
import os


# ============================================================
# 基本設定
# ============================================================

API_URL = "http://127.0.0.1:7860/"

REFERENCE_AUDIO = r"C:\Users\81903\Desktop\作成物\irodoriTTS\ffmpeg\reference.wav"
OUTPUT_DIR = Path(r"C:\Users\81903\Desktop\作成物\irodoriTTS\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"


# ============================================================
# 音声デバイス設定
# ============================================================

# 入力：実マイク
# 24 = マイク (Logi C270 HD WebCam), Windows WASAPI
# 22 = ヘッドセット (WH-CH510), Windows WASAPI
# 3  = マイク (Logi C270 HD WebCam), MME
INPUT_DEVICE = 24

# 出力：VB-CABLEの CABLE Input
# Discord側では CABLE Output を入力デバイスにする
#
# 21 = CABLE Input (VB-Audio Virtual Cable), Windows WASAPI
# 5  = CABLE Input (VB-Audio Virtual C...), MME
OUTPUT_DEVICE_CANDIDATES = [21, 5]


# ============================================================
# 録音・発話検出設定
# ============================================================

CHANNELS = 1
BLOCK_SIZE = 1024

# マイクの標準サンプルレートを自動取得する
INPUT_DEVICE_INFO = sd.query_devices(INPUT_DEVICE, "input")
SAMPLE_RATE = int(INPUT_DEVICE_INFO["default_samplerate"])

# 声を検出するしきい値
# 反応しないなら 0.008
# 勝手に反応するなら 0.025
VOICE_THRESHOLD = 0.015

# この秒数だけ無音なら発話終了
SILENCE_SECONDS = 1.0

# 最低録音秒数
MIN_RECORD_SECONDS = 0.7

# 最大録音秒数
MAX_RECORD_SECONDS = 12.0

# TTS再生後、マイク再開まで少し待つ
AFTER_PLAYBACK_SLEEP = 0.3


# ============================================================
# TTS設定
# ============================================================

NUM_STEPS = 30
NUM_CANDIDATES = 1
SEED_RAW = "1"

CFG_GUIDANCE_MODE = "independent"
CFG_SCALE_TEXT = 3
CFG_SCALE_SPEAKER = 5

CFG_MIN_T = 0.5
CFG_MAX_T = 1
CONTEXT_KV_CACHE = True

SPEAKER_KV_MIN_T_RAW = "0.9"


# ============================================================
# キュー
# ============================================================

audio_queue = queue.Queue()


# ============================================================
# ユーティリティ
# ============================================================

def print_devices():
    print("")
    print("========== 使用デバイス ==========")
    print("入力デバイス:", INPUT_DEVICE)
    print(INPUT_DEVICE_INFO)
    print("入力サンプルレート:", SAMPLE_RATE)

    print("")
    print("出力デバイス候補:")
    for device_id in OUTPUT_DEVICE_CANDIDATES:
        try:
            info = sd.query_devices(device_id, "output")
            print(device_id, info["name"])
        except Exception as e:
            print(device_id, "取得失敗:", e)

    print("==================================")
    print("")


def rms(audio_block):
    return float(np.sqrt(np.mean(audio_block ** 2)))


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    audio_queue.put(indata.copy())


def save_temp_wav(audio_np, sample_rate):
    temp_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    sf.write(temp_path, audio_np, sample_rate)
    return temp_path


def normalize_audio_for_playback(data):
    """
    sounddeviceで流しやすい形に整える。
    monoなら shape を (samples, 1) にする。
    音割れ防止のため軽く正規化。
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_abs = np.max(np.abs(data)) if data.size else 0

    if max_abs > 1.0:
        data = data / max_abs

    return data.astype(np.float32)


# ============================================================
# Whisper
# ============================================================

def load_whisper():
    print("Whisperモデル読み込み中...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("Whisperモデル読み込み完了")
    return model


def transcribe_audio(whisper_model, audio_path):
    print("文字起こし中...")

    segments, info = whisper_model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        vad_filter=True
    )

    text_parts = []

    for segment in segments:
        text_parts.append(segment.text)

    text = "".join(text_parts).strip()

    return text


# ============================================================
# Irodori-TTS
# ============================================================

def load_irodori_client():
    client = Client(API_URL)

    print("Irodori-TTS モデル読み込み中...")

    client.predict(
        checkpoint=CHECKPOINT,
        model_device="cuda",
        model_precision="fp32",
        codec_device="cuda",
        codec_precision="fp32",
        api_name="/_load_model"
    )

    print("Irodori-TTS モデル読み込み完了")

    return client


def generate_tts(client, text):
    print("Irodori-TTSで音声生成中...")
    print("入力テキスト:", text)

    result = client.predict(
        checkpoint=CHECKPOINT,
        model_device="cuda",
        model_precision="fp32",
        codec_device="cuda",
        codec_precision="fp32",

        text=text,
        uploaded_audio=handle_file(REFERENCE_AUDIO),

        num_steps=NUM_STEPS,
        num_candidates=NUM_CANDIDATES,
        seed_raw=SEED_RAW,

        cfg_guidance_mode=CFG_GUIDANCE_MODE,
        cfg_scale_text=CFG_SCALE_TEXT,
        cfg_scale_speaker=CFG_SCALE_SPEAKER,
        cfg_scale_raw="",

        cfg_min_t=CFG_MIN_T,
        cfg_max_t=CFG_MAX_T,
        context_kv_cache=CONTEXT_KV_CACHE,

        truncation_factor_raw="",
        rescale_k_raw="",
        rescale_sigma_raw="",
        speaker_kv_scale_raw="",
        speaker_kv_min_t_raw=SPEAKER_KV_MIN_T_RAW,
        speaker_kv_max_layers_raw="",

        api_name="/_run_generation"
    )

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

        out = OUTPUT_DIR / "discord_irodori_output.wav"
        shutil.copy(audio_path, out)

        print("生成音声保存:", out)
        return out

    return None


# ============================================================
# VB-CABLEへ出力
# ============================================================

def play_audio_to_virtual_mic(audio_path):
    print("仮想マイクへ出力中...")

    data, sr = sf.read(str(audio_path), dtype="float32")
    data = normalize_audio_for_playback(data)

    last_error = None

    for output_device in OUTPUT_DEVICE_CANDIDATES:
        try:
            device_info = sd.query_devices(output_device, "output")

            print("出力デバイス:", output_device, device_info["name"])
            print("wav sample rate:", sr)

            sd.play(
                data,
                sr,
                device=output_device,
                blocking=True
            )

            print("仮想マイクへの出力完了")
            return True

        except Exception as e:
            last_error = e
            print("この出力デバイスでは失敗:", output_device)
            print(e)

    print("すべての出力デバイスで失敗しました")
    print("最後のエラー:", last_error)

    return False


# ============================================================
# 自動録音
# ============================================================

def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def listen_once():
    print("")
    print("待機中... 話すと自動で録音します")

    clear_audio_queue()

    recording = []
    is_recording = False
    started_at = None
    last_voice_at = None

    while True:
        block = audio_queue.get()
        volume = rms(block)
        now = time.time()

        if not is_recording:
            if volume >= VOICE_THRESHOLD:
                print("録音開始")
                is_recording = True
                started_at = now
                last_voice_at = now
                recording.append(block)
            continue

        recording.append(block)

        if volume >= VOICE_THRESHOLD:
            last_voice_at = now

        recorded_seconds = now - started_at
        silence_seconds = now - last_voice_at

        if recorded_seconds >= MAX_RECORD_SECONDS:
            print("最大録音時間に到達")
            break

        if recorded_seconds >= MIN_RECORD_SECONDS and silence_seconds >= SILENCE_SECONDS:
            print("発話終了")
            break

    audio_np = np.concatenate(recording, axis=0)
    audio_np = np.squeeze(audio_np)

    return save_temp_wav(audio_np, SAMPLE_RATE)


# ============================================================
# メイン処理
# ============================================================

def main():
    print_devices()

    if not os.path.exists(REFERENCE_AUDIO):
        print("参照音声が見つかりません:")
        print(REFERENCE_AUDIO)
        return

    whisper_model = load_whisper()
    client = load_irodori_client()

    print("")
    print("=== Discord用 自動マイク変換モード開始 ===")
    print("終了するには Ctrl + C")
    print("")
    print("Discord側の入力デバイスは CABLE Output を選んでください")
    print("Python側は CABLE Input に出力します")
    print("")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback
        ):
            while True:
                mic_wav = listen_once()

                text = transcribe_audio(whisper_model, mic_wav)

                if not text:
                    print("文字起こしできませんでした")
                    continue

                print("文字起こし結果:")
                print(text)

                output_wav = generate_tts(client, text)

                if output_wav:
                    ok = play_audio_to_virtual_mic(output_wav)

                    if not ok:
                        print("仮想マイクへの出力に失敗しました")
                else:
                    print("音声生成に失敗しました")

                time.sleep(AFTER_PLAYBACK_SLEEP)

    except KeyboardInterrupt:
        print("")
        print("終了しました")

    except Exception as e:
        print("")
        print("エラーが発生しました:")
        print(e)
        print("")
        print("対処候補:")
        print("1. INPUT_DEVICE を 24 から 3 に変える")
        print("2. OUTPUT_DEVICE_CANDIDATES を [5, 21] に変える")
        print("3. Discord側の入力が CABLE Output になっているか確認")
        print("4. Discordのノイズ抑制・自動感度調整をOFFにする")


if __name__ == "__main__":
    main()