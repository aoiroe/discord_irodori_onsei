from gradio_client import Client, handle_file
from pathlib import Path
import shutil

API_URL = "http://127.0.0.1:7860/"
REFERENCE_AUDIO = r"C:\Users\81903\Desktop\作成物\irodoriTTS\ffmpeg\reference.wav"
OUTPUT_DIR = Path(r"C:\Users\81903\Desktop\作成物\irodoriTTS\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

client = Client(API_URL)

client.predict(
    checkpoint="Aratako/Irodori-TTS-500M-v2",
    model_device="cuda",
    model_precision="fp32",
    codec_device="cuda",
    codec_precision="fp32",
    api_name="/_load_model"
)

result = client.predict(
    checkpoint="Aratako/Irodori-TTS-500M-v2",
    model_device="cuda",
    model_precision="fp32",
    codec_device="cuda",
    codec_precision="fp32",

    text="これは複数候補を生成するテストです。",
    uploaded_audio=handle_file(REFERENCE_AUDIO),

    num_steps=40,
    num_candidates=4,
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

# 0〜31番目までが音声候補
audio_files = result[:32]
run_log = result[32]
timing = result[33]

saved = []

for i, audio_file in enumerate(audio_files, start=1):
    if not audio_file:
        continue

    if isinstance(audio_file, dict):
        audio_path = audio_file.get("path") or audio_file.get("value")
    else:
        audio_path = audio_file

    if not audio_path:
        continue

    out = OUTPUT_DIR / f"candidate_{i:02}.wav"
    shutil.copy(audio_path, out)
    saved.append(out)
for i, item in enumerate(result):
    print("index:", i)
    print("type:", type(item))
    print("value:", item)
    print("-" * 50)
print("保存した音声:")
for path in saved:
    print(path)

print("ログ:")
print(run_log)

print("Timing:")
print(timing)