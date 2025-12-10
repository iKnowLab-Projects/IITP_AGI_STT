import streamlit as st
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration
import torch
import torchaudio
from peft import PeftModel

# 모델과 프로세서 로딩
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
#model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config = quantization_config,
        device_map="auto",
        trust_remote_code=True
)

st.title("🗣️ Qwen2-Audio-7B-Instruct Speech-to-Text")
st.write("Upload an audio file and get the transcribed text.")

audio_file = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file)

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = processor(audios=waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # 사용자 지시어 (Prompt)
    instruction = "Transcribe this speech into English text."
    inputs.update({"text": instruction})

    # 모델 추론
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    st.subheader("📝 Transcription")
    st.write(result)
