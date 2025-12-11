import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import torch
import numpy as np
import torchaudio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from streamlit_webrtc import WebRtcMode


# Load model and processor
MODEL_NAME = "/data3/gkook/model/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained("/data3/gkook/model/Qwen2-Audio-7B-Instruct", dtype=torch.bfloat16)
model = Qwen2AudioForConditionalGeneration.from_pretrained("/data3/gkook/model/Qwen2-Audio-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
try:
    model.tie_weights()
except:
    model._tie_weights()

# 제목
st.title("🎙️ Real-Time Speech-to-Text (Qwen2-Audio)")
st.write("Speak into your mic. Transcription will appear below.")

# 세션 상태로 텍스트 저장
if "latest_text" not in st.session_state:
    st.session_state["latest_text"] = ""

# 텍스트 출력 공간
transcription_texts = st.empty()

# 오디오 프로세서 클래스 정의
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.buffer = np.concatenate((self.buffer, audio))

        # 약 3초 분량의 버퍼가 쌓이면 처리
        if len(self.buffer) > 48000:
            clip = self.buffer[:48000]
            self.buffer = self.buffer[48000:]

            # STT 변환
            inputs = processor(audios=clip, sampling_rate=16000, return_tensors="pt")
            inputs.update({"text": "Transcribe this speech into English text."})

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
                result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            st.session_state["latest_text"] = result

        return frame

# WebRTC 스트리밍 설정
webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,  # ❗ 여기를 수정!
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)


# 텍스트 출력
st.subheader("📝 Transcription:")
st.write(st.session_state["latest_text"])