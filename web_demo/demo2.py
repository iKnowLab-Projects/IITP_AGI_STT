import torchaudio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import io
import re, time
import os
import sys

# api_demo 경로를 sys.path에 추가하여 model_loader를 불러올 수 있게 함
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api_demo'))
from model_loader import get_model

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, processor = get_model()

# torchaudio backend를 ffmpeg로 설정
torchaudio.set_audio_backend("ffmpeg")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 파일 읽기
    start_time = time.time()
    file_bytes = await file.read()
    audio_buffer = io.BytesIO(file_bytes)

    # torchaudio로 오디오 로드 (자동 리샘플링 없음)
    try:
        waveform, sr = torchaudio.load(audio_buffer)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return {"error": str(e)}

    # 리샘플링 (16000 Hz)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # mono로 변환 (여러 채널인 경우 평균 또는 첫 번째 채널 사용)
    if waveform.shape[0] > 1:
        # stereo인 경우 mono로 변환 (평균 사용)
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # numpy로 변환 - 반드시 1D 배열로 변환
    audio_data = waveform.squeeze().numpy()
    
    # 차원 확인 및 보정
    if audio_data.ndim == 0:
        audio_data = audio_data.reshape(-1)
    elif audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # dtype 확인 (float32로 변환)
    if audio_data.dtype != 'float32':
        audio_data = audio_data.astype('float32')

    # 대화 템플릿 생성
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_data}]}]
    # instruction = "Transcribe this speech into English text."
    instruction = "Transcribe the following:"

    prompt = processor.apply_chat_template(
        conversation + [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False
    )

    # 모델 입력 준비 (audios는 Qwen2-Audio 최신 버전에서 audio로 변경되었을 수 있으나 기존 코드 호환성 유지)
    try:
        inputs = processor(
            text=prompt,
            audios=[audio_data],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to("cuda")
    except TypeError:
        # 최신 버전에서는 audio 인자 사용
        inputs = processor(
            text=prompt,
            audio=[audio_data],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to("cuda")

    inputs.input_ids = inputs.input_ids.to("cuda")

    # 모델 추론
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_length=1536, do_sample=False)

    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    result = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    # 출력 후처리: 불필요한 프리픽스 제거
    result = clean_transcription_output(result)

    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f}")

    return {"transcription": result}


def clean_transcription_output(text: str) -> str:
    """
    모델 출력에서 불필요한 프리픽스나 설명 문구를 제거하고 순수 STT 결과만 반환
    """
    if not text:
        return text
    
    # 1. 포괄적인 정규표현식 패턴들
    patterns_to_remove = [
        r"^The original content of (this |the )?(audio|speech) is:?\s*['\"]?",
        r"^The transcription of the (speech|audio) is:?\s*['\"]?",
        r"^The transcription (of the audio )?is:?\s*['\"]?",
        r"^The speech in the audio is (in .*? )?and translates to:?\s*['\"]?",
        r"^The speech in the audio is (in .*? ):?\s*['\"]?",
        r"^The speech is (in .*? ):?\s*['\"]?",
        r"^The speaker says( in .*? )?:?\s*['\"]?",
        r"^The (audio|speech) contains( .*? saying)?:?\s*['\"]?",
        r"^The recorded (speech|audio) is:?\s*['\"]?",
        r"^The transcribed text (for this audio )?is:?\s*['\"]?",
        r"^The output of the transcription (of the audio )?is:?\s*['\"]?",
        r"^Transcription( of the audio)?:?\s*['\"]?",
        r"^The audio says:?\s*['\"]?",
        r"^The text is:?\s*['\"]?",
        r"^Here is the transcription:?\s*['\"]?",
        r"^The audio transcription is:?\s*['\"]?",
        r"^The sentence is:?\s*['\"]?",
        r"^Translated text:?\s*['\"]?",
        r"^It translates (from .*? )?to:?\s*['\"]?",
        r"^This translates to:?\s*['\"]?",
        r"^translates to:?\s*['\"]?",
        r"^translates to?\s*['\"]?",
        r"^saying:?\s*['\"]?",
        r"^saying?\s*['\"]?",
        r"^audio is:?\s*['\"]?",
        r"^audio is?\s*['\"]?",
        r"^audio:?\s*['\"]?",
        r"^audio?\s*['\"]?",
        r"^The meaning of the (speech|audio) is:?\s*['\"]?",
        r"^The content of the audio is:?\s*['\"]?",
        r"^The person (in the audio )?says:?\s*['\"]?",
        r"^Spoken text:?\s*['\"]?",
    ]
    
    cleaned_text = text.strip()
    
    for pattern in patterns_to_remove:
        new_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE).strip()
        if new_text != cleaned_text:
            cleaned_text = new_text
            break
    
    cleaned_text = cleaned_text.strip().strip("'\"").strip()
    return cleaned_text

# 모바일 전용 정적 파일 서빙
static_dir = os.path.join(os.path.dirname(__file__), "static_mobile")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # HTTPS 설정을 위한 인증서 경로
    cert_path = os.path.join(os.path.dirname(__file__), "cert.pem")
    key_path = os.path.join(os.path.dirname(__file__), "key.pem")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=57658,
        ssl_keyfile=key_path,
        ssl_certfile=cert_path
    )
