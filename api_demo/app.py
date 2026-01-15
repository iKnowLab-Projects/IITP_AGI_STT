# from fastapi import FastAPI, UploadFile, File
# from model_loader import get_model
# import librosa
# import torch

# app = FastAPI()
# model, processor = get_model()

# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     # 오디오 로드 및 리샘플링
#     audio_data, sr = librosa.load(file.file, sr=16000)

#     # 대화 컨텍스트 + instruction 생성
#     conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_data}]}]
#     instruction = "Transcribe this speech into English text."
#     prompt = processor.apply_chat_template(
#         conversation + [{"role": "user", "content": instruction}],
#         add_generation_prompt=True,
#         tokenize=False
#     )

#     # 모델 입력 생성
#     inputs = processor(text=prompt, audios=[audio_data], sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
#     inputs.input_ids = inputs.input_ids.to("cuda")

#     # 추론
#     generate_ids = model.generate(**inputs, max_length=50)
#     generate_ids = generate_ids[:, inputs.input_ids.size(1):]
#     result = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

#     return {"transcription": result}

import torchaudio
from fastapi import FastAPI, UploadFile, File
from model_loader import get_model
import torch
import io
import re, time

app = FastAPI()
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
    waveform, sr = torchaudio.load(audio_buffer)

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
        # 스칼라인 경우 (거의 없음)
        audio_data = audio_data.reshape(-1)
    elif audio_data.ndim > 1:
        # 2D 이상인 경우 1D로 변환
        audio_data = audio_data.flatten()
    
    # dtype 확인 (float32로 변환)
    if audio_data.dtype != 'float32':
        audio_data = audio_data.astype('float32')

    # 대화 템플릿 생성
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_data}]}]
    instruction = "Transcribe this speech into English text. Output only the transcribed text without any prefix, explanation, or additional text."

    prompt = processor.apply_chat_template(
        conversation + [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False
    )

    # 모델 입력 준비
    inputs = processor(
        text=prompt,
        audios=[audio_data],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    inputs.input_ids = inputs.input_ids.to("cuda")

    # 모델 추론
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_length=1536)

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
    
    # 일반적인 프리픽스 패턴들 제거
    patterns_to_remove = [
        r"^The speech in the audio is:\s*['\"]?",  # "The speech in the audio is: '"
        r"^The speech is:\s*['\"]?",  # "The speech is: '"
        r"^Transcription:\s*['\"]?",  # "Transcription: '"
        r"^The transcribed text is:\s*['\"]?",  # "The transcribed text is: '"
        r"^The audio says:\s*['\"]?",  # "The audio says: '"
        r"^The text is:\s*['\"]?",  # "The text is: '"
        r"^Here is the transcription:\s*['\"]?",  # "Here is the transcription: '"
        r"^The audio transcription is:\s*['\"]?",  # "The audio transcription is: '"
    ]
    
    cleaned_text = text.strip()
    
    # 각 패턴에 대해 제거 시도
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    
    # 앞뒤 따옴표 제거
    cleaned_text = cleaned_text.strip().strip("'\"")
    
    # 앞뒤 공백 제거
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text
