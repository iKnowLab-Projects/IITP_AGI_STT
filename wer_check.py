import os
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from jiwer import wer
import warnings
import re # 정규표현식 사용을 위해 추가

import os
import math, random, torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM

from peft import LoraConfig, TaskType, get_peft_model

from no_think import NEW_CHAT_TEMPLATE

import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset
import librosa
from transformers import AutoProcessor,  TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torchaudio

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors

from transformers import Trainer as HFTrainer
import torch.nn.functional as F

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# --- 1. 모델 및 평가 데이터 경로 설정 ---
# BASE_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
BASE_MODEL_ID = "/data3/gkook/model/canary-qwen-2.5b"
# 학습된 LoRA 어댑터의 최종 체크포인트 경로를 지정합니다.
# 이 경로는 이전 학습 스크립트의 `output_dir` 내에 있는 마지막 `checkpoint-XXXXX` 폴더입니다.
# LORA_ADAPTER_PATH = '/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result/final/lora_adapters/new_ft'
# LORA_ADAPTER_PATH = '/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result_v2/final/lora_adapters/new_ft'
# LORA_ADAPTER_PATH = '/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result_v3/final/lora_adapters/new_ft'
LORA_ADAPTER_PATH = '/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result_v4/final/lora_adapters/new_ft'
# 데이터셋 경로
LIBRI_TRAIN_CLEAN_100_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/train/LibriSpeech/train-clean-100"
LIBRI_TRAIN_CLEAN_360_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/train/LibriSpeech/train-clean-360"
OURS_CSV_PATH = "/data1/gkook/agi/tts_generated.csv"

# **새로 추가된 LibriSpeech 공식 테스트 데이터셋 경로**
LIBRI_TEST_CLEAN_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/test-clean" # 이 경로를 확인하고 정확히 맞춰주세요.
LIBRI_TEST_OTHER_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/test-other" # 필요 시 추가 (현재는 test-clean만 사용 예정)


# --- 2. 모델 및 프로세서 로드 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용할 디바이스: {device}")

# 기본 모델 로드
print(f"Loading base model {BASE_MODEL_ID}")
model = SALM.from_pretrained(BASE_MODEL_ID).bfloat16()

print_trainable_param = 0
print_total_param = 0
for n , p in model.named_parameters():
    if p.requires_grad==True:
        print('trainable', n, p.numel())
        print_trainable_param += p.numel()
    else:
        print('not trainable', n, p.numel())
    print_total_param += p.numel()
print(f"Trainable parameters: {print_trainable_param}")
print(f"Total parameters: {print_total_param}")

model.llm = model.llm.merge_and_unload()

# LoRA 어댑터 결합
print(f"Loading and merging LoRA adapter from {LORA_ADAPTER_PATH}...")
model.llm = PeftModel.from_pretrained(model.llm, LORA_ADAPTER_PATH)
model.llm = model.llm.merge_and_unload() # 평가 시 메모리 효율성 및 성능을 위해 병합


model.eval() # 모델을 평가 모드로 설정
model.to(device)
# model.tokenizer.tokenizer.chat_template = NEW_CHAT_TEMPLATE
print("Fine-tuned model loaded successfully.")


# --- 3. 데이터셋 로드 함수 (학습 스크립트와 동일하지만, 공식 LibriSpeech 로드 로직 추가) ---
def load_librispeech_dataset(base_path, is_official_test=False):
    data = []
    # LibriSpeech의 디렉토리 구조 (예: base_path/speaker_id/chapter_id/...)를 탐색합니다.
    for speaker_id in os.listdir(base_path):
        speaker_path = os.path.join(base_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            
            # .trans.txt 파일 찾기
            transcript_file_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if os.path.exists(transcript_file_path):
                with open(transcript_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            audio_path = os.path.join(chapter_path, f"{audio_id}.flac")
                            if os.path.exists(audio_path):
                                data.append({"audio_path": audio_path, "text": transcript})
                            else:
                                # print(f"Warning: Audio file not found at {audio_path}") # 너무 많으면 주석 처리
                                pass
            else:
                # print(f"Warning: Transcript file not found at {transcript_file_path}") # 너무 많으면 주석 처리
                pass
    return data

def load_ours_dataset(csv_path):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        if os.path.exists(row["audio_path"]):
            data.append({"audio_path": row["audio_path"], "text": row["first_line"]})
        else:
            print(f"Warning: Audio file not found for ours dataset at {row['audio_path']}. Skipping.")
    return data


# --- 4. 평가 데이터셋 준비 ---
print("Preparing evaluation datasets...")

# LibriSpeech 학습/검증 분할은 더 이상 필요 없으므로 주석 처리하거나 제거합니다.
# 대신 공식 테스트 세트를 로드합니다.
# librispeech_data_100 = load_librispeech_dataset(LIBRI_TRAIN_CLEAN_100_PATH)
# librispeech_data_360 = load_librispeech_dataset(LIBRI_TRAIN_CLEAN_360_PATH)
# librispeech_full_df = pd.DataFrame(librispeech_data_100 + librispeech_data_360)
# _, librispeech_test_df = train_test_split(librispeech_full_df, test_size=0.05, random_state=42)
# librispeech_test_dataset = Dataset.from_pandas(librispeech_test_df)

print(f"Loading official LibriSpeech test-clean dataset from {LIBRI_TEST_CLEAN_PATH}...")
librispeech_official_test_clean_data = load_librispeech_dataset(LIBRI_TEST_CLEAN_PATH, is_official_test=True)
librispeech_official_test_clean_dataset = Dataset.from_pandas(pd.DataFrame(librispeech_official_test_clean_data))
print(f"LibriSpeech Test-Clean loaded: {len(librispeech_official_test_clean_dataset)} samples.")


print(f"Loading ours dataset from {OURS_CSV_PATH} and splitting...")
ours_full_data = load_ours_dataset(OURS_CSV_PATH)
ours_full_df = pd.DataFrame(ours_full_data)
# 학습 스크립트와 동일한 random_state와 test_size를 사용하여 정확히 동일한 테스트 세트를 가져옵니다.
_, ours_test_df = train_test_split(ours_full_df, test_size=0.1, random_state=42)
ours_test_dataset = Dataset.from_pandas(ours_test_df)
print(f"Ours Test Dataset (from split) loaded: {len(ours_test_dataset)} samples.")

print("\n--- All evaluation datasets prepared. ---")

generation_kwargs = dict(
    do_sample=False,
    # temperature=0.7,      # 0.7~1.0 범위에서 조절
    # top_p=0.9,            # 또는 top_k=50~100
    repetition_penalty=1.1,# 1.1~1.2
    no_repeat_ngram_size=3,# 3~4
    max_new_tokens=256,    # 과도하면 반복 확률↑
)


def evaluate_wer(dataset, model, dataset_name="Unnamed Dataset"):
    too_long_count = 0
    predictions = []
    references = []
    
    
    print(f"\n--- Starting WER Evaluation for: {dataset_name} ({len(dataset)} samples) ---")

    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
        try:
            audio_path = item['audio_path']
            reference_text = item['text']
            with torch.no_grad():
                generated_ids = model.generate(
                    prompts = [
                            [
                                {
                                    # "role": "user", "content": f"/nothink Transcribe the following directly without other words: {model.audio_locator_tag}",
                                    "role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}",
                                    "audio": [audio_path]
                                }
                            ]
                        ],
                    **generation_kwargs,
                )
                response_text = model.tokenizer.ids_to_text(generated_ids[0].cpu())
                
            # with torch.no_grad():
            #     prompt = f"Transcribe the following without thinking: {model.audio_locator_tag}"
                
            
            
            
            
            
            
            # print(f"Response text: {response_text}")
            # print(f"Original text: {reference_text}")
           
            # --- 예측 텍스트 추출 로직 (강화됨) ---
            
            # predicted_text_cleaned = response_text.replace('<think>', '').replace('</think>', '').strip()
                        # --- 예측 텍스트 추출 로직 (강화됨) ---
            predicted_text_cleaned = response_text.replace('<think>', '').replace('</think>', '').strip()
                            
            # print(f"Cleaned text : {predicted_text_cleaned}")
            # print(f"Original text: {reference_text}")            
            if len(predicted_text_cleaned) >= 2*len(reference_text) or '\n' in predicted_text_cleaned:
                print('Too long or new line Detected:', predicted_text_cleaned)
                too_long_count += 1
            predictions.append(predicted_text_cleaned)
            references.append(reference_text)

        except Exception as e:
            print(f"Error processing {item.get('audio_path', 'N/A')}: {e}. Skipping this sample.")
            # 오류 발생 시 빈 문자열을 예측으로 추가하여 WER 계산에 영향을 주지 않도록 합니다.
            # 이는 WER을 과대평가할 수 있지만, 전체 샘플 수 유지를 위함입니다.
            predictions.append("") 
            references.append(item['text'])

    # jiwer.wer 함수는 내부적으로 텍스트를 정규화합니다 (소문자 변환, 공백 처리, 일부 구두점 제거 등).
    # 따라서 여기서 추가적인 .upper()나 replace(" ", "")는 일반적으로 불필요하며, jiwer의 표준 정규화에 맡기는 것이 좋습니다.
    # 단, jiwer가 처리하지 않는 특정 정규화(예: 숫자 -> 단어)가 필요하다면, predictions와 references를 WER 계산 전에 변환해야 합니다.
    wer_score = wer(references, predictions)
    return wer_score, predictions, references, too_long_count

# --- 6. WER 평가 실행 ---

# 1. LibriSpeech Official Test-Clean 평가
wer_score_librispeech, _, _, too_long_count = evaluate_wer(librispeech_official_test_clean_dataset, model, dataset_name="LibriSpeech Test-Clean (Official)")
print(f"\n📈 LibriSpeech Official Test-Clean WER: {wer_score_librispeech:.4f}")

print("-" * 50)

# 2. Ours Test Dataset 평가
wer_score_ours, _, _, too_long_count2 = evaluate_wer(ours_test_dataset, model, dataset_name="Ours Test Dataset (Internal Split)")
print(f"\n📈 Ours Test Dataset WER: {wer_score_ours:.4f}")

print("\n--- Evaluation Complete ---")
print('Too long or new line count LibriSpeech: ', too_long_count)
print('Too long or new line count Ours: ', too_long_count2)