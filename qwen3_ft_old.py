import os
import math, random, torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM

from peft import LoraConfig, TaskType, get_peft_model

import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset
import librosa
from transformers import AutoProcessor,  TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb



model = SALM.from_pretrained("/data3/gkook/model/canary-qwen-2.5b")

device = next(model.parameters()).device
SR =  16000
AUDIO_TAG = model.audio_locator_tag
AUDIO_TAG_ID = model.audio_locator_tag_id
tok = model.tokenizer

model.llm = model.llm.merge_and_unload()
model.llm = get_peft_model(model.llm, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ),
    'new_ft'
)

for n, p in model.named_parameters():
    if p.requires_grad==True:
        p.requires_grad = False

for n, p in model.llm.named_parameters():
    if 'lora' in n.lower():
        p.requires_grad = True
        
model.llm.print_trainable_parameters()

# --- 2. 데이터셋 경로 설정 ---
LIBRI_TRAIN_CLEAN_100_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/train/LibriSpeech/train-clean-100"
LIBRI_TRAIN_CLEAN_360_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/train/LibriSpeech/train-clean-360"
OURS_CSV_PATH = "/data1/gkook/agi/tts_generated.csv"
OURS_AUDIO_DIR = "/data1/gkook/agi/output_audio"

# --- 3. 데이터셋 로드 함수 ---
def load_librispeech_dataset(base_path):
    data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(root, file)
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            audio_path = os.path.join(root, f"{audio_id}.flac") 
                            if os.path.exists(audio_path):
                                data.append({"audio_path": audio_path, "text": transcript, "source": "librispeech"})
                            else:
                                print(f"Warning: Audio file not found at {audio_path}")
    return data

def load_ours_dataset(csv_path):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        if os.path.exists(row["audio_path"]):
            data.append({"audio_path": row["audio_path"], "text": row["first_line"], "source": "ours"})
        else:
            print(f"Warning: Audio file not found for ours dataset at {row['audio_path']}")
    return data

# --- 4. 데이터 준비 ---
print("Loading LibriSpeech train-clean-100...")
librispeech_data_100 = load_librispeech_dataset(LIBRI_TRAIN_CLEAN_100_PATH)
print("Loading LibriSpeech train-clean-360...")
librispeech_data_360 = load_librispeech_dataset(LIBRI_TRAIN_CLEAN_360_PATH)
librispeech_full_df = pd.DataFrame(librispeech_data_100 + librispeech_data_360)
librispeech_train_df, librispeech_test_df = train_test_split(librispeech_full_df, test_size=0.05, random_state=42)
librispeech_train_dataset = Dataset.from_pandas(librispeech_train_df)
librispeech_test_dataset = Dataset.from_pandas(librispeech_test_df)

print("Loading ours dataset...")
ours_full_data = load_ours_dataset(OURS_CSV_PATH)
ours_full_df = pd.DataFrame(ours_full_data)
ours_train_df, ours_test_df = train_test_split(ours_full_df, test_size=0.1, random_state=42)
ours_train_dataset = Dataset.from_pandas(ours_train_df)
ours_test_dataset = Dataset.from_pandas(ours_test_df)

training_datasets_list = [librispeech_train_dataset, ours_train_dataset]
evaluation_datasets_list = [librispeech_test_dataset, ours_test_dataset]
print("\n--- Data Preparation Complete ---")


# --- 5. 데이터셋 및 데이터로더 구현 ---
class BalancedIterableDataset(IterableDataset):
    def __init__(self, dataset_list, seed=42):
        self.dataset_list = dataset_list
        self.num_datasets = len(dataset_list)
        self.all_indices = [list(range(len(ds))) for ds in dataset_list]
        self.lengths = [len(ds) for ds in dataset_list]
        self.seed = seed
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id)
        shuffled_indices_per_dataset = [list(indices) for indices in self.all_indices]
        for indices in shuffled_indices_per_dataset:
            rng.shuffle(indices)
        current_pointers = [0] * self.num_datasets
        while True:
            for i in range(self.num_datasets):
                if current_pointers[i] >= len(shuffled_indices_per_dataset[i]):
                    rng.shuffle(shuffled_indices_per_dataset[i])
                    current_pointers[i] = 0
                local_idx = shuffled_indices_per_dataset[i][current_pointers[i]]
                current_pointers[i] += 1
                yield (i, local_idx)
    def __len__(self):
        return sum(self.lengths)

# ⭐️ [수정된 부분] 학습/평가 데이터 형식을 모두 처리하도록 변경된 데이터 콜레이터
class CustomDataCollator:
    def __init__(self, processor_instance, datasets_list=None):
        self.processor = processor_instance
        self.datasets_list = datasets_list # 학습 시에만 사용됨

    def __call__(self, samples):
        if not samples:
            return {}

        batch_audios = []
        full_texts = []
        prompt_texts = []

        # 입력 데이터의 형태를 확인하여 분기 처리
        # 학습 시: samples = [(dataset_idx, local_idx), ...] -> 튜플 리스트
        # 평가 시: samples = [{'audio_path': ..., 'text': ...}, ...] -> 딕셔너리 리스트
        is_training_format = isinstance(samples[0], tuple)

        actual_samples = []
        if is_training_format:
            # 학습 데이터 처리: 인덱스를 사용하여 실제 데이터를 가져옴
            if self.datasets_list is None:
                raise ValueError("datasets_list must be provided to the collator for the training dataset.")
            for dataset_idx, local_idx in samples:
                actual_samples.append(self.datasets_list[dataset_idx][local_idx])
        else:
            # 평가 데이터 처리: samples 자체가 실제 데이터임
            actual_samples = samples

        for sample in actual_samples:
            try:
                target_sampling_rate = self.processor.feature_extractor.sampling_rate
                audio_input, _ = librosa.load(sample["audio_path"], sr=target_sampling_rate, mono=True)
                batch_audios.append(audio_input)

                transcription = sample["text"]
                
                prompt_messages = [
                    {"role": "user", "content": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe this audio."}
                    ]}
                ]
                
                full_messages = prompt_messages + [{"role": "assistant", "content": transcription}]

                prompt_str = self.processor.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
                full_str = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)
                
                prompt_texts.append(prompt_str)
                full_texts.append(full_str)
            except Exception as e:
                # 오류 발생 시 어떤 파일에서 문제가 생겼는지 명확히 출력
                print(f"Error processing audio {sample.get('audio_path', 'N/A')}: {e}. Skipping.")
                continue

        if not batch_audios: return {}

        features = self.processor(
            text=full_texts,
            audio=batch_audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt"
        )

        prompt_features = self.processor(
            text=prompt_texts,
            padding=True,
            return_tensors="pt"
        )
        prompt_lengths = prompt_features.attention_mask.sum(dim=1)

        labels = features.input_ids.clone()

        for i in range(len(labels)):
            prompt_len = prompt_lengths[i]
            labels[i, :prompt_len] = -100
        
        features['labels'] = labels
        return features


balanced_iterable_dataset = BalancedIterableDataset(training_datasets_list)
eval_dataset = concatenate_datasets(evaluation_datasets_list)

# DataCollator를 생성할 때 학습용 데이터셋 리스트를 넘겨줍니다.
data_collator = CustomDataCollator(processor_instance=processor, datasets_list=training_datasets_list)

BATCH_SIZE = 4
print("\n데이터 파이프라인 설정 완료.")

# --- 7. 훈련 인자 (TrainingArguments) 설정 ---
print("\n--- 훈련 인자 설정 중 ---")
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_OUTPUT_DIR, "qwen2_audio_finetune_output"), # Updated output directory
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    warmup_steps=500,
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=BATCH_SIZE,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    fp16=False,
    bf16=True if torch.cuda.is_bf16_supported() else False,
    report_to="wandb",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# --- 8. Trainer 인스턴스 생성 및 학습 ---
print("\n--- Trainer 인스턴스 생성 중 ---")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=balanced_iterable_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

print("\n--- 파인튜닝 시작 ---")
train_result = trainer.train()

print("\n--- 파인튜닝 완료 ---")
output_lora_path = os.path.join(training_args.output_dir, "lora_adapters")
model.save_pretrained(output_lora_path)
print(f"학습된 LoRA 어댑터가 '{output_lora_path}'에 저장되었습니다.")

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state() # This will save the trainer state in output_dir
print(f"최종 학습 결과: {metrics}")

print("\n--- 최종 평가 실행 중 ---")
eval_results = trainer.evaluate()
trainer.log_metrics("eval", eval_results)
trainer.save_metrics("eval", eval_results)
print(f"최종 평가 결과: {eval_results}")

wandb.finish()
print("Weights & Biases 실행이 종료되었습니다.")
