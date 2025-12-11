import os
import math, random, torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
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

import torchaudio

from transformers import Trainer as HFTrainer
import torch.nn.functional as F

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
        r=4,
        lora_alpha=8,
        lora_dropout=0.01,
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
# evaluation_datasets_list = [librispeech_test_dataset[:10], ours_test_dataset[:10]]
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

# class SALMTrainer(HFTrainer):
#     def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
#         prepared = model.prepare_inputs(inputs)  # -> input_embeds, attention_mask, target_ids
#         outputs = model(prepared["input_embeds"], attention_mask=prepared["attention_mask"])
#         logits = outputs["logits"]
#         num_frames = (prepared["target_ids"] != -100).long().sum()
#         loss = F.cross_entropy(
#             logits.flatten(0, 1),
#             prepared["target_ids"].flatten(0, 1),
#             reduction="sum",
#             ignore_index=-100,
#         ) / num_frames.clamp_min(1)
#         return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    with torch.no_grad():
        preds, labels = eval_pred   # preds: (B, T, V) or numpy
        if isinstance(preds, torch.Tensor):
            logits = preds
        else:
            logits = torch.from_numpy(preds)
        if isinstance(labels, torch.Tensor):
            refs = labels
        else:
            refs = torch.from_numpy(labels)

        pred_ids = logits.argmax(dim=-1)         # (B, T)
        mask = refs != -100
        if mask.sum().item() == 0:
            acc = 0.0
        else:
            acc = (pred_ids[mask] == refs[mask]).float().mean().item()
        return {"eval_acc": acc}


class SALMTrainer(HFTrainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # ✅ 병렬 래퍼 제거(DDP/DP/FSDP/Deepspeed 등 모두 안전)
        # core = self.accelerator.unwrap_model(model)

        # # NeMo 방식 전처리 -> forward
        # prepared = core.prepare_inputs(inputs)  # -> input_embeds, attention_mask, target_ids
        # outputs  = core(prepared["input_embeds"], attention_mask=prepared["attention_mask"])
        # logits   = outputs["logits"]
        
        core = self.accelerator.unwrap_model(model)

        # NeMo 방식 전처리 -> forward
        prepared = inputs
        outputs  = core(prepared["input_embeds"], attention_mask=prepared["attention_mask"])
        logits   = outputs["logits"]

        # NeMo training_step와 동일한 CE loss
        num_frames = (prepared["target_ids"] != -100).long().sum()
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            prepared["target_ids"].flatten(0, 1),
            reduction="sum",
            ignore_index=-100,
        ) / num_frames.clamp_min(1)

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        - prepared 입력( input_embeds, attention_mask, target_ids )을 받아
          SALM의 validation_step과 동일하게 토큰 평균 CE loss와 logits/labels를 반환.
        - collator가 raw 4키(audios, audio_lens, input_ids, loss_mask)를 주는 구조로 바꾸더라도
          주석의 한 줄만 바꾸면 그대로 동작하도록 안내 포함.
        """
        core = self.accelerator.unwrap_model(model)
        device = next(core.parameters()).device

        # ── 1) (지금 구조) collator가 이미 prepared_dict를 주는 경우 ──
        prepared = {
            "input_embeds": inputs["input_embeds"].to(device, non_blocking=True),
            "attention_mask": inputs["attention_mask"].to(device, non_blocking=True),
            "target_ids": inputs["target_ids"].to(device, non_blocking=True),
        }

        # ── 만약 collator가 raw 4키를 주도록 바꾼다면, 위 블록 대신 아래 한 줄로 대체하세요:
        # for k in ("audios","audio_lens","input_ids","loss_mask"): inputs[k] = inputs[k].to(device, non_blocking=True)
        # prepared = core.prepare_inputs(inputs)

        with torch.no_grad():
            forward_outputs = core(
                prepared["input_embeds"],
                attention_mask=prepared["attention_mask"],
            )
            logits = forward_outputs["logits"]                 # (B, T, V)
            labels = prepared["target_ids"]                    # (B, T)

            # SALM: 토큰 평균 CE loss (ignore_index=-100)
            num_frames = (labels != -100).long().sum()
            loss = F.cross_entropy(
                logits.flatten(0, 1),          # (*, V)
                labels.flatten(0, 1),          # (*)
                reduction="sum",
                ignore_index=-100,
            ) / num_frames.clamp_min(1)

        if prediction_loss_only or self.compute_metrics is None:
            # HF는 (loss, logits, labels)를 기대함
            return (loss, None, None)

        # compute_metrics에서 정확도 등 계산할 수 있도록 logits/labels 반환
        return (loss, logits.detach(), labels.detach())
    
    def save_model(self, output_dir=None, _internal_call=True):
        out = output_dir or self.args.output_dir
        os.makedirs(out, exist_ok=True)

        # (선택) 토크나이저 저장
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "save_pretrained"):
            self.model.tokenizer.save_pretrained(out)

        # ✅ LoRA 어댑터만 저장 + auto_mapping_dict 제거
        if hasattr(self.model, "llm") and hasattr(self.model.llm, "save_pretrained"):
            self.model.llm.save_pretrained(
                os.path.join(out, "lora_adapters"),
                selected_adapters=["new_ft"],
            )
        else:
            super().save_model(out, _internal_call)

class NeMoBatchCollator:
    """
    SALM.prepare_inputs가 요구하는 4개 키를 직접 만들어 준다:
      - audios: FloatTensor (B, T_samples)
      - audio_lens: LongTensor (B,)
      - input_ids: LongTensor (B, T_tokens)  *오디오 위치는 audio_locator_tag_id가 들어 있어야 함*
      - loss_mask: BoolTensor (B, T_tokens)
    """
    def __init__(self, model, datasets_list=None, prompt_template=None, max_length=2048):
        self.model = model.to('cpu')
        self.tok = model.tokenizer
        self.audio_tag = model.audio_locator_tag
        self.audio_tag_id = model.audio_locator_tag_id
        self.pad_id = model.text_pad_id
        self.sr = getattr(model, "sampling_rate", 16000)
        self.datasets_list = datasets_list
        self.max_length = max_length
        # self.prompt_template = prompt_template or (lambda text, tag: f"Transcribe the following: {tag}")
        self.prompt_template = prompt_template or (lambda text, tag: f"/nothink Transcribe the following: {tag}")
        self._has_chat = hasattr(self.tok, "apply_chat_template")

    def _load_audio_batch(self, paths):
        waves, lens = [], []
        for p in paths:
            wav, sr = torchaudio.load(p)              # (C, T)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = wav.mean(0)                         # mono (T,)
            waves.append(wav)
            lens.append(wav.shape[0])
        max_len = max(lens)
        B = len(waves)
        audios = torch.zeros(B, max_len, dtype=torch.float32)
        for i, w in enumerate(waves):
            audios[i, : w.shape[0]] = w
        return audios, torch.tensor(lens, dtype=torch.long)

    def _encode_prompts(self, texts):
        """PromptFormatter 없이: 문자열→ids, 좌측패딩 직접 구현."""
        ids_list = []
        for t in texts:
            if self._has_chat:
                chat = [{"role": "user", "content": self.prompt_template(t, self.audio_tag)}]
                s = self.tok.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            else:
                s = self.prompt_template(t, self.audio_tag)

            if hasattr(self.tok, "text_to_ids"):
                ids = torch.tensor(self.tok.text_to_ids(s), dtype=torch.long)
            elif hasattr(self.tok, "encode"):
                ids = torch.tensor(self.tok.encode(s), dtype=torch.long)
            else:
                raise RuntimeError("Tokenizer cannot tokenize string; provide text_to_ids or encode.")
            ids_list.append(ids[: self.max_length])

        # left pad (오른쪽 정렬)
        maxlen = max(x.numel() for x in ids_list)
        B = len(ids_list)
        input_ids = torch.full((B, maxlen), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(ids_list):
            L = seq.numel()
            input_ids[i, -L:] = seq
        return input_ids

    def __call__(self, samples):
        # 학습: [(dataset_idx, local_idx), ...]  /  평가: [{'audio_path':..., 'text':...}, ...]
        is_training_format = isinstance(samples[0], tuple)
        if is_training_format:
            if self.datasets_list is None:
                raise ValueError("datasets_list required for training format.")
            actual = [self.datasets_list[di][li] for (di, li) in samples]
        else:
            actual = samples

        audio_paths = [s["audio_path"] for s in actual]
        texts       = [s["text"]       for s in actual]

        audios, audio_lens = self._load_audio_batch(audio_paths)
        input_ids = self._encode_prompts(texts)

        attention_mask = (input_ids != self.pad_id)
        # 오디오 placeholder 토큰은 supervised 대상 아님 → 마스킹 제외
        loss_mask = attention_mask & (input_ids != self.audio_tag_id)

        with torch.no_grad():
            prepared_dict =  {
                "audios": audios.to(self.model.llm.device),
                "audio_lens": audio_lens.to(self.model.llm.device),
                "input_ids": input_ids.to(self.model.llm.device),
                "loss_mask": loss_mask.to(self.model.llm.device),
            }
            
            prepared_dict = self.model.prepare_inputs(prepared_dict)
            prepared_dict = {k: v.to('cpu') for k, v in prepared_dict.items()}
        
        return prepared_dict


balanced_iterable_dataset = BalancedIterableDataset(training_datasets_list)
# evaluation_datasets_list = [librispeech_test_dataset.select(range(10))]
eval_dataset = concatenate_datasets(evaluation_datasets_list)

# data_collator = NeMoBatchCollator(
#     tokenizer=model.tokenizer,
#     audio_tag_id=model.audio_locator_tag_id,
#     sampling_rate=SR,
#     datasets_list=training_datasets_list,
#     # 필요하면 prompt_template=lambda text, tag: f"Transcribe: {tag}\n\n{some_prefix}"
# )

BATCH_SIZE = 4
print("\n데이터 파이프라인 설정 완료.")

# --- 7. 훈련 인자 (TrainingArguments) 설정 ---
print("\n--- 훈련 인자 설정 중 ---")
training_args = TrainingArguments(
    output_dir='/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result_v2', # Updated output directory
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    warmup_steps=500,
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    # eval_steps=1,
    per_device_eval_batch_size=4,
    # eval_accumulation_steps=4,
    # dataloader_num_workers=4,
    dataloader_num_workers=0,
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
trainer = SALMTrainer(
    model=model,                         # SALM 인스턴스
    args=training_args,
    train_dataset=balanced_iterable_dataset,
    eval_dataset=eval_dataset,
    data_collator=NeMoBatchCollator(model, datasets_list=training_datasets_list),
    # compute_metrics=compute_metrics,
)

print("\n--- 파인튜닝 시작 ---")
train_result = trainer.train()

print("\n--- 파인튜닝 완료 ---")
output_lora_path = os.path.join('/data3/gkook/temp/agi/canary-qwen-2.5b_ft_result_v2/final', "lora_adapters")
# model.save_pretrained(output_lora_path)
# print(f"학습된 LoRA 어댑터가 '{output_lora_path}'에 저장되었습니다.")

print("\n--- 최종 평가 실행 중 ---")
eval_results = trainer.evaluate()
trainer.log_metrics("eval", eval_results)
trainer.save_metrics("eval", eval_results)
print(f"최종 학습 결과: {train_result.metrics}")
print(f"최종 평가 결과: {eval_results}")

model.llm.save_pretrained(
    output_lora_path,
    selected_adapters=["new_ft"],
    # 선택: 특정 어댑터만 저장하려면
    # selected_adapters=[model.llm.active_adapter]  # 필요시
)
print(f"LoRA 어댑터만 '{output_lora_path}'에 저장했습니다.")

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state() # This will save the trainer state in output_dir
print(f"최종 학습 결과: {metrics}")



wandb.finish()
print("Weights & Biases 실행이 종료되었습니다.")
