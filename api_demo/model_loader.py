from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration
import torch
import torchaudio
from peft import PeftModel

# 모델과 프로세서 로딩
MODEL_NAME = "/data3/gkook/model/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
LORA_ADAPTER_PATH = "/data1/jc/AGI/LibriSpeech/LibriSpeech/train/LibriSpeech/qwen2_audio_finetune_output/checkpoint-18321" # 혹은 lora_adapters 폴더 경로
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


#print(f"Loading and merging LoRA adapter from {LORA_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
model = model.merge_and_unload() # 메모리 효율성을 위해 모델을 병합하고 원본 어댑터는 언로드
model.eval() # 모델을 평가 모드로 설정

# 모델은 메모리에 유지한 채 재사용
def get_model():
    return model, processor
