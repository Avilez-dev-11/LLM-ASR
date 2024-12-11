from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
 
import torch
import evaluate

from pathlib import Path

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(batch):
    audio, sampling_rate = torchaudio.load(batch["audio_filepath"])

    if sampling_rate != 16000:
        audio = torchaudio.transforms.Resample(sampling_rate, 16000)(audio)
    
    inputs = processor(audio.squeeze().numpy(), text=batch['transcription'], return_tensors="pt", sampling_rate=16000, padding=True)
    
    input_features = inputs.input_features[0]
    labels = inputs.labels
    return {"input_features": input_features, "labels": labels}

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    save_total_limit=2,
    logging_dir="./logs",
)

train_path = Path("train-clean-100/LibriSpeech/train-clean-100")

folders = [item.name for item in train_path.iterdir() if item.is_dir()]

data = []
for i in range(5):
    cur_path = train_path / folders[i]

    for num_folder in cur_path.iterdir():
        sub_folder_path = cur_path / num_folder.name
        txt_file = next(sub_folder_path.glob("*.txt"))

        with txt_file.open('r') as file:
            lines = [ line.split(" ", 1) for line in file]
            print(lines)



