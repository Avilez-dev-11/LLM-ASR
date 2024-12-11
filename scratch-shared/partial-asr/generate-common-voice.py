import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from accelerate import PartialState

# Constants
MODEL_NAME = "whisper-large.en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CV_DATASET_DIR = "/Users/zuriel.a1/mounts/cci-home/scratch-shared/partial-asr/datasets/cv-corpus-19.0-2024-09-13"
CLIPS_DIR = os.path.join(CV_DATASET_DIR, "clips")
# Adjust if metadata file differs
TSV_FILE = os.path.join(CV_DATASET_DIR, "validated.tsv")
OUTPUT_DIR = "/Users/zuriel.a1/mounts/cci-home/scratch-shared/partial-asr/common-voice"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_common_voice(tsv_path, clips_dir):
    """
    Load Common Voice dataset into a format compatible with Whisper.
    """
    data = []
    with open(tsv_path, "r", encoding="utf-8") as tsv_file:
        lines = tsv_file.readlines()[1:]  # Skip header
        for line in lines:
            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue
            file_name, transcript = fields[0], fields[1]
            audio_path = os.path.join(clips_dir, file_name)
            if os.path.exists(audio_path):
                data.append({"audio_path": audio_path, "text": transcript})
    return Dataset.from_list(data)


def generate_whisper_outputs(dataset, model, processor, device):
    """
    Generate transcriptions for a dataset using Whisper.
    """
    predictions = []
    for example in tqdm(dataset):
        try:
            # Load audio and preprocess
            audio_input = processor(
                example["audio_path"], sampling_rate=16000, return_tensors="pt", sampling_rate_target=16000
            ).input_features
            audio_input = audio_input.to(device)

            # Generate transcription
            with torch.no_grad():
                output_ids = model.generate(audio_input, max_length=256)
            transcription = processor.batch_decode(
                output_ids, skip_special_tokens=True)[0]

            # Collect results
            predictions.append({
                "audio_path": example["audio_path"],
                "reference": example["text"],
                "prediction": transcription,
            })

        except Exception as e:
            print(f"Error processing {example['audio_path']}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    return predictions


def main():
    # Load model and processor
    processor = WhisperProcessor.from_pretrained(
        f"openai/{MODEL_NAME}", cache_dir="cache/transformers")
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/{MODEL_NAME}", cache_dir="cache/transformers")
    model.to(DEVICE)

    # Load dataset
    dataset = load_common_voice(TSV_FILE, CLIPS_DIR)

    # Generate outputs
    print(f"Processing dataset of size: {len(dataset)}")
    predictions = generate_whisper_outputs(dataset, model, processor, DEVICE)

    # Save predictions to file
    output_file = os.path.join(OUTPUT_DIR, f"common_voice_predictions.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
