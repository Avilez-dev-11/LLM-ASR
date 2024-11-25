import os
from tqdm import tqdm
import time
import torch
import gc
import csv
import json
import transformers
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperProcessor

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from evaluate import load
from datasets import load_dataset
from accelerate import PartialState

# TODO: SHOWS HOW TO DO INFERENCE ON PEFT MODELS: https://huggingface.co/blog/peft
# :/opt/conda/lib/:/gpfs/u/home/NLUG/NLUGcbsm/local/cuda-11.6/lib64
# module load gcc
# module load cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/u/home/NLUG/NLUGcbsm/scratch/miniconda3x86/lib/
# pip install torch==1.13.0
# pip uninstall nvidia_cublas_cu11
MODEL_NAME = "whisper-large.en"
DS_SPLIT = "test-other"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# wer = load("wer")
print("DEVICE", device)


def main(
    use_lora=True,
    load_8bit: bool = True,
    base_model: str = "FreedomIntelligence/phoenix-inst-chat-7b",
    # lora_weights: str = "/gpfs/u/home/NLUG/NLUGcbsm/scratch-shared/results/5_best/checkpoint-230",
    lora_weights: str = f"./models/{MODEL_NAME}/all/checkpoint-epoch-2",
    # lora hyperparams
    # The prompt template to use, will default to alpaca.
    prompt_template: str = "phoenix",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='FreedomIntelligence/phoenix-inst-chat-7b'"
    print("LOCAL RANK", int(os.environ.get("LOCAL_RANK", 0)))

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir="cache/transformers"
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map={'': local_rank},
            cache_dir="cache/transformers",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={'': local_rank},
                torch_dtype=torch.float16,
            )
        print("DEVICE MAP", model.hf_device_map)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float32,
            cache_dir="cache/transformers",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        print("MPSSS")
    else:
        print("CPUUU")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            cache_dir="cache/transformers"
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
            # model = get_peft_model(model, peft_config)
        print("DEVICE MAP", model.hf_device_map)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    distributed_state = PartialState()
    model.to(distributed_state.device)

    def evaluate(
        instruction,
        input=None,
        temperature=0.7,
        # temperature=0.7,
        top_p=0.75,
        # top_k=40,
        # do_sample=True,
        num_beams=4,
        max_new_tokens=512,
        stream_output=False,
        prompt_template: str = "phoenix",
        **kwargs,
    ):
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(f"cuda:{local_rank}")
        generation_config = GenerationConfig(
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break
                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # yield output
        yield prompter.get_response(output)

    def find_model_predictions(prompts_and_references, instruction):
        predictions = []
        processor = WhisperProcessor.from_pretrained(
            f"openai/{MODEL_NAME}", cache_dir="cache/transformers")

        i = 0
        num_mistakes = 0
        with distributed_state.split_between_processes(prompts_and_references) as prompts_and_references_per_process:
            print("NUM PROMPTS PER PROCESS", len(
                prompts_and_references_per_process))
            for prompt, reference in tqdm(prompts_and_references_per_process):
                # seconds = time.time()
                for attempt in range(3):
                    try:
                        for model_output in evaluate(instruction, prompt):
                            model_output = processor.tokenizer._normalize(
                                model_output.replace('</s>', ''))
                            print("MODEL OUTPUT", model_output,
                                  model_output == reference, num_mistakes)
                            print("REFERENCE", reference)
                            if model_output != reference:
                                num_mistakes += 1
                            # if i % 10 == 0 and i > 0:
                            #     print("RUNNING WER:", wer.compute(predictions=a, references=b))
                            predictions.append({
                                "prediction": model_output,
                                "reference": reference
                            })
                            # a.append(model_output)
                            # b.append(reference)

                        # if i % 500 == 0:
                        #     with open(f"final-predictions2/{PARTIAL_AMOUNT}/{DS_SPLIT}-checkpoint-{i}.json", "w") as pred_file:
                        #         json.dump(predictions, pred_file, indent=2)
                        i += 1
                    # model_output = evaluate(instruction, prompt)
                    except Exception as e:
                        print("ERROR:", e)
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        break
	    # epoch_times.append(seconds)
        # print("Average time per iteration: ", epoch_times/len((epoch_times)))

        return predictions

    def get_prompts_references_instruction(model_name, ds_split):
        ds = load_dataset("json", data_files=f"input-data/{model_name}/{
                          ds_split}.json", cache_dir="cache/datasets")["train"]
        prompts = ds["input"]
        references = ds["output"]
        instruction = ds["instruction"][0]
        print("PROMPTS", prompts[:10])

        return prompts, references, instruction

    prompts, references, instruction = get_prompts_references_instruction(
        MODEL_NAME, DS_SPLIT)
    prompts_and_references = [(prompt, reference)
                              for prompt, reference in zip(prompts, references)]
    predictions = find_model_predictions(prompts_and_references, instruction)
    # predictions_formatted = [{"prediction": prediction} for prediction in predictions]
    with open(f"output-data/{MODEL_NAME}/all/{DS_SPLIT}-{local_rank}.json", "w") as pred_file:
        json.dump(predictions, pred_file, indent=2)
    cer = load("cer")
    cer_ = 100 * cer.compute(predictions=[item["prediction"] for item in predictions],
                             references=[item["reference"] for item in predictions])
    print(f"CER: {ser_:.2f}%")
    wer = load("wer")
    wer_ = 100 * wer.compute(predictions=[item["prediction"] for item in predictions],
                             references=[item["reference"] for item in predictions])
    print(f"WER: {wer_:.2f}%")


if __name__ == "__main__":
    main()
