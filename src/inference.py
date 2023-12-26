import sys
import os
import time
from datetime import datetime
from tqdm import tqdm 
import fire
import torch
import transformers
import json
import jsonlines
from utils import *
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re 
from config import *

def evaluate(
        prompts,
        tokenizer,
        model,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    inputs = tokenizer(
        prompts, return_tensors="pt", max_length=max_new_tokens, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to('cuda')
    generation_config = GenerationConfig(
        temperature=temperature,
        # top_p=top_p,
        # top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output

def main(
    base_model: str = "Model_Path",
    input_data_path = "Input.jsonl",
    output_data_path = "Output.jsonl",
    prompt_type = None,
    k: str = 1,
    load_8bit: bool = False,
    overwrite: bool = False,
):
    print(f'start: {datetime.now()}')
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()

    model.eval()
    model = torch.compile(model)

    results = []
    if os.path.exists(output_data_path):
        results = read_jsonl(output_data_path)
    start_id = len(results)
    print(f'Saving to {output_data_path}, start from {start_id}')

    input_data = read_jsonl(input_data_path)
    output_data = jsonlines.open(output_data_path, mode='a')

    for num, one_data in tqdm(enumerate(input_data[start_id:])):
        prompts = generate_prompt(
            one_data["question"], 
            one_data["input"] if 'input' in one_data else None,
            prompt_type)

        final_output = []
        for i in range(k):
            _output = evaluate(prompts, tokenizer, model)
            output = _output[0].split('### RESPONSE:')[-1]
            final_output.append(output)

        new_data = {
            "id": one_data["id"],
            "question": prompts,
            "output": final_output,
            'difficulty': one_data["difficulty"],
        }

        results.append(new_data)
        output_data.write(new_data)

if __name__ == "__main__":
    fire.Fire(main)
