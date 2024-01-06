import argparse
import json
import re
import jsonlines
from vllm import LLM, SamplingParams
import sys
import os.path as osp 
import os 
from config import generate_prompt
from utils import read_jsonl 

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test(
    model_path, data_path, solution_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, prompt_type='PROBLEM_PROMPT', gpu_memory_utilization=0.9, max_model_len=2048, local_rank=0, **kwargs
    ):
    INVALID_ANS = "[invalid]"
    ins = []
    print('prompt =====', 
        generate_prompt('<input>', '<in_lan>', '<out_lan>', prompt_type))
    dataset = read_jsonl(jsonlines.Reader(f))
    with open(data_path,"r", encoding="utf8") as f:
        for idx, item in enumerate(dataset):
            temp_instr = generate_prompt(
                item["input"],
                item['in_lan'],
                item['out_lan'],
                prompt_type)
            ins.append(temp_instr)

    if osp.exists(solution_path):
        start = len(read_jsonl(solution_path))

    ins = ins[start:end]
    print('start ====', start, 'lenght ====', len(ins))
    if len(ins) == 0:
        print(f'{solution_path} has finished !!!')
        return 0

    batch_ins = batch_data(ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len
    )

    os.makedirs(osp.dirname(osp.abspath(solution_path)),
                exist_ok=True)
    output_data = jsonlines.open(solution_path, mode='a')
    for idx, (prompt, one_data) in enumerate(zip(batch_ins, dataset)):
        if not isinstance(prompt, list):
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            output_data.write({
                "id": one_data["id"],
                "question": prompt,
                'output': generated_text,
                'difficulty': one_data["difficulty"],
            })

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)  # model path
    parser.add_argument("--data_path", type=str, default='data/test.jsonl')  # data path
    parser.add_argument("--solution_path", type=str, default=None)  # result path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=64)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--prompt_type", type=str, default='PROMPT')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.97)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parse_args())
    test(**args)
