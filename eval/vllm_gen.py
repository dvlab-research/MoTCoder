import json
import os
from vllm import LLM, SamplingParams
import argparse
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from dataset.prompts import PROMPT
import torch 
torch.multiprocessing.set_start_method('spawn', force=True)# good solution !!!!

def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--model_id', type=str, default='/mnt/nas-alinlp/ljy/models/Qwen2.5-Coder-7B-Instruct', help='Path to the model')
    parser.add_argument('--lora_path', type=str, help='Path to the model')
    parser.add_argument('--data_path', type=str, default='/mnt/nas-alinlp/ljy/MoTCoder/data/prompts/prompts.jsonl', help='Path to the data')
    parser.add_argument('--save_path', type=str, default='/mnt/nas-alinlp/ljy/MoTCoder/data/prompts/Qwen2.5-Coder-7B-Instruct.jsonl', help='Path to save evaluation results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='')
    parser.add_argument('--key', type=str, default='prompt', help='')
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing prompts')
    parser.add_argument('--n', type=int, default=1, help='')
    args = parser.parse_args()
    return args

def process_item(item):
    query = item[args.key]
    if hasattr(tokenizer, 'app_chat_template') and callable(getattr(tokenizer, 'app_chat_template')):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": "Develop a Python solution for the provided problem. Provide only the python code, without explanations. Please wrap your code answer using ```python ```."}, 
            {"role": "user", "content": query}], 
            tokenize=False, add_generation_prompt=True
        )
    else:
        return PROMPT['test'].format(instruction=query)


def process_batch(batch, log=False):
    if args.apply_chat_template:
        with ThreadPoolExecutor() as executor:
            queries = list(executor.map(process_item, batch))
    else:
        queries = [item[args.key] for item in batch]

    if log:
        print('======================================== Example Query ==========================================')
        print(queries[0])
        # import pdb;pdb.set_trace()

    outputs = llm.generate(queries, sampling_params, lora_request=lora_request)
    results = []
    for i, (output, item) in enumerate(zip(outputs, batch)):
        if log and i == 0:
            print('======================================== Example Output ==========================================')
            print(output.outputs[0].text.strip())
        item.update({
            'call_args': {"model": args.model_id},
            'gen': [opt.text.strip() for opt in output.outputs],
        })
        results.append(item)
    return results

if __name__ == '__main__':
    args = parse_args()
    assert args.apply_chat_template

    # if os.path.exists(args.save_path) and not args.overwrite:
    #     print(f'{args.save_path} exists')
    #     import sys;sys.exit()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print('Model:', args.model_id)

    lora_request = LoRARequest("lora", 1, lora_path=args.lora_path) if args.lora_path is not None else None
    sampling_params = SamplingParams(temperature=0 if args.n==1 else 1.0, max_tokens=2048, n=args.n)
    llm = LLM(
        model=args.model_id, 
        max_model_len=8192, 
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1, #torch.cuda.device_count(),
        enable_lora=True if args.lora_path is not None else False,
    )

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    batch_size = args.batch_size
    total_batches = (len(data) + batch_size - 1) // batch_size

    with open(args.save_path, 'w', encoding='utf-8') as f:
        for i in range(total_batches):
            batch = data[i*batch_size : (i+1)*batch_size]
            results = process_batch(batch, log=True if i==0 else False)
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f'{args.save_path} saved')
