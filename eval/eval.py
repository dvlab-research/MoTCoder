import json
import argparse
import re
from utils import compute_metrics
import os 
from utils import extract_code_blocks
from tqdm import tqdm 

def read_data_jsonl(file_path):
    """Read data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc='loading'):
            item = json.loads(line.strip())

            item['gen'] = [extract_code_blocks(gen) for gen in item['gen']]
            if 'fix_gen' in item:
                item['fix_gen'] = [extract_code_blocks(gen) for gen in item['fix_gen']]

            # try:
            #     in_outs = json.loads(item["input_output"])
            # except ValueError:
            #     # import pdb;pdb.set_trace()
            #     in_outs = eval(item["input_output"])
            #     in_outs = {
            #         'inputs': in_outs['input'],
            #         'outputs': in_outs['output']
            #     }
            #     item["input_output"] = json.dumps(in_outs)
            data.append(item)
            # if len(item['gen']) < 5:
            #     import pdb;pdb.set_trace()
            # print(item['difficulty'])
            # import pdb;pdb.set_trace()
    return data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate code generations with specified metrics.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input JSONL data file.')
    parser.add_argument('--save_path', type=str, help='Path to save evaluation metrics.')
    parser.add_argument('--results_path', type=str, help='Path to save evaluation results.')
    parser.add_argument('--levels', type=str, nargs='+', default=['interview', 'competition', 'introductory', 'test', 'valid'], help='Difficulty levels to evaluate.')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--k_list', type=int, nargs='+', default=[1], help='')
    args = parser.parse_args()

    # Read data
    metrics = {}
    generations = read_data_jsonl(args.data_path)
    if args.save_path is not None and os.path.exists(args.save_path):
        with open(args.save_path, 'r', encoding='utf-8') as file:
            metrics = json.load(file)

    for i, level in enumerate(args.levels):
        results, metrics[level] = compute_metrics(generations, level=level, k_list=args.k_list, debug=args.debug)
        print(level, metrics[level])

        if args.results_path:
            with open(args.results_path, 'w' if i == 0 else 'a', encoding='utf-8') as file:
                for index in range(len(results)):
                    json.dump(results[index], file)
                    file.write('\n') 
                    # import pdb;pdb.set_trace()
            
    print(f"Results saved to {args.results_path}")

    if args.save_path:
        with open(args.save_path, 'w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {args.save_path}")
    
if __name__ == '__main__':
    main()
