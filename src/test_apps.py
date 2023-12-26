import json
from evaluate import load
from utils import *
import re
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--solutions_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--level', type=str, default='all', 
                        choices=["introductory", "interview", "competition", 'all'])
    parser.add_argument('--metric_path', type=str, default="../data/apps_metric")
    args = parser.parse_args()

    print(f'args: {args}\n')

    rows = read_jsonl(args.solutions_path)
    problems = read_jsonl(args.data_path)
    if len(rows) != len(problems):
        print(f'get {len(rows)} solutions but {len(problems)} problems')

    solutions = []
    dataset = []
    for i, (row, problem) in enumerate(zip(rows, problems)):
        if args.level == 'all' or row['difficulty'] == args.level:
            solutions.append([unwrap(output) for output in row['output']])
            dataset.append(problem)
            print(solutions[0][0])
    print(f'Test level: {args.level}, amount: {len(solutions)}')

    print(f'Saving results in {args.save_path}')
    metric = load(args.metric_path)
    result = metric.compute(
        predictions=solutions, level=args.level, 
        eval_dataset=dataset, k_list=[1])
    write_json(args.save_path, result)

if __name__ == "__main__":
    main()
