from leetcode_dataset.environment import LeetCodeEnv
from leetcode_dataset.types import LeetCodeSubmission, ProgrammingLanguage
from leetcode_dataset.utils.formatting import PythonSubmissionFormatter
import argparse
import jsonlines
from tqdm import tqdm
import os 
import json 
import time 
import numpy as np
import re 
from utils import *

def main(code, question_slug):
    sub = LeetCodeSubmission(code=code,
                            lang=ProgrammingLanguage.PYTHON3,
                            question_slug=question_slug)
    env = LeetCodeEnv()
    status, reward, done, submission_result = env.step(sub)
    return status, reward, done, submission_result
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation')
    parser.add_argument('--data_path', type=str, 
                        default='./data/leetcode_test.jsonl')
    parser.add_argument('--solutions_path', type=str, 
                        default="./output/WizardCoder-15B-V1.0/NORMAL_FORMAT_PROMPT_lc-5k-normal/leetcode_test_generations.jsonl")
    parser.add_argument('--save_path', type=str, 
                        default="./output/WizardCoder-15B-V1.0/NORMAL_FORMAT_PROMPT_lc-5k-normal/leetcode_test_results_easy.jsonl")
    parser.add_argument('--level', type=str, choices=['easy', 'medium', 'hard'],
                        default="easy")
    args = parser.parse_args()

    print(f'Data path: {args.data_path}')
    print(f'Solution path: {args.solutions_path}')
    print(f'Save path: {args.save_path}')
    print(f'Level: {args.level}')

    dataset = []
    solutions = []
    for data, solution in zip(
        read_jsonl(args.data_path), read_jsonl(args.solutions_path)):
        if data['level'] == args.level:
            dataset.append(data)
            solutions.append(solution)

    start_id = 0
    if os.path.exists(args.save_path):
        start_id = len(read_jsonl(args.save_path))
    print(f'Start from {start_id}, total: {len(dataset)}')

    for data, solution in tqdm(zip(dataset[start_id:], solutions[start_id:])):
        output = unwrap(solution['output'][0])
        
        if not output.startswith('def'):
            output = data['instruction'] + output

        code = PythonSubmissionFormatter.to_leetcode(output)
        status, reward, done, submission_result = main(code, question_slug=data['id'])
        
        print(f'{status}')
        write_jsonl(args.save_path, [submission_result], append=True)

        # wait for 5s to avoid 'Too many requests'
        time.sleep(5)
        
    last_line = read_jsonl(args.save_path)[-1]
    if 'avg_accuracy' not in last_line:
        rewards = read_jsonl(args.save_path)
        avg_accuracy = []
        strict_accuracy = []
        for i, reward in enumerate(rewards):
            if reward['state'] != "SUCCESS":
                avg_accuracy.append(0.0)
                strict_accuracy.append(False)
            else:
                avg_accuracy.append(reward['total_correct']/reward['total_testcases'])
                strict_accuracy.append(reward['status_msg'] == 'Accepted')
        result = {
            'avg_accuracy': np.mean(avg_accuracy),
            'strict_accuracy': np.mean(strict_accuracy),
        }
        write_jsonl(args.save_path, [result], append=True)

    else:
        result = last_line

    print(f"avg_accuracy: {result['avg_accuracy']*100:.2f}, strict_accuracy: {result['strict_accuracy']*100:.2f}")

        
        