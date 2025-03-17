import json
from datasets import load_dataset, Dataset, concatenate_datasets
import os 
from tqdm import tqdm 
import uuid
import re


def cc2apps(item, problem_id, difficulty, correct=False):
    input_output = {
        'inputs': item['generated_tests']['input'],
        'outputs': item['generated_tests']['output']
    }
    key = 'solutions' if not correct else 'incorrect_solutions'
    solutions = []
    for lan, sol in zip(item[key]['language'], item[key]['solution']):
        if lan in [3]: # python, python3
            solutions.append(sol)
            if len(solutions) >= 100:
                break
    converted_item = {
        'problem_id': problem_id,
        'question': item['description'],
        'solutions': json.dumps(solutions),
        'input_output': json.dumps(input_output),
        'difficulty': difficulty,
        'url': json.dumps(item['source']), 
        'starter_code': ''
    }
    return converted_item


def load_data(dname='cc', split='test', correct=False):
    print(f'=> Loading {split} Dataset {dname}')
    if dname in ['apps', 'all']:
        apps = load_dataset(APPS_PATH, split=split)
        # ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code']
        if dname == 'apps':
            return apps

    if dname in ['cc', 'all']:
        if split == 'train':
            cc_datasets = {
                'cc': load_dataset(CC_PATH, split="train"),
            }
        else:
            cc_datasets = {
                'test': load_dataset(CC_PATH, split="test"),
                'valid': load_dataset(CC_PATH, split="valid")
            }
        '''
        ['name', 'description', 'public_tests', 'private_tests', 'generated_tests', 'source', 
        'difficulty', 'solutions', 'incorrect_solutions', 'cf_contest_id', 'cf_index', 'cf_points', 
        'cf_rating', 'cf_tags', 'is_description_translated', 'untranslated_description', 'time_limit', 
        'memory_limit_bytes', 'input_file', 'output_file']
        '''

        converted_data = []
        i = 5000
        for key, cc_dataset in cc_datasets.items():
            for j, item in enumerate(tqdm(cc_dataset, desc=f'Converting {key}')):
                if split == 'train':
                    try:
                        from overlap import cc_overlap
                    except:
                        from .overlap import cc_overlap
                    if j in cc_overlap:
                        # import pdb;pdb.set_trace()
                        continue
                converted_item = cc2apps(item, problem_id=i, difficulty=key, correct=correct)
                converted_data.append(converted_item)
                i += 1
        
        cc = Dataset.from_list(converted_data)
        if dname == 'cc':
            return cc

    dataset = concatenate_datasets([apps, cc])
    # dataset = dataset.remove_columns(['solutions', 'url', 'starter_code'])
    print('Loaded')
    return dataset 


def save_with_uuid(dataset, output_file_path, split='test'):
    from prompts import PROMPT
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in tqdm(dataset, desc=f'Saving to {output_file_path}'):
            if split in ['label', 'correct_label', 'clean', 'mot']:
                solutions = json.loads(item['solutions'])
                for i, solution in enumerate(solutions):
                    prompt = PROMPT[split].format(instruction=item['question'], code=solution)
                    save_item = {
                        'uuid': f"{item['problem_id']}-{i}",
                        'prompt': prompt,
                    }
                    outfile.write(json.dumps(save_item) + '\n')    

            elif split in ['test', 'query']:
                prompt = PROMPT[split].format(instruction=item['question'])
                item.update({
                    'uuid': item['problem_id'],
                    'prompt': prompt
                })
                outfile.write(json.dumps(item) + '\n')

            elif split in ['alpaca']:
                solutions = json.loads(item['solutions'])
                for i, solution in enumerate(solutions):
                    save_item = {
                        'instruction': item['question'],
                        'output': solution,
                    }
                    outfile.write(json.dumps(save_item) + '\n')    

            else:
                raise NotImplementedError(split)
                
    print('Saved')


if __name__ == '__main__':

    CC_PATH = "data/code_contests"
    APPS_PATH = "data/apps"

    cc = load_data('cc', split='test')
    apps = load_data('apps', split='test')

    output_file_path = 'data/prompts/queries.jsonl'
    dataset = load_data('all', split='test')
    save_with_uuid(dataset, output_file_path, split='query')

    output_file_path = 'data/prompts/queries_cc.jsonl'
    dataset = load_data('cc', split='test')
    save_with_uuid(dataset, output_file_path, split='query')
