import json 
import os 
import csv
import re 

class Color:
    Pink = '\033[95m'  
    Blue = '\033[94m'  
    Green = '\033[92m'  
    Yellow = '\033[93m'  
    Red = '\033[91m'  
    Endc = '\033[0m' 

def has_non_english_characters(input_string):
    return any(ord(char) > 127 for char in input_string)

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        output = json.load(json_file)
    return output

def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    return file_contents

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data, append=False):
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def unwrap(output):
    if "```python" in output:
        match = re.search("```python([\s\S]*?)```", output, re.DOTALL)
        if match:
            output = match.group(1)
        else:
            output = output.split('```python')[-1]
    return output
