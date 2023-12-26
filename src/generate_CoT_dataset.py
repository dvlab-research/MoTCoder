from datasets import load_dataset
from tqdm import tqdm
import re
import random
from utils import *
import argparse

system_prompt = """Develop a well-structured Python solution for the provided original solution. Ensure modularity and considering potential edge cases and failures. Start by outlining the required code modules, including function headers and signatures. Subsequently, proceed to implement each module to create the final code.

In simpler terms, create a clean and organized Python solution for the given original solution. Break it down into smaller parts (modules) with clear function names and input/output specifications. Once the structure is ready, write the actual code for each module to complete the solution.

You will NOT return anything except for the program. Please wrap your code answer using ```.

### Example 1
### ORIGINAL CODE:
{original_code}
### RESPONSE:
{example_modularized_code}
"""

original_code = """```python
for _ in range(int(input())):
    n = int(input())
    mass = []
    zo = 0
    oz = 0
    zz = 0
    oo = 0
    ozs = []
    zos = []
    ozss = set()
    zoss = set()
    for j in range(n):
        k = input()
        mass.append(k)
        if k[0] == '0' and k[-1] == '1':
            zoss.add(k)
            zos.append(j + 1)
            zo += 1
        elif k[0] == '1' and k[-1] == '0':
            ozss.add(k)
            ozs.append(j + 1)
            oz += 1
        elif k[0] == '0' and k[-1] == '0':
            zz += 1
        else:
            oo += 1
    if zz and oo and not oz and not zo:
        print(-1)
        continue
    else:
        if zo > oz:
            print((zo - oz) // 2)
            ans = []
            need = (zo - oz) // 2
            i = 0
            while need:
                zzz = mass[zos[i] - 1][len(mass[zos[i] - 1]) - 1:: -1]
                if zzz not in ozss:
                    ans.append(zos[i])
                    need -= 1
                i += 1
            print(*ans)
        else:
            print((oz - zo) // 2)
            ans = []
            need = (oz - zo) // 2
            i = 0
            while need:
                zzz = mass[ozs[i] - 1][len(mass[ozs[i] - 1]) - 1:: -1]
                if zzz not in zoss:
                    ans.append(ozs[i])
                    need -= 1
                i += 1
            print(*ans)
```
"""

example_modularized_code = """STEP 1: GENERATE SUB-MODULES:
```module
def count_start_end_chars(words):
    \"\"\"
    Description: This function counts the number of words that start and end with each character.
    Input:
    words (list): A list of binary words.
    Output:
    start_count (defaultdict): A dictionary containing the count of words that start with each character.
    end_count (defaultdict): A dictionary containing the count of words that end with each character.
    \"\"\"
```

```module
def reverse_words(words, reversed_indices):
    \"\"\"
    Description: This function reverses the specified words in the given list.
    Input:
    words (list): A list of binary words.
    reversed_indices (list): A list of indices indicating the words to be reversed.
    Output:
    reversed_words (list): A new list of words with the specified words reversed.
    \"\"\"
```

STEP 2: GENERATE PYTHON CODE:
```python
import collections

def count_start_end_chars(words):
    start_count = collections.defaultdict(int)
    end_count = collections.defaultdict(int)
    for word in words:
    start_count[word[0]] += 1
    end_count[word[-1]] += 1
    return start_count, end_count

def reverse_words(words, reversed_indices):
    reversed_words = []
    for i, word in enumerate(words):
    if i in reversed_indices:
    reversed_words.append(word[::-1])
    else:
    reversed_words.append(word)
    return reversed_words

def solve_task(words):
    start_count, end_count = count_start_end_chars(words)
    characters_with_difference = []
    for char in start_count:
    if abs(start_count[char] - end_count[char]) > 1:
    characters_with_difference.append(char)
    reversed_indices = []
    for char in characters_with_difference:
    difference = abs(start_count[char] - end_count[char])
    reverse_count = difference // 2
    if start_count[char] < end_count[char]:
    indices = [i for i, word in enumerate(words) if word.startswith(char)]
    reversed_indices.extend(indices[:reverse_count])
    else:
    indices = [i for i, word in enumerate(words) if word.endswith(char)]
    reversed_indices.extend(indices[:reverse_count])
    reversed_words = reverse_words(words, reversed_indices)
    total_reversed = len(reversed_indices)
    return total_reversed, reversed_words

t = int(input())
for _ in range(t):
    n = int(input())
    words = []
    for _ in range(n):
    words.append(input())
    total_reversed, reversed_words = solve_task(words)
    print(total_reversed)
    if total_reversed != 0:
    print(*reversed_words)
```
"""

input_prompt = """
### Example 2
### ORIGINAL CODE:
{code}
### RESPONSE:
"""

def llm(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=messages,
    )
    reply = response['choices'][0]['message']['content']
    return reply

def generate_modular_dataset(dataset, save_path):
    start_id = 0
    if os.path.exists(save_path):
        start_id = read_jsonl(save_path)[-1]['id'] + 1

    print('Generating...')
    for data in tqdm(dataset[start_id:]):
        task_id = data['id']
        
        if 'solutions' in data:
            solutions = eval(data['solutions'])
        elif 'output' in data:
            if len(data['output']) > 0:
                solutions = data['output']
                solutions.extend(random.choices(solutions, k=50-len(solutions)))
            else:
                print(f'find no python solution for {task_id}')
                continue
        else:
            raise NotImplementedError(data.keys())

        for code_id, code in enumerate(solutions):
            flag = False
            system_content = system_prompt.format(
                original_code=original_code, example_modularized_code=example_modularized_code)
            input_content = input_prompt.format(code=f"```python\n{code}```")

            if len(input_content) > 4000:
                print(f'Input too long, retrying {code_id}th solution...')
                flag = True  
                continue

            messages = [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': input_content}
            ]
            # print(f'{system_content}-----------------\n{input_content}')
            generation = llm(messages).split('### RESPONSE:')[-1]
            
            # split steps
            split_generation = generation.split('STEP 2: GENERATE PYTHON CODE:')
            if len(split_generation) == 2:
                modules_generation, solution_generation = split_generation
            else:
                print(f'Failed to find two steps, retrying {code_id}th solution...')
                flag = True  
                continue

            # extract modules
            modules = re.findall(
                r'(```module|```python|```)\n([\s\S]*?)```', modules_generation)
            modules = [module for _, module in modules]

            if len(modules) == 0:
                print(f'Failed to find any modules, retrying {code_id}th solution...')
                flag = True
                continue

            for module in modules:
                if not module.strip().startswith('def'):
                    print(f'module failed includes functions only, retrying {code_id}th solution...')
                    flag = True
                    break

            data["input"] = '\n'.join(
                f"```python\n{module}```\n" for module in modules)

            # extract modular solution
            modular_solutions = re.findall(
                r'(```module|```python|```)\n([\s\S]*?)```', solution_generation)
            modular_solutions = [solution for _, solution in modular_solutions]

            if len(modular_solutions) > 1:
                print(f'Find {len(modular_solutions)} solutions, retrying {code_id}th solution...')
                flag = True
                continue

            if len(modular_solutions) == 0:
                print(f'Failed to find any generated modular solutions, retrying {code_id}th solution...')
                flag = True
                continue

            # data['modular_solutions'].append(modular_solutions[0])
            data["output"] = f"```python\n{modular_solutions[0]}```"

            # if find no bugs, continue to next problem
            if not flag:
                write_jsonl(save_path, [data], append=True)
                break

def main(args):
    dataset = read_jsonl(args.data_path)[:args.num]
    generate_modular_dataset(dataset, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CoT dataset')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--api_base', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--num', type=int, default=10000)

    import openai 
    openai.api_base = args.api_base
    openai.api_key = args.api_key
    args = parser.parse_args()
    print(f'args: {args}\n')

    main(args)
