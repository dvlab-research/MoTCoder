from typing import Dict
import transformers
from examples import *

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|endoftext|>"

NORMAL_FORMAT_PROMPT = """You are an expert Python programmer. You will be given a question (problem specification) and some Input-Output pairs. Generate a correct Python program that matches the specification and passes all tests. Your code reads Input with function 'input()' and returns Output with function 'print()'. Wrap your code using: 
```python
if __name__ == "__main__":
    # YOUR CODE HERE
```.
### QUESTION:
{instruction}
### RESPONSE:"""

NORMAL_FORMAT_PROMPT_CODELLAMA = """You are an expert Python programmer. You will be given a question (problem specification) and some Input-Output pairs. Generate a correct Python program that matches the specification and passes all tests. Your code reads Input with function 'input()' and returns Output with function 'print()'. Wrap your code using: 
[PYTHON]
if __name__ == "__main__":
    # YOUR CODE HERE
[/PYTHON].
### QUESTION:
{instruction}
### RESPONSE:"""

FORMAT_PROMPT = """You are an expert Python programmer. You will be given a question (problem specification) and some Input-Output pairs. Generate a correct Python program that matches the specification and passes all tests. Your code reads Input with function 'input()' and returns Output with function 'print()'. You will also be given a set of related utility Python functions. Try to complete and reuse them into your solution. Wrap your code using: 
```python
if __name__ == "__main__":
    # YOUR CODE HERE
```.
### QUESTION:
{instruction}
### RELEVANT FUNCTIONS:
{input}
### RESPONSE:
"""

FORMAT_PROMPT_CODELLAMA = """You are an expert Python programmer. You will be given a question (problem specification) and some Input-Output pairs. Generate a correct Python program that matches the specification and passes all tests. Your code reads Input with function 'input()' and returns Output with function 'print()'. You will also be given a set of related utility Python functions. Try to complete and reuse them into your solution. Wrap your code using: 
[PYTHON]
if __name__ == "__main__":
    # YOUR CODE HERE
[/PYTHON].
### QUESTION:
{instruction}
### RELEVANT FUNCTIONS:
{input}
### RESPONSE:
"""

NORMAL_PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. 

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### QUESTION:
{instruction}
### RESPONSE:"""

MODULAR_PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures. Start by outlining the required code modules, including function headers and signatures. Subsequently, proceed to implement each module to create the final code.

In simpler terms, create a clean and organized Python solution for the given problem. Break it down into smaller parts (modules) with clear function names and input/output specifications. Once the structure is ready, write the actual code for each module to complete the solution.

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### TASK:
{instruction}
### RESPONSE:
"""

PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures. Given a set of related utility Python functions, try to reuse or adapt them as much as possible into your solution (create new unique functions if needed). Start by outlining the required code modules, including function headers and signatures. Subsequently, proceed to implement each module to create the final code.

In simpler terms, create a clean and organized Python solution for the given problem. Break it down into smaller parts (modules) with clear function names and input/output specifications. Once the structure is ready, write the actual code for each module to complete the solution.

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### TASK:
{instruction}
### RELEVANT FUNCTIONS:
{input}
### RESPONSE:
"""

ONE_SHOT_NORMAL_PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. 

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### Example 1
### QUESTION:
{example_question}
### RESPONSE:
{example_code}
-----------------
### Example 2
### QUESTION:
{instruction}
### RESPONSE:""".format(
    example_question=example_question, 
    example_code=example_code, 
    instruction='{instruction}')

ONE_SHOT_NORMAL_PROMPT_LEETCODE = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. 

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### Task 1
### QUESTION:
{example_question}
### RESPONSE:
{example_question}
-----------------
### Task 2
### QUESTION:
{instruction}
### RESPONSE:""".format(
    example_question=example_question_leetcode, 
    example_code=example_code_leetcode, 
    instruction='{instruction}')

ONE_SHOT_MODULAR_PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures. Start by outlining the required code modules, including function headers and signatures. Subsequently, proceed to implement each module to create the final code.

In simpler terms, create a clean and organized Python solution for the given problem. Break it down into smaller parts (modules) with clear function names and input/output specifications. Once the structure is ready, write the actual code for each module to complete the solution.

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### Example 1
### TASK:
{example_question}
### RESPONSE:
STEP 1: GENERATE SUB-MODULES:
{example_modules}
STEP 2: GENERATE PYTHON CODE:
{example_modular_code}
-----------------
### Example 2
### TASK:
{instruction}
### RESPONSE:
""".format(
    example_modules=example_modules, 
    example_question=example_question, 
    example_modular_code=example_modular_code,
    instruction='{instruction}')

NONE = '{instruction}    # complete the code here'

ONE_SHOT_PROMPT = """Develop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures. Given a set of related utility Python functions, try to reuse or adapt them as much as possible into your solution (create new unique functions if needed). Start by outlining the required code modules, including function headers and signatures. Subsequently, proceed to implement each module to create the final code.

In simpler terms, create a clean and organized Python solution for the given problem. Break it down into smaller parts (modules) with clear function names and input/output specifications. Once the structure is ready, write the actual code for each module to complete the solution.

The output code needs to follow standard input streams. Please wrap your code answer using ```.
### Example 1
### TASK:
{example_question}
### RELEVANT FUNCTIONS:
{example_modules}
### RESPONSE:
{example_modular_code}
-----------------
### Example 2
### TASK:
{instruction}
### RELEVANT FUNCTIONS:
{input}
### RESPONSE:
""".format(
    example_modules=example_modules, 
    example_question=example_question, 
    example_modular_code=example_modular_code,
    instruction='{instruction}',
    input='{input}')

def generate_prompt(instruction, input, prompt_type=None):
    prompts = {
        'PROMPT': PROMPT,
        'MODULAR_PROMPT': MODULAR_PROMPT,
        'ONE_SHOT_PROMPT': ONE_SHOT_PROMPT,
        'ONE_SHOT_NORMAL_PROMPT': ONE_SHOT_NORMAL_PROMPT,
        'ONE_SHOT_MODULAR_PROMPT': ONE_SHOT_MODULAR_PROMPT,
        'FORMAT_PROMPT': FORMAT_PROMPT,
        'NORMAL_FORMAT_PROMPT': NORMAL_FORMAT_PROMPT,
        'NORMAL_PROMPT': NORMAL_PROMPT,
        'NORMAL_FORMAT_PROMPT_CODELLAMA': NORMAL_FORMAT_PROMPT_CODELLAMA,
        'FORMAT_PROMPT_CODELLAMA': FORMAT_PROMPT_CODELLAMA,
        'NONE': NONE, 
        'ONE_SHOT_NORMAL_PROMPT_LEETCODE': ONE_SHOT_NORMAL_PROMPT_LEETCODE,
    }
    output = prompts[prompt_type].format(instruction=instruction, input=input)
    return output

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
