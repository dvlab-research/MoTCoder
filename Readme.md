# MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks

<p align="center">
• 🤗 <a href="https://huggingface.co/datasets/JingyaoLi/MoTCode-Data" target="_blank">Data </a> • 🤗 <a href="https://huggingface.co/JingyaoLi/MoTCoder-15B-v1.0" target="_blank">Model </a> • 🐱 <a href="https://github.com/dvlab-research/MoTCoder" target="_blank">Code</a> • 📃 <a href="https://arxiv.org/abs/2312.15960" target="_blank">Paper</a> <br>
</p>

[![PWC](https://img.shields.io/endpoint?url=https%3A%2F%2Fpaperswithcode.com%2Fbadge%2Fmotcoder-elevating-large-language-models-with%2Fcode-generation-on-apps%3Fmetric%3DIntroductory%2520Pass%25401)](https://paperswithcode.com/sota/code-generation-on-apps?metric=Introductory%20Pass%401/motcoder-elevating-large-language-models-with) 
[![PWC](https://img.shields.io/endpoint?url=https%3A%2F%2Fpaperswithcode.com%2Fbadge%2Fmotcoder-elevating-large-language-models-with%2Fcode-generation-on-codecontests%3Fmetric%3DTest%2520Set%2520pass%25401)](https://paperswithcode.com/sota/code-generation-on-codecontests?metric=Test%20Set%20pass%401)

Large Language Models (LLMs) have showcased impressive capabilities in handling straightforward programming tasks. However, their performance tends to falter when confronted with more challenging programming problems. We observe that conventional models often generate solutions as monolithic code blocks, restricting their effectiveness in tackling intricate questions. To overcome this limitation, we present Modular-of-Thought Coder (MoTCoder). We introduce a pioneering framework for MoT instruction tuning, designed to promote the decomposition of tasks into logical sub-tasks and sub-modules. 
Our investigations reveal that, through the cultivation and utilization of sub-modules, MoTCoder significantly improves both the modularity and correctness of the generated solutions, leading to substantial relative *pass@1* improvements of 12.9% on APPS and 9.43% on CodeContests.

![MoTCoder Framework](./imgs/framework.png)

## Performance

![Performance on APPS](./imgs/impression.png)



**Performance on APPS**

| Model      | Size  | Pass@ | Introductory | Interview | Competition | All   |
|------------|-------|-------|--------------|-----------|-------------|-------|
| **CodeT5**     | 770M  | 1     | 6.60         | 1.03      | 0.30        | 2.00  |
| **GPT-Neo**    | 2.7B  | 1     | 14.68        | 9.85      | 6.54        | 10.15 |
|            |       | 5     | 19.89        | 13.19     | 9.90        | 13.87 |
| **GPT-2**      | 0.1B  | 1     | 5.64         | 6.93      | 4.37        | 6.16  |
|            |       | 5     | 13.81        | 10.97     | 7.03        | 10.75 |
|            | 1.5B  | 1     | 7.40         | 9.11      | 5.05        | 7.96  |
|            |       | 5     | 16.86        | 13.84     | 9.01        | 13.48 |
| **GPT-3**      | 175B  | 1     | 0.57         | 0.65      | 0.21        | 0.55  |
| **StarCoder**  | 15B   | 1     | 7.25         | 6.89      | 4.08        | 6.40  |
| **WizardCoder**| 15B   | 1     | 26.04        | 4.21      | 0.81        | 7.90  |
| **MoTCoder**               | 15B  | 1     | **33.80**        | **19.70**     | **11.09**       | **20.80** |
| **text-davinci-002** | - | 1 | -            | -         | -           | 7.48  |
| **code-davinci-002** | - | 1 | 29.30        | 6.40      | 2.50        | 10.20 |
| **GPT3.5**     | -     | 1     | 48.00     | 19.42     | 5.42        | 22.33 |


**Performance on CodeContests**
| Model | Size | Revision | Val pass@1 | Val pass@5 | Test pass@1 | Test pass@5 | Average pass@1 | Average pass@5 |
|-------|------|----------|------------|------------|-------------|-------------|----------------|----------------|
| **code-davinci-002** | - | - | - | - | 1.00 | - | 1.00 | - |
| **code-davinci-002 + CodeT** | - | 5 | - | - | 3.20 | - | 3.20 | - |
| **WizardCoder** | 15B | - | 1.11 | 3.18 | 1.98 | 3.27 | 1.55 | 3.23 |
| **WizardCoder + CodeChain** | 15B | 5 | 2.35 | 3.29 | 2.48 | 3.30 | 2.42 | 3.30 |
| **MoTCoder** | 15B | - | **2.39** | **7.69** | **6.18** | **12.73** | **4.29** | **10.21** |
| **GPT3.5** | - | - | 6.81 | 16.23 | 5.82 | 11.16 | 6.32 | 13.70 |

## Environment
Install the dependencies.
```bash
python -m pip install -e .
```

## Evaluation Datasets
### APPS Dataset
The APPs dataset [[github]](https://github.com/hendrycks/apps) can be download from [huggingface](https://huggingface.co/datasets/codeparrot/apps).

### CodeContests Dataset
The CodeContests dataset [[github]](https://github.com/google-deepmind/code_contests) can be download from [huggingface](https://huggingface.co/datasets/deepmind/code_contests).
For CodeContests, convert your dataset to the same format as APPs for utilizting APPs evaluation metrics:
```bash
python src/convert_codecontest_dataset.py $SRC_DIR $DST_DIR
```

## Inference
You can download our MoTCoder for evaluation from [huggingface](https://huggingface.co/JingyaoLi/MoTCoder-15B-v1.0). We provide the inference command to reproduce the results in our paper.
- If you want to use modular-of-thought inference prompt, set `prompt_type=FORMAT_PROMPT`.
- If you want to use normal inference prompt, set `prompt_type=NORMAL_FORMAT_PROMPT`.

First generate the solutions for you targeted evaluation dataset.
**Choice 1: VLLM (Recommended)**
To install the requreiments:
```bash
pip install vllm
```

Inference:
```bash
python src/inference_vllm.py \
    --model_path $model_path \
    --data_path $data_path \
    --solution_path $solution_path \
    --prompt_type $prompt_type
```

**Choice 2: transformers**
Inference:
```bash
python src/inference.py \
    $model_path \
    $data_path \
    $solution_path \
    $prompt_type
```
### APPs Evaluation
For APPs evaluation, choices of $level$ include $introductory, interview, competition$.
```bash
python src/test_leetcode.py \
    --solutions_path $solution_path \
    --data_path $data_path \
    --save_path $result_path \
    --level $level
```

### CodeContests Evaluation
```bash
python src/test_apps.py \
    --solutions_path $solution_path \
    --data_path $data_path \
    --save_path $result_path
```

## Training
### Modular-of-Thought Training Dataset
We provide an example python file to evolution a MoT dataset. 
Run the following command:
```bash
python src/generate_MoT_dataset.py \
    --data_path $data_path \
    --save_path $MoT_data_path \
    --api_base $api_base \
    --api_key $api_key
```

### MoTCode Dataset
Or, you can download our generated modular-of-thought code dataset.
```python
from datasets import load_dataset
load_dataset("JingyaoLi/MoTCode-Data")
```

### Modular-of-Thought Training
Run the following command to train the model 
```bash 
deepspeed src/train.py \
    --model_name_or_path $model_path \
    --data_path $MoT_data_path \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True \
    --prompt_type FORMAT_PROMPT
```
