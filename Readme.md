# MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks

This is the official code repository of MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks.

## Abstract
Large Language Models (LLMs) have showcased impressive capabilities in handling straightforward programming tasks. However, their performance tends to falter when confronted with more challenging programming problems. We observe that conventional models often generate solutions as monolithic code blocks, restricting their effectiveness in tackling intricate questions. To overcome this limitation, we present Modular-of-Thought Coder (MoTCoder). We introduce a pioneering framework for MoT instruction tuning, designed to promote the decomposition of tasks into logical sub-tasks and sub-modules. 
Our investigations reveal that, through the cultivation and utilization of sub-modules, MoTCoder significantly improves both the modularity and correctness of the generated solutions, leading to substantial relative *pass@1* improvements of 12.9% on APPS and 9.43% on CodeContests.

![MoTCoder Framework](./imgs/framework.png)

## Performance

![Performance on APPS](./imgs/impression.png)

**Performance on APPS**
| Model                     | Size | Pass@ | Introductory | Interview | Competition | All  |
|---------------------------|------|-------|--------------|-----------|-------------|------|
| **GPT-Neo**               | 2.7B | 1     | 3.90         | 0.57      | 0.00        | 1.12 |
|                           |      | 5     | 5.50         | 0.80      | 0.00        | 1.58 |
| **Codex**                 | 12B  | 1     | 4.14         | 0.14      | 0.02        | 0.92 |
|                           |      | 5     | 9.65         | 0.51      | 0.09        | 2.25 |
|                           |      | 1000  | 25.02        | 3.70      | 3.23        | 7.87 |
| **AlphaCode**             | 1B   | 1000  | 17.67        | 5.24      | 7.06        | 8.09 |
| **AlphaCode (Filtered 1k)**|      | 5     | 14.36        | 5.63      | 4.58        | 7.17 |
| **AlphaCode (Filtered 10k)**|     | 5     | 18.18        | 8.21      | 6.65        | 9.89 |
| **AlphaCode (Filtered 50k)**|     | 5     | 20.36        | 9.66      | 7.75        | 11.42 |
| **StarCoder**             | 15B  | 1     | 7.25         | 6.89      | 4.08        | 6.40 |
| **WizardCoder**           | 15B  | 1     | 26.04        | 4.21      | 0.81        | 7.90 |
| **CodeLlama**             | 7B   | 5     | 10.76        | 2.01      | 0.77        | 3.51 |
|                           |      | 10    | 15.59        | 3.12      | 1.41        | 5.27 |
|                           |      | 100   | 33.52        | 9.40      | 7.13        | 13.77|
|                           | 13B  | 5     | 23.74        | 5.63      | 2.05        | 8.54 |
|                           |      | 10    | 30.19        | 8.12      | 3.35        | 11.58|
|                           |      | 100   | 48.99        | 18.40     | 11.98       | 23.23|
|                           | 34B  | 5     | 32.81        | 8.75      | 2.88        | 12.39|
|                           |      | 10    | 38.97        | 12.16     | 4.69        | 16.03|
|                           |      | 100   | 56.32        | 24.31     | 15.39       | 28.93|
| **CodeLlama-Python**      | 7B   | 5     | 12.72        | 4.18      | 1.31        | 5.31 |
|                           |      | 10    | 18.50        | 6.25      | 2.24        | 7.90 |
|                           |      | 100   | 38.26        | 14.94     | 9.12        | 18.44|
|                           | 13B  | 5     | 26.33        | 7.06      | 2.79        | 10.06|
|                           |      | 10    | 32.77        | 10.03     | 4.33        | 13.44|
|                           |      | 100   | 51.60        | 21.46     | 14.60       | 26.12 |
|                           | 34B  | 5     | 28.94        | 7.80      | 3.45        | 11.16 |
|                           |      | 10    | 35.91        | 11.12     | 5.53        | 14.96 |
|                           |      | 100   | 54.92        | 23.90     | 16.81       | 28.69 |
| **CodeLlama-Instruct**    | 7B   | 5     | 12.85        | 2.07      | 1.13        | 4.04  |
|                           |      | 10    | 17.86        | 3.12      | 1.95        | 5.83  |
|                           |      | 100   | 35.37        | 9.44      | 8.45        | 14.43 |
|                           | 13B  | 5     | 24.01        | 6.93      | 2.39        | 9.44  |
|                           |      | 10    | 30.27        | 9.58      | 3.83        | 12.57 |
|                           |      | 100   | 48.73        | 19.55     | 13.12       | 24.10 |
|                           | 34B  | 5     | 31.56        | 7.86      | 3.21        | 11.67 |
|                           |      | 10    | 37.80        | 11.08     | 5.12        | 15.23 |
|                           |      | 100   | 55.72        | 22.80     | 16.38       | 28.10 |
| **MoTCoder**               | 15B  | 1     | **33.80**        | **19.70**     | **11.09**       | **20.80** |
| **code-davinci-002**      | -    | 1     | 29.30        | 6.40      | 2.50        | 10.20 |
| **GPT3.5**                | -    | 1     | 48.00        | 19.42     | 5.42        | 22.33 |

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
