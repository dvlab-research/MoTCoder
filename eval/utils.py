import itertools
import json
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset
from testing_util import run_test
from tqdm import tqdm 
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
TIMEOUT = 3

def extract_code_blocks(text):
    """Extract code blocks from text."""
    code_block_pattern = re.compile(
        r'```(?:python)?\n(.*?)\n```',  # (?:python)? non-capturing group for optional 'python'
        re.DOTALL  # Enable dot to match newlines
    )
    code_blocks = code_block_pattern.findall(text)
    if len(code_blocks) > 0:
        return code_blocks[0]
    else:
        # import pdb;pdb.set_trace()
        code = text.lstrip('```python').lstrip('```').rstrip('```')
        return code

def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    if debug:
        # Execute in single-threaded mode for easy debugging
        print("Running in debug mode (single-threaded)")
        result = []
        _temp_run(sample, generation, debug, result)
    else:
        # Execute in multiprocessing mode
        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            # print("Process timed out and will be terminated.")
            p.kill()
        
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [{
            'outputs': [f"global timeout" for i in range(len(in_outs["inputs"]))],
            'results': [-1 for i in range(len(in_outs["inputs"]))],
        }]
        if debug:
            print(f"global timeout")

    return result[0]


def evaluate_generations(samples: list, level: str = "all", debug: bool = False):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """
    results = {}
    ori_l = len(samples)
    samples = [sample for sample in samples if level == "all" or sample.get('difficulty', 'all') == level]
    print(f'Loaded {len(samples)} {level} samples from total {ori_l} samples')

    def evaluate_sample(index, sample):
        res = []
        opt = []
        fix = 'fix_gen' in sample
        if fix:
            problem_generations = sample['fix_gen']
        else:
            problem_generations = sample['gen']
        # if len(problem_generations) < 5:
        #     import pdb;pdb.set_trace()
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                both = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                curr_res = both['results']
                curr_opt = both['outputs']
                if debug:
                    print(f"\nSuccessful compilation of task {index}!")
                fixed = [bool(e) if isinstance(e, np.bool_) else e.item() if isinstance(e, np.ndarray) else e for e in curr_res]
                curr_res = fixed
                if not np.all(curr_res) and debug:
                    print(f"Results were not True for all test cases")
            except Exception as e:
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                curr_opt = f"Compilation failed, test framework exception = {repr(e)}{e}"
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
                opt.append(curr_opt)
        # if len(res) < 5:
        #     import pdb;pdb.set_trace()
        if fix:
            sample['fix_eval'] = res
        else:
            sample['eval'] = res
        sample['outputs'] = opt
        return index, sample

    if debug:
        # Single-threaded debug path
        for index, sample in enumerate(tqdm(samples, desc='Evaluating samples')):
            idx, sample = evaluate_sample(index, sample)
            results[idx] = sample
    else:
        # Multi-threaded path using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_sample, index, sample): index for index, sample in enumerate(samples)}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Evaluating samples'):
                # for future in as_completed(futures):
                idx, sample = future.result()
                results[idx] = sample
                
    return results

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def get_results(results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 10, 100]):
    """
    Given the results evaluated against the testcases we output some statistics.
    For single generations:
    >>> example_results = {0: [[-2]], 1: [[False,False]], 2: [[True,True]], 3: [[False,True,False,True]], 4: [[-1,-1]]}
    >>> get_results(example_results, count_errors=True)
    Computing accuracy metrics...
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of problems evaluated = 5
    Average Accuracy : 0.3
    Strict Accuracy : 0.2
    {'avg_accuracy': 0.3, 'strict_accuracy': 0.2, 'pass_at_k': None}

    For multiple generations:
    >>> example_results = {0: [[-2], [True, True, True]], 1: [[-1,-1, -1], [True, False, True]]}
    >>> get_results(example_results, k_list=[1, 2])
    Computing pass@k metric for multiple generations...
    {'pass@1': 0.25, 'pass@2': 0.5}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 0.25, 'pass@2': 0.5}}
    """

    metrics = {"num": len(results), "avg_accuracy": None, "strict_accuracy": None, "pass_at_k": None}

    if len(results) == 0:
        return metrics 

    key = 'fix_eval' if 'fix_eval' in results[0] else 'eval'

    # if len(results[0]['eval']) == 1:
    # if len(results[0]['fix_eval']if fix else results[0]['eval']) == 1:
    if True:
        # for single generations we compute average accuracy and stric accuracy: original APPS metrics
        print(f"Computing {key} accuracy metrics...")

        res = []
        per_prob_res = []
        all_correct = []

        # 新列表用于存储每个 index 的 pass@1 和 pass@5 的结果
        pass_at_1_list = []
        pass_at_5_list = []

        for index in results:
            first_result = None
            max_result_value = float('-inf')
            if len(results[0][key]) == 1:
                res.extend(np.asarray(results[index][key]))

            for idx, result in enumerate(results[index][key]):
                problem_results = np.asarray(result)
                
                # 计算每个 result 的 mean
                if np.any(problem_results > 0):
                    result_value = np.mean(problem_results > 0)
                else:
                    result_value = 0.0
                
                # 计算 pass@1
                if len(results[index]) > 1:
                    if idx == 0:
                        first_result = result_value
                    
                    # 更新最大值用于 pass@5
                    max_result_value = max(max_result_value, result_value)
                
                # 更新所有概率结果
                per_prob_res.append(result_value)
                all_correct.append(np.all(problem_results > 0))

            if len(results[index]) > 1:
                # 将计算的 pass@1 和 pass@5 添加到列表中
                pass_at_1_list.append(first_result)
                pass_at_5_list.append(max_result_value)

        # we count campilation and runtime errors once per pronlem
        if len(results[0][key]) == 1:
            compile_errors = len([e for e in res if -2 in e])
            runtime_errors = len([e for e in res if -1 in e])
            total_testcases = len(res)

            if count_errors:
                print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
                print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
                print(f"number of problems evaluated = {total_testcases}")

        print(f"Average Accuracy : {np.mean(per_prob_res)}")
        print(f"Strict Accuracy : {np.mean(all_correct)}")
        metrics["avg_accuracy"] = np.mean(per_prob_res)
        metrics["strict_accuracy"] = np.mean(all_correct)
        if len(pass_at_5_list) > 0:
            metrics["pass_at_k"] = {"pass@1": np.mean(pass_at_1_list), 'pass@5': np.mean(pass_at_5_list)}

    else:
        # for multiple generations we use pass@k metric used in the HumanEval benchmark
        # we use strict accuracy, a generation is valid if it has to pass all the tests
        print("Computing pass@k metric for multiple generations...")
        # total is list with nb generations per task (task=index)
        # correct is number of generations that passed all tests per task
        total = []
        correct = [] 
        for index in results:
            all_correct = []
            generations = results[index][key]
            for generation in generations:
                gen = np.array(generation)
                import pdb;pdb.set_trace()
                if np.any(gen > 0):
                    all_correct.append(np.mean(gen>0))
                else:
                    all_correct.append(0.0)
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
        metrics["pass_at_k"] = pass_at_k
    return metrics


def compute_metrics(generations, level="all", k_list=[1, 10, 100], count_errors=True, debug=False):
    """Return metrics for the given generations.
    Args:
        generations: list of code generations for each problem (each generation is a list of generations)
        k_list: list of k values to compute pass@k when using multiple generations
        count_errors: whether to count compilation and runtime errors when using single generations
        level: difficulty level in APPS dataset that was used for the given generations (from: "all", "introductory", "interview", "competition")
    Returns:
        metrics: dict of metrics  

    Examples:

    >>> import json
    >>> # lists of solutions to the two first APPS problems (note not all solutions pass all tests)
    >>> solution_sample1 = json.load(open("test_examples/solutions_problem_1.json", "r"))
    >>> solution_sample2 = json.load(open("test_examples/solutions_problem_2.json", "r"))
    >>> single_solutions = [solution_sample1[:1], solution_sample2[:1]]
    >>> compute_metrics(single_solutions, level="all")
    Computing accuracy metrics...
    number of compile errors = 0 avg = 0.0
    number of runtime errors = 0 avg = 0.0
    number of problems evaluated = 2
    Average Accuracy : 1.0
    Strict Accuracy : 1.0
    {'avg_accuracy': 1.0, 'strict_accuracy': 1.0, 'pass_at_k': None}
    >>> multiple_solutions = [solution_sample1[:3], solution_sample2[:3]]
    >>> compute_metrics(multiple_solutions, level="all", k_list=[1, 2, 3])
    Computing pass@k metric for multiple generations...
    {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}}
    """
    results = evaluate_generations(generations, level=level, debug=debug)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return results, metrics

if __name__ == '__main__':
    data_path = '/mnt/nas-alinlp/ljy/MoTCoder/data/prompts/code_contests/Qwen2.5-Coder-7B-Instruct.jsonl'
    level = 'test'
    generations = read_data_jsonl(data_path)
    results, metrics = compute_metrics(generations, level=level, k_list=[1], debug=False)
    # import doctest
    # doctest.testmod()
