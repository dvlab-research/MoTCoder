import os
import logging
import argparse
from leetcode_dataset.fetch_dataset import fetch_dataset, fetch_solutions
from leetcode_dataset.utils.utils import get_api_instance
from leetcode_dataset.clean_dataset import remove_class_dependent, remove_void, remove_class_impls, remove_examples
from leetcode_dataset.format_dataset import format_problems, to_jsonl

parser = argparse.ArgumentParser(description="Configuration for building uncontaminated Leetcode Hard dataset")
parser.add_argument('--langs', nargs='+', default=['python3'], help="List of languages. Possible options are: rust, python3")
parser.add_argument('--log_level', type=str, default='INFO', help="Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
parser.add_argument('--output_path', type=str, default="./build", help="Directory to save the built dataset.")
parser.add_argument('--extract_test_cases', action='store_true', help="If set, test cases will be extracted from problem descriptions using GPT.")
parser.add_argument('--remove_examples', action='store_true', help="If set, examples will be removed. Cannot be used with --extract_test_cases.")
parser.add_argument('--fetch_solutions', action='store_true', help="If set, solutions to problems will be fetched. Currently only supports lang=python3.")
parser.add_argument('--difficulty', default=0, type=int)
parser.add_argument('--uncontaminated_only', action='store_true', help="")
parser.add_argument('--num', default=1000, type=int, help="")
args = parser.parse_args()

langs = args.langs
log_level = getattr(logging, args.log_level.upper())
output_path = args.output_path
extract_test_cases_ = args.extract_test_cases
remove_examples_ = args.remove_examples
fetch_solutions_ = args.fetch_solutions

try:
    os.environ["LEETCODE_SESSION"]
except:
    print("Environment variable LEETCODE_SESSION is not set. Please refer to README")
    exit(1)

if extract_test_cases_:
    try:
        os.environ["OPENAI_API_KEY"]
        import openai
        import langchain
    except:
        print("Extra dependencies and setup are required for test case extraction. Please refer to README")
        exit(1)
    if remove_examples_:
        print("Cannot use --remove_examples with --extract_test_cases")
        exit(1)


logging.basicConfig(level=log_level)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

api_instance = get_api_instance()
dataset = fetch_dataset(api_instance, 
    uncontaminated_only=args.uncontaminated_only, 
    difficulty=args.difficulty, 
    num=args.num)

filtered_dataset = \
    remove_class_impls(
    remove_void(
    remove_class_dependent(dataset))).reset_index(drop=True)

if remove_examples_:
    filtered_dataset = remove_examples(filtered_dataset)

logging.info(f"Filtered out {len(dataset) - len(filtered_dataset)} problem(s)")

for lang in langs:
    logging.info(f"Formatting dataset for {lang}")
    formatted_dataset = format_problems(filtered_dataset, lang)
    if extract_test_cases_:
        logging.info(f"Extracting test cases for {lang}")
        from leetcode_dataset.add_test_cases import extract_test_cases
        formatted_dataset = extract_test_cases(formatted_dataset, lang)
    if fetch_solutions_:
        logging.info(f"Fetching solutions for {lang}")
        formatted_dataset = fetch_solutions(formatted_dataset, lang)
    to_jsonl(formatted_dataset, output_path)
