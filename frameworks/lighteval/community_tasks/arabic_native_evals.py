import re
import ast
from typing import List
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics


def natural_pfn(line, task_name: str = None, verbose: bool = False):
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"

    # Convert stringified list to actual list if needed
    raw_choices_input = line["choices"]
    try:
        raw_choices: List[str] = (
            ast.literal_eval(raw_choices_input)
            if isinstance(raw_choices_input, str)
            else raw_choices_input
        )
    except Exception as e:
        print(f"[ERROR] Failed to parse choices from: {raw_choices_input}")
        raise

    correct_label: str = line["correct_choice"].strip()
    question_text: str = line["question_text"].strip()

    valid_keys_arabic = []
    cleaned_choices = []

    for i, choice in enumerate(raw_choices):
        match = re.match(r"^\((.)\)\s*(.*)", choice)
        if match:
            label = match.group(1).strip()
            text = match.group(2).strip()
            valid_keys_arabic.append(label)
            cleaned_choices.append(text)
        else:
            print(f"[WARNING] Skipping malformed choice at index {i}: '{choice}' — expected format like '(أ) النص'")

    if len(valid_keys_arabic) < 2:
        print(f"[ERROR] Too few valid choices parsed for question: '{question_text}'")
        print(f"[ERROR] Raw choices: {raw_choices}")
        raise ValueError("Insufficient valid choices parsed.")

    try:
        answer_index = valid_keys_arabic.index(correct_label)
    except ValueError:
        print(f"[ERROR] Correct label '{correct_label}' not found in valid labels {valid_keys_arabic}")
        print(f"[ERROR] Full line: {line}")
        raise

    query = f"{instruction}{question_text}\n"
    # query += "".join([f"({label}) {text}\n" for label, text in zip(valid_keys_arabic, cleaned_choices)])
    query += "".join([f"{label}. {text}\n" for label, text in zip(valid_keys_arabic, cleaned_choices)])
    query += "الإجابة:"

    if verbose:
        print(f"\n[DEBUG] Processed question: {question_text}")
        print(f"[DEBUG] Labels: {valid_keys_arabic}")
        print(f"[DEBUG] Gold label: {correct_label} -> index {answer_index}")
        print(f"[DEBUG] Query preview:\n{query}")

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomTask(LightevalTaskConfig):
    def __init__(self, hf_repo, pfn_func, name, hf_subset):
        super().__init__(
            name=name,
            prompt_function=pfn_func,
            suite=["community"],
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            hf_avail_splits=["train", "test"],  # Make both available
            evaluation_splits=["train", "test"],  # Evaluate both splits together
            metric=[Metrics.loglikelihood_acc_norm],
            trust_dataset=True,
            version=0,
        )

subsets = [("original","falcon-arabic/naturalQA"), ("25percent","falcon-arabic/naturalQA_25percent")]

NATURAL_TASKS = [CustomTask(name=f"arabic_naturalqa:{sub}", hf_repo=repo, pfn_func=natural_pfn, hf_subset="")
                 for sub, repo in subsets]

TASKS_TABLE = (NATURAL_TASKS)