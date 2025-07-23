import random
import re
import ast
from typing import Any, Dict, List, Optional, Union

from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import Metric, Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

LETTER_INDICES_AR = [
    "أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"
]

SYN_SUBSETS = ["Math", "Biology", "Physics", "Chemistry", "General_Science"]

def parse_choices(raw: str, labels: List[str]) -> List[str]:
    if all(lbl in raw for lbl in labels):
        positions_and_labels = sorted((raw.find(lbl), lbl) for lbl in labels if raw.find(lbl) != -1)
        label_to_text = {
            lbl: raw[pos + len(lbl): positions_and_labels[i + 1][0] - 1 if i < 3 else len(raw)].strip()
            for i, (pos, lbl) in enumerate(positions_and_labels)
        }
        return [label_to_text[lbl] for lbl in labels]
    elif "," in raw:
        parts, buffer, depth = [], "", 0
        for ch in raw:
            if ch == "(": depth += 1
            elif ch == ")": depth = max(depth - 1, 0)
            if ch == "," and depth == 0:
                parts.append(buffer.strip()); buffer = ""
            else:
                buffer += ch
        parts.append(buffer.strip())

        if len(parts) != 4:
            raise ValueError(f"Expected 4 top-level commas, got {len(parts)}: {parts!r}")

        return [part[2:].strip() for part in parts]
    else:
        raise ValueError(f"Cannot determine how to split choices: {raw!r}")

def build_prompt_doc(line: dict, instruction: str, task_name: Optional[str], include_lettered_choices: bool = True, clean_choices: bool = False) -> Doc:
    labels = ["أ)", "ب)", "ج)", "د)"]
    raw = line["choices"]
    if isinstance(raw, list):
        raw = raw[0]
    elif not isinstance(raw, str):
        raise ValueError(f"Invalid choices type: {type(raw)}")

    choices = parse_choices(raw, labels)
    if clean_choices:
        choices = [re.sub(r"^\)?\s*", "", c).strip(" []'\"\n") for c in choices]
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
    arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
    valid_keys_arabic = list(latin_to_arabic.values())
    self_answer_arabic = line["self_answer"].strip()
    self_answer_latin = arabic_to_latin.get(self_answer_arabic)

    if self_answer_latin not in latin_to_arabic:
        raise ValueError(f"Invalid answer: {self_answer_arabic!r}")

    answer_index = list(latin_to_arabic.keys()).index(self_answer_latin)

    query = f"{instruction}{line['question']}\n"
    if include_lettered_choices:
        for arab_label, choice_text in zip(valid_keys_arabic, choices):
            choice_text = re.sub(r"^\)?\s*", "", choice_text).strip(" []'\"\n")
            query += f"{arab_label}. {choice_text}\n"
    query += "الإجابة:"
    print (query)
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic if include_lettered_choices else choices,
        gold_index=answer_index,
        instruction=instruction,
    )

def native_parse_choices(line):
    raw_choices_input = line["choices"]
    raw_choices = ast.literal_eval(raw_choices_input) if isinstance(raw_choices_input, str) else raw_choices_input
    labels, texts = [], []
    for i, choice in enumerate(raw_choices):
        match = re.match(r"^\((.)\)\s*(.*)", choice)
        if match:
            labels.append(match.group(1).strip())
            texts.append(match.group(2).strip())
        else:
            raise ValueError(f"Malformed choice: {choice}")
    return labels, texts

def native_prompt_function(line, task_name: str = None, completion: bool = False, verbose: bool = False) -> Doc:
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"
    labels, texts = native_parse_choices(line)
    correct_label = line["correct_choice"].strip()
    question_text = line["question_text"].strip()
    try:
        answer_index = labels.index(correct_label)
    except ValueError:
        raise ValueError(f"Correct label '{correct_label}' not found in labels {labels}")

    query = f"{instruction}{question_text}\n"
    if not completion:
        query += "".join([f"{label}. {text}\n" for label, text in zip(labels, texts)])
    query += "الإجابة:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=texts if completion else labels,
        gold_index=answer_index,
        instruction=instruction,
    )

class CustomTask(LightevalTaskConfig):
    def __init__(self, name, hf_repo, pfn_func, hf_subset=None, metrics=None):
        super().__init__(
            name=name,
            prompt_function=pfn_func,
            suite=["community"],
            hf_repo=hf_repo,
            hf_subset=hf_subset or None,
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            metric=metrics or [Metrics.loglikelihood_acc_norm],
            trust_dataset=True,
            version=0,
        )

subsets = [("NativeQA", "tiiuae/NativeQA"), ("NativeQA-RDP", "tiiuae/NativeQA-RDP")]

NATIVE_MCQ_TASKS = [CustomTask(f"MCQ_{sub}", repo, lambda x, tn=None: native_prompt_function(x, tn, completion=False)) for sub, repo in subsets]
NATIVE_COMPLETION_TASKS = [CustomTask(f"COMPLETION_{sub}", repo, lambda x, tn=None: native_prompt_function(x, tn, completion=True), metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm]) for sub, repo in subsets]
SYN_TASKS = [
    CustomTask(
        name=f"MCQ_Synthetic:{subset}",
        hf_repo="tiiuae/SyntheticQA",
        pfn_func=lambda x, tn=None: build_prompt_doc(
            x,
            "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n",
            tn
        ),
        hf_subset=subset  
    )
    for subset in SYN_SUBSETS
]

SYN_GEN_TASKS = [
    CustomTask(
        name=f"COMPLETION_Synthetic:{subset}",
        hf_repo="tiiuae/SyntheticQA",
        hf_subset=subset,  
        pfn_func=lambda x, tn=None: build_prompt_doc(
            x,
            "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n",
            tn,
            include_lettered_choices=False,
            clean_choices=True
        ),
        metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm]
    )
    for subset in SYN_SUBSETS
]

TASKS_TABLE = SYN_TASKS + SYN_GEN_TASKS + NATIVE_MCQ_TASKS + NATIVE_COMPLETION_TASKS
