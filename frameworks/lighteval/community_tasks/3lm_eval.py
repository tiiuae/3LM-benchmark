# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re
import ast 
from typing import Any, Dict, List, Optional, Union

from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import Metric, Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]

SYN_SUBSETS = [
    "Math","Biology","Physics","Chemistry","General_Science"
]
# SYN_SUBSETS = [
#     "math","biology","physics","chemistry","general_Science"
# ]


def syn_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"

    raw = line["choices"]
    if isinstance(raw, list):
        raw = raw[0]
    elif not isinstance(raw, str):
        raise ValueError(f"Invalid choices type: {type(raw)}")


    labels = ["أ)", "ب)", "ج)", "د)"]
    if all(lbl in raw for lbl in labels):
        positions_and_labels = []
        for lbl in labels:
            idx = raw.find(lbl)  # only the first occurrence
            if idx == -1:
                raise ValueError(f"Missing label {lbl} in: {raw!r}")
            positions_and_labels.append((idx, lbl))


        positions_and_labels.sort(key=lambda pair: pair[0])
        label_to_text = {}
        for i, (pos, lbl) in enumerate(positions_and_labels):
            start = pos + len(lbl)
            end = positions_and_labels[i+1][0]-1 if i < 3 else len(raw)
            chunk = raw[start:end].strip()
            label_to_text[lbl] = chunk

        choices = [
            label_to_text["أ)"],
            label_to_text["ب)"],
            label_to_text["ج)"],
            label_to_text["د)"],
        ]

        # Sanity check
        if any(not c for c in choices):
            raise ValueError(f"After slicing labels, got empty choice(s): {choices!r}")

    elif "," in raw:
        parts = []
        buffer = ""
        depth = 0
        for ch in raw:
            if ch == "(":
                depth += 1
                buffer += ch
            elif ch == ")":
                depth = depth - 1 if depth > 0 else 0
                buffer += ch
            elif ch == "," and depth == 0:
                parts.append(buffer.strip())
                buffer = ""
            else:
                buffer += ch
        parts.append(buffer.strip())

        if len(parts) != 4:
            raise ValueError(f"Expected 4 top-level commas, got {len(parts)}: {parts!r}")

        choices = []
        for idx, segment in enumerate(parts):
            label = segment[:2]
            if label not in labels:
                raise ValueError(f"Missing or malformed label at chunk {idx+1}: {segment!r}")
            stripped = segment[2:].strip()
            choices.append(stripped)
            print("these are choices", choices)
        if any(not c for c in choices):
            raise ValueError(f"After stripping labels, got empty choice(s): {choices!r}")
        
    else:
        raise ValueError(f"Cannot determine how to split choices: {raw!r}")

    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
    arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
    valid_keys_latin = ["A", "B", "C", "D"]
    valid_keys_arabic = [latin_to_arabic[k] for k in valid_keys_latin]

    self_answer_arabic = line["self_answer"].strip()
    self_answer_latin = arabic_to_latin.get(self_answer_arabic)
    if self_answer_latin not in valid_keys_latin:
        raise ValueError(f"Invalid answer: {self_answer_arabic!r}")
    answer_index = valid_keys_latin.index(self_answer_latin)


    query = f"{instruction}{line['question']}\n"
    for arab_label, choice_text in zip(valid_keys_arabic, choices):
        clean_choice = choice_text.strip(" []'\"\n")
        query += f"{arab_label}.{clean_choice[1:]}\n"
    query += "الإجابة:"
    # print(query)
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,
        gold_index=answer_index,
        instruction=instruction,
    )

def syn_gen_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    raw = line["choices"]
    if isinstance(raw, list):
        raw = raw[0]
    elif not isinstance(raw, str):
        raise ValueError(f"Invalid choices type: {type(raw)}")

    labels = ["أ)", "ب)", "ج)", "د)"]
    if all(lbl in raw for lbl in labels):
        positions_and_labels = []
        for lbl in labels:
            idx = raw.find(lbl)  # only the first occurrence
            if idx == -1:
                raise ValueError(f"Missing label {lbl} in: {raw!r}")
            positions_and_labels.append((idx, lbl))

        positions_and_labels.sort(key=lambda pair: pair[0])
        label_to_text = {}
        for i, (pos, lbl) in enumerate(positions_and_labels):
            start = pos + len(lbl)
            end = positions_and_labels[i+1][0]-1 if i < 3 else len(raw)
            chunk = raw[start:end].strip()
            label_to_text[lbl] = chunk

        choices = [
            label_to_text["أ)"],
            label_to_text["ب)"],
            label_to_text["ج)"],
            label_to_text["د)"],
        ]

        if any(not c for c in choices):
            raise ValueError(f"After slicing labels, got empty choice(s): {choices!r}")

    elif "," in raw:
        parts = []
        buffer = ""
        depth = 0
        for ch in raw:
            if ch == "(":
                depth += 1
                buffer += ch
            elif ch == ")":
                depth = depth - 1 if depth > 0 else 0
                buffer += ch
            elif ch == "," and depth == 0:
                parts.append(buffer.strip())
                buffer = ""
            else:
                buffer += ch
        parts.append(buffer.strip())

        if len(parts) != 4:
            raise ValueError(f"Expected 4 top-level commas, got {len(parts)}: {parts!r}")

        choices = []
        for idx, segment in enumerate(parts):
            label = segment[:2]
            if label not in labels:
                raise ValueError(f"Missing or malformed label at chunk {idx+1}: {segment!r}")
            stripped = segment[2:].strip()
            choices.append(stripped)

        if any(not c for c in choices):
            raise ValueError(f"After stripping labels, got empty choice(s): {choices!r}")

    else:
        raise ValueError(f"Cannot determine how to split choices: {raw!r}")

    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
    arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
    valid_keys_latin = ["A", "B", "C", "D"]
    valid_keys_arabic = [latin_to_arabic[k] for k in valid_keys_latin]

    self_answer_arabic = line["self_answer"].strip()
    self_answer_latin = arabic_to_latin.get(self_answer_arabic)
    if self_answer_latin not in valid_keys_latin:
        raise ValueError(f"Invalid answer: {self_answer_arabic!r}")
    answer_index = valid_keys_latin.index(self_answer_latin)

    choices = [c[1:] for c in choices]
    # print("these are the choices", choices)
    new_choices = []
    for arab_label, choice_text in zip(labels, choices):
        clean_choice = choice_text.strip(" []'\"\n").replace("'", "")  # Ensure commas are consistent
        new_choices.append(clean_choice)
    # print("these are the new choices", new_choices)
    question = line['question']
    # print("this is the question", question)
    gold_answer = choices[answer_index].strip(" []'\"\n").replace("'", "")
    print("this is the gold answer", gold_answer)
    query = f"{instruction}{question}\n"
    query += "الإجابة:"
    print(query)
    print("new_choices" ,new_choices)


    return Doc(
        task_name=task_name,
        query=query,
        instruction=instruction,
        choices=new_choices, # we give the text of all the choices without the letter
        gold_index=answer_index  # Index of the correct answer in choices
    )

def native_mcq_pfn(line, task_name: str = None, verbose: bool = False):
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"

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

def native_completion_pfn(line, task_name: str = None, verbose: bool = False):
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"

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
    query += "الإجابة:"

    if verbose:
        print(f"\n[DEBUG] Processed question: {question_text}")
        print(f"[DEBUG] Labels: {valid_keys_arabic}")
        print(f"[DEBUG] Gold label: {correct_label} -> index {answer_index}")
        print(f"[DEBUG] Query preview:\n{query}")

    return Doc(
        task_name=task_name,
        query=query,
        choices=cleaned_choices,
        gold_index=answer_index,
        instruction=instruction,
    )

class CustomNativeMCQTask(LightevalTaskConfig):
    def __init__(self, hf_repo, pfn_func, name, hf_subset=None):
        super().__init__(
            name=name,
            prompt_function=native_mcq_pfn,
            suite=["community"],
            hf_repo=hf_repo,
            hf_subset=hf_subset or None, 
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            metric=[Metrics.loglikelihood_acc_norm],
            trust_dataset=True,
            version=0,
        )

subsets = [("NativeQA","tiiuae/NativeQA"), ("NativeQA-RDP","tiiuae/NativeQA-RDP")]

NATIVE_MCQ_TASKS = [CustomNativeMCQTask(name=f"MCQ_{sub}", hf_repo=repo, pfn_func=native_mcq_pfn, hf_subset="")
                 for sub, repo in subsets]


class CustomNativeCompletionTask(LightevalTaskConfig):
    def __init__(self, hf_repo, pfn_func, name, hf_subset=None):
        super().__init__(
            name=name,
            prompt_function=native_completion_pfn,
            suite=["community"],
            hf_repo=hf_repo,
            hf_subset=hf_subset or None, 
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm],
            trust_dataset=True,
            version=0,
        )

NATIVE_COMPLETION_TASKS = [CustomNativeCompletionTask(name=f"COMPLETION_{sub}", hf_repo=repo, pfn_func=native_completion_pfn, hf_subset="")
                 for sub, repo in subsets]


class CustomSynGENTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=syn_gen_pfn,
            hf_repo="tiiuae/SyntheticQA",
            # hf_repo="falcon-arabic/ilm",
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["test"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


SYN_GEN_TASKS = [
    CustomSynGENTask(name=f"COMPLETION_Synthetic:{subset}", hf_subset=subset) for subset in SYN_SUBSETS
]



class CustomSyntheticTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=syn_pfn,
            hf_repo="tiiuae/SyntheticQA",
            # hf_repo="falcon-arabic/ilm",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["test"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


SYN_TASKS = [
    CustomSyntheticTask(name=f"MCQ_Synthetic:{subset}", hf_subset=subset) for subset in SYN_SUBSETS
]


TASKS_TABLE = (
    SYN_TASKS
    + SYN_GEN_TASKS
    + NATIVE_MCQ_TASKS
    + NATIVE_COMPLETION_TASKS
)

# import random
# import re
# import ast
# from functools import partial
# from typing import Any, Dict, List, Optional, Union

# from lighteval.metrics.metrics import Metrics
# from lighteval.tasks.lighteval_task import LightevalTaskConfig
# from lighteval.tasks.requests import Doc


# LETTER_INDICES_AR = ["أ", "ب", "ج", "د"]
# LATIN_TO_ARABIC = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
# ARABIC_TO_LATIN = {v: k for k, v in LATIN_TO_ARABIC.items()}


# SYN_SUBSETS = ["Math", "Biology", "Physics", "Chemistry", "General_Science"]
# subsets = [("NativeQA", "tiiuae/NativeQA"), ("NativeQA-RDP", "tiiuae/NativeQA-RDP")]


# def parse_arabic_choices(raw: str) -> List[str]:
#     labels = [f"{l})" for l in LETTER_INDICES_AR[:4]]
#     if all(lbl in raw for lbl in labels):
#         positions = [(raw.find(lbl), lbl) for lbl in labels]
#         positions.sort()
#         return [raw[pos + len(lbl): positions[i+1][0] if i < 3 else None].strip() for i, (pos, lbl) in enumerate(positions)]
#     elif "," in raw:
#         parts, buffer, depth = [], "", 0
#         for ch in raw:
#             if ch == "(": depth += 1
#             elif ch == ")": depth = max(depth - 1, 0)
#             if ch == "," and depth == 0:
#                 parts.append(buffer.strip()); buffer = ""
#             else:
#                 buffer += ch
#         parts.append(buffer.strip())
#         if len(parts) != 4:
#             raise ValueError(f"Expected 4 parts, got {len(parts)}")
#         return [p[2:].strip() for p in parts]
#     raise ValueError("Invalid choice format")

# def syn_doc_parser(line, task_name=None, include_labels=True):
#     instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
#     choices = parse_arabic_choices(line["choices"] if not isinstance(line["choices"], list) else line["choices"][0])
#     self_answer_latin = ARABIC_TO_LATIN.get(line["self_answer"].strip())
#     if self_answer_latin not in LATIN_TO_ARABIC:
#         raise ValueError(f"Invalid answer: {line['self_answer']!r}")
#     answer_index = ["A", "B", "C", "D"].index(self_answer_latin)

#     query = f"{instruction}{line['question']}\n"
#     formatted_choices = choices if not include_labels else [f"{LETTER_INDICES_AR[i]}. {c[1:].strip()}" for i, c in enumerate(choices)]
#     if include_labels:
#         query += "\n".join(formatted_choices) + "\n"
#     query += "الإجابة:"

#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=LETTER_INDICES_AR if include_labels else [c[1:].strip() for c in choices],
#         gold_index=answer_index,
#         instruction=instruction,
#     )

# syn_pfn = partial(syn_doc_parser, include_labels=True)
# syn_gen_pfn = partial(syn_doc_parser, include_labels=False)

# def parse_native_doc(line, task_name=None, verbose=False, include_choices=True):
#     instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"
#     raw_choices = ast.literal_eval(line["choices"]) if isinstance(line["choices"], str) else line["choices"]
#     correct_label = line["correct_choice"].strip()
#     question_text = line["question_text"].strip()
#     labels, texts = [], []
#     for choice in raw_choices:
#         match = re.match(r"^\((.)\)\s*(.*)", choice)
#         if match:
#             labels.append(match.group(1).strip())
#             texts.append(match.group(2).strip())

#     if correct_label not in labels:
#         raise ValueError(f"Correct label {correct_label} not found")
#     answer_index = labels.index(correct_label)

#     query = f"{instruction}{question_text}\n"
#     if include_choices:
#         query += "".join([f"{l}. {t}\n" for l, t in zip(labels, texts)])
#     query += "الإجابة:"

#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=labels if include_choices else texts,
#         gold_index=answer_index,
#         instruction=instruction,
#     )

# native_mcq_pfn = partial(parse_native_doc, include_choices=True)
# native_completion_pfn = partial(parse_native_doc, include_choices=False)

# def create_task_class(name, hf_repo, prompt_function, metrics, hf_subset=None):
#     return LightevalTaskConfig(
#         name=name,
#         prompt_function=prompt_function,
#         suite=["community"],
#         hf_repo=hf_repo,
#         hf_subset=hf_subset or None,
#         hf_avail_splits=["test"],
#         evaluation_splits=["test"],
#         metric=metrics,
#         trust_dataset=True,
#         version=0,
#     )

# NATIVE_MCQ_TASKS = [
#     create_task_class(f"MCQ_{sub}", repo, native_mcq_pfn, [Metrics.loglikelihood_acc_norm])
#     for sub, repo in subsets
# ]

# NATIVE_COMPLETION_TASKS = [
#     create_task_class(f"COMPLETION_{sub}", repo, native_completion_pfn, [Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm])
#     for sub, repo in subsets
# ]

# SYN_TASKS = [
#     create_task_class(f"MCQ_Synthetic:{subset}", "tiiuae/SyntheticQA", syn_pfn, [Metrics.loglikelihood_acc_norm], hf_subset=subset)
#     for subset in SYN_SUBSETS
# ]

# SYN_GEN_TASKS = [
#     create_task_class(f"COMPLETION_Synthetic:{subset}", "tiiuae/SyntheticQA", syn_gen_pfn, [Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm], hf_subset=subset)
#     for subset in SYN_SUBSETS
# ]

# TASKS_TABLE = SYN_TASKS + SYN_GEN_TASKS + NATIVE_MCQ_TASKS + NATIVE_COMPLETION_TASKS
