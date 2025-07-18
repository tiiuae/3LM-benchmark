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
# fmt: on

# ArabicMMLU
# fmt: off
ILM_SUBSETS = [
    "math","biology","physics","chemistry","general_science"
]
# fmt: on

def ilm_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    # 1) Extract the raw choices string
    raw = line["choices"]
    if isinstance(raw, list):
        raw = raw[0]
    elif not isinstance(raw, str):
        raise ValueError(f"Invalid choices type: {type(raw)}")

    # 2) Decide whether to split by Arabic labels or by commas
    labels = ["أ)", "ب)", "ج)", "د)"]
    if all(lbl in raw for lbl in labels):
        # --- Label-based splitting (allow multiple occurrences) ---
        positions_and_labels = []
        for lbl in labels:
            idx = raw.find(lbl)  # only the first occurrence
            if idx == -1:
                raise ValueError(f"Missing label {lbl} in: {raw!r}")
            positions_and_labels.append((idx, lbl))

        # Sort by the index so we know textual order
        positions_and_labels.sort(key=lambda pair: pair[0])
        # Extract each chunk between consecutive label‐indices
        label_to_text = {}
        for i, (pos, lbl) in enumerate(positions_and_labels):
            start = pos + len(lbl)
            end = positions_and_labels[i+1][0]-1 if i < 3 else len(raw)
            chunk = raw[start:end].strip()
            label_to_text[lbl] = chunk

        # Reassemble in the canonical order ["أ)", "ب)", "ج)", "د)"]
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
        # --- Comma-based splitting (unchanged) ---
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

    # 3) Map between Latin ⇄ Arabic labels, find gold index
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
    arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
    valid_keys_latin = ["A", "B", "C", "D"]
    valid_keys_arabic = [latin_to_arabic[k] for k in valid_keys_latin]

    self_answer_arabic = line["self_answer"].strip()
    self_answer_latin = arabic_to_latin.get(self_answer_arabic)
    if self_answer_latin not in valid_keys_latin:
        raise ValueError(f"Invalid answer: {self_answer_arabic!r}")
    answer_index = valid_keys_latin.index(self_answer_latin)

    # 4) Build the query text
    query = f"{instruction}{line['question']}\n"
    for arab_label, choice_text in zip(valid_keys_arabic, choices):
        query += f"{arab_label}.{choice_text[1:]}\n"
    query += "الإجابة:"
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,
        gold_index=answer_index,
        instruction=instruction,
    )



class CustomArabicMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=ilm_pfn,
            hf_repo="falcon-arabic/ilm",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ILM_TASKS = [
    CustomArabicMMLUTask(name=f"ilm:{subset}", hf_subset=subset) for subset in ILM_SUBSETS
]




TASKS_TABLE = (
    ILM_TASKS
    # + ARABIC_MMLU_HT_TASKS
    # + ARABIC_MMLU_MT_TASKS
)
