import hashlib
import json
import os
from typing import Dict

from evalplus.data.utils import (
    CACHE_DIR,
    completeness_check,
    stream_jsonl,
)

HUMANEVAL_PLUS_VERSION = "v0.1.0"
HUMANEVAL_PLUS_PATH = os.path.join(os.path.dirname(__file__), "data_files", "HumanEvalPlus.jsonl.gz")


def _ready_human_eval_plus_path(*_, **__) -> str:
    if not os.path.exists(HUMANEVAL_PLUS_PATH):
        raise FileNotFoundError(f"HumanEvalPlus dataset not found at {HUMANEVAL_PLUS_PATH}")
    return HUMANEVAL_PLUS_PATH


def get_human_eval_plus_hash(*_, **__) -> str:
    with open(HUMANEVAL_PLUS_PATH, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_human_eval_plus(err_incomplete=True, *_, **__) -> Dict[str, Dict]:
    plus_path = _ready_human_eval_plus_path()
    plus = {task["task_id"]: task for task in stream_jsonl(plus_path)}
    if err_incomplete:
        completeness_check("HumanEval+", plus)
    return plus


def get_human_eval() -> Dict[str, Dict]:
    import gzip
    from evalplus.data.utils import make_cache

    # Path to raw HumanEval (no need to change)
    from evalplus.data.utils import CACHE_DIR
    human_eval_path = os.path.join(CACHE_DIR, "HumanEval.jsonl")
    make_cache(
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        human_eval_path,
    )

    with open(human_eval_path, "r") as f:
        human_eval = [json.loads(line) for line in f if line.strip()]

    # Handle 115_max_fill.py to make its docstring well-formed
    human_eval[115]["prompt"] = "import math\n" + human_eval[115]["prompt"].replace("import math\n", "")

    # Get enhanced prompts from HumanEvalPlus
    plus = get_human_eval_plus()
    for task in human_eval:
        task_id = task["task_id"]
        if task_id in plus:
            task["prompt"] = plus[task_id]["prompt"]

    return {task["task_id"]: task for task in human_eval}
