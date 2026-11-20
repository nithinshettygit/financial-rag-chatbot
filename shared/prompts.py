from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

import yaml

from shared.settings import general_settings


_PROMPTS_DIR = Path(general_settings.PROMPTS_PATH)
if not _PROMPTS_DIR.is_absolute():
    _PROMPTS_DIR = Path(__file__).resolve().parents[1] / _PROMPTS_DIR


@lru_cache(maxsize=None)
def load_prompt_file(file_name: str) -> dict[str, Any]:
    if Path(file_name).name != file_name or Path(file_name).suffix != ".yaml":
        raise ValueError(f"Prompt file must be a YAML filename: {file_name}")

    prompt_path = _PROMPTS_DIR / file_name
    with prompt_path.open(encoding="utf-8") as prompt_file:
        prompt_document = yaml.safe_load(prompt_file)
    if not isinstance(prompt_document, dict):
        raise ValueError(f"Expected a YAML mapping in {prompt_path}")
    return prompt_document


def load_prompt(file_name: str, name: str) -> str:
    prompt_document = load_prompt_file(file_name)
    prompt_definitions = prompt_document.get("prompts")

    if isinstance(prompt_definitions, dict):
        prompt: Any = prompt_definitions
        for key in name.split("."):
            if not isinstance(prompt, dict) or key not in prompt:
                raise ValueError(f"Prompt '{name}' not found in {_PROMPTS_DIR / file_name}")
            prompt = prompt[key]
    elif isinstance(prompt_definitions, list):
        matches = [
            item.get("content")
            for item in prompt_definitions
            if isinstance(item, dict) and item.get("task") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one '{name}' prompt in {_PROMPTS_DIR / file_name}")
        prompt = matches[0]
    else:
        raise ValueError(f"Expected a 'prompts' mapping or list in {_PROMPTS_DIR / file_name}")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt '{name}' in {_PROMPTS_DIR / file_name} must be a non-empty string")
    return prompt


def render_prompt(file_name: str, name: str, **values: str) -> str:
    return Template(load_prompt(file_name, name)).substitute(**values)
