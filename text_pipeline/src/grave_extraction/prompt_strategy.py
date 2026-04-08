from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import UndefinedError


class PromptStrategy(ABC):
    @abstractmethod
    def format(self, **kwargs) -> str:
        pass


class Jinja2FilePromptStrategy(PromptStrategy):
    """
    Loads a prompt strategy from a file and uses a strict Jinja2 environment
    to ensure all variables are provided.
    """

    def __init__(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                template_str = f.read()
        except FileNotFoundError:
            raise ValueError(f"Template file not found at: {file_path}")

        self.file_name = Path(file_path).name

        env = Environment(undefined=StrictUndefined)
        self.template = env.from_string(template_str)

    def format(self, **kwargs: Any) -> str:
        try:
            return self.template.render(**kwargs)
        except UndefinedError as e:
            raise ValueError(f"Missing required variable in prompt: {e}")
