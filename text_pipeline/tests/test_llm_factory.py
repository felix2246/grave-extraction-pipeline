from unittest.mock import patch

import pytest
from grave_extraction.config import Settings
from grave_extraction.llm_factory import get_llm


def test_get_llm_raises_for_missing_openai_api_key():
    with patch("grave_extraction.llm_factory.settings", Settings(OPENAI_API_KEY="")):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm("openai", "gpt-4.1-mini")


def test_get_llm_raises_for_missing_gwdg_api_key():
    with patch("grave_extraction.llm_factory.settings", Settings(GWDG_API_KEY="")):
        with pytest.raises(ValueError, match="GWDG_API_KEY"):
            get_llm("gwdg", "mistral-large-instruct")
