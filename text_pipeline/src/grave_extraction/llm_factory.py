from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic.types import SecretStr

from grave_extraction.config import load_settings, validate_provider_api_key
from grave_extraction.models import ModelProvider

settings = load_settings()


def get_llm(provider: ModelProvider, model_name: str, **kwargs) -> BaseChatModel:
    validate_provider_api_key(provider, settings)

    match provider:
        case "openai":
            return ChatOpenAI(
                model=model_name,
                name=model_name,
                api_key=SecretStr(settings.OPENAI_API_KEY),
                **kwargs,
            )
        case "gwdg":
            return ChatOpenAI(
                model=model_name,
                base_url="https://chat-ai.academiccloud.de/v1",
                api_key=SecretStr(settings.GWDG_API_KEY),
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported provider: {provider}")
