from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from grave_extraction.models import ModelProvider


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    GWDG_API_KEY: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_settings() -> Settings:
    load_dotenv()
    return Settings()


def validate_provider_api_key(provider: ModelProvider, settings: Settings) -> None:
    provider_to_env_var = {
        "openai": "OPENAI_API_KEY",
        "gwdg": "GWDG_API_KEY",
    }

    required_env_var = provider_to_env_var[provider]

    if (
        not (api_key := getattr(settings, required_env_var, ""))
        or not str(api_key).strip()
    ):
        raise ValueError(
            f"Missing required API key for provider '{provider}'. Please set {required_env_var} in .env."
        )
