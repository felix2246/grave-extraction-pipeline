import os

from grave_extraction.logger import logger

_PHOENIX_CHECK_URL = os.environ.get("PHOENIX_CHECK_URL", "http://127.0.0.1:6006")


def _phoenix_reachable(url: str = _PHOENIX_CHECK_URL, timeout: float = 2.0) -> bool:
    """Return True if the Phoenix server responds to a simple GET (e.g. UI on 6006)."""
    try:
        import requests

        r = requests.get(url.rstrip("/") + "/", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def init_tracing(project_name: str) -> None:
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        if not _phoenix_reachable():
            logger.warning(
                "Tracing disabled – Phoenix not reachable at %s (is the server running?)",
                _PHOENIX_CHECK_URL,
            )
            return

        tracer_provider = register(project_name=project_name)

        tracer_provider = register(project_name=project_name)
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument(
            skip_dep_check=True
        )
        logger.info("Initialized tracing with phoenix")
    except Exception as e:
        logger.warning("Tracing disabled – Phoenix not available: %s", e)
