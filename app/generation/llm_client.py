import time

from openai import OpenAI

from app.config.settings import settings
from app.utils.logger import logger


class LLMClient:
    """OpenRouter client with model fallback. The API key is never logged."""

    def is_configured(self, api_key=None) -> bool:
        return bool(api_key or settings.OPENROUTER_API_KEY)

    def generate(self, prompt: str, api_key=None) -> dict:
        key = api_key or settings.OPENROUTER_API_KEY
        if not key:
            return {"answer": None, "model": None, "error": "No API key configured."}

        client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=key,
            timeout=settings.LLM_TIMEOUT,
        )
        last_error = None

        for model in settings.LLM_MODELS:
            for attempt in range(settings.LLM_RETRIES):
                try:
                    logger.info("Trying model: %s", model)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    text = (response.choices[0].message.content or "").strip()
                    if text:
                        return {"answer": text, "model": model, "error": None}
                except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                    last_error = str(exc)
                    logger.warning("Attempt %d failed for %s: %s", attempt + 1, model, last_error)
                    if getattr(exc, "status_code", None) in (401, 403):
                        # Bad key: retrying or switching models will not help.
                        return {"answer": None, "model": None, "error": "Invalid API key."}
                    time.sleep(attempt + 1)

        return {"answer": None, "model": None, "error": last_error or "LLM returned no answer."}
