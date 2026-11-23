from app.config.settings import settings

BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "system prompt",
    "developer mode",
    "jailbreak",
    "bypass security",
    "reveal confidential",
    "delete database",
    "shutdown server",
)


class InputGuardrails:
    def __init__(self, max_chars: int = None):
        self.blocked_patterns = BLOCKED_PATTERNS
        self.max_chars = max_chars or settings.MAX_QUERY_CHARS

    def validate_query(self, query: str) -> dict:
        normalized = " ".join(query.lower().split())

        if not normalized:
            return {"allowed": False, "reason": "Query is empty."}
        if len(query) > self.max_chars:
            return {"allowed": False, "reason": f"Query exceeds {self.max_chars} characters."}

        for pattern in self.blocked_patterns:
            if pattern in normalized:
                return {"allowed": False, "reason": f"Blocked pattern detected: {pattern}"}

        return {"allowed": True, "reason": None}
