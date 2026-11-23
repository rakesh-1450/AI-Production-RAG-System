import re

_AND = re.compile(r"\s+\band\b\s+", re.IGNORECASE)

EXPANSIONS = {
    "office timings": "working hours office schedule",
    "leave": "annual leave paid leave vacation policy",
    "remote work": "work from home hybrid work",
    "authentication": "login security access control",
}


class QueryProcessor:
    def rewrite_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def expand_query(self, query: str) -> str:
        lower = query.lower()
        extra = [value for key, value in EXPANSIONS.items() if key in lower]
        return " ".join([query, *extra]) if extra else query

    def decompose_query(self, query: str):
        """Split multi-part questions on the *word* 'and' (not inside words like 'standard')."""
        parts = [p.strip(" ?.") for p in _AND.split(query)]
        # Only split when every part is a real sub-question (>= 2 words)
        if len(parts) > 1 and all(len(p.split()) >= 2 for p in parts):
            return parts
        return [query]

    def process_query(self, query: str):
        rewritten = self.rewrite_query(query)
        sub_queries = self.decompose_query(rewritten)
        expanded = [self.expand_query(q) for q in sub_queries]
        # de-duplicate while keeping order
        return list(dict.fromkeys(expanded))
