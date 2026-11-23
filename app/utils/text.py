"""Small, dependency-free text helpers shared across modules."""
import re

STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on for and or with by at as
    from that this these those it its do does did what which who whom how when where
    why can could should would will shall may might must i you we they he she them
    our your their about into over under than then so if not no""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?::[0-9]+)?")


def stem(word: str) -> str:
    """Very light stemmer: enough to match 'leaves'/'leave', 'timings'/'timing'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, drop_stopwords: bool = False):
    tokens = [stem(t) for t in _TOKEN.findall(text.lower())]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
