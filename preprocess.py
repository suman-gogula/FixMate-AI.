"""
preprocess.py - NLP preprocessing pipeline for FixMate AI

Pipeline (in order):
    1. Lowercasing
    2. Punctuation / symbol removal
    3. Tokenization
    4. Stop-word removal
    5. Stemming (Porter Stemmer)

Every step is a separate small function so students can explain it easily.
"""

import re

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# ---------------------------------------------------------------------------
# Stop-word list
# ---------------------------------------------------------------------------
# Built-in backup list, used only if the NLTK stop-word corpus is unavailable
# (for example when there is no internet to download it).
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself of
off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves please
""".split())


def _load_stopwords():
    """Try NLTK's English stop-word list; fall back to the built-in list."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)   # one-time download
            words = set(stopwords.words("english"))
    except Exception:
        words = set(_FALLBACK_STOPWORDS)
    # Negation words change the meaning ("not working"), so we KEEP them.
    words -= {"not", "no", "nor"}
    words.add("please")
    return words


STOP_WORDS = _load_stopwords()
_TOKENIZER = RegexpTokenizer(r"[a-z]+")      # a token = a run of letters
_STEMMER = PorterStemmer()


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------
def to_lowercase(text: str) -> str:
    """Step 1: 'Water' and 'water' must be treated as the same word."""
    return text.lower().strip()


def remove_punctuation(text: str) -> str:
    """Step 2: keep only letters and spaces (drops . , ! ? digits symbols)."""
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    """Step 3: split the sentence into individual words (tokens)."""
    return _TOKENIZER.tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Step 4: drop very common words (the, is, in ...) that carry little meaning."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem_tokens(tokens: list) -> list:
    """Step 5: reduce words to their root, e.g. leaking -> leak, working -> work."""
    return [_STEMMER.stem(t) for t in tokens]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def preprocess_steps(text: str) -> dict:
    """Run the pipeline and return the output of EVERY step (for display)."""
    lowered = to_lowercase(text)
    no_punct = remove_punctuation(lowered)
    tokens = tokenize(no_punct)
    no_stop = remove_stopwords(tokens)
    stemmed = stem_tokens(no_stop)
    return {
        "original": text,
        "lowercase": lowered,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "no_stopwords": no_stop,
        "stemmed": stemmed,
        "clean_text": " ".join(stemmed),
    }


def preprocess(text: str) -> str:
    """Return only the final cleaned string (used for training and prediction)."""
    return preprocess_steps(text)["clean_text"]


if __name__ == "__main__":
    demo = "Water is leaking continuously from my bathroom ceiling."
    for key, value in preprocess_steps(demo).items():
        print(f"{key:15}: {value}")
