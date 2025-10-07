import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is available (train script will also download)
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Basic cleaning: remove URLs, HTML tags, non-alpha, lower, remove stopwords."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)