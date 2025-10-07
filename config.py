import os

# Path settings
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_detector.joblib")

# Optional: put your OpenAI API key in environment variable OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY)