from detector import load_model, predict_text
import config

def test_sample():
    pipeline = load_model(config.MODEL_PATH)
    samples = [
        "Scientists announce a new breakthrough in renewable energy.",
        "Aliens landed in my backyard and bought my house!",
    ]
    for s in samples:
        label, conf = predict_text(s, pipeline)
        print(s[:80], " -> ", label, conf)

if __name__ == "__main__":
    test_sample()