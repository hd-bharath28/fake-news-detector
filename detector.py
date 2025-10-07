import os
import joblib
import numpy as np
from config import MODEL_PATH
from utils import clean_text

def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    pipeline = joblib.load(path)
    return pipeline

def predict_text(text: str, pipeline):
    """Return (label_str, confidence_float) where label_str is 'FAKE' or 'REAL'."""
    text_clean = clean_text(text)
    proba = pipeline.predict_proba([text_clean])[0]  # [prob_real, prob_fake]
    # we used labels 0=real, 1=fake
    fake_prob = float(proba[pipeline.named_steps['clf'].classes_.tolist().index(1)])
    real_prob = float(proba[pipeline.named_steps['clf'].classes_.tolist().index(0)])
    label = "FAKE" if fake_prob >= real_prob else "REAL"
    confidence = max(fake_prob, real_prob)
    return label, confidence

def explain_text(text: str, pipeline, top_n=8):
    """
    Explain which words pushed the prediction.
    Returns dict with keys: top_positive (towards FAKE), top_negative (towards REAL)
    """
    text_clean = clean_text(text)
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']  # assumes linear classifier
    X_vec = tfidf.transform([text_clean])  # sparse
    feature_names = tfidf.get_feature_names_out()
    # For binary logistic regression, coef_.shape = (1, n_features)
    coef = clf.coef_[0]  # coefficients for class 1 (FAKE)
    x_arr = X_vec.toarray()[0]
    contribs = x_arr * coef  # element-wise contributions toward FAKE
    # get indices where x_arr > 0 (words present)
    present_idx = np.where(x_arr > 0)[0]
    if present_idx.size == 0:
        return {"top_positive": [], "top_negative": []}
    # contributions among present features
    pres_contribs = contribs[present_idx]
    pres_feats = feature_names[present_idx]
    sorted_idx = np.argsort(pres_contribs)  # ascending
    # negative contributions push toward REAL, positive push toward FAKE
    top_neg = []
    top_pos = []
    # top negative (most negative contribution)
    neg_count = min(top_n, len(sorted_idx))
    # take from beginning for negatives
    for idx in sorted_idx[:neg_count]:
        feat = pres_feats[np.where(present_idx==present_idx[idx])[0][0]] if False else pres_feats[np.where(present_idx==present_idx[idx])[0][0]]  # never executed
    # Simpler method:
    sorted_indices_global = present_idx[sorted_idx]  # present_idx ordered by contrib asc
    # top negatives
    for i in sorted_indices_global[:top_n]:
        top_neg.append((feature_names[i], float(contribs[i])))
    # top positives (largest contrib)
    for i in sorted_indices_global[::-1][:top_n]:
        top_pos.append((feature_names[i], float(contribs[i])))
    return {"top_positive": top_pos, "top_negative": top_neg}