import re

def score(text: str, model, threshold: float) -> (bool, float):

    # Preprocess input text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Transform the input text using the same vectorizer used during training
    vectorizer = model.named_steps['vectorizer']
    text_vectorized = vectorizer.transform([text])

    # Predict the probability of the positive class
    propensity_score = model['model'].predict_proba(text_vectorized)[:, 1].item()

    # Classify based on the threshold
    prediction = propensity_score >= threshold

    return prediction, propensity_score