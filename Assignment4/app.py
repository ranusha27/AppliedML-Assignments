from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)
model = joblib.load("logit_model.joblib")


@app.route('/score', methods=['POST'])
def get_score():
    data = request.json
    text = data.get('text', '')
    
    # Get prediction and propensity score using your score function
    prediction, propensity_score = score(text, model, 0.5)

    response = {
        'prediction': prediction,
        'propensity': propensity_score
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

