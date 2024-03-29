import os
import requests
import json
import subprocess
import time
import pytest
import joblib

from score import score


@pytest.mark.skip(reason="This test should be run in python, not with pytest")
def test_score():
    model = joblib.load("logit_model.joblib")
    
    unit_tests = [test_smoke_test, 
                  test_format_test, 
                  test_prediction_value,
                  test_threshold_0,
                  test_threshold_1,
                  test_obvious_spam, 
                  test_obvious_non_spam]
    
    result = {}
    for testfun in unit_tests:
        try:
            testfun(model)
        except AssertionError:
            result[testfun.__name__] = False
        else:
            result[testfun.__name__] = True
            
    for fname in result:
        print(("PASSED:" if result[fname] else "FAILED:"), fname, '\n')
    
    nPassed = sum(result.values())
    nFailed = len(result) - nPassed
    print("{} PASSED, {} FAILED".format(nPassed, nFailed))
        

@pytest.mark.skip(reason="This test should be run in python, not with pytest")
def test_flask():
    try:
        test_flask_app()
    except AssertionError:
        print("Integration test FAILED")
    else:
        print("Integration test PASSED")
    
    
@pytest.fixture
def trained_model():
    # Load the trained pipeline from the saved file
    pipeline = joblib.load("logit_model.joblib")
    return pipeline


def test_flask_app():
    # Launch the Flask app in a separate process
    flask_process = subprocess.Popen(['python', 'app.py'])
   
    time.sleep(1)  # Wait for Flask app to start

    # Send a POST request to the /score endpoint
    data = {'text': 'Test text for prediction'}
    response = requests.post('http://localhost:5000/score', json=data)
    result = response.json()

    # Check if response contains 'prediction' and 'propensity' keys
    assert 'prediction' in result
    assert 'propensity' in result

    # Close the Flask app by terminating the process
    flask_process.terminate()
    flask_process.wait()
    
def test_smoke_test(trained_model):
    text = "This is a test"
    threshold = 0.5
    prediction, propensity_score = score(text, trained_model, threshold)
    assert isinstance(prediction, bool)
    assert isinstance(propensity_score, float)

def test_format_test(trained_model):
    text = "This is a test"
    threshold = 0.5
    prediction, propensity_score = score(text, trained_model, threshold)
    assert 0 <= propensity_score <= 1

def test_prediction_value(trained_model):
    text = "This is a test"
    threshold = 0.5
    prediction, propensity_score = score(text, trained_model, threshold)
    assert prediction in [0, 1]

def test_threshold_0(trained_model):
    text = "This is a test"
    threshold = 0
    prediction, propensity_score = score(text, trained_model, threshold)
    assert prediction == 1

def test_threshold_1(trained_model):
    text = "This is a test"
    threshold = 1
    prediction, propensity_score = score(text, trained_model, threshold)
    assert prediction == 0

def test_obvious_spam(trained_model):
    text = "Buy now!"
    threshold = 0.5
    prediction, propensity_score = score(text, trained_model, threshold)
    assert prediction == 1

def test_obvious_non_spam(trained_model):
    text = "Thanks and regards"
    threshold = 0.5
    prediction, propensity_score = score(text, trained_model, threshold)
    assert prediction == 0
    
    

