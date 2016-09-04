import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Load Data
train_df = pd.read_csv("scripts/data/train.csv")
test_df = pd.read_csv("scripts/data/test.csv")
validation_df = pd.read_csv("scripts/data/valid.csv")
y_test = np.zeros(len(test_df))

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)

# Evaluate Random predictor
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))