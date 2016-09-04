import pandas as pd
import numpy as np
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


class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predict(self, context, utterances):
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        return np.argsort(result, axis=0)[::-1]

# Evaluate TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)
y = [pred.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))