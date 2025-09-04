import json
import boto3
import pandas as pd
import io
import math
from collections import defaultdict

# Initialize S3 client
s3 = boto3.client('s3')

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.total_docs = 0

    def train(self, data):
        for text, label in data:
            words = str(text).lower().split()
            self.class_counts[label] += 1
            self.total_docs += 1
            for word in words:
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)

    def predict(self, text):
        words = str(text).lower().split()
        scores = {}
        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / self.total_docs)
            total_words = sum(self.class_word_counts[label].values())
            for word in words:
                count = self.class_word_counts[label].get(word, 0)
                log_prob += math.log((count + 1) / (total_words + len(self.vocab)))
            scores[label] = log_prob
        return max(scores, key=scores.get)

# Lambda handler
def lambda_handler(event, context):
    bucket = 'youtubedata123'  # Your S3 bucket name
    key = 'YoutubeCommentsDataSet.csv'  # Your CSV file in the bucket

    try:
        # Parse input JSON body
        if event.get('body'):
            body = json.loads(event['body'])
            comment_to_classify = body.get("comment", None)
        else:
            comment_to_classify = None

        # Load and preprocess dataset
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data))

        if 'Comment' not in df.columns or 'Sentiment' not in df.columns:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "CSV must contain 'Comment' and 'Sentiment' columns."})
            }

        df.dropna(subset=['Comment', 'Sentiment'], inplace=True)

        # Train Naive Bayes model
        training_data = list(zip(df['Comment'], df['Sentiment']))
        nb = NaiveBayesClassifier()
        nb.train(training_data)

        # Make prediction
        if comment_to_classify:
            prediction = nb.predict(comment_to_classify)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "input_comment": comment_to_classify,
                    "predicted_sentiment": prediction
                })
            }
        else:
            # If no input comment, predict for first 5 rows as sample
            df['Predicted_Sentiment'] = df['Comment'].apply(nb.predict)
            result = df[['Comment', 'Predicted_Sentiment']].head().to_dict(orient='records')
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "predictions": result
                })
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
