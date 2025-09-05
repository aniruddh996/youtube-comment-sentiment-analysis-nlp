# YouTube Sentiment API (AWS Lambda)

Serverless REST API that predicts the sentiment of a YouTube comment using a from‑scratch Multinomial Naive Bayes model. The model is (re)trained from a labeled CSV in S3 on cold start and served via Amazon API Gateway → AWS Lambda.

# Features

Serverless inference (API Gateway → Lambda)

From‑scratch Naive Bayes (Laplace smoothing)

Training data in S3 (Comment, Sentiment columns)

Simple HTTP POST interface (JSON)

Works with Postman / curl

# API 

#### Endpoint: 
https://kfks2dfrmg.execute-api.us-east-1.amazonaws.com/predict?

#### Request Body: 
{
  "comment": "This video is amazing! Loved the editing and pacing."
}

#### Success Response
{
"input_comment": "This video is amazing! Loved the editing and pacing.",
"predicted_sentiment": "positive"
}

