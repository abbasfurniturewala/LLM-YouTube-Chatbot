import os
import re
import ast
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import open
#import Levenshtein

import pandas as pd
import numpy as np
from dateutil import parser
#import isodate

# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import seaborn as sns
#sns.set(style="darkgrid", color_codes=True)

# Google API
from googleapiclient.discovery import build

#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud

from IPython.display import JSON
from datetime import datetime, timedelta
# ##Constants##


"""model_name = "your_model_name"  # Replace with the actual model name
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)"""
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

answers = []

def ans_generator(question, inputdata):
    #filtered_data = videos1[videos1['col1'] == 'abc']
    inputs = tokenizer(
        inputdata,
        question,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # You may need to adjust max_length as per your model's requirements
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the answer from the model's output
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    #answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end + 1]))
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end + 1])

    answers.append({
        "question": question,
        "answer": answer})
    final_answer(answer)
    return answers

def final_answer(answer):
   
    text = answer

    # Preprocess the text (remove special tokens)
    text = text.replace("[CLS]", "").replace("[SEP]", "")
    comments = [comment.strip() for comment in text.split(",")]

    # Define your question
    question = "What is the sentiment of these comments?"

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # Combine comments into a single string
    combined_comments = " ".join(comments)

    # Encode the combined comments and question
    inputs = tokenizer(
        question,
        combined_comments,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    # Use the BERT model to get the answer
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer start and end positions
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Extract the answer text
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end + 1])

    #answers_df = pd.DataFrame(answer)
    #answer.to_excel("finans.xlsx", index=False)
    # You can now analyze the answer as needed for your specific question.
    print("Final Answer:", answer)


def checkSimilarity(inputques, questions):
    #distance = Levenshtein.distance(str1, str2)
    #similarity = 1 - (distance / max(len(str1), len(str2))
    #print("Similarity:", similarity)
    matchedques = ""
    for question in questions:
        vectorizer = CountVectorizer().fit_transform([inputques, question])
        vectors = vectorizer.toarray()

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(vectors)
        if cosine_sim[0][1] >= 0.8:
            #similarity = 1
            matchedques = question
            break;
    return matchedques

if __name__ == '__main__':
    data = pd.read_excel("trends.xlsx", header=0)
    """questions = ["What is the video about based on the comments?",
    "How did users respond to this video",
    "Tell me the key trends discussed in the video.",
    "What are the trending topics discussed in the video.",
    "What are the positive sentiments related to this video?",
    "Tell me about the negative sentiments.",
    "List the more discussed topics",
    "Explain the overall sentiment for this video.",
    "Give me insights into the neutral sentiment."]
    for index, row in data.iterrows():
        video_id = row["video_ID"]
        comments = row["comments"]
        neutral_score = row["neutral_scores"]
        positive_score = row["positive_scores"]
        negative_score = row["negative_scores"]
        compound_score = row["sentiment_scores"]
        trends = row["trends"]
        for question in questions:
            question_encoding = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
            inputs = tokenizer(question, return_tensors="pt")
            start_logits, end_logits = model(**inputs)
            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits) + 1
            #answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

            answers.append({
            "video_id": video_id,
            "question": question,
            "answer": answer,
            "comments": comments,
            "neutral_sentiment_score": neutral_score,
            "positive_sentiment_score": positive_score,
            "negative_sentiment_score": negative_score,
            "compound_score": compound_score,
            "trends": trends})
    answers_df = pd.DataFrame(answers)
    answers_df.to_excel("answers.xlsx", index=False)
    
    num_answers_to_print = 5  # Adjust as needed
    for answer in answers[:num_answers_to_print]:
        print(f"Video ID: {answer['video_id']}")
        print(f"Question: {answer['question']}")
        print(f"Answer: {answer['answer']}")
        print(f"Comments: {answer['comments']}")
        print(f"Neutral Sentiment Scores: {answer['neutral_sentiment_scores']}")
        print(f"Positive Sentiment Scores: {answer['positive_sentiment_scores']}")
        print(f"Negative Sentiment Scores: {answer['negative_sentiment_scores']}")
        print(f"Compound Scores: {answer['compound_scores']}")
        print(f"Trends: {answer['trends']}")
        print("\n")"""
    # Filter the data for video_id 976
    video_data = pd.DataFrame(data)

    inputques = "List the most discussed topics."
    video_id='iw6WXQ_-iLQ'
    #EaSLfSGdGPM, iw6WXQ_-iLQ, 'XYM217gqWdk'
    # Combine comments into a single text
    """video = video_data['video_id']
    comments_text = video_data['comments']
    neutral_score = video_data["neutral_scores"]
    positive_score = video_data["positive_scores"]
    negative_score = video_data["negative_scores"]
    compound_score = video_data["sentiment_scores"]"""
            
    # Define questions for the BERT model
    questions1 = ["What is the video about based on the comments?", "List the most discussed topics."]
    similarity = checkSimilarity(inputques, questions1)
    if len(similarity)>1:
        #filtered_data = videos1[videos1['col1'] == 'abc']
        inputdata= video_data[video_data['VIDEO_ID']==video_id]
        comments_list = inputdata['COMMENTS'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
        answers = ans_generator(similarity, " ".join(comments_list))

    """questions3 = ["What are the top trends in the comments?", "Identify trends in the comments."]
    similarity = checkSimilarity(inputques, questions3)
    if len(similarity)>1:
        #filtered_data = videos1[videos1['col1'] == 'abc']
        inputdata= video_data[video_data['VIDEO_ID']==video_id]
        answers = ans_generator(similarity, inputdata['trends'])

    questions2 = ["How did users respond to this video","Explain the overall sentiment for this video."]
    #if inputques in questions:
    similarity = checkSimilarity(inputques, questions2)
    if len(similarity)>1:
        inputdata= video_data[video_data['VIDEO_ID']==video_id]
        #inputdata= video_data[['VIDEO_ID','neutral_scores','positive_scores','negative_scores','sentiment_scores']]
        answers = ans_generator(similarity, inputdata['sentiment_scores'])

    questions4 = ["What are the positive sentiments related to this video?"]
    #if inputques in questions:
    similarity = checkSimilarity(inputques, questions4)
    if len(similarity)>1:
        inputdata= video_data[video_data['VIDEO_ID']==video_id]
        #inputdata= video_data[['VIDEO_ID','neutral_scores','positive_scores','negative_scores','sentiment_scores']]
        answers = ans_generator(similarity, inputdata['positive_scores'])
    
    questions5 = ["Was the response negative?"]
    #if inputques in questions:
    similarity = checkSimilarity(inputques, questions5)
    if len(similarity)>1:
        inputdata= video_data[video_data['VIDEO_ID']==video_id]
        #inputdata= video_data[['VIDEO_ID','neutral_scores','positive_scores','negative_scores','sentiment_scores']]
        answers = ans_generator(similarity, inputdata['negative_scores'])"""
    
    #print(answers)

    answers_df = pd.DataFrame(answers)
    answers_df.to_excel("answers.xlsx", index=False)



