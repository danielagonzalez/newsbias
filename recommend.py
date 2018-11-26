from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model
from keras.models import model_from_json
import numpy as np
import os
import pickle

from newsapi import NewsApiClient
import json

API_KEY = 'ea9f3ca698cb4cad9277ef52d085111d'
SOURCES = ['the-new-york-times', 'politico', 'the-washington-post', 'the-hill', 'national-review', 'fox-news']
MAX_SEQUENCE_LENGTH = 150

# 0 is liberal, 1 is conservative

newsapi = NewsApiClient(api_key=API_KEY)


def get_top_articles ():
	top_headlines = newsapi.get_top_headlines(language='en', page_size=20, sources='the-huffington-post,google-news')
	headlines = top_headlines["articles"]
	article_list = []
	full_articles = []
	for article in headlines:
		if article["content"]:
			article_list.append (article["content"])
			full_articles.append (article)

	# loading tokenizer
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	
	sequences = tokenizer.texts_to_sequences(article_list)
	sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	return full_articles, sequences

def build_model (filename):
	# load json and create model
	json_file = open(filename + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(filename + ".h5")
	print("Loaded model from disk")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return loaded_model

def make_predictions (sequences, model, articles, prob_liberal):
	preds = model.predict(sequences)
	liberal_probs = preds[:, 0]
	# Get indices of top 5 liberal and top 5 conservative

	target_liberal_prob = (prob_liberal + 0.5) if prob_liberal <= 0.5 else (prob_liberal - 0.5)
	dist = np.abs (liberal_probs - target_liberal_prob)

	rec_idx = np.argsort (dist)[:3]
	for idx in rec_idx:
		print("Recommended article (liberal probability " + str(liberal_probs[idx]) + "): " + str(articles[idx]) + "\n")

	# top_conservative = np.argsort (dist)[0]
	# top_liberal = np.argsort (dist)[-1]
	
	# print("Top conservative article (probability " + str(1 - liberal_probs[top_conservative]) + "): " + str(articles[top_conservative]) + "\n")
	# print("Top liberal article: (probability " + str(liberal_probs[top_liberal]) + "): " + str(articles[top_liberal]))


articles, text = get_top_articles ()
model = build_model ('model')
make_predictions (text, model, articles, 0.2)