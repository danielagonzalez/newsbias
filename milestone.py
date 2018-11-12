# CS229 Project Milestone
import requests
import random
import numpy as np

def get_data():
	nytimes_url = (
		'https://newsapi.org/v2/everything?'
		'sources=the-new-york-times&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)
	politico_url = (
		'https://newsapi.org/v2/everything?'
		'sources=politico&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)
	washpo_url = (
		'https://newsapi.org/v2/everything?'
		'sources=washington-post&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)
	thehill_url = (
		'https://newsapi.org/v2/everything?'
		'sources=the-hill&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)
	review_url = (
		'https://newsapi.org/v2/everything?'
		'sources=national-review&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)
	fox_url = (
		'https://newsapi.org/v2/everything?'
		'sources=fox-news&'
		'pageSize=100&'
		'apiKey=5694075ebf3948a7b12095c01d95c2f4'
	)

	liberal = requests.get(nytimes_url).json()['articles']
	liberal.extend(requests.get(politico_url).json()['articles'])
	liberal.extend(requests.get(washpo_url).json()['articles'])

	conservative = requests.get(thehill_url).json()['articles']
	conservative.extend(requests.get(review_url).json()['articles'])
	conservative.extend(requests.get(fox_url).json()['articles'])

	data = {}
	article_id = 0
	for article in liberal:
		if article['description']:
			data[article_id] = (article['description'], 0)
			article_id += 1
	for article in conservative:
		if article['description']:
			data[article_id] = (article['description'], 1)
			article_id += 1

	return data

def get_words(article):
	try:
		words = article.split(' ')
		normalized_words = [word.lower() for word in words]
		return normalized_words
	except:
		return []

def create_dictionary(articles):
	dictionary = {}

	index = 0
	for i in range(len(articles)):
		article = articles[i]
		words = get_words(article)

		for word in words:
			if word not in dictionary:
				dictionary[word] = index
				index += 1

	return dictionary

def transform_text(articles, dictionary):
	m, n = len(articles), len(dictionary)
	array = np.zeros((m, n))
	for i in range(len(articles)):
		article = get_words(articles[i])
		for word in article:
			if word in dictionary:
				index = dictionary[word]
				array[i][index] += 1

	return array

def fit_naive_bayes_model(matrix, labels):
	all_count_conservative = 0
	all_count_liberal = 0
	word_counts_conservative = {}
	word_counts_liberal = {}
	count_conservative = 0
	count_liberal = 0

	words_conservative = set()
	words_liberal = set()

	for i in range(len(labels)):
		message = matrix[i]
		if labels[i] == 0:
			count_liberal += 1
			for word in range(len(message)):
				if message[word] != 0.0:
					words_liberal.add(word)
				all_count_liberal += message[word]
				if word not in word_counts_liberal:
					word_counts_liberal[word] = message[word]
				else:
					word_counts_liberal[word] += message[word]
		else:
			count_conservative += 1
			for word in range(len(message)):
				if message[word] != 0.0:
					words_conservative.add(word)
				all_count_conservative += message[word]
				if word not in word_counts_conservative:
					word_counts_conservative[word] = message[word]
				else:
					word_counts_conservative[word] += message[word]

	k = matrix.shape[0]
	word_p_conservative = [(count+1.0)/(all_count_conservative+k) for word, count in word_counts_conservative.items()]
	word_p_liberal = [(count+1.0)/(all_count_liberal+k) for word, count in word_counts_liberal.items()]
	p_conservative = count_conservative*1.0/len(labels)
	p_liberal = count_liberal*1.0/len(labels)

	return word_p_conservative, word_p_liberal, p_conservative, p_liberal

def predict_from_naive_bayes_model(model, matrix):
	word_p_conservative, word_p_liberal, p_conservative, p_liberal = model
	predictions = []
	for i in range(matrix.shape[0]):
		message = matrix[i] 
		p_yes = 0
		p_no = 0

		for word in range(len(message)):
			p_yes += np.log(word_p_conservative[word])*message[word]
			p_no += np.log(word_p_liberal[word])*message[word]
		
		p_yes += np.log(p_conservative)
		p_no += np.log(p_liberal)

		if p_yes > p_no:
			predictions.append(1)
		else:
			predictions.append(0)

	return predictions

def main():
	data = get_data()

	train_size = int(len(data)*0.6)
	train_ids = random.sample(list(data.keys()), train_size)
	train_articles = []
	train_labels = []
	for train_id in train_ids:
		train_articles.append(data[train_id][0])
		train_labels.append(data[train_id][1])
		del data[train_id]

	val_size = train_size/3
	val_ids = random.sample(list(data.keys()), val_size)
	val_articles = []
	val_labels = []
	for val_id in val_ids:
		val_articles.append(data[val_id][0])
		val_labels.append(data[val_id][1])
		del data[val_id]

	test_articles = []
	test_labels = []
	for article in data.values():
		test_articles.append(article[0])
		test_labels.append(article[1])
	
	dictionary = create_dictionary(train_articles)
	train_matrix = transform_text(train_articles, dictionary)

	val_matrix = transform_text(val_articles, dictionary)
	test_matrix = transform_text(test_articles, dictionary)
 
	naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
	naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
	
	print naive_bayes_predictions
	print test_labels

	naive_bayes_accuracy = 0.0
	for i in range(len(test_labels)):
		prediction = naive_bayes_predictions[i]
		label = test_labels[i]
		if prediction == label:
			naive_bayes_accuracy += 1

	naive_bayes_accuracy /= len(test_labels)
	print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

if __name__ == "__main__":
	main()