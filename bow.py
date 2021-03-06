import numpy as np
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

SOURCES = ['the-new-york-times', 'politico', 'the-washington-post', 'the-hill', 'national-review', 'fox-news']
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 100
N_CLASSES = 2
MAX_NB_WORDS = 50000
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.2

def use_existing_data ():
	data = []
	labels = []

	for idx, source in enumerate (SOURCES):
		with open('data/' + source + '.json', "r") as read_file:
			articles = json.load(read_file)
			label = 0 if idx < 3 else 1
			for article in articles:
				if article['content']:
					data.append (article['content'].encode("ascii", "ignore").lower())
					labels.append (label)

	return data, labels

def preprocess (text, labels, save=True):
	train_text, test_text, train_y, test_y = train_test_split (text, labels,test_size = 0.2)

	texts_train = train_text
	# print(texts_train)
	texts_test = test_text
	# print texts_test[0]
	# print texts_test[1]
	# print texts_test[-1]
	# print test_y[0]
	# print test_y[1]
	# print test_y[-1]
	
	tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
	tokenizer.fit_on_texts(texts_train)
	sequences = tokenizer.texts_to_sequences(texts_train)
	sequences_test = tokenizer.texts_to_sequences(texts_test)

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	print('Found %s unique docs.' % tokenizer.document_count)

	# index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
	# print(" ".join([index_to_word[i] for i in sequences[0]]))

	x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
	print('Shape of data tensor:', x_train.shape)
	print('Shape of data test tensor:', x_test.shape)

	y_train = train_y
	y_test = test_y

	y_train = to_categorical(np.asarray(y_train))
	print('Shape of train label tensor:', y_train.shape)

	if save:
		with open('tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return x_train, x_test, y_train, y_test

def train_bow (x_train, x_test, y_train, y_test):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

	embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True)
	embedded_sequences = embedding_layer(sequence_input)

	average = GlobalAveragePooling1D()(embedded_sequences)
	predictions = Dense(N_CLASSES, activation='softmax')(average)

	model = Model(sequence_input, predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

	model.fit(x_train, y_train, epochs=6, batch_size=128, validation_data=(x_test, to_categorical(np.asarray(y_test))))

	output_test = model.predict(x_test)
	# # print(output_test[0])
	# # print(output_test[1])
	# # print(output_test[-1])
	print("test auc:", roc_auc_score(y_test,output_test[:,1]))

def train_lstm (x_train, x_test, y_train, y_test):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True)
	embedded_sequences = embedding_layer(sequence_input)

	x = LSTM(HIDDEN_SIZE, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB)(embedded_sequences)
	predictions = Dense(2, activation='softmax')(x)

	model = Model(sequence_input, predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, to_categorical(np.asarray(y_test))))

	output_test = model.predict(x_test)
	# print(output_test[0])
	# print(output_test[1])
	# print(output_test[-1])
	print("test auc:", roc_auc_score(y_test,output_test[:,1]))

def train_cnn_lstm (x_train, x_test, y_train, y_test):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True)
	embedded_sequences = embedding_layer(sequence_input)

	# 1D convolution with 64 output channels
	x = Conv1D(HIDDEN_SIZE/2, 5)(embedded_sequences)
	# MaxPool divides the length of the sequence by 5
	x = MaxPooling1D(5)(x)
	x = Dropout(DROPOUT_PROB)(x)
	x = Conv1D(HIDDEN_SIZE/2, 5)(x)
	x = MaxPooling1D(5)(x)
	# LSTM layer with a hidden size of 64
	x = Dropout(DROPOUT_PROB)(x)
	x = LSTM(HIDDEN_SIZE)(x)
	predictions = Dense(2, activation='softmax')(x)

	model = Model(sequence_input, predictions)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	model.fit(x_train, y_train, validation_data=(x_test, to_categorical(np.asarray(y_test))), epochs=6, batch_size=128)
	output_test = model.predict(x_test)
	# # print(output_test)
	# print(y_test)
	# print(output_test[0])
	# print(output_test[1])
	# print(output_test[-1])
	print("test auc:", roc_auc_score(y_test,output_test[:,1]))

	return model

def save_model (model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
			json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")

data, labels = use_existing_data ()
x_train, x_test, y_train, y_test = preprocess (data, labels)

train_bow (x_train, x_test, y_train, y_test)
train_lstm (x_train, x_test, y_train, y_test)
model = train_cnn_lstm (x_train, x_test, y_train, y_test)
save_model (model)