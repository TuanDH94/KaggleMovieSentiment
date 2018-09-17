from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer

stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()
from string import punctuation
import re

train_data = pd.read_csv('train.tsv/train.tsv', '\t')
test_data = pd.read_csv('test.tsv/test.tsv', '\t')

test_data['Sentiment'] = -999
df = pd.concat([train_data, test_data], ignore_index=True)


def clean_review(review_col):
    review_corpus = []
    for i in range(0, len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        # review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = ' '.join(review)
        review_corpus.append(review)
    return review_corpus


df['clean_review'] = clean_review(df.Phrase.values)
print(df.head())

df_train = df[df.Sentiment != -999]
#df_train.shape
df_test = df[df.Sentiment == -999]
df_test.drop('Sentiment', axis=1, inplace=True)

# split train/test
train_text = df_train.clean_review.values
test_text = df_test.clean_review.values
target = df_train.Sentiment.values
y = to_categorical(target)
print(train_text.shape, target.shape, y.shape)

X_train_text,X_val_text,y_train,y_val = train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)


all_words=' '.join(X_train_text)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)

r_len = []
for text in X_train_text:
    word = word_tokenize(text)
    l = len(word)
    r_len.append(l)

MAX_REVIEW_LEN = np.max(r_len)

max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 10
num_classes=y.shape[1]

# to sequence
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)

# padding
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)

# lstm model
model=Sequential()
model.add(Embedding(max_features,250,mask_zero=True))
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()

history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=5, batch_size=batch_size, verbose=1)

y_pred=model.predict_classes(X_test)
