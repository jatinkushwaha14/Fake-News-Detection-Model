# ## Access DataSet
# https://www.kaggle.com/c/fake-news/data

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding, TextVectorization, Concatenate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'train.csv')
df.head()

df.drop(columns=['id'], inplace=True)
df = df.dropna()
df.head()

def make_transform(X):
    title=X['title']
    author=X['author']
    text=X['text']

    title_vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1000, output_mode='int')
    author_vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=500, output_mode='int')
    text_vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1800, output_mode='int')

    title_vectorizer.adapt(title.values)
    author_vectorizer.adapt(author.values)
    text_vectorizer.adapt(text.values)

    vectorized_titles = np.array(title_vectorizer(title.values))
    vectorized_authors = np.array(author_vectorizer(author.values))
    vectorized_texts = np.array(text_vectorizer(text.values))
    vectorized = np.concatenate([vectorized_titles, vectorized_authors, vectorized_texts], axis=1)

    return vectorized

X=df.iloc[:,0:3]
y=df.iloc[:,-1].values

vectorized=make_transform(X)

X_train, X_val, y_train, y_val = train_test_split(vectorized,y,test_size=0.2,random_state=1)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16).prefetch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16).prefetch(8)

model = Sequential([
    Embedding(20000+1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_dataset, epochs=3, validation_data=val_dataset)

import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

test_df=pd.read_csv(r'test.csv')
y=pd.read_csv(r'submit.csv')

result=pd.concat([test_df, y], axis=1)
result.columns = ['id1', 'title', 'author', 'text', 'id2', 'label']
result.head()

mismatched_entries = result[result['id1'] != result['id2']]
mismatched_entries

result.drop(columns=['id1','id2'], inplace=True)
result = result.dropna()

X_test=df.iloc[:,0:3]
y_test=df.iloc[:,-1].values

vectorized=make_transform(X_test)

test_predictions = (model.predict(vectorized) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')

