# Fake News Detection Model

## Overview
This project is a deep learning model for detecting fake news articles. The model is built using Tensorflow and Keras, and it takes in a combination of article title, author, and text as input features to predict whether the article is fake or real.

## Dataset
This project aims to classify news articles as fake or real using a Bidirectional LSTM neural network. The dataset used is from the [Kaggle Fake News Challenge](https://www.kaggle.com/c/fake-news/data).

## Preprocessing
The dataset is preprocessed using the following steps:
1. Handling missing values: Rows with missing values are dropped.
2. Text vectorization: The article title, author, and text are vectorized using Tensorflow's `TextVectorization` layer. The maximum number of tokens for the title and text is set to 20,000, and the maximum sequence length is set to 1,000 and 1,800, respectively. The maximum number of tokens for the author is set to 10,000, and the maximum sequence length is set to 500.
3. Concatenating the vectorized features: The vectorized title, author, and text features are concatenated into a single input feature vector.

## Model Architecture
The model architecture consists of the following layers:
1. Embedding layer: This layer maps the input feature vector to a dense representation.
2. Bidirectional LSTM layer: This layer captures the sequential information in the input features.
3. Dense layers: These layers perform the final classification task, with ReLU activation functions.
4. Output layer: This layer uses a sigmoid activation function to output the probability of the article being fake.

## Training and Evaluation
The model is trained using the Adam optimizer and binary cross-entropy loss function. The training is performed on the training set, and the validation set is used for monitoring the model's performance during training.

The model's performance is evaluated on the test set, and the test accuracy is reported.

## Usage
To use the model, you can follow these steps:
1. Clone the repository.
2. Install the required dependencies (Numpy, Pandas, Tensorflow, Sklearn, Matplotlib).
3. Load the dataset and preprocess the data using the provided functions.
4. Train the model using the `model.fit()` function.
5. Evaluate the model's performance on the test set.

## Limitations and Future Improvements
- The model's performance can be further improved by exploring different model architectures, hyperparameter tuning.
- The model's robustness can be tested on more diverse and challenging datasets.
- The model can be deployed as a web application or integrated into a larger system for real-world fake news detection.

## Conclusion
This project demonstrates the use of deep learning techniques for detecting fake news articles. The model's performance can be further improved, and the approach can be extended to other text classification tasks.
