"""
Simple RNN Example using TensorFlow/Keras
Task: Sentiment Classification on toy text data
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = [
    "I love deep learning",
    "This course is amazing",
    "I hate bugs",
    "The project is terrible",
    "Machine learning is great",
    "I dislike errors"
]

labels = np.array([1, 1, 0, 0, 1, 0])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
max_len = 5
X = pad_sequences(sequences, maxlen=max_len)

vocab_size = len(tokenizer.word_index) + 1

# Build RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=max_len),
    SimpleRNN(16, activation="tanh"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(X, labels, epochs=20, verbose=1)

# Prediction
test_text = ["deep learning is amazing"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=max_len)

prediction = model.predict(test_pad)
print("Prediction:", prediction[0][0])
