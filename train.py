import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the intents file
with open("intents.json") as file:
    data = json.load(file)

# Preprocess data
training_sentences = []
training_labels = []
labels = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Encode labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Tokenize and pad sequences
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Build the model
model = Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(set(training_labels)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# Save the model
model.save("chatbot_model.h5")

# Save tokenizer and label encoder for future use
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(label_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Model training complete and saved as chatbot_model.h5")
