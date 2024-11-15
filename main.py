from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import json  # Added import for json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model, tokenizer, and label encoder
model = tf.keras.models.load_model('chatbot_model.h5')

with open('words.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('classes.pkl', 'rb') as enc_file:
    label_encoder = pickle.load(enc_file)

# Initialize FastAPI app
app = FastAPI()

# Define request body for the message
class Message(BaseModel):
    message: str

# Define the endpoint for chatbot response
@app.post("/chatbot")
def get_response(message: Message):
    try:
        # Preprocess the input message
        sequence = tokenizer.texts_to_sequences([message.message])
        padded_sequence = pad_sequences(sequence, maxlen=20, truncating='post')
        
        # Get the predicted class
        prediction = model.predict(padded_sequence)
        predicted_class_index = np.argmax(prediction)
        
        # Get the intent label (tag)
        intent = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get a response for the predicted intent
        with open("intents.json") as file:
            data = json.load(file)
        
        response = ""
        for intent_data in data["intents"]:
            if intent_data["tag"] == intent:
                response = np.random.choice(intent_data["responses"])  # Random response from intent
                break
        
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

