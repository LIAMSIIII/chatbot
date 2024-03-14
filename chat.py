import random
import json
import pickle
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load pre-trained machine learning model using pickle
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

neural_net_model = NeuralNet(input_size, hidden_size, output_size).to(device)
neural_net_model.load_state_dict(model_state)
neural_net_model.eval()

bot_name = "Sam"
print("Welcome to the Heart Disease Risk Assessment Chatbot! I'll be asking you a series of questions to assess your risk of heart disease. Let's get started!")

# Initialize dictionary to store user inputs
user_inputs = {}

for intent in intents['intents']:
    if intent['tag'] in ['age_input', 'sex_input', 'chest_pain_input', 'blood_pressure_input', 'cholesterol_input', 'blood_sugar_input', 'ecg_results_input', 'heart_rate_input', 'angina_input', 'st_depression_input', 'st_slope_input', 'vessels_input', 'thal_input']:
        # Ask the user the question specified in the patterns
        pattern = intent['patterns'][0]
        formatted_pattern = pattern.format(**user_inputs) if user_inputs else pattern  # Format the question with user inputs if available
        print(formatted_pattern)  # Prompt user with the question
        user_input = input("You: ")
        user_inputs[intent['tag']] = user_input  # Store user's response with corresponding tag

# Convert user inputs to the appropriate data types
# Assuming all inputs are numeric for simplicity
model_inputs = [float(user_inputs[tag]) for tag in tags if tag in user_inputs]

# Make predictions using the pre-trained model
prediction = model.predict([model_inputs])

# Display prediction result to the user
if prediction == 1:
    print(f"{bot_name}: Based on your inputs, you might have a heart disease. Please consult a medical professional for further evaluation.")
else:
    print(f"{bot_name}: Based on your inputs, you are less likely to have a heart disease. However, it's always important to maintain a healthy lifestyle.")

# End the conversation
print(f"{bot_name}: Thank you for using the Heart Disease Risk Assessment Chatbot. Take care!")
