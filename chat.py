import random
import json
import pickle
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import warnings

# Suppress sklearn warning
warnings.filterwarnings("ignore", category=UserWarning)

# Load pre-trained machine learning model using pickle
with open('heardisease_model.pkl', 'rb') as f:
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

# Define the desired order of inputs
input_order = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestoral',
    'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_induced_angina',
    'st_depression', 'st_slope', 'num_major_vessels', 'thal'
]

# Ask questions and collect user inputs
for input_name in input_order:
    # Find the intent corresponding to the input
    intent = next((intent for intent in intents['intents'] if intent['tag'] == input_name), None)
    if intent:
        # Ask the user the question specified in the patterns
        pattern = intent['patterns'][0]
        print(pattern)
        while True:
            user_input = input("You: ")
            
            # Translate 'yes'/'no' responses to numerical values
            if user_input.lower() in ['yes', 'y']:
                user_input = 1
                break
            elif user_input.lower() in ['no', 'n']:
                user_input = 0
                break
            elif input_name == 'sex':  # Translate gender inputs
                if user_input.lower() == 'male':
                    user_input = 1
                    break
                elif user_input.lower() == 'female':
                    user_input = 0
                    break
            elif user_input.isdigit():  # Check if input is numeric
                user_input = float(user_input)
                break
            else:
                print("Invalid input! Please enter a valid input.")

        # Store user's response with corresponding input name
        user_inputs[input_name] = float(user_input)
    else:
        pass  # Skip error message

# Sort the inputs according to the desired order
sorted_inputs = [
    user_inputs.get('age', ''), 
    user_inputs.get('sex', ''), 
    user_inputs.get('chest_pain_type', ''), 
    user_inputs.get('resting_blood_pressure', ''), 
    user_inputs.get('serum_cholestoral', ''), 
    user_inputs.get('fasting_blood_sugar', ''), 
    user_inputs.get('resting_ecg', ''), 
    user_inputs.get('max_heart_rate', ''), 
    user_inputs.get('exercise_induced_angina', ''), 
    user_inputs.get('st_depression', ''), 
    user_inputs.get('st_slope', ''), 
    user_inputs.get('num_major_vessels', ''), 
    user_inputs.get('thal', '')
]

# Convert sorted inputs to float values
model_inputs = [float(input_value) for input_value in sorted_inputs]

# Make predictions using the pre-trained model
prediction = model.predict([model_inputs])

# Display prediction result to the user
if prediction == 1:
    print(f"{bot_name}: Based on your inputs, you might have a heart disease. Please consult a medical professional for further evaluation.")
else:
    print(f"{bot_name}: Based on your inputs, you are less likely to have a heart disease. However, it's always important to maintain a healthy lifestyle.")

# End the conversation
print(f"{bot_name}: Thank you for using the Heart Disease Risk Assessment Chatbot. Take care!")
