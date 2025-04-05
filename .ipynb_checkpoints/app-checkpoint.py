from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd


model = T5ForConditionalGeneration.from_pretrained("./t5_qa_model/")
tokenizer = T5Tokenizer.from_pretrained("./t5_qa_model/")
model.eval()
print(model.eval())
app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/get_response')
def get_response():
    user_message = request.args.get('message')
    response = generate_response(user_message)
    
    return response

def generate_response(user_input):
    # Your custom logic here
    # Example:
    input_text = "question:"+ user_input 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Response:", response)
    if "hello" in input_text.lower():
        return "Hey there! How can I help you today?"
    elif "bye" in input_text.lower():
        return "Goodbye! See you soon."
    else:
        return response

if __name__ == '__main__':
    app.run(debug=True)