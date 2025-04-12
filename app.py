from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)
# Allow CORS for your frontend's origin (adjust if needed)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:5000"]}})

# Load model and tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained("./t5_qa_model/")
    tokenizer = T5Tokenizer.from_pretrained("./t5_qa_model/")
    model.eval()
    print("Model loaded successfully:", model.training)
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['GET'])
def get_response():
    try:
        user_message = request.args.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        response = generate_response(user_message)
        return response  # Return plain text for frontend compatibility
    except Exception as e:
        print(f"Error in get_response: {e}")
        return jsonify({"error": "Internal server error"}), 500

def generate_response(user_input):
    try:
        # Format input for T5
        input_text = f"question: {user_input}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        # Generate response
        with torch.no_grad():  # Disable gradient computation for inference
            output = model.generate(
                input_ids,
                max_length=100,  # Adjust based on your needs
                num_beams=5,     # Beam search for better responses
                early_stopping=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated Response:", response)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
