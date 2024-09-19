from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os

app = Flask(__name__)

# Set a folder to save uploaded images
UPLOAD_FOLDER = 'uploaded_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the BLIP processor and VQA model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Function to answer questions
def answer_question(image_path, question, max_new_tokens=50):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and question
    inputs = blip_processor(images=image, text=question, return_tensors="pt")
    
    # Generate the answer
    output = blip_qa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode the answer
    answer = blip_processor.decode(output[0], skip_special_tokens=True)
    
    return answer or "Sorry, I couldn't generate an answer."

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Process image uploads and questions
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')

    # Check if the user uploaded a file
    if 'image' not in request.files:
        return jsonify({'answer': 'No image uploaded.'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'answer': 'No image selected.'})

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Get the answer for the question
    answer = answer_question(image_path, question)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
