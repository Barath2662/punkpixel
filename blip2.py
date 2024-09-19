from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the BLIP processor and the VQA model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def answer_question(image_path, question, max_new_tokens=50):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and question, prepare it for the model
    inputs = blip_processor(images=image, text=question, return_tensors="pt")
    
    # Generate the answer using the VQA model
    output = blip_qa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode the answer
    answer = blip_processor.decode(output[0], skip_special_tokens=True)
    
    if not answer:
        return "Sorry, I couldn't generate an answer."
    return answer

# Define a function to interact with the VQA chatbot
def vqa_chatbot(image_path):
    print("Welcome to the VQA chatbot!")
    while True:
        question = input("Ask a question about the image (or type 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            print("Exiting chatbot.")
            break
        answer = answer_question(image_path, question)
        print(f"Answer: {answer}")

# Upload and define the image
image_path = "C:/Users/Harivenkat/Downloads/cat.jpg"

# Start the VQA chatbot
vqa_chatbot(image_path)