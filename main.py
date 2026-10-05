import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("HF_TOKEN not found. Please check your .env file.")
    exit()

# Create Inference Client
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

# Take topic from user
topic = input("Enter topic for flashcards: ")

# Create prompt
prompt = f"""
Create 10 flashcards about {topic}.

Use this format:

Card 1:
Q: question
A: answer

Card 2:
Q: question
A: answer

Card 3:
Q: question
A: answer

Card 4:
Q: question
A: answer

Card 5:
Q: question
A: answer

Card 6:
Q: question
A: answer

Card 7:
Q: question
A: answer

Card 8:
Q: question
A: answer

Card 9:
Q: question
A: answer

Card 10:
Q: question
A: answer


Keep the questions and answers short and easy to understand.
"""

try:
    # Send request to Hugging Face Inference Service
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    # Display generated flashcards
    print("\n========== FLASHCARDS ==========\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("Error:", e)