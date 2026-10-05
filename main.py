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

# Take question from user
question = input("Ask your question: ")

# Create prompt
prompt = f"""
Answer the following question:

{question}

Give a short, simple, and easy-to-understand answer.
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

    # Display answer
    print("\n========== ANSWER ==========\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("Error:", e)