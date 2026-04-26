import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Safely fetch the API key
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not LLM_API_KEY:
    print("Warning: LLM_API_KEY not found in environment variables. Make sure your .env file is set up.")