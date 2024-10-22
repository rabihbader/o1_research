from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Retrieve the token
token = os.getenv('PERPLEXITY_API_KEY')

# Use the token
print(f"My token is: {token}")
