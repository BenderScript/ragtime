import openai
import numpy as np
from dotenv import load_dotenv


# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


load_dotenv(override=True, dotenv_path=".env")  # take environment variables from .env.
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "your api key"

# The string to generate an embedding for
main_string = "I love machine learning"

# Other strings to compare against
other_strings = ["I enjoy artificial intelligence", "The weather is sunny", "Learning about AI is fascinating",
                 "I love reading books"]

# Generate embeddings for the main string and other strings
embeddings = openai.embeddings.create(input=[main_string] + other_strings, model="text-similarity-babbage-001")

# Extract the main string's embedding
main_embedding = embeddings.data[0].embedding

# Iterate over other strings and compute similarities
for i, other_string in enumerate(other_strings):
    other_embedding = embeddings.data[i + 1].embedding
    similarity = cosine_similarity(main_embedding, other_embedding)
    print(f"Similarity between \"{main_string}\" and \"{other_string}\": {similarity}")
