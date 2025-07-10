import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embedding_function():
    """
    Returns the embedding function for the RAG system.

    This function is configured to use the Google Gemini API for embeddings.
    It uses the "models/embedding-001" model.
    """

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please get one from https://aistudio.google.com/app/apikey.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    return embeddings