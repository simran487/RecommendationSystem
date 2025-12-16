import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# --- Configuration ---
# Define the path where the model assets were saved by data_analysis.py
MODEL_ASSET_DIR = Path("model_assets")

# Filenames for the saved assets (Must match data_analysis.py)
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE = "query_features.pkl" 
TRAIN_DATA_FILE = "trained_df.pkl" 

# --- FastAPI Setup ---
app = FastAPI(
    title="Assessment Recommendation API",
    description="API for recommending SHL assessments based on a job query.",
    version="1.0.0"
)

# --- State Variables ---
# These will hold the loaded ML components and data
tfidf_vectorizer = None
query_features = None
trained_df = None


# --- Input Schema for API ---
class Query(BaseModel):
    """Schema for the incoming user query."""
    user_query: str = Field(
        ...,
        example="Looking for a candidate assessment for a senior data scientist proficient in Python, SQL, and machine learning models.",
        description="The job description or candidate requirement query."
    )


# --- Initialization Function (Runs on startup) ---
@app.on_event("startup")
async def load_model_assets():
    """
    Load the trained TF-IDF vectorizer, the TF-IDF matrix, and the original 
    training DataFrame from the model_assets folder.
    """
    global tfidf_vectorizer, query_features, trained_df
    
    print("--- Loading Model Assets ---")
    
    try:
        # 1. Load the TF-IDF Vectorizer
        vectorizer_path = MODEL_ASSET_DIR / TFIDF_VECTORIZER_FILE
        tfidf_vectorizer = joblib.load(vectorizer_path)
        print(f"Loaded TF-IDF Vectorizer from: {vectorizer_path}")
        
        # 2. Load the TF-IDF Matrix 
        tfidf_matrix_path = MODEL_ASSET_DIR / TFIDF_MATRIX_FILE
        query_features = joblib.load(tfidf_matrix_path)
        print(f"Loaded TF-IDF Matrix from: {tfidf_matrix_path}")

        # 3. Load the Original Training Data (for mapping URLs)
        train_data_path = MODEL_ASSET_DIR / TRAIN_DATA_FILE
        trained_df = joblib.load(train_data_path)
        print(f"Loaded Training Data (DataFrame) from: {train_data_path}")

    except FileNotFoundError as e:
        print(f"Error loading asset: {e}")
        # Raise an exception to prevent the application from starting if assets are missing
        raise RuntimeError(
            f"Required model asset not found. Ensure 'model_assets' folder exists "
            f"and contains the necessary files: {e.filename}"
        ) from e
    except Exception as e:
        print(f"An unexpected error occurred during asset loading: {e}")
        raise RuntimeError(f"Failed to load model assets: {e}") from e

    print("--- Model Assets Loaded Successfully ---")


# --- Prediction Logic ---
def get_recommendations(query: str, top_n: int = 10) -> List[str]:
    """
    Calculates cosine similarity between the user query and all training queries,
    and returns the top N recommended assessment URLs.
    """
    if tfidf_vectorizer is None or query_features is None or trained_df is None:
        raise RuntimeError("Model assets are not fully loaded. API is not ready.")

    # 1. Vectorize the user query using the *trained* vectorizer
    query_vec = tfidf_vectorizer.transform([query])
    
    # 2. Calculate Cosine Similarity
    # Use the global variable 'query_features' (the loaded TF-IDF matrix)
    cosine_sims = cosine_similarity(query_vec, query_features).flatten()
    
    # 3. Get the indices of the top N most similar queries
    top_indices = cosine_sims.argsort()[::-1][:top_n]
    
    # 4. Map indices back to the assessment URLs
    # Use the global variable 'trained_df'
    recommended_urls = trained_df.iloc[top_indices]['Assessment_url'].tolist()
    
    return recommended_urls


# --- API Endpoint ---
@app.post("/recommend", response_model=List[str])
async def recommend_assessments(query: Query):
    """
    Accepts a user query (job description) and returns a list of up to 10 
    recommended SHL assessment URLs.
    """
    try:
        # The user query is available at query.user_query
        urls = get_recommendations(query.user_query)
        return urls
    except RuntimeError as e:
        # Handles cases where model loading failed during startup
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Handles unexpected prediction errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Simple health check endpoint for deployment purposes
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Recommendation API is running."}