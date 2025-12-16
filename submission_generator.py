import joblib
import pandas as pd
import os
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 
from typing import List

# --- Configuration (Must match data_analysis.py) ---
MODEL_DIR = 'model_assets'
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'query_features.pkl')
DATAFRAME_PATH = os.path.join(MODEL_DIR, 'trained_df.pkl') 

# --- Input/Output Files ---
TEST_FILE_NAME = 'Gen_AI_Dataset_Train.csv'
SUBMISSION_FILE_NAME = 'submission_results.csv' 
TOP_N_RECOMMENDATIONS = 10 # Get top 10 as a robust default

# --- Global Variables for Loaded Assets ---
tfidf_vectorizer = None
query_features = None
trained_df = None


# --- Helper Functions (Copied from data_analysis.py) ---

def load_csv_with_encoding(file_path):
    """Attempts to load a CSV file using common encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            return df
        except Exception:
            continue
    print(f"Error: Failed to load the file with all common encodings: {file_path}")
    return None

def clean_text(text):
    """Cleans text for vectorization."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_recommendations(query: str, top_n: int = 10) -> List[str]:
    """
    Calculates cosine similarity and returns the top N recommended assessment URLs.
    Uses the globally loaded model assets.
    """
    cleaned_query = clean_text(query)
    query_vec = tfidf_vectorizer.transform([cleaned_query])
    
    # Calculate Cosine Similarity
    cosine_sims = cosine_similarity(query_vec, query_features).flatten()
    
    # Get the indices of the top N most similar queries
    top_indices = cosine_sims.argsort()[::-1][:top_n]
    
    # Map indices back to the assessment URLs
    # Note: trained_df contains 'Assessment_url'
    recommended_urls = trained_df.iloc[top_indices]['Assessment_url'].tolist()
    
    # Return unique URLs while preserving rank order
    unique_urls = []
    for url in recommended_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    return unique_urls


# --- Main Execution ---
if __name__ == "__main__":
    
    print("--- Starting Submission Generation ---")
    
    # 1. Load Model Assets
    try:
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        query_features = joblib.load(FEATURES_PATH)
        trained_df = joblib.load(DATAFRAME_PATH)
        print("Model assets loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model assets. Did you run data_analysis.py? Error: {e}")
        exit()
        
    # 2. Load Test Data
    df_test = load_csv_with_encoding(TEST_FILE_NAME)
    if df_test is None:
        print(f"FATAL ERROR: Could not load Test Set file: {TEST_FILE_NAME}")
        exit()

    print(f"Test Set loaded with {len(df_test)} queries.")
    
    # 3. Generate Predictions
    submission_data = []
    
    # Iterate through unique queries in the test set
    unique_test_queries = df_test['Query'].unique()
    
    print(f"Processing {len(unique_test_queries)} unique test queries...")

    for i, query in enumerate(unique_test_queries):
        if pd.isna(query) or query.strip() == "":
            continue # Skip empty or NaN queries

        # Get recommendations using the model
        recommendations = get_recommendations(query, top_n=TOP_N_RECOMMENDATIONS)
        
        # Format results for submission CSV
        for url in recommendations:
            submission_data.append({
                'Query': query,
                'Assessment_url': url
            })
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} queries...")
            
    # 4. Save Submission File
    df_submission = pd.DataFrame(submission_data)
    
    # Ensure the required column order
    df_submission = df_submission[['Query', 'Assessment_url']] 
    
    # Save the file. index=False prevents writing the DataFrame index to the CSV.
    df_submission.to_csv(SUBMISSION_FILE_NAME, index=False)
    
    print("\n--- Submission Complete ---")
    print(f"Successfully generated {len(df_submission)} recommendation rows.")
    print(f"Submission file saved as: {SUBMISSION_FILE_NAME}")