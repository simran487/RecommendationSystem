import pandas as pd
import os
import re 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
import joblib # Library to save and load Python objects

# --- File Paths for Saved Model ---
MODEL_DIR = 'model_assets'
# These paths match the filenames expected by api.py
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'query_features.pkl')
DATAFRAME_PATH = os.path.join(MODEL_DIR, 'trained_df.pkl') 

# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


def load_csv_with_encoding(file_path):
    """
    Attempts to load a CSV file using a list of common encodings.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    print(f"Attempting to load file: {file_path}")
    
    # Check if the file exists before trying to load
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None

    for encoding in encodings_to_try:
        try:
            print(f"  Trying encoding: {encoding}...")
            # Use the specified encoding for reading the file
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"Successfully loaded file using {encoding} encoding.")
            return df
        except UnicodeDecodeError as e:
            # If a UnicodeDecodeError occurs, we just move to the next encoding
            print(f"  Failed with {encoding}: {e}")
        except Exception as e:
            # Catch other potential errors (like file structure issues)
            print(f"An unexpected error occurred: {e}")
            return None

    print("\nError: Failed to load the file with all common encodings.")
    return None

# --- Text Preprocessing Function ---
def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing non-word characters (punctuation, symbols, etc.).
    3. Removing extra whitespace.
    """
    if isinstance(text, str):
        # 1. Convert to lowercase
        text = text.lower()
        # 2. Replace non-word characters (except letters, numbers, spaces) with a space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # 3. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    return text
    
# --- Recommendation Function ---
def recommend_assessments(new_query, df, tfidf_model, feature_matrix, top_n=5):
    """
    Generates assessment recommendations for a new query based on Cosine Similarity.
    
    Args:
        new_query (str): The new job description query from the user.
        df (pd.DataFrame): The trained DataFrame containing 'Assessment_url'.
        tfidf_model (TfidfVectorizer): The fitted vectorizer instance.
        feature_matrix (sparse matrix): The TF-IDF matrix of the training data.
        top_n (int): The number of top recommendations to return.
    """
    # 1. Clean and vectorize the new query using the trained model
    cleaned_query = clean_text(new_query)
    
    # Use transform() on the trained model
    new_query_vector = tfidf_model.transform([cleaned_query])
    
    # 2. Calculate the cosine similarity between the new query and ALL training queries
    cosine_scores = cosine_similarity(new_query_vector, feature_matrix).flatten()
    
    # 3. Get the indices of the top-N most similar items
    # argsort gives ascending order, [::-1] reverses it, [-top_n:] gets the top N
    top_indices = cosine_scores.argsort()[-top_n:][::-1]
    
    # 4. Use the indices to get the corresponding assessment URLs from the original DataFrame
    recommended_urls = df.iloc[top_indices]['Assessment_url'].tolist()
    
    # 5. Return unique URLs while trying to maintain rank order
    unique_urls = []
    for url in recommended_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    return unique_urls

# --- Implementation ---

# 1. Define the correct file path for the training data.
file_name = 'Gen_AI_Dataset_Train.csv' 

# 2. Call the function to load the data
df_train = load_csv_with_encoding(file_name)

# 3. Proceed with Data Inspection and Cleaning if the loading was successful
if df_train is not None:
    print("\n--- Data Inspection (EDA) ---")
    print(f"Shape of the DataFrame (Rows, Columns): {df_train.shape}")
    
    # Check and remove duplicate rows
    duplicate_count = df_train.duplicated().sum()
    print(f"\nTotal duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        print("--- Data Cleaning: Removing Duplicates ---")
        df_train.drop_duplicates(inplace=True)
        print(f"Total rows after removing duplicates: {len(df_train)}")
    else:
        print("No exact duplicate rows found. Data is clean.")

    # --- Feature Engineering: Text Vectorization (TF-IDF) ---
    print("\n--- Feature Engineering: Text Vectorization (TF-IDF) ---")
    
    # 4. Apply the text cleaning function to the Query column
    df_train['Cleaned_Query'] = df_train['Query'].astype(str).apply(clean_text)
    print("Step 1: Queries have been cleaned.")
    
    # 5. Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # 6. Fit the vectorizer to the cleaned queries and transform the data
    query_features = tfidf_vectorizer.fit_transform(df_train['Cleaned_Query'])
    
    print(f"Step 2: TF-IDF Vectorizer trained and applied.")
    print(f"Shape of Query Features matrix: {query_features.shape}")
    
    # --- Save the trained model assets ---
    print("\n--- Saving Trained Model Assets using joblib ---")
    
    # 1. Save the fitted vectorizer
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)
    
    # 2. Save the resulting TF-IDF matrix (for similarity calculations)
    joblib.dump(query_features, FEATURES_PATH)
    
    # 3. Save the necessary columns of the dataframe for quick access in the API
    joblib.dump(df_train[['Query', 'Assessment_url', 'Cleaned_Query']], DATAFRAME_PATH)
    
    print(f"Model assets saved successfully in the '{MODEL_DIR}' folder.")

    # 7. Define a new, hypothetical user query for a quick demo
    user_query = "Looking for a candidate assessment for a senior data scientist proficient in Python, SQL, and machine learning models."
    
    # 8. Get recommendations
    recommendations = recommend_assessments(
        new_query=user_query, 
        df=df_train, 
        tfidf_model=tfidf_vectorizer, 
        feature_matrix=query_features, 
        top_n=10 
    )
    
    # 9. Output the results
    print(f"\nUser Query: '{user_query}'")
    print("\nRecommended Assessment URLs (Demo):")
    if recommendations:
        for i, url in enumerate(recommendations):
            print(f"{i+1}. {url}")
    else:
        print("No recommendations found.")
    
    print("\nEnd of Recommendation System Demo.")