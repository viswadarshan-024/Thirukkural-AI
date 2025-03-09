import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the Thirukkural dataset
dataset = load_dataset("Selvakumarduraipandian/Thirukural")
df = pd.DataFrame(dataset['train'])

# Initialize the sentence transformer model for embedding generation
# We'll use a multilingual model to handle both Tamil and English
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Create a combined text field for embedding generation
# This combines the kural, its meaning, and explanation to create rich embeddings
df['combined_text_tamil'] = df['Kural'] + ' ' + df['Vilakam'] + ' ' + df['Kalaingar_Urai']
df['combined_text_english'] = df['Couplet'] + ' ' + df['M_Varadharajanar'] + ' ' + df['Solomon_Pappaiya']

# Generate embeddings for Tamil and English texts
tamil_embeddings = model.encode(df['combined_text_tamil'].tolist(), batch_size=32, show_progress_bar=True)
english_embeddings = model.encode(df['combined_text_english'].tolist(), batch_size=32, show_progress_bar=True)

# Normalize the embeddings (required for Faiss cosine similarity)
tamil_embeddings = tamil_embeddings / np.linalg.norm(tamil_embeddings, axis=1)[:, np.newaxis]
english_embeddings = english_embeddings / np.linalg.norm(english_embeddings, axis=1)[:, np.newaxis]

# Create Faiss indices for fast similarity search
dimension = tamil_embeddings.shape[1]

# Create and train the Faiss index for Tamil
tamil_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
tamil_index.add(tamil_embeddings.astype('float32'))

# Create and train the Faiss index for English
english_index = faiss.IndexFlatIP(dimension)
english_index.add(english_embeddings.astype('float32'))

# Save the indices for later use
faiss.write_index(tamil_index, "thirukkural_tamil_index.faiss")
faiss.write_index(english_index, "thirukkural_english_index.faiss")

# Save the dataframe for later lookup
df.to_pickle("thirukkural_data.pkl")

print("Vector database setup complete. Files saved: thirukkural_tamil_index.faiss, thirukkural_english_index.faiss, thirukkural_data.pkl")
