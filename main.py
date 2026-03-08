import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Current folder contains all txt files
folder = "."

documents = []
filenames = []

# Read only txt files
for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            documents.append(content)
            filenames.append(filename)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert documents into embeddings
doc_embeddings = model.encode(documents).astype("float32")

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS index
index.add(doc_embeddings)

# User question
question = input("Ask your accounting question: ")

# Question embedding
query_embedding = model.encode([question]).astype("float32")

# Search nearest match
distance, result = index.search(query_embedding, k=1)

best_match = result[0][0]

# Display answer
print(f"\nBest answer from {filenames[best_match]}:")
print(documents[best_match])