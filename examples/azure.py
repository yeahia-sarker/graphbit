"""
This script demonstrates the integration of Azure services with Graphbit.

Key operations:

- Text generation with Azure OpenAI.
- Embedding generation and storage in Azure AI Search.
- Vector similarity search using Graphbit.
"""

import ast
import json
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from graphbit import EmbeddingClient as gb_etc
from graphbit import EmbeddingConfig as gb_ecg

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")


# Simple text completion with Azure OpenAI
prompt = "Hello, how are you?"
response = openai.ChatCompletion.create(
    engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=128,
)
print(response["choices"][0]["message"]["content"])


# Generate embeddings using Azure OpenAI
text = "This is a sample document for vector search."
response = openai.Embedding.create(
    engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    input=text,
)
print(response["data"][0]["embedding"])

# Azure AI Search vectorstore Integration
# Configure environment variables for Azure AI Search

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX]):
    raise ValueError("Missing required environment variables for Azure Search.")

search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=AzureKeyCredential(AZURE_SEARCH_KEY))


# Generate and Store single embedding

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_config = gb_ecg.openai(OPENAI_API_KEY, "text-embedding-3-small")
embedding_client = gb_etc(embedding_config)

# Insert a single embedding
doc_text = "This is a sample document for vector search."
embedding = embedding_client.embed(doc_text)

document = {"id": "item_1", "text": doc_text, "embedding": embedding, "metadata": json.dumps({"category": "test"})}
search_client.upload_documents(documents=[document])
print("Inserted embedding for item_1.")

# Batch embedding insert
batch_texts = [
    "Graph databases are great for relationships.",
    "Vector search enables semantic retrieval.",
    "OpenAI provides powerful embedding models.",
]
batch_embeddings = embedding_client.embed_many(batch_texts)
batch_docs = []
for idx, (text, emb) in enumerate(zip(batch_texts, batch_embeddings)):
    batch_docs.append({"id": f"batch_{idx}", "text": text, "embedding": emb, "metadata": json.dumps({"text": text})})
search_client.upload_documents(documents=batch_docs)
print(f"Inserted {len(batch_texts)} documents with embeddings.")


# Vector search using Graphbit

query_text = "Find documents related to vector search."
query_embedding = embedding_client.embed(query_text)

# Get all documents (with embeddings) from Azure Search
results = search_client.search(search_text=None, vector=query_embedding, top_k=10, vector_fields="embedding")  # Limit the results to top 10

# Retrieve all rows from the search query results
all_rows = [(result["id"], result["embedding"], result["metadata"]) for result in results]

# Apply custom vector similarity search using Graphbit logic
best_score = -1
best_item = None
for item_id, embedding_vec, metadata in all_rows:
    # Convert the embedding from string to list if needed
    if isinstance(embedding_vec, str):
        embedding_vec = ast.literal_eval(embedding_vec)
    score = embedding_client.similarity(query_embedding, embedding_vec)
    if score > best_score:
        best_score = score
        best_item = (item_id, metadata)

if best_item is not None:
    print(f"[Vector Search] Most similar document: {best_item[0]} with score {best_score:.4f}")
else:
    print("[Vector Search] No documents found in vector table.")

print("Done.")
