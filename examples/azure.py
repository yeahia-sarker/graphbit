"""Azure integration with Graphbit."""

import ast
import json
import os

import psycopg2
from dotenv import load_dotenv
from openai import AzureOpenAI

from graphbit import EmbeddingClient, EmbeddingConfig

load_dotenv()


# Connect to PostgreSQL
conn = psycopg2.connect(dbname="vector_db", user="postgres", password="postgres", host="localhost", port=5432)
cur = conn.cursor()

# Load configuration from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_chat = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
deployment_embeddings = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS")

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# -------------------------------
# Simple text completion (Chat)
# -------------------------------
prompt = "Hello, how are you?"
chat_response = client.chat.completions.create(
    model=deployment_chat,  # <-- deployment name (a.k.a. "model" for Azure clients)
    messages=[{"role": "user", "content": prompt}],
    max_tokens=128,
)
print("Chat completion:\n", chat_response.choices[0].message.content)

# -------------------------------
# Generate embeddings
# -------------------------------
texts = [
    "GraphBit is a framework for LLM workflows and agent orchestration.",
    "ChromaDB is a fast, open-source embedding database for AI applications.",
    "OpenAI offers tools for LLMs and embeddings.",
]

embed_response = client.embeddings.create(
    model=deployment_embeddings,  # <-- deployment name for your embedding model
    input=texts,
)

embeddings = [d.embedding for d in embed_response.data]
print("\nEmbedding vector (truncated):\n", embed_response.data[0].embedding[:5])  # preview only

# Insert embeddings into PGVector table
for idx, (text, emb) in enumerate(zip(texts, embeddings)):
    cur.execute(
        """
        INSERT INTO vector_data (item_id, embedding, metadata)
        VALUES (%s, %s, %s)
        """,
        (f"batch_{idx}", emb, json.dumps({"text": text})),
    )
conn.commit()

query = "What is ChromaDB?"
query_embedding = client.embeddings.create(model=deployment_embeddings, input=query).data[0].embedding

embedding_config = EmbeddingConfig.openai(os.getenv("OPENAI_API_KEY"), "text-embedding-3-small")
embedding_client = EmbeddingClient(embedding_config)

cur.execute("SELECT item_id, embedding, metadata FROM vector_data;")
all_rows = cur.fetchall()
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
    print(f"Most similar document: {best_item[0]} with score {best_score:.4f}")
else:
    print("No documents found in vector table.")
conn.close()
