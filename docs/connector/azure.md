# Azure AI Integration with Graphbit

## Overview

This guide explains how to connect Azure AI (Azure OpenAI) services to Graphbit, enabling you to leverage Azure-hosted GPT models within your Graphbit workflows. This integration allows you to use enterprise-grade LLMs with the security and compliance of Azure.

---

## Prerequisites

- **Azure Subscription** with access to [Azure OpenAI Service](https://portal.azure.com/).
- **Python environment** with `openai`, `graphbit`, `azure-search-documents`, and `python-dotenv` installed.
- **.env file** in your project root with the following variables:
  ```env
  AZURE_OPENAI_API_KEY=your-azure-openai-api-key
  AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT=your-deployment-name
  AZURE_OPENAI_API_VERSION=2024-03-01-preview
  AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
  AZURE_SEARCH_KEY=your-search-service-key
  AZURE_SEARCH_INDEX=your-index-name  
  ```

---

## Step 1: Set Up Azure OpenAI Authentication

Configure the OpenAI Python client for Azure endpoints:

```python
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")
```

---

## Step 2: Basic Usage of Azure OpenAI

### Simple text completion with Azure OpenAI

```python
prompt = "Hello, how are you?"
response = openai.ChatCompletion.create(
    engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=128,
)
print(response["choices"][0]["message"]["content"])
```

### Generate embeddings using Azure OpenAI

```python
text = "This is a sample document for vector search."
response = openai.Embedding.create(
    engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    input=text,
)
print(response["data"][0]["embedding"])
```


---

## Step 3: Azure AI Search vectorstore Integration

### Configure environment variables for Azure AI Search
```python
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX]):
    raise ValueError("Missing required environment variables for Azure Search.")

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT, 
    index_name=AZURE_SEARCH_INDEX, 
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)
```

### Generate and store single embedding

```python
from graphbit import EmbeddingClient as gb_etc
from graphbit import EmbeddingConfig as gb_ecg

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_config = gb_ecg.openai(OPENAI_API_KEY, "text-embedding-3-small")
embedding_client = gb_etc(embedding_config)

# Insert a single embedding
doc_text = "This is a sample document for vector search."
embedding = embedding_client.embed(doc_text)

document = {
    "id": "item_1",
    "text": doc_text,
    "embedding": embedding,
    "metadata": json.dumps({"category": "test"})
}
search_client.upload_documents(documents=[document])
print("Inserted embedding for item_1.")
```

### Batch Embedding 

```python
batch_texts = [
    "Graph databases are great for relationships.",
    "Vector search enables semantic retrieval.",
    "OpenAI provides powerful embedding models.",
]
batch_embeddings = embedding_client.embed_many(batch_texts)
batch_docs = []
for idx, (text, emb) in enumerate(zip(batch_texts, batch_embeddings)):
    batch_docs.append({
        "id": f"batch_{idx}",
        "text": text,
        "embedding": emb,
        "metadata": json.dumps({"text": text})
    })
search_client.upload_documents(documents=batch_docs)
print(f"Inserted {len(batch_texts)} documents with embeddings.")
```
### Vector Search using Graphbit

```python
import ast

query_text = "Find documents related to vector search."
query_embedding = embedding_client.embed(query_text)

# Get all documents (with embeddings) from Azure Search
results = search_client.search(
    search_text=None,
    vector=query_embedding,
    top_k=10,  # Limit the results to top 10
    vector_fields="embedding"
)

# Retrieve all rows from the search query results
all_rows = [(result['id'], result['embedding'], result['metadata']) for result in results]

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
```

---

**This integration enables you to leverage Azure's enterprise-grade AI capabilities within your Graphbit workflows, providing access to powerful models and scalable infrastructure for production AI applications.** 
