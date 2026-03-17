#!/usr/bin/env python3
"""
КОРОТКИЙ поиск с query_embeddings
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Подключение
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Запрос → эмбеддинг → поиск
query = "Енуроф владыка цепи"
query_emb = model.encode([query])  # ← 384 dim

# 🔥 query_embeddings
results = collection.query(
    query_embeddings=query_emb.tolist(),
    n_results=10,
    include=['documents', 'metadatas', 'distances']
)

# Вывод
for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
    score = 1 - dist
    print(f"{i+1}. [{score:.3f}] {meta['filename']}")
    print(f"   {doc[:100]}...\n")
