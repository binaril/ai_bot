from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
# Ваш текущий код + reranker

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("documents")

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # 0.5s на 50 docs
model = SentenceTransformer('all-MiniLM-L6-v2')

query = "Веронайз Трибри Фил Веронайз (Фил) Трибри"
query_emb = model.encode([query], normalize_embeddings=True)

# Больше кандидатов для reranking
results = collection.query(
    query_embeddings=query_emb.tolist(),
    n_results=50,  # ↑ Было 10
    include=['documents', 'metadatas', 'distances']
)

# 🔥 RERANKING
docs = results['documents'][0]
pairs = [[query, doc] for doc in docs]
scores = reranker.predict(pairs)  # [0.85, 0.72, 0.65, ...]

# Топ-10 по reranker (не по distance!)
top_idx = np.argsort(scores)[::-1][:10]
for i, idx in enumerate(top_idx):
    score = scores[idx]
    doc, meta, dist = docs[idx], results['metadatas'][0][idx], results['distances'][0][idx]
    print(f"{i+1}. [{score:.3f}] {meta['filename']} (orig_dist={dist:.3f})")
