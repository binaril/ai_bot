#!/usr/bin/env python3
"""
Эмбеддинги → Chroma (ФИНАЛЬНАЯ версия — все ошибки исправлены)
"""

import json
import numpy as np
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import warnings
import argparse

warnings.filterwarnings("ignore")


class ChromaIndexer:
    def __init__(self, embeddings_dir: str, chroma_path: str = "./chroma_db"):
        self.embeddings_dir = Path(embeddings_dir)
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(exist_ok=True)

        self.embeddings_file = self.embeddings_dir / "embeddings.npz"
        self.metadata_file = self.embeddings_dir / "enriched_chunks.json"

    def load_data(self):
        embeddings = np.load(self.embeddings_file)['embeddings']
        print(f"📊 Эмбеддинги: {embeddings.shape}")

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"📦 Метаданные: {len(metadata)} чанков")

        return embeddings, metadata

    def create_chroma_index(self, collection_name: str = "documents"):
        """✅ ФИНАЛЬНАЯ версия — без ошибок"""
        print(f"\n🔗 Коллекция '{collection_name}'...")

        client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Получаем или создаем коллекцию
        try:
            collection = client.get_collection(name=collection_name)
            print(f"📂 Найдена: {collection.count()} документов")
        except:
            collection = client.create_collection(name=collection_name)
            print(f"✨ Создана новая")

        # ✅ ИСПРАВЛЕНИЕ: удаляем ПО ID
        if collection.count() > 0:
            all_data = collection.get(include=[])
            all_ids = all_data.get('ids', [])
            if all_ids:
                collection.delete(ids=all_ids)
                print(f"🗑️  Удалено: {len(all_ids)} документов")

        # Загружаем данные
        embeddings, metadata = self.load_data()

        ids = [f"chunk_{meta.get('chunk_id', i)}" for i, meta in enumerate(metadata)]
        metadatas = [{
            'filename': meta.get('filename', 'unknown'),
            'word_count': meta.get('word_count', 0),
            'char_count': meta.get('char_count', 0),
            'start_position': meta.get('start_position', 0),
            'chunk_id': meta.get('chunk_id', 0)
        } for meta in metadata]
        documents = [meta.get('content', '')[:800] for meta in metadata]

        # Батчевая загрузка
        BATCH_SIZE = 100
        for i in range(0, len(ids), BATCH_SIZE):
            collection.add(
                ids=ids[i:i + BATCH_SIZE],
                embeddings=embeddings[i:i + BATCH_SIZE].tolist(),
                metadatas=metadatas[i:i + BATCH_SIZE],
                documents=documents[i:i + BATCH_SIZE]
            )
            print(f"📦 Батч {(i // BATCH_SIZE) + 1}: {len(ids[i:i + BATCH_SIZE])}")

        print(f"\n🎉 ГОТОВО: {collection.count()} документов")
        return collection, client

    def test_search(self, collection, top_k: int = 3):
        print("\n🔍 ТЕСТ ПОИСКА:")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        queries = ["асинхронный контроллер", "middleware", "OpenTelemetry"]

        for query in queries:
            query_emb = model.encode([query])  # ✅ Плоский список

            results = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            print(f"\n'{query}':")
            for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                score = 1 - dist
                print(f"  {i + 1}. [{score:.3f}] {meta['filename']}")
                print(f"     {doc[:100]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_dir", nargs='?', default="./embeddings")
    parser.add_argument("-c", "--chroma-path", default="./chroma_db")
    args = parser.parse_args()

    indexer = ChromaIndexer(args.embeddings_dir, args.chroma_path)
    collection, client = indexer.create_chroma_index()
    indexer.test_search(collection)


if __name__ == "__main__":
    main()
