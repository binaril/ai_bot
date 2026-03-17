#!/usr/bin/env python3
"""
JSON с чанками → Эмбеддинги all-MiniLM-L6-v2 + метаданные
Вход: chunks.json с форматом [{"chunk_id": 1, "content": "...", "filename": "..."}]
Выход: embeddings.npz + enriched_chunks.json
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)


def count_words(text: str) -> int:
    import re
    return len(re.findall(r'\b\w+\b', text))


class ChunkEmbedder:
    def __init__(self, input_file: str, output_dir: str = "./embeddings"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("🚀 Загрузка all-MiniLM-L6-v2...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Модель готова (384 dim)")

        self.chunks = []
        self.embeddings = None
        self.valid_indices = []

    def load_chunks(self):
        """Загружает JSON с точным форматом"""
        print(f"📂 Загрузка: {self.input_file}")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        print(f"📦 Найдено: {len(self.chunks)} чанков")

        # Проверяем структуру
        sample = self.chunks[0] if self.chunks else {}
        print(f"📋 Пример: chunk_id={sample.get('chunk_id', 'N/A')}")
        print(f"📄 Файл: {sample.get('filename', 'N/A')}")
        print(f"📏 Слов: {sample.get('word_count', 'N/A')}")

    def generate_embeddings(self, batch_size: int = 64):
        """Генерирует эмбеддинги для content полей"""
        print("\n🔢 Генерация эмбеддингов...")

        # Извлекаем content из всех чанков
        texts = []
        self.valid_indices = []

        for i, chunk in enumerate(self.chunks):
            content = chunk.get('content', '').strip()
            if content and len(content) > 10 and count_words(content) > 5:
                texts.append(content)
                self.valid_indices.append(i)
            else:
                print(f"⚠️  Пропуск chunk_id={chunk.get('chunk_id', i)} (пустой)")

        print(f"📝 Тексты для обработки: {len(texts)}")

        # Батчевая генерация
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Нормализованные для косинусного сходства
        )

        print(f"✅ {len(self.embeddings)} эмбеддингов ({self.embeddings.shape})")

    def save_embeddings(self):
        """Сохраняет эмбеддинги + обогащенные метаданные"""
        print("\n💾 СОХРАНЕНИЕ:")

        # 1. Эмбеддинги в numpy (компактно)
        embeddings_file = self.output_dir / "embeddings.npz"
        np.savez_compressed(embeddings_file, embeddings=self.embeddings)
        print(f"   📊 {embeddings_file.name} ({self.embeddings.nbytes / 1e6:.1f} MB)")

        # 2. Обогащенные чанки (с эмбеддингами)
        enriched_chunks = []
        for i, orig_idx in enumerate(self.valid_indices):
            chunk = self.chunks[orig_idx].copy()
            chunk['embedding'] = self.embeddings[i].tolist()
            chunk['embedding_norm'] = float(np.linalg.norm(self.embeddings[i]))
            enriched_chunks.append(chunk)

        enriched_file = self.output_dir / "enriched_chunks.json"
        with open(enriched_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
        print(f"   💎 {enriched_file.name} ({len(enriched_chunks)} чанков)")

        # 3. Только эмбеддинги (JSON)
        embeddings_json = self.output_dir / "embeddings.json"
        with open(embeddings_json, 'w', encoding='utf-8') as f:
            json.dump(self.embeddings.tolist(), f)
        print(f"   🔢 {embeddings_json.name}")

        # 4. Статистика
        stats = {
            'total_chunks_original': len(self.chunks),
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': 'all-MiniLM-L6-v2',
            'avg_chunk_words': np.mean([c.get('word_count', 0) for c in self.chunks]),
            'files': len(set(c.get('filename', '') for c in self.chunks))
        }
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   📈 {stats_file.name}")

    def run(self):
        """Полный пайплайн"""
        print("🤖 JSON ЧАНКИ → ЭМБЕДДИНГИ")
        print("=" * 50)

        self.load_chunks()
        self.generate_embeddings()
        self.save_embeddings()

        print(f"\n🎉 ГОТОВО!")
        print(f"📂 Результаты: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="JSON чанки → Эмбеддинги")
    parser.add_argument("input_json", help="JSON файл с чанками")
    parser.add_argument("-o", "--output", default="./embeddings",
                        help="Папка для вывода")
    parser.add_argument("-b", "--batch", type=int, default=64,
                        help="Размер батча")

    args = parser.parse_args()

    embedder = ChunkEmbedder(args.input_json, args.output)
    embedder.run()


# Быстрый запуск
if __name__ == "__main__":
    # Если файл существует, запускаем
    default_input = "chunks\data.json"
    if Path(default_input).exists():
        embedder = ChunkEmbedder(default_input)
        embedder.run()
    else:
        main()
