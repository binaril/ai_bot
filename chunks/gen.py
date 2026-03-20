import warnings
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=FutureWarning)


def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

class PreciseChunker:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=10,
            length_function=count_words,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
            add_start_index=True
        )

        self.final_chunks = []

    def load_documents(self):
        """Загрузка файлов"""
        supported_exts = {'.txt', '.md', '.py', '.cs', '.js', '.json', '.html', '.xml'}
        documents = []

        for file_path in self.folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if len(content) > 50:
                            doc = Document(
                                page_content=content,
                                metadata={
                                    'filename': file_path.name,
                                    'total_chars': len(content),
                                    'total_words': count_words(content)
                                }
                            )
                            documents.append(doc)
                except Exception as e:
                    print(f"❌ {file_path.name}: {e}")

        return documents

    def calculate_position(self, chunk, full_text):
        """Вычисляем позицию начала чанка"""
        try:
            # Берем первые 100 символов для поиска
            chunk_start = chunk.page_content[:100].strip()
            pos = full_text.find(chunk_start)
            return max(0, pos) if pos != -1 else 0
        except:
            return 0

    def process_folder(self):
        docs = self.load_documents()
        if not docs:
            print("❌ Нет файлов!")

        full_texts = {doc.metadata['filename']: doc.page_content for doc in docs}

        print("✂️  RecursiveCharacterTextSplitter...")
        all_chunks = self.splitter.split_documents(docs)

        for chunk in all_chunks:
            word_count = count_words(chunk.page_content)

            filename = chunk.metadata['filename']
            position = self.calculate_position(chunk, full_texts[filename])

            chunk.metadata.update({
                'word_count': word_count,
                'char_count': len(chunk.page_content),
                'start_position': position
            })
            self.final_chunks.append(chunk)

        # Статистика
        word_counts = [c.metadata['word_count'] for c in self.final_chunks]
        print(f"\n   {len(self.final_chunks)} чанков")
        print(f"   Размер: {min(word_counts):.0f}-{max(word_counts):.0f} слов")
        print(f"   Средний: {np.mean(word_counts):.0f} слов")
        print(f"   Всего: {sum(word_counts):.0f} слов")

    def save_chunks(self, output_file='chunks\data.json'):
        """💾 Сохраняем с filename + позицией"""
        import json
        result = []
        for i, chunk in enumerate(self.final_chunks):
            result.append({
                'chunk_id': i,
                'content': chunk.page_content,
                'filename': chunk.metadata['filename'],
                'word_count': chunk.metadata['word_count'],
                'char_count': chunk.metadata['char_count'],
                'start_position': chunk.metadata['start_position']
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 {output_file}")
        return result


# 🚀 ЗАПУСК
if __name__ == "__main__":
    import numpy as np

    chunker = PreciseChunker("./knowledge_base")
    chunker.process_folder()
    chunker.save_chunks()
