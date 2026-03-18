#!/usr/bin/env python3
# RAG-бот: ChromaDB + Qwen 2.5 7B (Ollama) + Reranking
# pip install chromadb sentence-transformers numpy rich ollama requests

import chromadb
import requests
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

console = Console()
OLLAMA_URL = "http://localhost:11434"


class QwenRAGBot:
    def __init__(self, chroma_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("documents")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.test_ollama()

    def test_ollama(self):
        """Проверка доступности Ollama"""
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate",
                                     json={"model": "qwen2.5:7b-instruct", "prompt": "test"},
                                     timeout=10)
            console.print("✅ Qwen 2.5 7B подключен!", style="green")
        except:
            console.print("❌ Ollama недоступен! Запустите: ollama run qwen2.5:7b-instruct",
                          style="red")
            raise

    def ollama_chat(self, context: str, query: str) -> str:
        """Запрос к Qwen 2.5 через Ollama API"""
        prompt = f"""Ты эксперт по коду и документации. Используй только предоставленный контекст.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

Ответь точно по контексту, без выдумок. Если информации нет — скажи "Не нашёл в документах"."""

        payload = {
            "model": "qwen2.5:7b-instruct",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512
            }
        }

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task = progress.add_task("Генерирую ответ...", total=None)
            try:
                response = requests.post(f"{OLLAMA_URL}/api/generate",
                                         json=payload, stream=True, timeout=300)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                        if data.get("done", False):
                            break
                progress.remove_task(task)
                return full_response.strip()
            except Exception as e:
                progress.remove_task(task)
                return f"Ошибка Ollama: {e}"

    def search(self, query: str, n_results: int = 10) -> list:
        """Retrieval + Reranking"""
        query_emb = self.model.encode([query], normalize_embeddings=True)

        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=50,
            include=['documents', 'metadatas', 'distances']
        )

        docs = results['documents'][0]
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs)

        top_idx = np.argsort(scores)[::-1][:n_results]
        return [(scores[i], docs[i], results['metadatas'][0][i])
                for i in top_idx]

    def chat(self):
        console.print(Panel("🤖 Qwen 2.5 7B RAG-бот готов!\n(./chroma_db + all-MiniLM + reranking)",
                            title="🚀 RAG + Ollama", style="bold green"))

        while True:
            try:
                query = Prompt.ask("\n[bold cyan]Твой вопрос[/bold cyan]", console=console)
                if query.lower() in ['exit', 'выход', 'quit', 'q']:
                    break

                if not query.strip():
                    continue

                console.print(f"\n🔍 Ищу: [italic]{query}[/italic]")

                # Поиск
                top_docs = self.search(query, n_results=5)

                # Таблица результатов
                table = Table(title="Топ документы")
                table.add_column("Score", style="cyan", no_wrap=True)
                table.add_column("Файл", style="magenta")
                table.add_column("Превью", style="white")

                context = []
                for score, doc, meta in top_docs:
                    filename = meta.get('filename', 'unknown')
                    preview = (doc[:120] + "...") if len(doc) > 120 else doc
                    table.add_row(f"{score:.3f}", filename, preview)
                    context.append(f"Файл: {filename}\n{doc}\n")

                console.print(table)

                # LLM запрос
                full_context = "\n".join(context[:5])  # Топ-5
                console.rule("🤖 Qwen 2.5 ответ")

                answer = self.ollama_chat(full_context, query)
                console.print(Panel(answer, title="Ответ", border_style="green"))

            except KeyboardInterrupt:
                console.print("\n👋 До свидания!")
                break
            except Exception as e:
                console.print(f"[red]Ошибка: {e}[/red]")


def main():
    try:
        bot = QwenRAGBot("./chroma_db")
        bot.chat()
    except Exception as e:
        console.print(f"[red]Инициализация не удалась: {e}\n"
                      "[yellow]Запустите:[/yellow]\n"
                      "ollama run qwen2.5:7b-instruct\n"
                      "python rag_qwen_bot.py", style="bold")


if __name__ == "__main__":
    main()
