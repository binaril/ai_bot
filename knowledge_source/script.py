#!/usr/bin/env python3
"""
ПРОСТОЙ скрипт замены слов из terms_map.json
input/ → output/
"""

import json
import os
import re
from pathlib import Path


def load_terms(filename='knowledge_source\\terms_map.json'):
    """Загружает словарь замен"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_file(filepath, terms):
    """Обрабатывает один файл"""
    try:
        # Пробуем UTF-8, потом CP1251
        encodings = ['utf-8', 'cp1251', 'latin1']
        content = None
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"❌ {filepath}")
            return

        # Заменяем все слова из словаря
        result = content
        for old_word, new_word in terms.items():
            result = re.sub(rf'\b{re.escape(old_word)}\b', new_word,
                            result, flags=re.IGNORECASE)

        return result
    except Exception as e:
        print(f"❌ {filepath}: {e}")
        return None


def main():
    # ПАПКИ
    input_folder = "knowledge_source\\files"
    output_folder = "knowledge_base"

    # Загружаем словарь замен
    terms = load_terms()
    print("🔤 Замены:", list(terms.items()))
    print()

    # Проверяем папку input
    if not os.path.exists(input_folder):
        print(f"❌ Создайте папку '{input_folder}' с файлами")
        return

    # Создаем папку output
    os.makedirs(output_folder, exist_ok=True)

    processed = 0
    changed = 0

    # Обрабатываем все файлы рекурсивно
    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)

            # Создаем структуру папок
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Обрабатываем
            result = process_file(input_path, terms)
            if result is not None:
                # Сравниваем с оригиналом
                with open(input_path, 'rb') as f:
                    original_bytes = f.read()

                if result.encode('utf-8') != original_bytes:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"✅ {rel_path}")
                    changed += 1
                else:
                    # Копируем без изменений
                    shutil.copy2(input_path, output_path)
                    print(f"⚪ {rel_path}")

                processed += 1

    print(f"\n🎉 Готово!")
    print(f"📊 Обработано: {processed} файлов")
    print(f"✨ Изменено: {changed} файлов")
    print(f"📁 Результат: {output_folder}/")


if __name__ == "__main__":
    import shutil

    main()
