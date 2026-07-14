# ⚡ OptiCo (Optimal Context Compressor)

**Author:** AlexDorado

**OptiCo** is a professional, high-performance local context compression engine for LLMs. It compresses massive prompts (100k–200k tokens) by **15–25x** in **~1–2 seconds on a single CPU** without requiring expensive GPUs, while maintaining up to 100% information retrieval accuracy (as proven by "Needle In A Haystack" benchmarks).

This engine is designed to be easily deployed as a microservice or imported as a library for chat-bots, RAG pipelines, and log analyzers where you need to answer questions based on a large volume of dynamic text (e.g. chat history, log files, codebase) without overloading your LLM's KV-cache.

---

## 🚀 How It Works (Cascading Retrieval)

The compression pipeline runs in 3 stages:

1. **Semantic Chunking:** The context is divided into logical chunks (~300 tokens) while preserving Markdown headers as contextual anchor points.
2. **Two-Stage Cascade Hybrid Routing:**
   * **Query Cleaning & Synonym Expansion:** Stopwords and conversational noise are stripped (`who`, `what`, `wrote`, `something`, etc.). Synonyms are expanded dynamically via config file (e.g., `user` ↔ `client` or `error` ↔ `bug`).
   * **BM25 Lexical Filter:** Rapidly filters out hundreds of clearly irrelevant chunks in milliseconds.
   * **Semantic Embeddings:** A fast local CPU model (or a remote endpoint) calculates vector similarity **only for the top 45 candidates**, saving CPU cycles.
   * **Semantic Fallback:** If there is no keyword overlap (BM25 returns zero matches), the system automatically falls back to full vector search.
3. **Recency Priority & Assembly:**
   * The last **4 chunks** (~1000 tokens) of the document (latest chat messages or log entries) are always kept at the end of the prompt to preserve current chronological history.
   * Selected chunks are stitched back together using missing-text indicators (`... [NERELEVANT TEXT REMOVED] ...`) and passed to your LLM.

---

## 🛠️ Installation & Setup

### 1. Install dependencies:
OptiCo requires Python 3.9+.

```bash
pip install -r requirements.txt
```

### 2. Run the API Server:
Start the FastAPI server:

```bash
python server.py
```
The server will start at `http://localhost:5005`.

---

## 📡 API Specification

### 1. Compress Context (`POST /api/compress`)

#### Request Payload:
```json
{
  "context": "Your huge log file, chat history, or documentation text...",
  "query": "Who completed the service update yesterday?",
  "target_tokens": 6000
}
```

#### Response (`200 OK`):
```json
{
  "compressed_text": "... [NERELEVANT TEXT REMOVED] ...\nUser AlexDorado: service update completed successfully...",
  "original_tokens": 145090,
  "final_tokens": 6265,
  "compression_ratio": 23.16,
  "compress_latency_sec": 1.542
}
```

---

## ⚙️ Environment Variables (Configuration)

Configure the embedding model settings via environment variables:
* `OPTICO_EMBEDDING_MODE` — Embeddings engine mode.
  * `"local"` (Default) — Runs the ultra-fast `all-MiniLM-L6-v2` model locally on CPU.
  * `"remote"` — Calls a remote OpenAI-compatible embeddings API (e.g., LM Studio serving Nomic Embed).
  * `"auto"` — Tries to use the remote API, and automatically falls back to the local MiniLM model on failure.
* `OPTICO_REMOTE_EMBED_URL` — Address of the remote embeddings API (Default: `http://localhost:5001/v1/embeddings`).

---
---

# ⚡ OptiCo (Optimal Context Compressor) [UA]

**Автор розробки:** AlexDorado

**OptiCo** — це професійний двигун локального стиснення контексту для LLM. Він дозволяє стискати великі промпти (100k–200k токенів) у **15–25 разів** за **~1–2 секунди на звичайному процесорі (CPU)** без використання відеокарт, зберігаючи точність відповідей на рівні 100%.

Двигун розроблений для швидкого розгортання як мікросервіс або інтегрування в RAG-системи та чат-ботів, коли потрібно відповісти на питання за великим обсягом тексту (історія чату, файли логів тощо), не перевантажуючи KV-кеш LLM.

---

## 🚀 Як це працює?

Процес стиснення відбувається у 3 етапи:

1. **Порціонування (Chunking):** Текст розбивається на чанки (~300 токенів) зі збереженням Markdown заголовків.
2. **Двоетапна гібридна маршрутизація (Two-Stage Cascade Hybrid Routing):**
   * **Очищення запиту (Conversational Stopwords Filter):** Прибирається розмовний шум із питання користувача (`хто`, `написав`, `щось`, `там` / `who`, `wrote`, `something`) та розширюються технічні синоніми через файл конфігурації (наприклад, `користувач` ↔ `user` або `помилка` ↔ `error`).
   * **BM25 Фільтрація:** За лічені мілісекунди відсіюються сотні явно нерелевантних чанків.
   * **Semantic Embeddings:** Швидка векторна модель прораховує семантичну схожість **виключно для 45 кандидатів**.
   * **Семантичний Fallback:** Якщо лексичних збігів немає, система автоматично перемикається в чистий векторний пошук по всьому документу.
3. **Хронологічна актуальність (Recency Priority):**
   * Останні **4 чанки** документу (найсвіжіші логи чи повідомлення) завжди додаються в кінець кінцевого тексту без фільтрації для збереження хронології.
   * Всі вибрані блоки зшиваються спеціальними маркерами вилучення та передаються у вашу ЛЛМ.

---

## 🛠️ Встановлення та запуск

### 1. Встановлення залежностей:
Рекомендується Python 3.9+.

```bash
pip install -r requirements.txt
```

### 2. Запуск локального API сервера:
```bash
python server.py
```
Сервер запуститься за адресою `http://localhost:5005`.

---

## 📡 Специфікація API

### 1. Стиснення контексту (`POST /api/compress`)

#### Запит (Payload):
```json
{
  "context": "Тут ваш великий текст логів, чату або підручника...",
  "query": "Запитання користувача (наприклад: Хто написав про оновлення?)",
  "target_tokens": 6000
}
```

#### Відповідь (`200 OK`):
```json
{
  "compressed_text": "... [NERELEVANT TEXT REMOVED] ...\nКористувач AlexDorado: оновлення завершено успішно...",
  "original_tokens": 145090,
  "final_tokens": 6265,
  "compression_ratio": 23.16,
  "compress_latency_sec": 1.542
}
```
