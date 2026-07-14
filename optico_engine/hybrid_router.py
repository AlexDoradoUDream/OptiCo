import os
import torch
import requests
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRouter:
    def __init__(
        self,
        mode: str = "auto",
        remote_url: str = "http://localhost:5001/v1/embeddings",
        semantic_model_name: str = "all-MiniLM-L6-v2",
        num_threads: int = None
    ):
        """
        Гібридний маршрутизатор із підтримкою двох режимів роутингу:
        1. "local"  — Локальна оптимізована CPU-модель (all-MiniLM-L6-v2).
        2. "remote" — Віддалений LM Studio API (text-embedding-nomic-embed-text-v2) на http://localhost:5001.
        3. "auto"   — Пріоритет віддаленому Nomic API з автоматичним фолбеком на локальний CPU.
        """
        self.mode = mode.lower()
        self.remote_url = remote_url
        self.local_encoder = None

        if num_threads is None:
            num_threads = max(1, min(os.cpu_count() or 4, 8))
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass

        if self.mode in ["local", "auto"]:
            try:
                self.local_encoder = SentenceTransformer(semantic_model_name, device="cpu")
                with torch.inference_mode():
                    _ = self.local_encoder.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
            except Exception as e:
                print(f"⚠️ [HybridRouter] Помилка завантаження локальної модель MiniLM: {e}")

        if self.mode == "remote":
            print(f"📡 [HybridRouter] Режим: Remote Nomic Embeddings API ({self.remote_url})")
        elif self.mode == "auto":
            print(f"⚡ [HybridRouter] Режим: AUTO (Remote Nomic + Local MiniLM Fallback)")
        else:
            print(f"💻 [HybridRouter] Режим: Local CPU MiniLM")

        # Завантажуємо розширені правила синонімів та стоп-слів із synonyms.json
        self.conversational_stops = set()
        self.synonym_expansion = {}
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(current_dir, "synonyms.json")
        if os.path.exists(rules_path):
            try:
                import json
                with open(rules_path, "r", encoding="utf-8") as f:
                    rules_data = json.load(f)
                    self.conversational_stops = set(rules_data.get("conversational_stops", []))
                    self.synonym_expansion = rules_data.get("synonyms", {})
            except Exception as e:
                print(f"⚠️ [HybridRouter] Помилка завантаження synonyms.json: {e}")

        # Дефолтні правила на випадок відсутності файлу конфігурації
        if not self.conversational_stops:
            self.conversational_stops = {"хто", "що", "як", "де", "коли", "там", "це", "who", "what", "where", "when", "how", "is", "are", "and", "the"}
        if not self.synonym_expansion:
            self.synonym_expansion = {
                "помилка": ["error", "exception", "failed", "bug"],
                "error": ["failed", "exception", "bug", "err"]
            }

    def _encode_remote(self, texts: list[str]) -> np.ndarray:
        """
        Отримання та L2-нормалізація векторів від віддаленого LM Studio API.
        """
        resp = requests.post(
            self.remote_url,
            json={"input": texts},
            timeout=20
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            sorted_data = sorted(data, key=lambda x: x.get("index", 0))
            raw_embs = [item["embedding"] for item in sorted_data]
            embs = np.array(raw_embs, dtype=np.float32)
            
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return embs / norms
        else:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    def route(self, query: str, chunks: list[str], top_k: int = 10, bm25_weight: float = 0.3, semantic_weight: float = 0.7) -> list[str]:
        """
        Двоетапна каскадна маршрутизація (Two-Stage Cascade Hybrid Routing) з підтримкою:
        1. Метод 1: Chronological Recency Priority (завжди беремо останні N чанків логів/чатів).
        2. Метод 3: Semantic Fallback (якщо немає лексичного перетину з запитом, шукаємо семантикою по всьому тексту).
        """
        if not chunks:
            return []
            
        # --- Метод 1: Chronological Recency Priority ---
        # Завжди тримаємо останні 4 блоки (це ~1000 токенів найсвіжішої історії в кінці промпту)
        recency_count = 4
        
        if len(chunks) <= recency_count:
            return chunks
            
        # Розділяємо чанки на зону пошуку та зону обов'язкового збереження (хронологічний хвіст)
        search_chunks = chunks[:-recency_count]
        recency_indices = list(range(len(chunks) - recency_count, len(chunks)))
        
        # Скільки чанків нам залишилося знайти за пошуком
        top_k_search = max(1, top_k - recency_count)

        raw_query_tokens = query.lower().split()
        tokenized_query = []
        for token in raw_query_tokens:
            clean_token = token.strip("?,.:;!\"'()[]{}")
            if not clean_token or clean_token in self.conversational_stops:
                continue
            tokenized_query.append(clean_token)
            
            if clean_token in self.synonym_expansion:
                tokenized_query.extend(self.synonym_expansion[clean_token])
                
        if not tokenized_query:
            tokenized_query = [t.strip("?,.:;!\"'()[]{}") for t in raw_query_tokens if t.strip("?,.:;!\"'()[]{}")]

        tokenized_chunks = [chunk.lower().split() for chunk in search_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores_raw = np.array(bm25.get_scores(tokenized_query))
        
        # --- Метод 3: Semantic Fallback ---
        # Якщо максимальний скор BM25 дорівнює 0, немає жодного ключового слова в тексті.
        # Тоді ігноруємо фільтр кандидатів і кодуємо ВСІ чанки семантично (чистий векторний пошук).
        is_fallback = float(np.max(bm25_scores_raw)) == 0.0
        
        if is_fallback:
            candidate_indices = np.arange(len(search_chunks))
        else:
            # Звичайний двоетапний каскад: відсіюємо кандидатів по BM25
            candidate_limit = max(top_k_search * 2, 45)
            if len(search_chunks) > candidate_limit:
                candidate_indices = np.argsort(bm25_scores_raw)[::-1][:candidate_limit]
            else:
                candidate_indices = np.arange(len(search_chunks))

        candidate_chunks = [search_chunks[i] for i in candidate_indices]
        candidate_bm25 = bm25_scores_raw[candidate_indices]
        max_bm25 = max(candidate_bm25) if max(candidate_bm25) > 0 else 1.0
        bm25_scores_norm = candidate_bm25 / max_bm25

        # --- Stage 2: Semantic Embedding Scoring (Only for Candidates) ---
        semantic_scores = None
        used_remote = False

        if self.mode in ["remote", "auto"]:
            try:
                all_texts = [query] + candidate_chunks
                all_embs = self._encode_remote(all_texts)
                query_emb = all_embs[0]
                chunk_embs = all_embs[1:]
                semantic_scores = np.dot(chunk_embs, query_emb)
                used_remote = True
            except Exception as e:
                if self.mode == "remote":
                    raise e
                else:
                    print(f"⚠️ [HybridRouter Auto Fallback] Віддалений Nomic недоступний ({e}). Використовуємо локальний CPU MiniLM...")

        if not used_remote:
            if self.local_encoder is None:
                raise RuntimeError("Локальний енкодер не ініціалізований.")
            with torch.inference_mode():
                query_emb = self.local_encoder.encode(query, normalize_embeddings=True, show_progress_bar=False)
                chunk_embs = self.local_encoder.encode(candidate_chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            semantic_scores = np.dot(chunk_embs, query_emb)
        
        # --- Stage 3: Cascade Hybrid Calculation ---
        hybrid_scores = (bm25_scores_norm * bm25_weight) + (semantic_scores * semantic_weight)
        
        # Вибираємо кращі K індексів з пулу кандидатів
        top_cand_sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k_search]
        original_top_indices = candidate_indices[top_cand_sorted_indices]
        
        # Об'єднуємо знайдені індекси з хронологічним хвостом
        final_indices = set(original_top_indices).union(set(recency_indices))
        
        # Сортуємо за початковою послідовністю у тексті (для збереження хронології)
        final_indices_in_order = sorted(list(final_indices))
        
        selected_chunks = [chunks[i] for i in final_indices_in_order]
        return selected_chunks
