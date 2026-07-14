import time
import sys
import tiktoken
from .chunker import chunk_text
from .hybrid_router import HybridRouter

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

class OptiCoCompressor:
    def __init__(self, embedding_mode: str = "local", remote_url: str = "http://localhost:5001/v1/embeddings"):
        """
        Головний клас рушія OptiCo.
        embedding_mode: "local" (Ультрашвидкий CPU MiniLM ~0.3s), "auto" (Remote + Local Fallback), "remote" (LM Studio Nomic)
        """
        print(f"⚙️ [OptiCo Engine] Ініціалізація векторних моделей (Режим: {embedding_mode})...")
        self.router = HybridRouter(mode=embedding_mode, remote_url=remote_url)
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        print("✅ [OptiCo Engine] Готовий до стиснення.")

    def compress(self, context_text: str, query: str, target_tokens: int = 6000) -> dict:
        """
        Головний метод: стискає вхідний контекст до цільової кількості токенів, 
        залишаючи лише найрелевантніші до запиту частини.
        """
        start_time = time.time()
        
        # 1. Розбиття на блоки (Chunking)
        chunks = chunk_text(context_text, max_tokens=300)
        
        # 2. Розрахунок кількості блоків (в середньому блок = 200-300 токенів)
        top_k = max(1, target_tokens // 250)
        
        # Якщо контекст і так малий, просто повертаємо його
        original_tokens = len(self.enc.encode(context_text))
        if original_tokens <= target_tokens:
            return {
                "compressed_text": context_text,
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
                "compression_ratio": 1.0,
                "compress_latency_sec": round(time.time() - start_time, 4)
            }
        
        # 3. Маршрутизація (Відбір)
        selected_chunks = self.router.route(query, chunks, top_k=top_k)
        
        # 4. Об'єднання (з маркерами пропущеного тексту для LLM)
        compressed_text = "\n\n... [ВИЛУЧЕНО НЕРЕЛЕВАНТНИЙ ТЕКСТ] ...\n\n".join(selected_chunks)
        final_tokens = len(self.enc.encode(compressed_text))
        
        return {
            "compressed_text": compressed_text,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "compression_ratio": round(original_tokens / final_tokens if final_tokens else 1.0, 2),
            "compress_latency_sec": round(time.time() - start_time, 4)
        }
