import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Додаємо поточну директорію до sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optico_engine import OptiCoCompressor

app = FastAPI(
    title="OptiCo Context Compression API",
    description="Професійний сервіс стиснення надвеликих контекстів для прискорення LLM.",
    version="2.0.0"
)

# Дозволяємо CORS для легкого інтегрування у веб-застосунки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ініціалізуємо компресор при запуску
# За замовчуванням використовуємо швидкий локальний режим ("local"). 
# Можна переключити на "auto" або "remote", якщо використовується LM Studio Nomic API.
EMBEDDING_MODE = os.getenv("OPTICO_EMBEDDING_MODE", "local")
REMOTE_EMBED_URL = os.getenv("OPTICO_REMOTE_EMBED_URL", "http://localhost:5001/v1/embeddings")

print(f"🚀 Запуск OptiCo API Server (Режим: {EMBEDDING_MODE})...")
compressor = OptiCoCompressor(embedding_mode=EMBEDDING_MODE, remote_url=REMOTE_EMBED_URL)


class CompressionRequest(BaseModel):
    context: str = Field(..., description="Повний довгий вхідний текст контексту.")
    query: str = Field(..., description="Питання або запит користувача, під який потрібно стиснути текст.")
    target_tokens: int = Field(6000, description="Цільовий розмір стисненого контексту в токенах (за замовчуванням 6000).")


class CompressionResponse(BaseModel):
    compressed_text: str = Field(..., description="Стиснений текст із збереженими релевантними шматками та маркерами пропусків.")
    original_tokens: int = Field(..., description="Кількість токенів у вихідному тексті.")
    final_tokens: int = Field(..., description="Кількість токенів у стисненому тексті.")
    compression_ratio: float = Field(..., description="Коефіцієнт стиснення (наприклад, 15.5 означає стиснення в 15.5 разів).")
    compress_latency_sec: float = Field(..., description="Затримка роботи алгоритму стиснення у секундах.")


@app.post("/api/compress", response_model=CompressionResponse, summary="Стиснути контекст під запит")
async def compress_context(req: CompressionRequest):
    """
    Приймає великий контекст і питання користувача. Повертає відфільтрований стиснений контекст,
    який містить лише релевантну інформацію та хронологічний хвіст.
    """
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="Контекст не може бути порожнім.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Запит не може бути порожнім.")
        
    try:
        res = compressor.compress(req.context, req.query, target_tokens=req.target_tokens)
        return CompressionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка стиснення: {str(e)}")


@app.get("/health", summary="Перевірка працездатності")
async def health_check():
    return {"status": "healthy", "engine": "OptiCo", "mode": EMBEDDING_MODE}


if __name__ == "__main__":
    # Запускаємо сервер на порту 5005
    uvicorn.run("server:app", host="0.0.0.0", port=5005, reload=False)
