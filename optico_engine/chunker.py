import re
import tiktoken

def chunk_text(text: str, max_tokens: int = 300, model: str = "gpt-3.5-turbo") -> list[str]:
    """
    Розбиває довгий текст на логічні блоки (chunks) до `max_tokens` токенів, 
    зберігаючи контекстні якорі (Markdown заголовки).
    """
    enc = tiktoken.encoding_for_model(model)
    
    # Rozbyvayemo tekst na abzatsy za podviynym perenosom
    paragraphs = re.split(r'\n{2,}', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_anchor = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Якщо параграф є Markdown-заголовком, фіксуємо його як "якір" (anchor)
        if para.startswith("#"):
            current_anchor = para
            # Не додаємо заголовок як окремий чанк, а приклеюємо до наступних
            continue
            
        para_tokens = len(enc.encode(para))
        
        # Якщо параграф завеликий сам по собі (що буває в логах чи монолітному тексті)
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append((current_anchor, "\n\n".join(current_chunk)))
                current_chunk = []
                current_tokens = 0
                
            # Розбиваємо великий параграф по реченнях
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub_chunk = []
            sub_tokens = 0
            for sent in sentences:
                sent_t = len(enc.encode(sent))
                if sub_tokens + sent_t > max_tokens and sub_chunk:
                    chunks.append((current_anchor, " ".join(sub_chunk)))
                    sub_chunk = []
                    sub_tokens = 0
                sub_chunk.append(sent)
                sub_tokens += sent_t
            if sub_chunk:
                chunks.append((current_anchor, " ".join(sub_chunk)))
            continue
            
        # Якщо параграф вміщається в поточний чанк
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append((current_anchor, "\n\n".join(current_chunk)))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
            
    if current_chunk:
        chunks.append((current_anchor, "\n\n".join(current_chunk)))
        
    # Формуємо фінальні текстові чанки, додаючи якорі для LLM
    final_chunks = []
    for anchor, content in chunks:
        if anchor:
            final_chunks.append(f"{anchor}\n{content}")
        else:
            final_chunks.append(content)
            
    return final_chunks
