import azure.functions as func
import json
from venv import logger
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import redis
from dotenv import load_dotenv
import os
from typing import Optional, List
from modules import embedding_openai

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Redis (puedes usar variables de entorno para mayor seguridad)
_redis_client: Optional[redis.StrictRedis] = None
_CACHE_ENABLED = False
_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "86400"))
_TRESHOLD = float(os.getenv("AUTH_THRESHOLD", "0.7"))

def _init_redis_once(db) -> None:
    global _redis_client, _CACHE_ENABLED
    if _redis_client is not None:
        return
    host = os.getenv("REDIS_HOST", "localhost")
    pwd = os.getenv("REDIS_PASSWORD")
    port = os.getenv("REDIS_PORT", 6380)
    if not (host and pwd and port):
        logger.warning("Redis not fully configured (REDIS_HOST/PORT/PASSWORD). Cache disabled.")
        _CACHE_ENABLED = False
        _redis_client = None
        return
    try:
        _redis_client = redis.StrictRedis(
            host=str(host),
            port=int(port),
            password=str(pwd),
            db=int(db) if db else 0,
            ssl=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=True
        )
        try:
            _redis_client.ping()
            _CACHE_ENABLED = True
            logger.info(f"Redis cache enabled. DB: {db}")
        except Exception as ping_err:
            logger.warning(f"Redis ping failed. Cache disabled. Err: {ping_err}")
            _CACHE_ENABLED = False
            _redis_client = None

    except Exception as e:
        logger.error(f"Error creating Redis client. Cache disabled. Err: {e}")
        _CACHE_ENABLED = False
        _redis_client = None

def get_redis_client(db) -> Optional[redis.StrictRedis]:
    if _redis_client is None:
        _init_redis_once(db)
    return _redis_client if _CACHE_ENABLED else None

app = FastAPI(title="Microservicio de Cache con Redis")

def build_cache_key(
    scope: str,
    role: str,
    cache_key: str,
    thread_id: Optional[str] = None
) -> str:
    """
    Construye la clave de la cache segun la convencion: 
    {scope}:{role}:{thread_id}:{cache_key}
    """
    thread_id = thread_id or ""
    return f"{scope}:{role}:{thread_id}:{cache_key}"

async def embedding_request(text) -> Optional[List[float]]:
    query_embedding = await embedding_openai.get_embedding(text)
    if not query_embedding:
        print("[DEBUG] ❌ No se pudo generar embedding para la consulta")
        return None
        
    print(f"[DEBUG] ✅ Embedding generado, longitud: {len(query_embedding)}")
    return query_embedding

async def get_embedding_cache(text: str, db: int) -> Optional[List[float]]:
    global _TRESHOLD
    redis_client = get_redis_client(db)
    if not redis_client:
        print("[DEBUG] ❌ Redis no está disponible, no se puede obtener el embedding de la cache")
        return None

    query_embedding = await embedding_openai.get_embedding(text)
    if not query_embedding:
        print("[DEBUG] ❌ No se pudo generar embedding para la consulta")
        return None

    cached_keys = redis_client.keys("*")
    print(f"[DEBUG] 📁 Claves encontradas en caché: {len(cached_keys)}")

    best_similarity = 0
    best_response = None
    similarities = []
    for i, key in enumerate(cached_keys):
        cached_data = redis_client.get(key)
        if cached_data:
            cache_entry = json.loads(cached_data)
            cached_embedding = cache_entry.get("embedding")
            cached_text = cache_entry.get("text", "")[:50]
                
            if cached_embedding:
                similarity = embedding_openai.cosine_similarity(query_embedding, cached_embedding)
                similarities.append((key, similarity, cached_text))
                print(f"[DEBUG] 🔍 #{i+1} Similitud: {similarity:.4f} | Texto: '{cached_text}...'")
                    
                if similarity > best_similarity and similarity >= _TRESHOLD:
                    best_similarity = similarity
                    best_response = cache_entry.get("response")
                    print(f"[DEBUG] 🎯 ¡Nueva mejor coincidencia! Similitud: {similarity:.4f}")
    # Mostrar ranking de similitudes
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] 📊 Top 3 similitudes:")
    for i, (key, sim, text) in enumerate(similarities[:3]):
        print(f"[DEBUG]   {i+1}. {sim:.4f} - '{text}...'")
        
    if best_response:
        print(f"[DEBUG] ✅ Respuesta encontrada con similitud: {best_similarity:.4f}")
    else:
        print(f"[DEBUG] ❌ No se encontró respuesta similar (mejor: {max([s[1] for s in similarities]) if similarities else 0:.4f})")
        
    return best_response
        

class CacheItem(BaseModel):
    scope: str  # shared o unique
    role: str   # anon, empresa, persona, ejecutivo
    cache_key: str
    value: str
    thread_id: Optional[str] = None
    db_id: Optional[int]

@app.route(route="cache", methods=["POST"], response_model=CacheItem)
async def set_cache(item: CacheItem):
    redis_client=get_redis_client(item.db_id)
    embedded_text = await embedding_openai.get_embedding(item.cache_key)
    print(redis_client)
    """Guarda un valor en la cache de Redis con estructura flexible."""
    key = build_cache_key(item.scope, item.role, item.cache_key, item.thread_id)
    value = {
        "query": item.cache_key,
        "value": item.value,
        "embedding": embedded_text,
    }
    redis_client.set(key, json.dumps(value))
    return item

@app.route(route="cache", methods=["GET"], response_model=CacheItem)
def get_cache(
    scope: str = Query(..., description="shared o unique"),
    role: str = Query(..., description="anon, empresa, persona, ejecutivo"),
    cache_key: str = Query(..., description="Nombre logico de la informacion cacheada"),
    thread_id: Optional[str] = Query(None, description="ID del usuario si aplica"),
    db_id: Optional[int] = Query(None, description="ID de la BD")
):
    """Obtiene un valor de la cache de Redis con estructura flexible."""
    redis_client=get_redis_client(db_id)
    key = build_cache_key(scope, role, cache_key, thread_id)
    value = redis_client.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Clave no encontrada en la cache")
    return CacheItem(scope=scope, role=role, cache_key=cache_key, value=value, thread_id=thread_id)

@app.route(route="cache", methods=["DELETE"])
def delete_cache(
    scope: str = Query(..., description="shared o unique"),
    role: str = Query(..., description="anon, empresa, persona, ejecutivo"),
    cache_key: str = Query(..., description="Nombre logico de la informacion cacheada"),
    thread_id: Optional[str] = Query(None, description="ID del usuario si aplica"),
    db_id: Optional[int] = Query(None, description="ID de la BD")
):
    """Elimina un valor de la cache de Redis con estructura flexible."""
    redis_client=get_redis_client(db_id)
    key = build_cache_key(scope, role, cache_key, thread_id)
    result = redis_client.delete(key)
    if result == 0:
        raise HTTPException(status_code=404, detail="Clave no encontrada en la cache")
    return {"detail": "Clave eliminada correctamente"}