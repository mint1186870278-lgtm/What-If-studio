"""Cross-session memory: user preferences (Mem0) + script vector search (Chroma).

Gracefully degrades: Mem0 → JSON file, Chroma → in-memory TF-IDF.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_MEMORY_DIR: Path | None = None


def _get_memory_dir() -> Path:
    global _MEMORY_DIR
    if _MEMORY_DIR is None:
        _MEMORY_DIR = Path(settings.storage_path) / "memory"
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return _MEMORY_DIR


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------

class MemoryService:
    """Cross-session memory for user preferences and script retrieval."""

    _instance: MemoryService | None = None

    def __new__(cls) -> MemoryService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._mem0: Any = None
        self._chroma: Any = None
        self._collection: Any = None
        self._embedding_fn: Any = None
        self._prefs_cache: dict[str, list[dict]] = {}

    # -- Mem0 (user preferences) ----------------------------------------------

    async def _ensure_mem0(self) -> Any:
        if self._mem0 is not None:
            return self._mem0
        key = settings.mem0_api_key
        if key:
            try:
                from mem0 import Memory
                os.environ.setdefault("MEM0_API_KEY", key)
                self._mem0 = Memory()
                logger.info("Mem0 client initialized")
                return self._mem0
            except Exception as exc:
                logger.warning("Mem0 init failed (%s), using JSON fallback", exc)
        self._mem0 = False  # sentinel: use fallback
        return False

    async def store_user_preference(self, user_id: str, preference: dict) -> None:
        """Store a user preference observation."""
        mem0 = await self._ensure_mem0()
        if mem0:
            try:
                text = json.dumps(preference, ensure_ascii=False)
                await asyncio.to_thread(
                    mem0.add,
                    text,
                    user_id=user_id,
                    metadata={"source": preference.get("source", "inferred")},
                )
                logger.debug("Mem0: stored preference for %s", user_id)
                return
            except Exception as exc:
                logger.warning("Mem0 store failed (%s), using JSON fallback", exc)

        # JSON fallback
        prefs = await self.get_user_preferences(user_id)
        prefs.append(preference)
        self._prefs_cache[user_id] = prefs
        await self._save_preferences_json(user_id, prefs)

    async def get_user_preferences(self, user_id: str) -> list[dict]:
        """Retrieve all stored preferences for a user."""
        mem0 = await self._ensure_mem0()
        if mem0:
            try:
                results = await asyncio.to_thread(
                    mem0.search, "", user_id=user_id, limit=20,
                )
                return [
                    {"key": r.get("memory", ""), "value": r.get("memory", ""),
                     "source": (r.get("metadata") or {}).get("source", "mem0")}
                    for r in (results or [])
                ]
            except Exception as exc:
                logger.debug("Mem0 search failed (%s), using JSON fallback", exc)

        return self._prefs_cache.get(user_id) or await self._load_preferences_json(user_id)

    async def _save_preferences_json(self, user_id: str, prefs: list[dict]) -> None:
        path = _get_memory_dir() / f"prefs_{_safe_filename(user_id)}.json"
        await asyncio.to_thread(
            lambda: path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8"),
        )

    async def _load_preferences_json(self, user_id: str) -> list[dict]:
        path = _get_memory_dir() / f"prefs_{_safe_filename(user_id)}.json"
        if path.exists():
            try:
                data = await asyncio.to_thread(lambda: json.loads(path.read_text(encoding="utf-8")))
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    # -- Chroma (script vector storage) ---------------------------------------

    async def _ensure_chroma(self) -> Any | None:
        if self._chroma is not None:
            return self._chroma
        try:
            import chromadb
            persist = settings.chroma_persist_path
            Path(persist).mkdir(parents=True, exist_ok=True)
            # Run in thread to prevent ONNX model download from blocking event loop
            self._chroma = await asyncio.to_thread(
                chromadb.PersistentClient, path=persist,
            )
            self._collection = await asyncio.to_thread(
                self._chroma.get_or_create_collection,
                name="scripts",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Chroma client initialized at %s", persist)
            return self._chroma
        except Exception as exc:
            logger.warning("Chroma init failed (%s), using in-memory fallback", exc)
            return None

    async def store_script(self, script: str, metadata: dict[str, Any]) -> str:
        """Store a script in vector DB. Returns a document ID."""
        import uuid
        doc_id = str(uuid.uuid4())

        chroma = await self._ensure_chroma()
        if chroma and self._collection is not None:
            try:
                embeddings = await self._embed(script)
                if embeddings:
                    self._collection.add(
                        ids=[doc_id],
                        documents=[script],
                        metadatas=[{k: str(v)[:512] for k, v in metadata.items()}],
                        embeddings=[embeddings],
                    )
                    logger.debug("Chroma: stored script %s", doc_id)
                    return doc_id
            except Exception as exc:
                logger.warning("Chroma store failed (%s), using JSON fallback", exc)

        # JSON fallback
        record = {"id": doc_id, "script": script, "metadata": metadata}
        _get_memory_dir().mkdir(parents=True, exist_ok=True)
        path = _get_memory_dir() / "scripts_fallback.jsonl"
        def _append():
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_append)
        return doc_id

    async def search_similar_scripts(
        self, query: str, k: int = 3, filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Search for similar historical scripts."""
        chroma = await self._ensure_chroma()
        if chroma and self._collection is not None:
            try:
                embeddings = await self._embed(query)
                where = (
                    {k: str(v) for k, v in (filter_metadata or {}).items()}
                    if filter_metadata else None
                )
                results = self._collection.query(
                    query_embeddings=[embeddings] if embeddings else None,
                    query_texts=[query] if not embeddings else None,
                    n_results=k,
                    where=where,
                )
                out = []
                ids_list = results.get("ids", [[]])[0]
                docs_list = results.get("documents", [[]])[0]
                metas_list = results.get("metadatas", [[]])[0]
                for i in range(min(len(ids_list), len(docs_list))):
                    out.append({
                        "id": ids_list[i],
                        "script": docs_list[i][:500],
                        "metadata": metas_list[i] if i < len(metas_list) else {},
                    })
                return out
            except Exception as exc:
                logger.warning("Chroma search failed (%s), using TF-IDF fallback", exc)

        return await self._search_tfidf(query, k)

    async def _search_tfidf(self, query: str, k: int) -> list[dict]:
        """TF-IDF fallback search over the JSONL file."""
        path = _get_memory_dir() / "scripts_fallback.jsonl"
        if not path.exists():
            return []

        def _do():
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if not records:
                return []
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                docs = [r["script"] for r in records]
                vec = TfidfVectorizer(max_features=1000)
                tfidf = vec.fit_transform([query] + docs)
                sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                top = sim.argsort()[::-1][:k]
                return [records[i] for i in top if sim[i] > 0.05]
            except ImportError:
                return records[:k]
        return await asyncio.to_thread(_do)

    # -- Embedding helper -----------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        try:
            from openai import AsyncOpenAI
            key = settings.openai_api_key
            base = settings.openai_base_url or "https://api.openai.com/v1"
            if not key:
                raise RuntimeError("No OpenAI key for embeddings")
            client = AsyncOpenAI(api_key=key, base_url=base)
            resp = await client.embeddings.create(
                model="text-embedding-3-small", input=text[:8000],
            )
            return resp.data[0].embedding
        except Exception as exc:
            logger.debug("OpenAI embedding failed (%s)", exc)
            return None

    # -- Context builder ------------------------------------------------------

    async def build_context_for_new_session(self, user_id: str, current_prompt: str) -> dict:
        """Build a context string to inject into a new discussion.

        Returns a dict with keys: context (str), preferences_count (int), similar_scripts (int).

        Only uses fast JSON preferences — Chroma vector search is deferred to
        post-discussion storage to avoid blocking the SSE stream on model download.
        """
        parts: list[str] = []

        prefs = await self.get_user_preferences(user_id)
        preferences_count = len(prefs)
        if prefs:
            parts.append("## 用户历史偏好")
            for p in prefs[-10:]:
                val = p.get("value") or p.get("key", "")
                if val:
                    parts.append(f"- {str(val)[:200]}")

        # Use TF-IDF fallback search (fast, no ONNX download needed).
        # Chroma is used only for post-discussion storage (off hot path).
        similar = await self._search_tfidf(current_prompt, k=2)
        similar_scripts_count = len(similar)
        if similar:
            parts.append("## 历史相关剧本参考")
            for s in similar:
                meta = s.get("metadata", {})
                style = meta.get("style", "")
                parts.append(f"- [{style}] {s.get('script', '')[:300]}")

        context_str = "\n".join(parts) if parts else ""
        return {
            "context": context_str,
            "preferences_count": preferences_count,
            "similar_scripts": similar_scripts_count,
        }

    # -- Feedback -------------------------------------------------------------

    async def record_feedback(self, user_id: str, script_id: str, feedback: dict) -> None:
        """Record user feedback on a generated script."""
        await self.store_user_preference(user_id, {
            "key": f"feedback_on_{script_id}",
            "value": json.dumps(feedback, ensure_ascii=False),
            "source": "explicit",
            "confidence": 90,
        })
        logger.info("Recorded feedback for user %s on script %s", user_id, script_id)


def detect_script_tone(script: str) -> str:
    """Detect the dominant tone/genre from a generated script."""
    script_lower = script.lower()
    tones = {
        "comedy": ["搞笑", "幽默", "喜剧", "笑话", "funny", "comedy", "轻松"],
        "dramatic": ["紧张", "悬疑", "冲突", "矛盾", "dramatic", "tense", "压抑"],
        "romantic": ["爱情", "浪漫", "恋爱", "romantic", "love", "温馨"],
        "action": ["动作", "打斗", "战斗", "action", "fight", "激烈"],
        "horror": ["恐怖", "惊悚", "horror", "scary", "诡异"],
        "fantasy": ["奇幻", "魔法", "fantasy", "magic", "神话"],
        "tragedy": ["悲剧", "牺牲", "死亡", "tragedy", "离别"],
        "slice_of_life": ["日常", "平淡", "生活", "温馨", "治愈"],
    }
    for tone, keywords in tones.items():
        if any(kw in script_lower for kw in keywords):
            return tone
    return "narrative"


def _safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in "._-")[:64]


# Module-level singleton
memory_service = MemoryService()
