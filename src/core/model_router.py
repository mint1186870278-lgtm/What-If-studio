"""Configurable model router for text LLM and video generation.

Abstracts provider differences behind a unified interface so users can
plug in their own API keys for any supported provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    SILICONFLOW = "siliconflow"


class VideoProvider(str, Enum):
    HAPPYHORSE = "happyhorse"
    KLING = "kling"
    WAN = "wan"
    SEEDANCE = "seedance"
    LOCAL = "local"
    OPENAI_NEXT = "openai_next"  # generic video gen (wan/kling/seedance)
    GROK_V2V = "grok_v2v"  # xAI Grok video editing (currently API error)
    GROK_I2V = "grok_i2v"  # xAI Grok image-to-video (confirmed working, ~$0.40/8s)
    DOUBAO = "doubao"  # doubao-seedance T2V ($2/5s)
    FFMPEG = "ffmpeg"


# ---------------------------------------------------------------------------
# Provider metadata — base URLs, env-var keys, default models
# ---------------------------------------------------------------------------

_PROVIDER_META: dict[ModelProvider, dict[str, str]] = {
    ModelProvider.OPENAI: {
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "default_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    ModelProvider.CLAUDE: {
        "key_env": "ANTHROPIC_API_KEY",
        "base_env": "ANTHROPIC_BASE_URL",
        "model_env": "ANTHROPIC_MODEL",
        "default_base": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
    },
    ModelProvider.ZHIPU: {
        "key_env": "ZHIPU_API_KEY",
        "base_env": "ZHIPU_BASE_URL",
        "model_env": "ZHIPU_MODEL",
        "default_base": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
    },
    ModelProvider.DEEPSEEK: {
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_BASE_URL",
        "model_env": "DEEPSEEK_MODEL",
        "default_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
    },
    ModelProvider.SILICONFLOW: {
        "key_env": "SILICONFLOW_API_KEY",
        "base_env": "SILICONFLOW_BASE_URL",
        "model_env": "SILICONFLOW_MODEL",
        "default_base": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
    },
}


def _resolve_api_key(provider: ModelProvider) -> str | None:
    meta = _PROVIDER_META[provider]
    attr = meta["key_env"].lower()
    val = getattr(settings, attr, None) or os.getenv(meta["key_env"])
    return val.strip() if val else None


def _resolve_base_url(provider: ModelProvider) -> str:
    meta = _PROVIDER_META[provider]
    attr = (meta["base_env"].lower().removesuffix("_base_url") + "_base_url")
    val = getattr(settings, attr, None) or os.getenv(meta["base_env"])
    return (val or meta["default_base"]).strip().rstrip("/")


def _resolve_model(provider: ModelProvider) -> str:
    meta = _PROVIDER_META[provider]
    env_val = os.getenv(meta.get("model_env", ""))
    if env_val:
        return env_val.strip()
    return meta["default_model"]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ProviderNotAvailableError(RuntimeError):
    """Raised when a model provider has no configured API key."""


class VideoGenerationError(RuntimeError):
    """Raised when video generation fails."""


# ---------------------------------------------------------------------------
# Text model router
# ---------------------------------------------------------------------------

class TextModelRouter:
    """Singleton router for text LLM calls across providers."""

    _instance: TextModelRouter | None = None
    _clients: dict[ModelProvider, Any] = {}

    def __new__(cls) -> TextModelRouter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_openai_client(self, provider: ModelProvider) -> Any:
        if provider not in self._clients:
            from openai import AsyncOpenAI
            key = _resolve_api_key(provider)
            if not key:
                raise ProviderNotAvailableError(f"No API key configured for {provider.value}")
            base = _resolve_base_url(provider)
            self._clients[provider] = AsyncOpenAI(api_key=key, base_url=base)
        return self._clients[provider]

    def get_available_providers(self) -> list[ModelProvider]:
        return [p for p in ModelProvider if _resolve_api_key(p)]

    async def generate(
        self,
        provider: ModelProvider,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Non-streaming text generation."""
        model = _resolve_model(provider)

        if provider == ModelProvider.CLAUDE:
            return await self._generate_claude(messages, model, temperature, max_tokens)

        client = self._get_openai_client(provider)
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def generate_stream(
        self,
        provider: ModelProvider,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Streaming text generation."""
        model = _resolve_model(provider)

        if provider == ModelProvider.CLAUDE:
            async for chunk in self._generate_claude_stream(messages, model, temperature, max_tokens):
                yield chunk
            return

        client = self._get_openai_client(provider)
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # -- Claude via native Anthropic API --------------------------------------

    async def _generate_claude(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int,
    ) -> str:
        key = _resolve_api_key(ModelProvider.CLAUDE)
        if not key:
            raise ProviderNotAvailableError("No API key configured for Claude")
        base = _resolve_base_url(ModelProvider.CLAUDE)
        system, chat_msgs = _split_system_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_claude_messages(chat_msgs),
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.post(
                f"{base}/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""

    async def _generate_claude_stream(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        key = _resolve_api_key(ModelProvider.CLAUDE)
        if not key:
            raise ProviderNotAvailableError("No API key configured for Claude")
        base = _resolve_base_url(ModelProvider.CLAUDE)
        system, chat_msgs = _split_system_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_claude_messages(chat_msgs),
            "stream": True,
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            async with client.stream(
                "POST",
                f"{base}/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            return
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif event.get("type") == "error":
                            raise VideoGenerationError(event.get("error", {}).get("message", "Claude stream error"))


def _split_system_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts = []
    chat = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content", "")))
        else:
            chat.append(m)
    return "\n".join(system_parts), chat


def _to_claude_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role == "assistant":
            role = "assistant"
        else:
            role = "user"
        out.append({"role": role, "content": str(m.get("content", ""))})
    return out


# ---------------------------------------------------------------------------
# Video model router
# ---------------------------------------------------------------------------

class VideoModelRouter:
    """Singleton router for video generation across providers."""

    _instance: VideoModelRouter | None = None

    def __new__(cls) -> VideoModelRouter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_available_providers(self) -> list[VideoProvider]:
        available: list[VideoProvider] = [VideoProvider.FFMPEG]
        if settings.happyhorse_api_key:
            available.append(VideoProvider.HAPPYHORSE)
        if settings.kling_api_key:
            available.append(VideoProvider.KLING)
        if settings.wan_api_key:
            available.append(VideoProvider.WAN)
        if settings.seedance_api_key:
            available.append(VideoProvider.SEEDANCE)
        if settings.local_video_model_path or settings.local_video_model_name:
            available.append(VideoProvider.LOCAL)
        if settings.openai_next_api_key:
            available.append(VideoProvider.OPENAI_NEXT)
            available.append(VideoProvider.GROK_V2V)
            available.append(VideoProvider.GROK_I2V)  # image-to-video!
            available.append(VideoProvider.DOUBAO)
        return available

    async def generate_storyboard(
        self,
        script: str,
        provider: ModelProvider | None = None,
        **kwargs,
    ) -> dict:
        """Generate a storyboard/preview from a script using a text LLM.

        Returns {"frames": [{"description": str, "timing": str}], "total_duration": str}
        """
        text_router = TextModelRouter()
        prompt = (
            "你是一位资深分镜师。请将以下剧本拆分为分镜脚本，为每一幕写出画面描述和预计时长。\n\n"
            f"剧本：\n{script}\n\n"
            "请输出JSON格式，keys: frames (数组, 每项含description和timing), total_duration。只输出JSON，不要其他文字。"
        )
        # Use specified provider or auto-detect the first available
        if provider is None:
            available = text_router.get_available_providers()
            provider = available[0] if available else ModelProvider.OPENAI
        try:
            raw = await text_router.generate(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
        except ProviderNotAvailableError:
            # Try any available provider
            available = text_router.get_available_providers()
            if not available:
                raise
            raw = await text_router.generate(
                provider=available[0],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
        try:
            return json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse storyboard JSON, using heuristic fallback")
            return _heuristic_storyboard(script)

    async def generate_video(
        self,
        prompt: str,
        provider: VideoProvider,
        source_video_url: str | None = None,
        reference_images: list[str] | None = None,
        output_path: str | None = None,
    ) -> str:
        """Generate a video. Returns the output file path."""
        if provider == VideoProvider.HAPPYHORSE:
            return await self._generate_happyhorse(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.KLING:
            return await self._generate_kling(prompt, output_path)
        elif provider == VideoProvider.WAN:
            return await self._generate_wan(prompt, output_path)
        elif provider == VideoProvider.SEEDANCE:
            return await self._generate_seedance(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.LOCAL:
            return await self._generate_local(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.OPENAI_NEXT:
            return await self._generate_openai_next(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.GROK_V2V:
            return await self._generate_grok_v2v(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.GROK_I2V:
            return await self._generate_grok_i2v(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.DOUBAO:
            return await self._generate_doubao(prompt, source_video_url, reference_images, output_path)
        elif provider == VideoProvider.FFMPEG:
            return await self._generate_ffmpeg(prompt, output_path)
        raise VideoGenerationError(f"Unknown video provider: {provider}")

    # -- HappyHorse (delegate to existing pipeline) ---------------------------

    async def _generate_happyhorse(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        from src.core.video_pipeline import _submit_happyhorse, _poll_happyhorse

        if not source_video_url:
            raise VideoGenerationError("HappyHorse requires a source video URL")

        task_id = await _submit_happyhorse(prompt, source_video_url, reference_images)
        logger.info("HappyHorse task submitted: %s", task_id[:12])
        result_url = await _poll_happyhorse(task_id)

        out = Path(output_path or settings.storage_temp_path / f"hh_{task_id[:12]}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(result_url, timeout=300, follow_redirects=True)
            resp.raise_for_status()
            out.write_bytes(resp.content)
        logger.info("HappyHorse video saved: %s", out)
        return str(out)

    # -- Kling ---------------------------------------------------------------

    async def _generate_kling(self, prompt: str, output_path: str | None) -> str:
        key = settings.kling_api_key
        base = settings.kling_base_url or "https://api.kling.kuaishou.com"
        if not key:
            raise ProviderNotAvailableError("No KLING_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"kling_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # Submit
            submit_resp = await client.post(
                f"{base.rstrip('/')}/v1/videos/text2video",
                headers=headers,
                json={
                    "model_name": "kling-v1",
                    "prompt": prompt,
                    "duration": "5",
                    "mode": "std",
                },
                timeout=60,
            )
            submit_resp.raise_for_status()
            task_id = submit_resp.json()["data"]["task_id"]
            logger.info("Kling task submitted: %s", task_id[:12])

            # Poll
            for i in range(60):
                await asyncio.sleep(10)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/v1/videos/text2video/{task_id}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data["data"]["task_status"]
                if status == "succeed":
                    video_url = data["data"]["task_result"]["videos"][0]["url"]
                    dl = await client.get(video_url, timeout=300)
                    dl.raise_for_status()
                    out.write_bytes(dl.content)
                    logger.info("Kling video saved: %s", out)
                    return str(out)
                elif status == "failed":
                    raise VideoGenerationError(f"Kling task failed: {data}")
            raise TimeoutError("Kling task timed out")

    # -- Wan -----------------------------------------------------------------

    async def _generate_wan(self, prompt: str, output_path: str | None) -> str:
        key = settings.wan_api_key
        base = settings.wan_base_url
        if not key:
            raise ProviderNotAvailableError("No WAN_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"wan_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            submit_resp = await client.post(
                f"{base.rstrip('/')}/api/v1/video/generate",
                headers=headers,
                json={"prompt": prompt, "duration": 5},
                timeout=60,
            )
            submit_resp.raise_for_status()
            task_id = submit_resp.json().get("task_id") or submit_resp.json().get("id")
            logger.info("Wan task submitted: %s", str(task_id)[:12])

            for i in range(60):
                await asyncio.sleep(10)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/api/v1/tasks/{task_id}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data.get("status") or data.get("state", "")
                if status in ("completed", "succeeded", "done"):
                    video_url = data.get("video_url") or data.get("output_url", "")
                    if video_url:
                        dl = await client.get(video_url, timeout=300)
                        dl.raise_for_status()
                        out.write_bytes(dl.content)
                        logger.info("Wan video saved: %s", out)
                        return str(out)
                    raise VideoGenerationError("Wan task completed but no video URL found")
                elif status in ("failed", "error"):
                    raise VideoGenerationError(f"Wan task failed: {data}")
            raise TimeoutError("Wan task timed out")

    # -- Seedance (legacy / self-hosted) -----------------------------------

    async def _generate_seedance(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        key = settings.seedance_api_key
        base = settings.seedance_api_url or "http://localhost:8000"
        if not key:
            raise ProviderNotAvailableError("No SEEDANCE_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"seedance_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {"prompt": prompt}
        if source_video_url:
            body["video_url"] = source_video_url
        if reference_images:
            body["reference_images"] = reference_images

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # Submit
            submit_resp = await client.post(
                f"{base.rstrip('/')}/api/v1/video/generate",
                headers=headers,
                json=body,
                timeout=60,
            )
            submit_resp.raise_for_status()
            data = submit_resp.json()
            task_id = data.get("task_id") or data.get("id") or data.get("job_id")
            if not task_id:
                raise VideoGenerationError(f"Seedance submit returned no task_id: {data}")
            logger.info("Seedance task submitted: %s", str(task_id)[:12])

            # Poll
            for _ in range(60):
                await asyncio.sleep(10)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/api/v1/tasks/{task_id}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data.get("status") or data.get("state", "")
                if status in ("completed", "succeeded", "done"):
                    video_url = data.get("video_url") or data.get("output_url", "")
                    if video_url:
                        dl = await client.get(video_url, timeout=300)
                        dl.raise_for_status()
                        out.write_bytes(dl.content)
                        logger.info("Seedance video saved: %s", out)
                        return str(out)
                    raise VideoGenerationError("Seedance completed but no video URL found")
                elif status in ("failed", "error"):
                    raise VideoGenerationError(f"Seedance task failed: {data}")
            raise TimeoutError("Seedance task timed out")

    # -- Local GPU inference (Wan2.1-VACE-14B) -----------------------------

    async def _generate_local(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        """Edit video locally using Wan2.1-VACE-14B.

        Input: source video + editing prompt → Output: edited video.
        Supports multi-GPU via device_map=\"auto\", FP8/BF16/FP16.
        """
        model_path = settings.local_video_model_path or settings.local_video_model_name
        if not model_path:
            raise ProviderNotAvailableError(
                "No LOCAL_VIDEO_MODEL_PATH or LOCAL_VIDEO_MODEL_NAME configured"
            )

        out = Path(output_path or settings.storage_temp_path / f"vace_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            from src.core.vace_pipeline import get_vace_pipeline

            logger.info("Loading VACE pipeline from: %s", model_path)
            logger.info("GPU count: %d", _count_gpus())

            # Use singleton — model stays loaded across requests
            pipe = await asyncio.to_thread(get_vace_pipeline, model_path)
            if not pipe._ready:
                await asyncio.to_thread(pipe.setup)

            # Require source video for editing
            if not source_video_url or not Path(source_video_url).exists():
                raise VideoGenerationError(
                    "LOCAL/VACE video editing requires a source video. "
                    "Set source_video_url to a local file path."
                )

            logger.info("Source video: %s", source_video_url)
            logger.info("Edit prompt: %s", prompt[:120])

            # Determine size based on available VRAM
            gpu_count = _count_gpus()
            if gpu_count >= 2:
                size = "1280*720"  # 720P with 2+ GPUs
            else:
                size = "832*480"   # 480P with 1 GPU

            # Run VACE video editing (fast test params: 33 frames, 15 steps)
            result = await asyncio.to_thread(
                pipe.edit,
                source_video=source_video_url,
                prompt=prompt,
                num_inference_steps=15,
                guidance_scale=5.0,
                size=size,
                num_frames=33,
                seed=42,
            )

            # Export to MP4
            await asyncio.to_thread(
                pipe.export_video, result, str(out), 16,
            )
            logger.info("VACE video saved: %s (size=%s)", out, size)

        except ImportError as e:
            raise ProviderNotAvailableError(
                f"Local inference requires torch, diffusers, transformers: {e}. "
                "Install with: pip install torch diffusers transformers safetensors accelerate"
            )
        except FileNotFoundError as e:
            raise ProviderNotAvailableError(str(e))
        except Exception as e:
            logger.error("VACE inference failed: %s", e, exc_info=True)
            raise VideoGenerationError(f"Local video generation failed: {e}")

        return str(out)

    # -- OpenAI-Next (draw.openai-next.com) ---------------------------------

    async def _generate_openai_next(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        """Generate video via draw.openai-next.com API (per-use billing)."""
        key = settings.openai_next_api_key
        base = settings.openai_next_base_url or "https://draw.openai-next.com"
        model = settings.openai_next_video_model or "wan2.7-videoedit"
        if not key:
            raise ProviderNotAvailableError("No OPENAI_NEXT_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"on_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # Auto-size based on available model
        size = "832x480"  # 480p for cost efficiency

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # 1. Submit video generation task
            body: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "size": size,
            }
            if source_video_url:
                if source_video_url.startswith("http"):
                    body["video_url"] = source_video_url
                else:
                    body["video_url"] = source_video_url  # local path — API may reject
            if reference_images:
                body["reference_images"] = reference_images[:3]

            logger.info("OpenAI-Next submit: model=%s size=%s", model, size)
            submit_resp = await client.post(
                f"{base.rstrip('/')}/v1/video/generations",
                headers=headers,
                json=body,
                timeout=60,
            )
            if submit_resp.status_code != 200:
                err = submit_resp.json()
                raise VideoGenerationError(
                    f"OpenAI-Next submit failed: {err.get('message', err)}"
                )
            data = submit_resp.json()
            task_id = data.get("task_id") or data.get("id")
            if not task_id:
                # Some models return direct result
                video_url = data.get("video_url") or data.get("output", {}).get("video_url")
                if video_url:
                    dl = await client.get(video_url, timeout=300)
                    dl.raise_for_status()
                    out.write_bytes(dl.content)
                    logger.info("OpenAI-Next video saved: %s", out)
                    return str(out)
                raise VideoGenerationError(f"OpenAI-Next: no task_id in response: {data}")

            logger.info("OpenAI-Next task: %s", str(task_id)[:24])

            # 2. Poll for completion
            for i in range(120):  # up to 20 min
                await asyncio.sleep(10)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/v1/tasks/{task_id}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                result = poll_resp.json()
                status = str(result.get("status") or result.get("task_status", "")).lower()
                if status in ("succeeded", "completed", "done"):
                    video_url = result.get("video_url") or result.get("output", {}).get("video_url")
                    if not video_url:
                        raise VideoGenerationError(
                            f"OpenAI-Next task {task_id} completed but no video_url"
                        )
                    dl = await client.get(video_url, timeout=300, follow_redirects=True)
                    dl.raise_for_status()
                    out.write_bytes(dl.content)
                    logger.info("OpenAI-Next video saved: %s (%d bytes)", out, out.stat().st_size)
                    return str(out)
                elif status in ("failed", "cancelled", "error"):
                    raise VideoGenerationError(
                        f"OpenAI-Next task {task_id} {status}: {result.get('message', result)}"
                    )
                if (i + 1) % 12 == 0:
                    logger.info("  polling... (%d/%d)", i + 1, 120)

            raise TimeoutError(f"OpenAI-Next task {task_id} timed out")

    # -- Grok (xAI) video editing via OpenAI-Next ---------------------------

    async def _generate_grok_v2v(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        """Edit video via xAI Grok on draw.openai-next.com (per-use billing).

        If source_video_url is a local path, the video is served via the
        project's public URL (nginx proxy at 218.13.42.36/whatif/).
        """
        key = settings.openai_next_api_key
        base = settings.openai_next_base_url or "https://draw.openai-next.com"
        if not key:
            raise ProviderNotAvailableError("No OPENAI_NEXT_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"grok_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        # Resolve video URL: local path → public URL via nginx proxy
        video_url = source_video_url or ""
        if video_url and not video_url.startswith("http"):
            # Local path like "45f14c6f-.../assets/video.mp4" →
            #   http://218.13.42.36/whatif/api/assets/{asset_id}/download
            # We extract asset_id from the URL that was passed
            # jobs.py passes _source_video which may be a local file path
            # Try to resolve it to a public URL
            from src.db import SessionLocal
            from src.models import Asset
            db = SessionLocal()
            try:
                # Search assets by file_path suffix
                asset = db.query(Asset).filter(
                    Asset.file_path.like(f"%{Path(video_url).name}"),
                ).order_by(Asset.created_at.desc()).first()
                if asset:
                    base = settings.public_base_url or "http://218.13.42.36/whatif"
                    video_url = f"{base.rstrip('/')}/api/assets/{asset.id}/download"
                    logger.info("Grok: resolved public URL for asset %s", asset.id)
                else:
                    raise VideoGenerationError(
                        f"Grok needs a public URL but could not find asset for {video_url}"
                    )
            finally:
                db.close()

        logger.info("Grok video URL: %s", video_url[:100])
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        body: dict[str, Any] = {
            "model": "grok-imagine-video",
            "prompt": prompt,
            "video": {"url": video_url},
        }

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # 1. Submit
            logger.info("Grok submit: prompt=%s...", prompt[:80])
            submit_resp = await client.post(
                f"{base.rstrip('/')}/xai-video/v1/videos/edits",
                headers=headers,
                json=body,
                timeout=60,
            )
            submit_resp.raise_for_status()
            data = submit_resp.json()
            if data.get("code") == -1:
                raise VideoGenerationError(f"Grok submit failed: {data.get('message', data)}")

            rid = data.get("request_id") or data.get("task_id")
            if not rid:
                raise VideoGenerationError(f"Grok: no request_id in response: {data}")
            logger.info("Grok task submitted: %s", str(rid)[:32])

            # 2. Poll
            for i in range(240):  # up to 60 min
                await asyncio.sleep(15)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/v1/tasks/{rid}",
                    headers=headers,
                    timeout=30,
                )
                if poll_resp.status_code == 200:
                    result = poll_resp.json()
                    if result.get("success") is False:
                        continue
                    dl_url = result.get("video_url") or result.get("output", {}).get("video_url")
                    if dl_url:
                        dl = await client.get(dl_url, timeout=300, follow_redirects=True)
                        dl.raise_for_status()
                        out.write_bytes(dl.content)
                        logger.info("Grok video saved: %s (%d bytes)", out, out.stat().st_size)
                        return str(out)
                if (i + 1) % 12 == 0:
                    logger.info("  grok poll %d/240", i + 1)

            raise TimeoutError(f"Grok task {rid} timed out after 60 min")

    # -- Grok I2V (xAI image-to-video, ~$0.40/8s) -------------------------

    async def _generate_grok_i2v(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        """Generate video from image+text via xAI Grok. Perfect for movie ending rewrites.

        Extracts the first frame from the uploaded video as the image input,
        combines it with the rewritten ending prompt, and generates an 8s video.
        """
        key = settings.openai_next_api_key
        base = settings.openai_next_base_url or "https://draw.openai-next.com"
        if not key:
            raise ProviderNotAvailableError("No OPENAI_NEXT_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"grok_i2v_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        # Resolve image URL: use reference_image, or extract frame from source video
        image_url = None
        if reference_images and len(reference_images) > 0:
            image_url = reference_images[0]
        elif source_video_url:
            # Use the source video as the image — extract first frame URL
            # For local files, construct public HTTPS URL
            from src.db import SessionLocal
            from src.models import Asset
            db = SessionLocal()
            try:
                asset = db.query(Asset).filter(
                    Asset.file_path.like(f"%{Path(source_video_url).name}"),
                ).order_by(Asset.created_at.desc()).first()
                if asset:
                    base_pub = settings.public_base_url or "https://tissue-stranger-love-suspension.trycloudflare.com"
                    image_url = f"{base_pub.rstrip('/')}/api/assets/{asset.id}/download"
                    logger.info("Grok I2V: using video as image source: %s", image_url[:80])
            finally:
                db.close()

        if not image_url:
            # Fallback: generate without image (text-only, may still work)
            logger.warning("Grok I2V: no image source available, using text-only mode")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        body: dict[str, Any] = {
            "model": "grok-imagine-video",
            "prompt": prompt,
        }
        if image_url:
            body["image_url"] = image_url

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # 1. Submit
            print(f"[GROK_I2V] Submitting: {prompt[:60]}...", flush=True)
            submit_resp = await client.post(
                f"{base.rstrip('/')}/xai-video/v1/videos/generations",
                headers=headers,
                json=body,
                timeout=60,
            )
            submit_resp.raise_for_status()
            data = submit_resp.json()
            rid = data.get("request_id") or data.get("task_id")
            if not rid:
                raise VideoGenerationError(f"Grok I2V: no request_id: {data}")
            print(f"[GROK_I2V] Task submitted: {rid}", flush=True)

            # 2. Poll
            for i in range(120):
                await asyncio.sleep(10)
                try:
                    poll_resp = await client.get(
                        f"{base.rstrip('/')}/xai-video/v1/videos/{rid}",
                        headers=headers,
                        timeout=30,
                    )
                    poll_resp.raise_for_status()
                    result = poll_resp.json()
                    status = result.get("status", "?")
                    if (i + 1) % 6 == 0:
                        print(f"[GROK_I2V] poll {i+1}/120 status={status}", flush=True)
                    if status == "done":
                        video_url = result.get("video", {}).get("url")
                        if video_url:
                            print(f"[GROK_I2V] Downloading video from {video_url[:60]}...", flush=True)
                            try:
                                dl = await client.get(video_url, timeout=15, follow_redirects=True)
                                dl.raise_for_status()
                                out.write_bytes(dl.content)
                                print(f"[GROK_I2V] Saved: {out} ({out.stat().st_size} bytes)", flush=True)
                                return str(out)
                            except Exception as dl_err:
                                print(f"[GROK_I2V] Download failed: {dl_err}", flush=True)
                                # Save the video URL for manual download
                                url_file = out.with_suffix(".url.txt")
                                url_file.write_text(video_url)
                                print(f"[GROK_I2V] Video URL saved to {url_file}", flush=True)
                                raise VideoGenerationError(
                                    f"Grok I2V video generated but download blocked (vidgen.x.ai CDN unreachable). "
                                    f"URL saved to {url_file}"
                                )
                        raise VideoGenerationError("Grok I2V done but no video URL")
                    elif status == "failed":
                        raise VideoGenerationError(f"Grok I2V failed: {result}")
                except Exception as poll_err:
                    print(f"[GROK_I2V] Poll error at {i+1}/120: {poll_err}", flush=True)
                    raise

            raise TimeoutError(f"Grok I2V task {rid} timed out")

    async def _translate_prompt(self, prompt: str) -> str:
        """Translate Chinese prompt to English, strip copyrighted names for content moderation."""
        msgs = [{
            "role": "user",
            "content": (
                f"Translate this video prompt to English and make it generic (NOT about any specific movie, book, or character). "
                f"Replace any proper names (characters, places, movies) with generic descriptions like 'a young wizard', 'a dark-haired man', 'an ancient chamber'. "
                f"Keep ALL visual details, lighting, camera angles, and cinematic style. Output ONLY the English description, 2-3 sentences:\n\n{prompt}"
            ),
        }]
        eng = await model_router.text.generate(
            provider=ModelProvider.DEEPSEEK,
            messages=msgs,
            max_tokens=300,
        )
        return eng.strip().strip('"').strip("'")

    # -- Doubao-Seedance I2V (draw.openai-next.com) --------------------------

    async def _generate_doubao(
        self,
        prompt: str,
        source_video_url: str | None,
        reference_images: list[str] | None,
        output_path: str | None,
    ) -> str:
        """I2V via doubao-seedance on draw.openai-next.com (~$2/5s 480p).

        Extracts first frame from local video as base64, auto-translates Chinese
        prompts to English, downloads result from ByteDance CDN (accessible in CN).
        """
        import base64
        import subprocess
        import tempfile

        key = settings.openai_next_api_key
        base = settings.openai_next_base_url or "https://draw.openai-next.com"
        if not key:
            raise ProviderNotAvailableError("No OPENAI_NEXT_API_KEY configured")

        out = Path(output_path or settings.storage_temp_path / f"doubao_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- Build image from source video ---
        image_b64: str | None = None
        frame_path: str | None = None

        if reference_images and len(reference_images) > 0:
            # Use provided reference image
            ref = reference_images[0]
            if ref.startswith("data:image"):
                image_b64 = ref.split(",", 1)[1]
            elif Path(ref).exists():
                image_b64 = base64.b64encode(Path(ref).read_bytes()).decode()
            else:
                image_b64 = ref  # assume already base64
        elif source_video_url:
            src = Path(source_video_url)
            if src.exists():
                print(f"[DOUBAO] Extracting frame from {src.name}...", flush=True)
                fd, frame_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", str(src), "-vframes", "1", "-q:v", "5", "-f", "mjpeg", frame_path],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode == 0 and Path(frame_path).exists():
                        image_b64 = base64.b64encode(Path(frame_path).read_bytes()).decode()
                        print(f"[DOUBAO] Frame encoded: {len(image_b64)} chars base64", flush=True)
                    else:
                        print(f"[DOUBAO] ffmpeg failed: {result.stderr[:200]}", flush=True)
                finally:
                    Path(frame_path).unlink(missing_ok=True)

        # --- Auto-translate Chinese prompt to English ---
        eng_prompt = prompt
        if any('一' <= c <= '鿿' for c in prompt):
            print(f"[DOUBAO] Translating Chinese prompt...", flush=True)
            try:
                eng_prompt = await self._translate_prompt(prompt)
                print(f"[DOUBAO] Translated: {eng_prompt[:120]}", flush=True)
            except Exception as e:
                print(f"[DOUBAO] Translation failed: {e}, using original", flush=True)

        # --- Submit ---
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        body: dict[str, Any] = {
            "model": "doubao-seedance-2-0-fast-260128",
            "prompt": eng_prompt,
            "size": "832x480",
        }
        if image_b64:
            body["image"] = image_b64
            print(f"[DOUBAO] I2V mode (image {len(image_b64)} chars)", flush=True)
        else:
            print(f"[DOUBAO] T2V mode (no image)", flush=True)

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            print(f"[DOUBAO] Submitting: {eng_prompt[:80]}...", flush=True)
            submit_resp = await client.post(
                f"{base.rstrip('/')}/v1/video/generations",
                headers=headers,
                json=body,
                timeout=60,
            )
            submit_resp.raise_for_status()
            data = submit_resp.json()
            tid = data.get("task_id") or data.get("id")
            if not tid:
                if data.get("error"):
                    raise VideoGenerationError(
                        f"Doubao submit failed: {data['error'].get('message', data['error'])}"
                    )
                raise VideoGenerationError(f"Doubao: no task_id in response: {data}")
            print(f"[DOUBAO] Task submitted: {tid}", flush=True)

            # --- Poll ---
            for i in range(60):  # up to 10 min
                await asyncio.sleep(15)
                poll_resp = await client.get(
                    f"{base.rstrip('/')}/v1/tasks/{tid}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                result = poll_resp.json()
                status = str(result.get("status", "")).lower()
                if (i + 1) % 4 == 0:
                    print(f"[DOUBAO] poll {i+1}/60 status={status}", flush=True)
                if status == "completed":
                    video_url = (result.get("result_url")
                                 or result.get("output", {}).get("content", {}).get("video_url")
                                 or result.get("output", {}).get("video_url")
                                 or result.get("video_url"))
                    if video_url:
                        print(f"[DOUBAO] Downloading from {video_url[:60]}...", flush=True)
                        dl = await client.get(video_url, timeout=120, follow_redirects=True)
                        dl.raise_for_status()
                        out.write_bytes(dl.content)
                        print(f"[DOUBAO] Saved: {out} ({out.stat().st_size} bytes)", flush=True)
                        return str(out)
                    raise VideoGenerationError(f"Doubao completed but no video_url")
                elif status == "failed":
                    raise VideoGenerationError(
                        f"Doubao task failed (may be content moderation). "
                        f"Try a simpler/different prompt."
                    )

            raise TimeoutError(f"Doubao task {tid} timed out")

    # -- ffmpeg fallback -----------------------------------------------------

    async def _generate_ffmpeg(self, prompt: str, output_path: str | None) -> str:
        from src.core.video_pipeline import _generate_ffmpeg_video
        out = output_path or str(settings.storage_temp_path / f"ffmpeg_{hash(prompt) & 0xFFFFFFF:07x}.mp4")
        return await asyncio.to_thread(_generate_ffmpeg_video, prompt, "", out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract JSON from text that may have markdown fences or surrounding prose."""
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    return text


def _heuristic_storyboard(script: str) -> dict:
    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()][:8]
    frames = []
    for i, para in enumerate(paragraphs):
        frames.append({
            "description": para[:200],
            "timing": f"{3 + i * 2}s-{5 + i * 2}s",
        })
    return {
        "frames": frames,
        "total_duration": f"{max(5, len(frames) * 5)}s",
    }


def _count_gpus() -> int:
    """Return the number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _best_dtype():
    """Pick the best torch dtype based on GPU count and Blackwell support.

    Strategy:
    - 2+ GPUs (e.g. 2x5090=64GB): use BF16 with device_map=\"auto\"
    - 1 GPU 32GB+: try BF16, fall back to FP8 if OOM
    - 1 GPU 24GB: FP8 or INT4
    - No GPU / old GPU: FP16 CPU offload (will be very slow)
    """
    import torch
    gpu_count = _count_gpus()
    total_vram = 0
    for i in range(gpu_count):
        total_vram += torch.cuda.get_device_properties(i).total_memory

    # 14B model at BF16 ≈ 28GB weights + ~10GB activations/buffers ≈ 38GB peak
    # 2x 5090 = 64GB → BF16 safe
    # 1x 5090 = 32GB → borderline at BF16, FP8 is safe
    # 1x 4090 = 24GB → need INT4 or heavy offload

    if total_vram >= 60 * (1024**3):  # 60GB+ → 2 cards or high-end
        return torch.bfloat16
    elif total_vram >= 30 * (1024**3):  # 30-60GB → single 5090/4090
        try:
            # Blackwell supports native FP8
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    elif total_vram >= 20 * (1024**3):
        return torch.float16
    else:
        return torch.float32  # CPU offload scenario


# ---------------------------------------------------------------------------
# Convenience composite
# ---------------------------------------------------------------------------

class ModelRouter:
    def __init__(self):
        self.text = TextModelRouter()
        self.video = VideoModelRouter()


model_router = ModelRouter()
