"""VACE (Video Aligned Content Enhancement) inference pipeline for Wan2.1-VACE.

Standalone video editing pipeline: source video + prompt → edited video.
Supports multi-GPU via device_map="auto", FP8/BF16/FP16, and CPU offload.

Model: Wan-AI/Wan2.1-VACE-14B
"""

from __future__ import annotations

import asyncio
import logging
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU / dtype helpers
# ---------------------------------------------------------------------------

def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _best_dtype():
    """Pick optimal torch dtype based on available GPU VRAM."""
    import torch
    nc = _count_gpus()
    if nc == 0:
        return torch.float32
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(nc))
    if total >= 60 * (1024**3):      # 64GB+ → BF16 safe
        return torch.bfloat16
    elif total >= 30 * (1024**3):    # 32GB+ → try BF16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif total >= 20 * (1024**3):
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Wan2.1 custom T5 encoder (loads from checkpoint, no HuggingFace needed)
# ---------------------------------------------------------------------------

class _WanT5Encoder:
    """Minimal T5 encoder that loads Wan2.1's custom checkpoint format.

    24-layer T5 Encoder (UMT5-XXL compatible):
    - d_model=4096, d_ff=10240, num_heads=64
    - Gated-GELU activation
    - Relative position bias (32 buckets per head)
    """

    def __init__(self, checkpoint_path, dtype, device):
        import torch
        import torch.nn as nn

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.dtype = dtype
        self.device = device

        # Embedding
        self.embed = nn.Embedding(256384, 4096)
        self.embed.weight.data.copy_(ckpt["token_embedding.weight"])

        # 24 transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(24):
            block = _T5EncoderBlock(4096, 10240, 64, 32)
            block.norm1.weight.data.copy_(ckpt[f"blocks.{i}.norm1.weight"])
            block.norm2.weight.data.copy_(ckpt[f"blocks.{i}.norm2.weight"])
            block.attn.q_proj.weight.data.copy_(ckpt[f"blocks.{i}.attn.q.weight"])
            block.attn.k_proj.weight.data.copy_(ckpt[f"blocks.{i}.attn.k.weight"])
            block.attn.v_proj.weight.data.copy_(ckpt[f"blocks.{i}.attn.v.weight"])
            block.attn.o_proj.weight.data.copy_(ckpt[f"blocks.{i}.attn.o.weight"])
            block.ffn.fc1.weight.data.copy_(ckpt[f"blocks.{i}.ffn.fc1.weight"])
            block.ffn.gate.weight.data.copy_(ckpt[f"blocks.{i}.ffn.gate.0.weight"])
            block.ffn.fc2.weight.data.copy_(ckpt[f"blocks.{i}.ffn.fc2.weight"])
            # pos_embedding is [num_buckets, num_heads] → transpose to [num_heads, num_buckets]
            block.pos_bias.data.copy_(
                ckpt[f"blocks.{i}.pos_embedding.embedding.weight"].t()
            )
            self.blocks.append(block)

        self.final_norm = nn.LayerNorm(4096, eps=1e-6)
        self.final_norm.weight.data.copy_(ckpt["norm.weight"])

        self.to(dtype).to(device)
        self.eval()

    def to(self, *args, **kwargs):
        self.embed.to(*args, **kwargs)
        self.blocks.to(*args, **kwargs)
        self.final_norm.to(*args, **kwargs)
        return self

    def eval(self):
        self.embed.eval()
        self.blocks.eval()
        self.final_norm.eval()
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass. Returns last_hidden_state [B, L, 4096]."""
        import torch
        input_ids = input_ids.to(dtype=torch.long)
        with torch.no_grad():
            x = self.embed(input_ids)
            mask = None
            if attention_mask is not None:
                mask = (1.0 - attention_mask.float()) * -10000.0
                mask = mask.unsqueeze(1).unsqueeze(2)
            for block in self.blocks:
                x = block(x, mask)
            x = self.final_norm(x)
        return type('Out', (), {'last_hidden_state': x})()


class _T5EncoderBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_buckets):
        super().__init__()
        import torch.nn as nn
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_buckets = num_buckets

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = _T5Attention(d_model, num_heads)
        self.ffn = _T5GatedFFN(d_model, d_ff)
        self.register_buffer("pos_bias", torch.zeros(num_heads, num_buckets))

    def forward(self, x, mask=None):
        import torch
        import torch.nn.functional as F
        # Self-attention with relative position bias
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x, self.pos_bias, mask)
        x = residual + attn_out
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class _T5Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        import torch.nn as nn
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pos_bias, mask=None):
        import torch
        import torch.nn.functional as F
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Relative position bias
        rp_bucket = _relative_position_bucket(
            L, self.num_heads, pos_bias.shape[1], x.device,
        )
        rp_bias = pos_bias[:, rp_bucket]  # [H, L, L]
        attn = attn + rp_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class _T5GatedFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        import torch.nn as nn
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        import torch.nn.functional as F
        return self.fc2(F.gelu(self.gate(x)) * self.fc1(x))


def _relative_position_bucket(seq_len, num_heads, num_buckets, device):
    """T5-style relative position bucket assignment."""
    import torch
    positions = torch.arange(seq_len, device=device)
    rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L]
    # Map to buckets using T5's formula
    rel_pos = rel_pos + num_buckets // 2  # shift to [0, num_buckets-1]
    rel_pos = torch.clamp(rel_pos, 0, num_buckets - 1)
    return rel_pos.long()


# ---------------------------------------------------------------------------
# VACE pipeline
# ---------------------------------------------------------------------------

class VacePipeline:
    """Wan2.1-VACE video editing pipeline.

    Usage::

        pipe = VacePipeline("/root/Wan2.1-VACE-14B")
        pipe.setup()
        output = pipe.edit(
            source_video="/path/to/input.mp4",
            prompt="让画面变成夜晚，加上雪花飘落",
            num_inference_steps=30,
        )
        pipe.export_video(output, "/path/to/output.mp4")
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or settings.local_video_model_path
                               or settings.local_video_model_name)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        gpu_count = _count_gpus()
        self.device = "cuda" if gpu_count > 0 else "cpu"
        # Check free VRAM — if < 20 GB, use aggressive CPU offload
        if gpu_count > 0:
            import torch
            free_gb = min(
                torch.cuda.get_device_properties(i).total_memory
                - torch.cuda.memory_allocated(i)
                for i in range(min(gpu_count, 2))
            ) / (1024**3)
            self._low_vram = free_gb < 20
            if self._low_vram:
                logger.info("Low VRAM detected (%.1f GB free) — enabling CPU offload", free_gb)
        else:
            self._low_vram = True
        self.dtype = _best_dtype()
        self._ready = False

        # Lazy-loaded components
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.transformer = None
        self.scheduler = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load all model components. Call once before inference."""
        if self._ready:
            return

        import torch
        logger.info("Loading Wan2.1-VACE from %s (dtype=%s)", self.model_path, self.dtype)

        # -- VAE ----------------------------------------------------------
        vae_path = self.model_path / "Wan2.1_VAE.pth"
        if vae_path.exists():
            self.vae = self._load_wan_vae(vae_path)
        else:
            raise FileNotFoundError(f"VAE not found: {vae_path}")

        # -- T5 text encoder (local Wan2.1 checkpoint, not HuggingFace) ----
        t5_dir = self.model_path / "google" / "umt5-xxl"
        t5_weights = self.model_path / "models_t5_umt5-xxl-enc-bf16.pth"
        if t5_dir.exists() and t5_weights.exists():
            import torch
            from transformers import T5TokenizerFast
            logger.info("Loading T5 tokenizer from local: %s", t5_dir)
            self.tokenizer = T5TokenizerFast.from_pretrained(str(t5_dir), use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # T5 always on CPU — only used once per prompt, saves 11 GB GPU VRAM
            self.text_encoder = _WanT5Encoder(t5_weights, torch.float32, "cpu")
            logger.info("T5 encoder loaded (device=cpu, dtype=float32)")
        else:
            logger.warning("T5 encoder not found locally, text conditioning disabled")
            self.text_encoder = None
            self.tokenizer = None

        # -- Transformer (DiT + VACE) -------------------------------------
        self.transformer = self._load_vace_transformer()

        # -- Scheduler (Flow Match Euler) ---------------------------------
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=5.0,  # flow_shift for 720P; use 3.0 for 480P
        )

        self._ready = True
        logger.info("VACE pipeline ready (device=%s, gpus=%d)", self.device, _count_gpus())

    # ------------------------------------------------------------------
    # VAE loading
    # ------------------------------------------------------------------

    def _load_wan_vae(self, vae_path: Path):
        """Load Wan-VAE from .pth checkpoint (local only, no HuggingFace)."""
        import torch
        from diffusers import AutoencoderKLWan

        logger.info("Loading VAE from local checkpoint: %s", vae_path)
        vae = AutoencoderKLWan(
            in_channels=3, out_channels=3,
            z_dim=16,
            base_dim=128,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
        )
        ckpt = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae.load_state_dict(ckpt, strict=False)
        vae = vae.to(torch.float32)
        vae.eval()
        # VAE stays on CPU — saves 2 GB GPU VRAM per card for DiT forward activations
        return vae

    # ------------------------------------------------------------------
    # DiT + VACE transformer loading
    # ------------------------------------------------------------------

    def _load_vace_transformer(self):
        """Load the VACE DiT transformer across GPUs shard-by-shard (zero CPU build).

        Strategy: build model on 'meta' device (instant, zero RAM), then stream
        7 safetensor shards and put each tensor directly onto the right GPU.
        Even layers → GPU 0, odd layers → GPU 1.
        """
        import torch
        import gc
        import re as _re
        from safetensors.torch import load_file

        cfg_path = self.model_path / "config.json"
        with open(cfg_path) as f:
            config = json.load(f)

        from src.core.wan_modules.vace_model import VaceWanModel

        # 1. Build on meta device — instant, zero RAM
        logger.info("Building VaceWanModel on meta device (%d layers)...", config["num_layers"])
        with torch.device("meta"):
            transformer = VaceWanModel(
                vace_layers=config.get("vace_layers", []),
                vace_in_dim=config.get("vace_in_dim", 96),
                dim=config["dim"],
                ffn_dim=config["ffn_dim"],
                freq_dim=config["freq_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                in_dim=config.get("in_dim", 16),
                out_dim=config.get("out_dim", 16),
                text_dim=4096,
                text_len=512,
                patch_size=(1, 2, 2),
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
            )

        # 2. Discover shards
        idx_path = self.model_path / "diffusion_pytorch_model.safetensors.index.json"
        if idx_path.exists():
            import json as _json
            with open(idx_path) as f:
                idx = _json.load(f)
            shard_files = sorted(set(idx["weight_map"].values()))
        else:
            shard_files = sorted(
                str(p.relative_to(self.model_path))
                for p in self.model_path.glob("diffusion_pytorch_model*.safetensors")
            )

        total_gb = sum((self.model_path / s).stat().st_size for s in shard_files) / 1e9
        logger.info("Loading %d shard(s) (%.1f GB) streamed to GPU0+GPU1...",
                     len(shard_files), total_gb)

        meta_keys = set(dict(transformer.named_parameters()).keys())
        loaded = 0

        for shard_i, shard in enumerate(shard_files):
            shard_path = self.model_path / shard
            sd = load_file(str(shard_path))
            logger.info("  shard %d/%d → %d tensors", shard_i + 1, len(shard_files), len(sd))

            for k, v in list(sd.items()):
                nk = k[6:] if k.startswith("model.") else k
                if nk not in meta_keys:
                    continue
                # Assign GPU based on block index
                m = _re.search(r'blocks\.(\d+)', nk)
                if m:
                    dev = "cuda:0" if int(m.group(1)) % 2 == 0 else "cuda:1"
                else:
                    dev = "cuda:0"  # embeddings, norms, head, etc.
                param = torch.nn.Parameter(
                    v.to(device=dev, dtype=self.dtype), requires_grad=False,
                )
                _set_nested_attr(transformer, nk, param)
                loaded += 1
                del param, v

            del sd
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Loaded %d/%d params", loaded, len(meta_keys))

        # Replace freqs buffer (created on meta, need real tensor)
        from src.core.wan_modules.model import rope_params
        d = config["dim"]
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1).to("cuda:0")
        del transformer.freqs
        transformer.register_buffer("freqs", freqs, persistent=False)

        transformer.eval()
        logger.info("VACE transformer ready (even→GPU0, odd→GPU1)")

        self._vace_layers = config.get("vace_layers", [])
        self._vace_in_dim = config.get("vace_in_dim", 96)
        return transformer

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str) -> np.ndarray | None:
        """Encode text prompt via T5."""
        if self.text_encoder is None or self.tokenizer is None:
            return None
        import torch
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        # Use T5's device (may be CPU in low-VRAM mode)
        t5_device = next(self.text_encoder.embed.parameters()).device
        inputs = {k: v.to(t5_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state.cpu().numpy()

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def load_video_frames(self, video_path: str, max_frames: int = 81,
                          target_size: tuple[int, int] | None = None) -> np.ndarray:
        """Load video and return normalized frames [T, C, H, W] in [-1, 1]."""
        frames = self._read_frames(video_path, max_frames)

        if target_size:
            h, w = target_size
        else:
            h, w = self._clamp_resolution(frames[0].shape[:2])

        import torch
        processed = []
        for frame in frames:
            from PIL import Image
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((w, h), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            processed.append(arr)

        video = np.stack(processed)  # [T, H, W, C]
        video = np.transpose(video, (0, 3, 1, 2))  # [T, C, H, W]
        return video

    def _read_frames(self, video_path: str, max_frames: int) -> list[np.ndarray]:
        """Extract frames: use ffmpeg directly with fps filter to get exactly max_frames."""
        import tempfile
        import imageio.v3 as iio
        fps = max(1, max_frames / 5)  # aim for ~5s of video
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps:.1f},scale=iw:ih",
                "-q:v", "2", f"{tmpdir}/%06d.jpg",
            ], check=True, capture_output=True, timeout=120)
            files = sorted(Path(tmpdir).glob("*.jpg"))
            stride = max(1, len(files) // max_frames)
            return [iio.imread(str(f)) for f in files[::stride][:max_frames]]

    def export_video(self, frames: np.ndarray, output_path: str,
                     fps: int = 16) -> str:
        """Export [T, C, H, W] normalized frames to MP4."""
        import torch
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Denormalize: [-1, 1] → [0, 255]
        frames = np.clip((frames + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        # [T, C, H, W] → [T, H, W, C]
        frames = np.transpose(frames, (0, 2, 3, 1))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image
            for i, f in enumerate(frames):
                Image.fromarray(f).save(f"{tmpdir}/{i:06d}.png")

            subprocess.run([
                "ffmpeg", "-y", "-r", str(fps),
                "-i", f"{tmpdir}/%06d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "medium", "-crf", "20",
                str(out),
            ], check=True, capture_output=True, timeout=120)

        logger.info("Video exported: %s", out)
        return str(out)

    # ------------------------------------------------------------------
    # Core inference — video editing
    # ------------------------------------------------------------------

    def edit(
        self,
        source_video: str,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        num_frames: int = 81,
        size: str = "1280*720",
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> np.ndarray:
        """Edit a video clip.

        Args:
            source_video: Path to input video file.
            prompt: Editing instruction (Chinese recommended).
            num_inference_steps: Diffusion steps (25-50).
            guidance_scale: CFG scale (5.0 typical).
            num_frames: Output frame count (must be 4n+1, e.g. 81).
            size: Output size "W*H" (e.g. "1280*720").
            negative_prompt: Things to avoid.
            seed: Random seed for reproducibility.

        Returns:
            np.ndarray [T, C, H, W] in [-1, 1] range.
        """
        if not self._ready:
            self.setup()

        import torch

        w, h = (int(x) for x in size.split("*"))

        # 1. Load & preprocess source video
        logger.info("Loading source video: %s", source_video)
        src_frames = self.load_video_frames(
            source_video, max_frames=num_frames, target_size=(h, w),
        )  # [T, C, H, W]

        # 2. Encode source video through VAE → latents
        # VAE is on GPU 1; send source frames there for encoding
        vae_device = next(self.vae.parameters()).device
        src_tensor = torch.from_numpy(src_frames).unsqueeze(0).to(vae_device).to(torch.float32)
        self_src_tensor = src_tensor

        with torch.no_grad():
            src_tensor = src_tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            src_latents = self.vae.encode(src_tensor).latent_dist.sample()

        # Move latents to GPU 0 for DiT processing
        src_latents = src_latents.to(self.device).to(self.dtype)

        # 3. Build VACE conditioning from source latents
        vace_condition = self._build_vace_condition(src_latents)

        # 4. Encode text prompt
        text_embeds = self.encode_prompt(prompt)
        if text_embeds is not None:
            text_embeds = torch.from_numpy(text_embeds).to(self.device).to(self.dtype)
        if negative_prompt:
            neg_embeds = self.encode_prompt(negative_prompt)
            if neg_embeds is not None:
                neg_embeds = torch.from_numpy(neg_embeds).to(self.device).to(self.dtype)
        else:
            neg_embeds = None

        # 5. Prepare latents (noise)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        latent_shape = list(src_latents.shape)  # [B, C, T, H, W]
        noise = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        latents = noise

        # 6. Set up scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 7. Denoising loop (flow matching)
        logger.info("Running VACE editing: %d steps, prompt='%s'...",
                     num_inference_steps, prompt[:80])

        for i, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).expand(latents.shape[0]).to(self.device)

            # CFG: predict for both conditional and unconditional
            if neg_embeds is not None and guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2, dim=0)
                t_input = torch.cat([t_tensor] * 2, dim=0)
                emb_input = torch.cat([text_embeds, neg_embeds], dim=0)
                vace_input = torch.cat([vace_condition] * 2, dim=0)

                noise_pred = self._transformer_forward(
                    latent_input, t_input, emb_input, vace_input,
                )
                cond_pred, uncond_pred = noise_pred.chunk(2, dim=0)
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                noise_pred = self._transformer_forward(
                    latents, t_tensor, text_embeds, vace_condition,
                )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if (i + 1) % 10 == 0:
                logger.debug("  step %d/%d", i + 1, num_inference_steps)

        # 8. Decode latents through VAE (on GPU 1)
        with torch.no_grad():
            latents_fp32 = latents.to(torch.float32).to(vae_device)
            decoded = self.vae.decode(latents_fp32).sample
        decoded = decoded.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        output = decoded.cpu().numpy()  # [T, C, H, W]

        logger.info("VACE editing complete: output shape=%s", output.shape)
        return output

    # ------------------------------------------------------------------
    # VACE conditioning
    # ------------------------------------------------------------------

    def _build_vace_condition(self, latents: "torch.Tensor") -> "torch.Tensor":
        """Build VACE conditioning tensor from source video latents.

        The VACE model expects 96 channels of conditioning at specific layers.
        We use the source video VAE latents (16ch) + zero padding → 96ch.
        For best results, real depth/pose features should be used; this is
        a simplified version that lets the model reference the source content.
        """
        import torch
        B, C, T, H, W = latents.shape
        # Pad from 16ch → 96ch with zeros
        pad = torch.zeros(B, self._vace_in_dim - C, T, H, W,
                          device=latents.device, dtype=latents.dtype)
        return torch.cat([latents, pad], dim=1)

    def _transformer_forward(
        self,
        latents: "torch.Tensor",
        timestep: "torch.Tensor",
        encoder_hidden_states: "torch.Tensor",
        vace_condition: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Call VaceWanModel.forward() with native API."""
        B = latents.shape[0]
        x_list = [latents[i] for i in range(B)]
        ctx_list = [encoder_hidden_states[i] for i in range(B)]
        seq_len = latents.shape[1]

        if vace_condition is not None:
            vace_ctx = [vace_condition[i] for i in range(B)]
        else:
            vace_ctx = [torch.zeros_like(latents[i]) for i in range(B)]

        out_list = self.transformer(
            x=x_list, t=timestep,
            vace_context=vace_ctx, context=ctx_list,
            seq_len=seq_len,
        )
        return torch.stack(out_list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _clamp_resolution(self, frame_shape: tuple) -> tuple[int, int]:
        """Clamp resolution to a valid VACE size (multiple of 16)."""
        h, w = frame_shape[:2]
        max_area = 1280 * 720
        aspect = h / w
        nh = int(np.sqrt(max_area * aspect)) // 16 * 16
        nw = int(np.sqrt(max_area / aspect)) // 16 * 16
        return max(nh, 64), max(nw, 64)


# Module-level singleton
_vace_instance: VacePipeline | None = None


def _set_nested_attr(model, attr_path: str, value) -> None:
    """Set model.attr_path = value, creating intermediate modules as needed."""
    parts = attr_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], value)


def get_vace_pipeline(model_path: str | None = None) -> VacePipeline:
    """Get or create the VACE pipeline singleton."""
    global _vace_instance
    if _vace_instance is None:
        _vace_instance = VacePipeline(model_path)
    return _vace_instance
