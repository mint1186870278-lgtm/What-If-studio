from .attention import flash_attention
from .model import WanModel
from .vace_model import VaceWanModel
from .vae import WanVAE

# T5 is optional (heavy, needs extra deps)
try:
    from .t5 import T5Decoder, T5Encoder, T5EncoderModel, T5Model
except Exception:
    T5Model = T5Encoder = T5Decoder = T5EncoderModel = None  # type: ignore[assignment]

try:
    from .tokenizers import HuggingfaceTokenizer
except Exception:
    HuggingfaceTokenizer = None  # type: ignore[assignment]

__all__ = [
    'WanVAE',
    'WanModel',
    'VaceWanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'flash_attention',
]
