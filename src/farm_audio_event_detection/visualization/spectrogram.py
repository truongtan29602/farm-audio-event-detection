"""Spectrogram visualization utilities.

This module owns the image-like Log-Mel spectrogram representation used by the
preprocessing pipeline. Keeping this code in the visualization package keeps
dataset scanning, audio windowing, and pickle serialization separate from the
feature visualization logic.
"""

import numpy as np


DEFAULT_SPECTROGRAM_SPECS: tuple[tuple[int, int], ...] = (
    (1024, 256),
    (2048, 512),
    (4096, 1024),
)


def _import_librosa():
    """Import librosa only when spectrogram extraction is actually needed."""

    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "librosa is required for spectrogram extraction. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return librosa


def _import_resize():
    """Import scikit-image resize only when spectrogram extraction is needed."""

    try:
        from skimage.transform import resize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-image is required for spectrogram resizing. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return resize


def _normalize_channel(values: np.ndarray) -> np.ndarray:
    """Normalize a spectrogram channel to the 0-1 range."""

    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def extract_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 44_100,
    n_mels: int = 128,
    image_width: int = 250,
    specs: tuple[tuple[int, int], ...] = DEFAULT_SPECTROGRAM_SPECS,
) -> np.ndarray:
    """Create a 3-channel Log-Mel spectrogram image.

    Each channel uses a different `(n_fft, hop_length)` pair:

    - short window: catches quick changes like a bark attack
    - medium window: balanced detail
    - long window: smoother view for longer calls like cow/sheep sounds

    The final shape is `(128, 250, 3)` by default.
    """

    librosa = _import_librosa()
    resize = _import_resize()
    channels: list[np.ndarray] = []
    for n_fft, hop_length in specs:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        resized = resize(
            log_mel,
            output_shape=(n_mels, image_width),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        channels.append(_normalize_channel(resized))

    return np.stack(channels, axis=-1).astype(np.float32)