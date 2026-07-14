import numpy as np
import pytest

from farm_audio_event_detection.preprocessing.audio_preprocessing import (
    CLASS_TO_INDEX,
    build_records,
    extract_spectrogram,
    iter_audio_windows,
)


def test_build_records_scans_dataset():
    records = build_records("dataset")

    assert len(records) == 280
    assert {record.label for record in records} == set(CLASS_TO_INDEX)
    assert {record.fold for record in records} == {1, 2, 3, 4, 5}


def test_extract_spectrogram_shape():
    pytest.importorskip("librosa")
    pytest.importorskip("skimage")
    sample_rate = 44_100
    audio = np.zeros(sample_rate, dtype=np.float32)

    spectrogram = extract_spectrogram(audio, sample_rate=sample_rate, image_width=250)

    assert spectrogram.shape == (128, 250, 3)
    assert spectrogram.dtype == np.float32


def test_iter_audio_windows_uses_half_second_hop():
    sample_rate = 10
    audio = np.ones(50, dtype=np.float32)


    windows = list(iter_audio_windows(audio, sample_rate=sample_rate, window_seconds=3.0, hop_seconds=0.5))

    assert windows[0][1].start_seconds == 0.0
    assert windows[1][1].start_seconds == 0.5
    assert windows[-1][1].end_seconds == 5.0