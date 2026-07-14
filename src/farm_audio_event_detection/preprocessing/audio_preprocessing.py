"""Data preprocessing utilities for the farm audio event detection project.

This module intentionally handles preprocessing only:

- scan the seed dataset folder structure
- read fold numbers from ESC-50 style filenames
- load audio at a consistent sampling rate
- delegate 3-channel Log-Mel spectrogram creation to the visualization package
- save train/validation pickle files for the 5 official folds
- prepare overlapping windows for future long-recording inference

It does not train a model and it does not make predictions.
"""

import argparse
import csv
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from farm_audio_event_detection.visualization.spectrogram import extract_spectrogram


CLASS_TO_INDEX = {
    "dog": 0,
    "cat": 1,
    "cow": 2,
    "rooster": 3,
    "sheep": 4,
    "others": 5,
}
INDEX_TO_CLASS = {index: label for label, index in CLASS_TO_INDEX.items()}

FOLD_PATTERN = re.compile(r"^(?P<fold>[1-5])-.*\.wav$", re.IGNORECASE)


def _import_librosa():
    """Import librosa only when audio feature extraction is actually needed."""

    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "librosa is required for audio loading and spectrogram extraction. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return librosa


@dataclass(frozen=True)
class AudioRecord:
    """One audio file found in the dataset."""

    path: str
    filename: str
    label: str
    label_index: int
    fold: int


@dataclass(frozen=True)
class WindowRecord:
    """One window cut from a longer recording."""

    source_path: str
    window_index: int
    start_seconds: float
    end_seconds: float
    padded: bool
    rms_db: float
    is_low_energy: bool


def build_records(dataset_dir: str | Path, class_to_index: dict[str, int] | None = None) -> list[AudioRecord]:
    """Scan `dataset_dir` and return one record for each `.wav` file.

    The seed data is organized as:

    ```text
    dataset/cat/*.wav
    dataset/cow/*.wav
    dataset/dog/*.wav
    dataset/others/*.wav
    dataset/rooster/*.wav
    dataset/sheep/*.wav
    ```

    The target label comes from the folder name. The fold number comes from the
    first number in the filename, for example `2-110010-A-5.wav` belongs to fold 2.
    """

    dataset_path = Path(dataset_dir)
    labels = class_to_index or CLASS_TO_INDEX

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    records: list[AudioRecord] = []
    for label, label_index in sorted(labels.items(), key=lambda item: item[1]):
        label_dir = dataset_path / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Expected class folder is missing: {label_dir}")

        for wav_path in sorted(label_dir.glob("*.wav")):
            match = FOLD_PATTERN.match(wav_path.name)
            if not match:
                raise ValueError(f"Could not read fold number from filename: {wav_path.name}")

            records.append(
                AudioRecord(
                    path=str(wav_path),
                    filename=wav_path.name,
                    label=label,
                    label_index=label_index,
                    fold=int(match.group("fold")),
                )
            )

    return records


def load_audio(path: str | Path, sample_rate: int = 44_100) -> tuple[np.ndarray, int]:
    """Load one audio file as mono audio at `sample_rate`."""

    librosa = _import_librosa()
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32), sr


def extract_features(
    records: Iterable[AudioRecord],
    sample_rate: int = 44_100,
    n_mels: int = 128,
    image_width: int = 250,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract spectrogram features and labels for many dataset records."""

    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict] = []

    for record in tqdm(list(records), desc="Extracting spectrograms"):
        audio, sr = load_audio(record.path, sample_rate=sample_rate)
        features.append(extract_spectrogram(audio, sr, n_mels=n_mels, image_width=image_width))
        labels.append(record.label_index)
        metadata.append(asdict(record) | {"duration_seconds": round(len(audio) / sr, 4)})

    return np.stack(features), np.asarray(labels, dtype=np.int64), metadata


def split_by_validation_fold(records: list[AudioRecord], validation_fold: int) -> tuple[list[AudioRecord], list[AudioRecord]]:
    """Return `(train_records, validation_records)` for one ESC-50 fold."""

    train_records = [record for record in records if record.fold != validation_fold]
    validation_records = [record for record in records if record.fold == validation_fold]
    return train_records, validation_records


def save_records_manifest(records: list[AudioRecord], output_dir: str | Path) -> None:
    """Save dataset metadata as both CSV and JSON for easy checking."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]

    with (output_path / "records.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    with (output_path / "records.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_fold_pickles(
    records: list[AudioRecord],
    output_dir: str | Path,
    sample_rate: int = 44_100,
    n_mels: int = 128,
    image_width: int = 250,
) -> None:
    """Create one pickle file per validation fold.

    Each file contains `X_train`, `y_train`, `X_val`, `y_val`, plus simple metadata.
    Generated pickle files are ignored by git because they can always be rebuilt.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_records_manifest(records, output_path)

    for validation_fold in range(1, 6):
        train_records, validation_records = split_by_validation_fold(records, validation_fold)
        X_train, y_train, train_metadata = extract_features(
            train_records,
            sample_rate=sample_rate,
            n_mels=n_mels,
            image_width=image_width,
        )
        X_val, y_val, validation_metadata = extract_features(
            validation_records,
            sample_rate=sample_rate,
            n_mels=n_mels,
            image_width=image_width,
        )

        payload = {
            "validation_fold": validation_fold,
            "sample_rate": sample_rate,
            "feature_shape": [n_mels, image_width, 3],
            "class_to_index": CLASS_TO_INDEX,
            "index_to_class": INDEX_TO_CLASS,
            "X_train": X_train,
            "y_train": y_train,
            "train_metadata": train_metadata,
            "X_val": X_val,
            "y_val": y_val,
            "validation_metadata": validation_metadata,
        }

        output_file = output_path / f"fold_{validation_fold}.pkl"
        with output_file.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def rms_db(audio: np.ndarray) -> float:
    """Return an RMS loudness estimate in dBFS-like units."""

    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
    return 20.0 * np.log10(rms)


def iter_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float = 3.0,
    hop_seconds: float = 0.5,
    low_energy_db_threshold: float = -45.0,
) -> Iterable[tuple[np.ndarray, WindowRecord]]:
    """Yield overlapping windows from a continuous recording.

    A 3-second window matches the clean training clips. A 0.5-second hop gives
    the later detector many chances to place event boundaries within the
    project's +/- 500 ms margin.
    """

    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    starts = list(range(0, max(len(audio) - window_samples + 1, 1), hop_samples))
    last_start = max(len(audio) - window_samples, 0)
    if starts[-1] != last_start:
        starts.append(last_start)

    for index, start in enumerate(starts):
        end = start + window_samples
        window = audio[start:end]
        padded = len(window) < window_samples
        if padded:
            window = np.pad(window, (0, window_samples - len(window)))

        loudness = rms_db(window)
        yield window.astype(np.float32), WindowRecord(
            source_path="",
            window_index=index,
            start_seconds=round(start / sample_rate, 3),
            end_seconds=round(min(end, len(audio)) / sample_rate, 3),
            padded=padded,
            rms_db=round(loudness, 3),
            is_low_energy=loudness < low_energy_db_threshold,
        )


def preprocess_continuous_audio(
    audio_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 44_100,
    window_seconds: float = 3.0,
    hop_seconds: float = 0.5,
    low_energy_db_threshold: float = -45.0,
    n_mels: int = 128,
    image_width: int = 250,
) -> None:
    """Preprocess a long recording into overlapping windows for future inference."""

    audio, sr = load_audio(audio_path, sample_rate=sample_rate)
    features: list[np.ndarray] = []
    window_metadata: list[dict] = []

    for window, metadata in tqdm(
        iter_audio_windows(audio, sr, window_seconds, hop_seconds, low_energy_db_threshold),
        desc="Preprocessing windows",
    ):
        metadata = WindowRecord(
            source_path=str(audio_path),
            **{key: value for key, value in asdict(metadata).items() if key != "source_path"},
        )
        features.append(extract_spectrogram(window, sr, n_mels=n_mels, image_width=image_width))
        window_metadata.append(asdict(metadata))

    payload = {
        "source_path": str(audio_path),
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "low_energy_db_threshold": low_energy_db_threshold,
        "feature_shape": [n_mels, image_width, 3],
        "X": np.stack(features),
        "window_metadata": window_metadata,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess farm audio data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-dataset", help="Preprocess the labeled seed dataset into fold pickle files.")
    seed.add_argument("--dataset-dir", default="dataset")
    seed.add_argument("--output-dir", default="preprocessed")
    seed.add_argument("--sample-rate", type=int, default=44_100)
    seed.add_argument("--n-mels", type=int, default=128)
    seed.add_argument("--image-width", type=int, default=250)

    continuous = subparsers.add_parser("continuous-audio", help="Preprocess a long recording into overlapping windows.")
    continuous.add_argument("audio_path")
    continuous.add_argument("output_path")
    continuous.add_argument("--sample-rate", type=int, default=44_100)
    continuous.add_argument("--window-seconds", type=float, default=3.0)
    continuous.add_argument("--hop-seconds", type=float, default=0.5)
    continuous.add_argument("--low-energy-db-threshold", type=float, default=-45.0)
    continuous.add_argument("--n-mels", type=int, default=128)
    continuous.add_argument("--image-width", type=int, default=250)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seed-dataset":
        records = build_records(args.dataset_dir)
        save_fold_pickles(
            records,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            image_width=args.image_width,
        )
    elif args.command == "continuous-audio":
        preprocess_continuous_audio(
            args.audio_path,
            args.output_path,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            low_energy_db_threshold=args.low_energy_db_threshold,
            n_mels=args.n_mels,
            image_width=args.image_width,
        )


if __name__ == "__main__":
    main()