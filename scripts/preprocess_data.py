"""Command-line wrapper for preprocessing.

Examples:

    python scripts/preprocess_data.py seed-dataset

    python scripts/preprocess_data.py continuous-audio path/to/long.wav preprocessed/long_windows.pkl
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from farm_audio_event_detection.preprocessing.audio_preprocessing import main


if __name__ == "__main__":
    main()