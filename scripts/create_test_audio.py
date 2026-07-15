import sys
import csv
from pathlib import Path
import json
import random
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from farm_audio_event_detection.preprocessing.audio_preprocessing import load_audio

def main():
    metadata_path = PROJECT_ROOT / "preprocessed" / "records.csv"
    if not metadata_path.exists():
        print(f"Error: metadata not found at {metadata_path}")
        return
        
    records = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(row)
            
    # Pick 12 random clips (~60 seconds + silence)
    sample_clips = random.sample(records, 12)
    
    audio_segments = []
    sample_rate = 44100
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    
    print("Building 1-minute audio file...")
    for row in sample_clips:
        file_path = PROJECT_ROOT / row['path']
        print(f"  Adding: {row['label']} ({row['path']})")
        audio, sr = load_audio(file_path, sample_rate=sample_rate)
        audio_segments.append(audio)
        audio_segments.append(silence)
        
    final_audio = np.concatenate(audio_segments)
    
    out_path = PROJECT_ROOT / "test_1min.wav"
    sf.write(out_path, final_audio, sample_rate)
    
    print(f"\nSaved 1-minute test audio to: {out_path}")

if __name__ == "__main__":
    main()
