import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from farm_audio_event_detection.preprocessing.audio_preprocessing import load_audio
from farm_audio_event_detection.inference import FarmAudioDetector
from farm_audio_event_detection.visualization.events import plot_audio_events, export_events_to_json

def _import_librosa():
    import librosa
    return librosa
    
def concatenate_audio_files(metadata_list, sample_rate=44100):
    """
    Concatenate a list of audio files into a single continuous numpy array.
    Adds a small 0.5s silence between files to separate events.
    """
    audio_segments = []
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    
    for item in metadata_list:
        path = PROJECT_ROOT / item["path"]
        audio, sr = load_audio(path, sample_rate=sample_rate)
        audio_segments.append(audio)
        audio_segments.append(silence)
        
    return np.concatenate(audio_segments), sample_rate

def main():
    parser = argparse.ArgumentParser(description="Evaluate k-fold validation sets as continuous audio streams.")
    parser.add_argument("--preprocessed-dir", default="preprocessed")
    parser.add_argument("--outputs-dir", default="outputs/kfold")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device_str = args.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    preprocessed_dir = Path(args.preprocessed_dir)
    outputs_dir = Path(args.outputs_dir)
    
    for fold in range(1, 6):
        print(f"\nProcessing Fold {fold}...")
        
        checkpoint_path = outputs_dir / f"fold_{fold}_best.pt"
        pickle_path = preprocessed_dir / f"fold_{fold}.pkl"
        
        if not checkpoint_path.exists() or not pickle_path.exists():
            print(f"Missing files for fold {fold}. Skipping.")
            continue
            
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            
        val_metadata = data["validation_metadata"]
        
        # Sort metadata by label to group similar animals in the continuous file
        val_metadata_sorted = sorted(val_metadata, key=lambda x: x["label"])
        
        # We take a subset (e.g., 2 files per class) to avoid the audio being too long to plot legibly
        subset_metadata = []
        counts = {}
        for item in val_metadata_sorted:
            label = item["label"]
            counts[label] = counts.get(label, 0) + 1
            if counts[label] <= 2:
                subset_metadata.append(item)
                
        print(f"Creating continuous audio from {len(subset_metadata)} validation clips...")
        continuous_audio, sr = concatenate_audio_files(subset_metadata, sample_rate=44100)
        
        print("Running event detection...")
        detector = FarmAudioDetector(checkpoint_path, device=device)
        events = detector.detect_events(continuous_audio, sr, threshold=args.threshold)
        
        json_out = outputs_dir / f"fold_{fold}_events_report.json"
        plot_out = outputs_dir / f"fold_{fold}_events_visualization.png"
        
        export_events_to_json(events, json_out)
        print(f"Exported JSON report to {json_out}")
        
        plot_audio_events(
            continuous_audio, 
            sr, 
            events, 
            plot_out, 
            title=f"Fold {fold} - Continuous Validation Set Evaluation"
        )
        print(f"Exported visualization to {plot_out}")

if __name__ == "__main__":
    main()
