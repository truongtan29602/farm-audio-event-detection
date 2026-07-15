import argparse
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from farm_audio_event_detection.inference import FarmAudioDetector
from farm_audio_event_detection.visualization.events import plot_audio_events, export_events_to_json

def main():
    parser = argparse.ArgumentParser(description="Run Farm Audio Event Detection on a single file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (.wav)")
    parser.add_argument("--model-path", type=str, default="outputs/final_model/final_model.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/prediction", help="Directory to save JSON and visualization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
        
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Did you run train.py?")
        sys.exit(1)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path} on {args.device}...")
    import torch
    device = torch.device(args.device)
    detector = FarmAudioDetector(model_path, device=device)
    
    print(f"Processing audio: {audio_path}...")
    from farm_audio_event_detection.preprocessing.audio_preprocessing import load_audio
    audio, sr = load_audio(audio_path)
    
    print("Running event detection and post-processing...")
    events = detector.detect_events(audio, sr, threshold=args.threshold)
    
    if not events:
        print("No events detected!")
    else:
        print(f"Detected {len(events)} events.")
        
    # Export JSON
    json_path = output_dir / f"{audio_path.stem}_events.json"
    export_events_to_json(events, json_path)
    print(f"Saved JSON report to {json_path}")
    
    # Export Visualization
    viz_path = output_dir / f"{audio_path.stem}_visualization.png"
    plot_audio_events(audio, sr, events, viz_path, title="Detected Events (Prediction)")
    print(f"Saved visualization to {viz_path}")
    
    print("\nDone! Your pipeline is fully ready for presentation day.")

if __name__ == "__main__":
    main()
