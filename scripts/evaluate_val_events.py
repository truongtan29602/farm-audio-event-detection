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
    Returns the continuous audio, sample rate, and ground truth events.
    """
    audio_segments = []
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    gt_events = []
    current_time = 0.0
    
    for item in metadata_list:
        path = PROJECT_ROOT / item["path"]
        audio, sr = load_audio(path, sample_rate=sample_rate)
        duration = len(audio) / sample_rate
        audio_segments.append(audio)
        
        if item["label"] != "others":
            gt_events.append({
                "label": item["label"],
                "start": current_time,
                "end": current_time + duration
            })
            
        current_time += duration
        audio_segments.append(silence)
        current_time += 0.5
        
    return np.concatenate(audio_segments), sample_rate, gt_events

def evaluate_events_with_iou(detected_events, gt_events, iou_threshold=0.3):
    """
    Evaluates detected events against ground truth events using Intersection over Union (IoU).
    A detected event is a True Positive if it overlaps with a ground truth event of the same label
    with an IoU >= iou_threshold, and the ground truth event hasn't been matched yet.
    """
    tp = 0
    fp = 0
    matched_gt = set()
    
    for det in detected_events:
        matched = False
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_events):
            if i in matched_gt:
                continue
                
            if det.animal == gt["label"]:
                intersection = max(0.0, min(det.event_end, gt["end"]) - max(det.event_start, gt["start"]))
                union = max(det.event_end, gt["end"]) - min(det.event_start, gt["start"])
                iou = intersection / union if union > 0 else 0.0
                
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched = True
            
        if not matched:
            fp += 1
            
    fn = len(gt_events) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

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
        continuous_audio, sr, gt_events = concatenate_audio_files(subset_metadata, sample_rate=44100)
        
        print("Running event detection...")
        detector = FarmAudioDetector(checkpoint_path, device=device)
        events = detector.detect_events(continuous_audio, sr, threshold=args.threshold)
        
        print("Evaluating detections...")
        eval_metrics = evaluate_events_with_iou(events, gt_events, iou_threshold=0.3)
        print(f"Metrics - F1: {eval_metrics['f1']:.3f}, Precision: {eval_metrics['precision']:.3f}, Recall: {eval_metrics['recall']:.3f}")
        
        json_out = outputs_dir / f"fold_{fold}_events_report.json"
        plot_out = outputs_dir / f"fold_{fold}_events_visualization.png"
        
        report_data = {
            "metrics": eval_metrics,
            "detected_events": [
                {
                    "animal": e.animal,
                    "event_start": e.event_start,
                    "event_end": e.event_end,
                    "confidence": e.confidence
                } for e in events
            ],
            "gt_events": gt_events
        }
        with open(json_out, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Exported JSON report to {json_out}")
        
        plot_audio_events(
            continuous_audio, 
            sr, 
            events, 
            plot_out, 
            title=f"Fold {fold} - Eval (F1: {eval_metrics['f1']:.2f})"
        )
        print(f"Exported visualization to {plot_out}")

if __name__ == "__main__":
    main()
