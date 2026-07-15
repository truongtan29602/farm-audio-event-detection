import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from farm_audio_event_detection.model import build_model
from farm_audio_event_detection.preprocessing.audio_preprocessing import (
    iter_audio_windows,
    extract_spectrogram,
    load_audio,
)

@dataclass
class DetectedEvent:
    event_start: float
    event_end: float
    animal: str
    confidence: float

class FarmAudioDetector:
    def __init__(self, checkpoint_path: str | Path, device: torch.device):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.class_to_index = checkpoint["class_to_index"]
        self.index_to_class = checkpoint["index_to_class"]
        self.norm_mean = np.array(checkpoint["norm_mean"])
        self.norm_std = np.array(checkpoint["norm_std"])
        
        num_classes = len(self.class_to_index)
        self.model = build_model(num_classes=num_classes, dropout=0.0).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
    def detect_events(
        self,
        audio: np.ndarray,
        sample_rate: int,
        window_seconds: float = 3.0,
        hop_seconds: float = 0.5,
        threshold: float = 0.5,
    ) -> List[DetectedEvent]:
        
        features = []
        windows = []
        
        # 1. Extract windows and spectrograms
        for window_audio, metadata in iter_audio_windows(
            audio, sample_rate, window_seconds, hop_seconds
        ):
            spec = extract_spectrogram(window_audio, sample_rate)
            features.append(spec)
            windows.append(metadata)
            
        if not features:
            return []
            
        X = np.stack(features) # (N, 128, 250, 3)
        
        # 2. Normalize using training stats
        X_norm = (X - self.norm_mean) / self.norm_std
        
        # 3. Predict
        X_tensor = torch.from_numpy(X_norm).float().permute(0, 3, 1, 2).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # 4. Post-process to merge consecutive events (Multi-label, with gap filling)
        events = []
        max_gap_seconds = 1.0
        min_duration_seconds = 0.5
        
        for class_idx in range(len(self.index_to_class)):
            label = self.index_to_class[class_idx]
            if label == "others":
                continue
                
            class_probs = probs[:, class_idx]
            
            active_windows = []
            for prob, window in zip(class_probs, windows):
                if prob >= threshold:
                    # Center-timestamp mapping:
                    # Assign the prediction to the 0.5s block at the center of the 3-second window
                    # to achieve finer temporal resolution and avoid the 3-second smearing effect.
                    center = window.start_seconds + (window.end_seconds - window.start_seconds) / 2.0
                    start_mapped = max(0.0, center - 0.25)
                    end_mapped = center + 0.25
                    active_windows.append((start_mapped, end_mapped, prob))
                    
            if not active_windows:
                continue
                
            merged_events = []
            current_start, current_end, current_probs = active_windows[0][0], active_windows[0][1], [active_windows[0][2]]
            
            for start, end, prob in active_windows[1:]:
                # If gap is within max_gap_seconds, merge them
                if start - current_end <= max_gap_seconds:
                    current_end = max(current_end, end)
                    current_probs.append(prob)
                else:
                    merged_events.append({
                        "label": label,
                        "start": current_start,
                        "end": current_end,
                        "probs": current_probs
                    })
                    current_start, current_end, current_probs = start, end, [prob]
                    
            merged_events.append({
                "label": label,
                "start": current_start,
                "end": current_end,
                "probs": current_probs
            })
            
            # Apply minimum duration threshold
            for event in merged_events:
                if event["end"] - event["start"] >= min_duration_seconds:
                    events.append(event)
                    
        # Sort events by start time
        events.sort(key=lambda x: x["start"])
        
        # Convert to DetectedEvent objects
        detected_events = []
        for e in events:
            detected_events.append(DetectedEvent(
                event_start=round(float(e["start"]), 3),
                event_end=round(float(e["end"]), 3),
                animal=e["label"],
                confidence=round(float(np.mean(e["probs"])), 3)
            ))
            
        return detected_events

    def process_file(
        self,
        audio_path: str | Path,
        window_seconds: float = 3.0,
        hop_seconds: float = 0.5,
        threshold: float = 0.5
    ) -> List[DetectedEvent]:
        audio, sr = load_audio(audio_path)
        return self.detect_events(audio, sr, window_seconds, hop_seconds, threshold)
