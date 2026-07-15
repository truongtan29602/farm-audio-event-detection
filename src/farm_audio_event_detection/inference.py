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
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        # 4. Post-process to merge consecutive events
        events = []
        current_event = None
        
        for i, (prob, window) in enumerate(zip(probs, windows)):
            pred_idx = np.argmax(prob)
            max_prob = prob[pred_idx]
            label = self.index_to_class[pred_idx]
            
            # If the network predicts a non-background class with high enough confidence
            if label != "others" and max_prob >= threshold:
                if current_event is None or current_event["label"] != label:
                    if current_event is not None:
                        events.append(current_event)
                    current_event = {
                        "label": label,
                        "start": window.start_seconds,
                        "end": window.end_seconds,
                        "probs": [max_prob]
                    }
                else:
                    # Extend current event
                    current_event["end"] = window.end_seconds
                    current_event["probs"].append(max_prob)
            else:
                if current_event is not None:
                    events.append(current_event)
                    current_event = None
                    
        if current_event is not None:
            events.append(current_event)
            
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
