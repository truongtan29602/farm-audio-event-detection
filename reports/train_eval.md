# Training & Evaluation Strategy

## 1. Training Architecture & Cross-Validation
The training pipeline (`scripts/train_kfold.py`) is designed to train the sound event detection model robustly using **5-fold cross-validation**:
- **Dataset Splitting**: The dataset is split into 5 folds. In each iteration, one fold is held out as the validation set, while the other 4 folds are used to train the model. This guarantees that every audio file is tested as unseen data exactly once.
- **Model Training**: The model takes the preprocessed log-mel spectrograms as input. During training, the model learns to classify the pre-cut windows into one of the 5 animal classes or "others" (background noise).
- **Output Artifacts**: For each fold, the script outputs the best model checkpoint (`fold_X_best.pt`), the training/validation loss and accuracy curves (`fold_X_history.png`), and a comprehensive summary report (`summary.json`).

## 2. Inference & Evaluation Pipeline
To detect events in long, unsegmented audio files (the core challenge of this project), we developed a continuous inference engine (`src/farm_audio_event_detection/inference.py`).
- **Sliding Window Approach**: The inference engine extracts overlapping 3-second windows with a 0.5-second hop size from the continuous audio signal. 
- **Continuous Event Merging**: The model predicts the probability of each animal class per window. We apply a confidence threshold (e.g., 0.5) to filter out noise. If the same animal class is detected in consecutive overlapping windows, our post-processing logic automatically merges them into a single continuous event. The output is a list of structured events containing `start_time`, `end_time`, and `animal` label.

## 3. Evaluation on the Validation Set (Concatenation Strategy)
The validation clips from the ESC-50 dataset are pre-cut 5-second files. However, the final presentation requires processing a continuous 1-minute audio recording.
To rigorously test our continuous event detection pipeline on the validation data:
- **Audio Merging**: The evaluation script (`scripts/evaluate_val_events.py`) artificially concatenates multiple 5-second validation clips (separated by a 0.5s silence) to synthesize a continuous audio stream of over 60 seconds per fold.
- **Stress Testing**: This simulated "presentation-style" stream is passed to the inference engine. This proves that the pipeline's sliding-window approach and merging logic can successfully transition between silence and different animals over time, rather than just returning a single label for a pre-cut clip.

## 4. Visualization & Reporting
For every fold evaluated on the continuous stream, two key outputs are produced by the visualization module (`src/farm_audio_event_detection/visualization/events.py`):
- **JSON Event Report (`fold_X_events_report.json`)**: A highly structured file listing every single animal vocalization detected, providing precise timestamps (`event_start`, `event_end`) and the `animal` label.
- **Event Visualization (`fold_X_events_visualization.png`)**: A clear, legible visual plot of the full continuous audio waveform. Detected events are marked segment-by-segment with transparent colored blocks over the waveform (e.g., orange for cat, green for cow). This provides immediate visual confirmation of the model's start/end time boundaries and classification accuracy against the real signal.
