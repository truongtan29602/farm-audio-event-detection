# Farm Audio Event Detection

This project detects animal sound events in longer farm audio recordings.

Target animal classes:

- dog
- cat
- sheep
- cow
- rooster
- others / background

The final pipeline should read a full `.wav` file and produce:

1. a JSON event report with start time, end time, and animal label
2. a visualization of the audio with detected events marked clearly

See [`project_description.md`](project_description.md) for the full project brief.

## Project structure

```text
configs/                         Configuration files
scripts/                         Command-line scripts for pipeline steps
src/farm_audio_event_detection/   Python package source code
  preprocessing/                 Dataset scanning and preprocessing
  features/                      Audio feature extraction
  modeling/                      Model training and evaluation code
  inference/                     Prediction code for full recordings
  postprocessing/                Convert frame/window predictions into events
  visualization/                 Plot waveform/spectrogram and events
  utils/                         Shared helper code
dataset/                         Seed ESC-50 subset, organized by class
preprocessed/                    Generated preprocessing outputs, ignored by git
outputs/                         Generated predictions/figures, ignored by git
notebooks/                       Experiments and exploration
reports/                         Notes, figures, and presentation material
tests/                           Automated tests
```

## Current focus

The first implementation branch focuses only on data preprocessing.

## Data preprocessing

Build fold-based preprocessing files from the seed dataset:

```bash
python scripts/preprocess_data.py seed-dataset
```

Preprocess a longer recording into overlapping 3-second windows:

```bash
python scripts/preprocess_data.py continuous-audio path/to/recording.wav preprocessed/recording_windows.pkl
```

See [`working_note.md`](working_note.md) for the reasoning behind the preprocessing choices.

## Slide visuals

Preprocessing explanation visuals are available in [`reports/preprocessing_visuals.md`](reports/preprocessing_visuals.md).
