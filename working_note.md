# Working note — data preprocessing

This note explains only the preprocessing part of the project. I am not training a model here yet, and I am not doing final event prediction yet.

The goal of this part is simple:

1. organize the audio files,
2. turn each sound into a useful feature representation,
3. keep the ESC-50 fold split clean,
4. prepare a reasonable way to scan longer audio later.

---

## 1. What the dataset looks like

The seed dataset is stored like this:

```text
dataset/
  cat/
  cow/
  dog/
  others/
  rooster/
  sheep/
```

Each folder name is used as the label.

Example:

```text
dataset/cat/2-110010-A-5.wav
```

For this file:

- label = `cat`, because it is inside the `cat` folder
- fold = `2`, because the filename starts with `2-`

The preprocessing code scans all `.wav` files and creates a record like this:

```python
{
    "path": "dataset/cat/2-110010-A-5.wav",
    "filename": "2-110010-A-5.wav",
    "label": "cat",
    "label_index": 1,
    "fold": 2,
}
```

I used this label mapping:

```text
dog     -> 0
cat     -> 1
cow     -> 2
rooster -> 3
sheep   -> 4
others  -> 5
```

The `others` class is important. It represents background sounds or non-target sounds. Later, this helps the system learn when it should say “nothing important is happening”.

---

## 2. Audio loading choice

Every file is loaded as mono audio at:

```text
44,100 Hz
```

Why 44,100 Hz?

- It keeps the same common audio rate as many `.wav` files.
- It preserves enough detail for animal sounds.
- It avoids mixing files with different sample rates, which can make features inconsistent.

Mono means we use one audio channel. For this project, the main question is “what animal sound is present?”, not “where is the animal located?”, so mono is enough for preprocessing.

---

## 3. Feature representation: 3-channel Log-Mel spectrogram

Raw audio is just a long list of numbers. A model usually works better if we convert the audio into a time-frequency representation.

I use a **Log-Mel spectrogram**.

Simple explanation:

- time goes left to right,
- frequency goes bottom to top,
- brightness means how strong the sound is.

It is like an image of the sound.

### Why Log-Mel?

The Mel scale is closer to how humans hear sound. The log scale makes loud and quiet parts easier to compare.

Animal sounds have different shapes:

- a dog bark can be short and sharp,
- a cow moo can be longer and smoother,
- a rooster call can have changing pitch.

A spectrogram shows these patterns better than raw waveform values.

### Why 3 channels?

The code creates 3 Log-Mel spectrograms using 3 different window/hop settings:

```text
short  view: n_fft=1024, hop=256
medium view: n_fft=2048, hop=512
long   view: n_fft=4096, hop=1024
```

Then the 3 views are stacked together like RGB channels in an image.

Why do this?

- The short view catches fast changes, like a bark starting suddenly.
- The medium view is a balanced view.
- The long view smooths the sound more, which can help with longer calls.

This is useful because animal sounds do not all happen at the same speed.

---

## 4. Fixed feature size

Every spectrogram is resized to:

```text
128 x 250 x 3
```

Meaning:

- 128 Mel frequency bins,
- 250 time steps,
- 3 channels.

Why resize?

A future model needs all inputs to have the same shape. Even if one audio file is slightly different in duration, the saved feature shape stays consistent.

Example final feature shape:

```python
(128, 250, 3)
```

---

## 5. Cross-validation split

ESC-50 uses 5 folds. I keep that rule.

For each fold:

- validation set = files from that fold,
- training set = files from the other 4 folds.

Example for fold 1:

```text
validation: fold 1
training:   folds 2, 3, 4, 5
```

The script saves one pickle per fold:

```text
preprocessed/fold_1.pkl
preprocessed/fold_2.pkl
preprocessed/fold_3.pkl
preprocessed/fold_4.pkl
preprocessed/fold_5.pkl
```

These files are generated outputs, so they are ignored by git. Anyone can rebuild them from the dataset.

---

## 6. How this connects to the hard part: continuous audio

The project is not only about classifying clean 3-second clips. The final system must work on a longer recording, maybe around 1 minute.

That is harder because the recording is continuous.

### Choice: sliding windows

For a long recording, I preprocess it using:

```text
window size = 3.0 seconds
hop size    = 0.5 seconds
```

This means the system looks at 3 seconds at a time, then moves forward by 0.5 seconds.

Example:

```text
window 1: 0.0s -> 3.0s
window 2: 0.5s -> 3.5s
window 3: 1.0s -> 4.0s
window 4: 1.5s -> 4.5s
```

Why 3 seconds?

- The seed training clips are clean short clips.
- Using 3-second windows keeps the long-audio input similar to training input.

Why 0.5 second hop?

- The project allows a `+/- 500 milliseconds` margin.
- A 0.5 second hop means we check the audio often enough to place event boundaries close to the required margin.

Tradeoff:

- Smaller hop = more precise but slower.
- Bigger hop = faster but less precise.

For this project, 0.5 seconds is a good balance.

---

## 7. Overlapping animal sounds

The preprocessing does not force one label for a time region in long audio. It only creates overlapping windows.

This matters because two animals can be active at the same time.

Example:

```text
2.0s -> 4.0s: dog barking
3.0s -> 5.0s: sheep bleating
```

Those events overlap between 3.0s and 4.0s.

A future model should ideally be multi-label, meaning one window can contain more than one active animal. This preprocessing keeps that option open because it prepares windows without assuming only one final animal event.

---

## 8. Silence and ambient noise

Most real recordings are not full of animal sounds. There can be silence, wind, footsteps, microphone noise, or far-away background sounds.

For long audio windows, the preprocessing stores an RMS loudness value:

```text
rms_db
```

It also marks very quiet windows:

```text
is_low_energy = True or False
```

Default threshold:

```text
-45 dB
```

This does not delete quiet windows automatically. It only marks them. I prefer marking instead of deleting because sometimes quiet animal sounds might still matter.

Later, the inference/post-processing step can use this metadata to reduce false alarms.

Example:

```python
{
    "start_seconds": 10.0,
    "end_seconds": 13.0,
    "rms_db": -52.4,
    "is_low_energy": True,
}
```

This means the window is probably silence or quiet background.

---

## 9. Frame/window predictions into events later

This branch does not implement final post-processing yet, but preprocessing is designed with it in mind.

Later, if the model predicts:

```text
cow from 6.0s -> 6.5s
cow from 7.0s -> 8.0s
```

we need to decide whether that is:

1. one cow event with a short gap, or
2. two separate cow events.

My planned strategy for later is:

- smooth predictions across nearby windows,
- fill small gaps, for example gaps shorter than 0.5 seconds,
- remove events that are too short to be reliable,
- keep separate labels independent so overlapping animals can both exist.

This matches the preprocessing choice because windows are spaced every 0.5 seconds.

---

## 10. How to run preprocessing

From the project root:

```bash
python scripts/preprocess_data.py seed-dataset
```

This creates the fold pickle files inside:

```text
preprocessed/
```

For a future long recording:

```bash
python scripts/preprocess_data.py continuous-audio path/to/recording.wav preprocessed/recording_windows.pkl
```

That creates overlapping 3-second windows with 0.5-second spacing.

---

## 11. Visuals for presentation slides

I added simple slide visuals in `reports/preprocessing_visuals.md` and `reports/figures/`. They explain the preprocessing pipeline, the 3-second / 0.5-second sliding window choice, and the idea of cleaning raw window hits into events later.

These are only for slides and explanation; they are not model code.

---

## 12. Summary of choices

| Choice | Value | Why |
|---|---:|---|
| Sample rate | 44,100 Hz | consistent and keeps audio detail |
| Feature | Log-Mel spectrogram | good sound image for models |
| Channels | 3 | short, medium, long sound details |
| Feature size | 128 x 250 x 3 | fixed shape for future model |
| Validation | ESC-50 5 folds | respects dataset standard |
| Long audio window | 3.0 seconds | matches training clip style |
| Long audio hop | 0.5 seconds | supports +/- 500 ms timing margin |
| Silence handling | RMS metadata | helps future system know when to say nothing |

The main idea is to make preprocessing useful for the real challenge: long audio, overlapping sounds, silence, and event timing.