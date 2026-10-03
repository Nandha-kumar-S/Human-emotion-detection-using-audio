# Audio Emotion Classification from Spectral Features

Classifying short audio clips into emotion categories (happy / sad / anger) by
extracting spectral features from the waveform and training a support vector
machine on them.

A hands-on exercise in audio feature engineering — the focus is on *how raw
audio becomes something a classifier can use*, rather than on building a
production-grade emotion recognizer.

---

## The problem

A raw waveform is just amplitude over time. A classifier can't use it directly:
two clips of the same emotion can have completely different sample values while
sharing the frequency characteristics a human ear picks up on.

So the real work isn't the model — it's reducing each clip to a fixed-length
vector that captures both its **frequency** and **time** characteristics.

## Approach

**1. Explore what the signal looks like**

- Waveform display (`librosa.display.waveshow`)
- Spectrogram via STFT, converted to decibels — shows signal strength across
  frequency and time
- **Spectral centroid** — where the "centre of mass" of the sound sits
- **Zero-crossing rate** — how often the signal crosses zero within a window

**2. Extract features (MFCCs)**

Mel-Frequency Cepstral Coefficients summarise the frequency distribution within
each analysis window, so both frequency *and* time characteristics survive into
the feature vector.

```python
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
```

MFCCs come out as `(40, n_frames)` — a variable number of frames depending on
clip length. Taking the mean across the time axis collapses that to a fixed
40-dimensional vector per clip, which is what makes clips of different lengths
comparable.

**3. Train**

- `LabelEncoder` to turn class names into integers
- 80/20 train/test split
- `SVC(kernel="linear")`

## Running it

```bash
pip install librosa scikit-learn pandas numpy matplotlib tqdm
jupyter notebook "Audio Classification model.ipynb"
```

The notebook currently uses absolute Windows paths — point
`audio_dataset_path` and the `Metadata.csv` path at this directory before
running.

## Dataset

`Metadata.csv` maps each file to its class:

| File | Duration (s) | Class |
|---|---|---|
| BabyElephant.wav | 4 | happy |
| Cantina.wav | 2 | happy |
| StarWars.wav | 3 | happy |
| Fanfare60.wav | 2 | sad |
| taunt.wav | 1 | sad |
| PinkPanther30.wav | 1 | anger |
| preamble10.wav | 3 | anger |

**On the dataset:** this is 7 clips of film music and sound effects, hand-labelled
with an emotional tone — not human speech, and far too small to generalise from.
It exists to exercise the pipeline end to end, not to produce a usable model. Any
accuracy figure from a 5-sample training set would be noise.

## What I'd change

- **A real dataset.** RAVDESS or CREMA-D would make this an actual emotion
  recognition task rather than a pipeline demo.
- **Mean-pooling loses the time dimension.** Averaging MFCCs across frames throws
  away how the sound *evolves* — which is arguably where emotional signal lives.
  Keeping the sequence and using a model that handles it would be the obvious
  next step.
- **Cross-validation instead of a single split**, given how few samples there are.
- **Fix the scoping bug** in `features_extractor(file)` — it takes `file` as a
  parameter but reads the outer-scope `file_name`.

## Built with

Python · librosa · scikit-learn · pandas · NumPy · matplotlib
