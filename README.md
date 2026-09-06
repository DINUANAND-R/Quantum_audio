# Quantum-Audio-Emotion

## Hybrid Quantum-Classical Speech Emotion Recognition Using Acoustic Features and Quantum Machine Learning

---

## Project Objective

Design, implement, and experimentally evaluate a **hybrid quantum-classical pipeline** for automatic speech emotion recognition (SER) using the RAVDESS dataset.

The system extracts acoustic features from raw speech audio, compresses them with PCA, and feeds them into both classical (SVM) and quantum (QSVC, VQC) classifiers.  
Results are compared **honestly** — no quantum advantage is claimed without experimental evidence.

---

## Problem Statement

### Why Speech Emotion Recognition?

Recognising emotion from speech is a fundamental problem in human-computer interaction.  
Speech carries affective information in its **acoustic structure** (pitch, energy, rhythm, timbre) as well as its **linguistic content** (words spoken).  
Accurate SER enables applications in mental health monitoring, customer service, human-robot interaction, and education.

### Why Acoustic Features?

Rather than transcribing speech to text first, acoustic features capture the **paralinguistic** properties of speech — the *how* of communication rather than the *what*.  
Emotions such as anger vs. sadness are often more distinguishable from spectral and temporal acoustic properties than from words alone.

### Why Quantum Machine Learning?

Current quantum computers (NISQ devices) offer the potential for **quantum kernel methods** that operate in exponentially large feature spaces.  
This project investigates whether quantum encoding of audio features provides any classification advantage over classical methods at small qubit counts.

> **Important caveat:** This is an experimental comparison. Quantum advantage is not guaranteed (or claimed a priori). Results are reported as-is from actual execution.

---

## Dataset: RAVDESS

**Ryerson Audio-Visual Database of Emotional Speech and Song**  
- 24 professional actors (12 male, 12 female)  
- 1440 speech audio files (60 per actor)  
- 8 emotion categories

### Emotion Labels

| Code | Emotion   |
|------|-----------|
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

### Filename Format

```
03-01-05-01-02-01-12.wav
│   │  │  │  │  │  └─ Actor ID (01–24)
│   │  │  │  │  └──── Repetition (01–02)
│   │  │  │  └─────── Statement (01–02)
│   │  │  └────────── Intensity (01=normal, 02=strong)
│   │  └───────────── Emotion (01–08)
│   └──────────────── Vocal channel (01=speech)
└──────────────────── Modality (03=audio-visual)
```

### Dataset Setup

1. Download the speech audio zip from [Zenodo RAVDESS](https://zenodo.org/record/1188976)  
   → File: `Audio_Speech_Actors_01-24.zip`

2. Extract so that the directory structure is:

```
data/
└── raw/
    └── RAVDESS/
        ├── Actor_01/
        │   └── 03-01-*.wav
        ├── Actor_02/
        ...
        └── Actor_24/
```

3. Verify: `python main.py --stage dataset`

---

## Architecture

### Full System Pipeline

```
Audio Speech
    ↓
[1] Audio Preprocessing (librosa)
    – Load at 22050 Hz, mono
    – Trim silence, pad/trim to 3 seconds
    – Amplitude normalise
    ↓
[2] Acoustic Feature Extraction
    – MFCC (40 coefficients)
    – Chroma
    – RMS Energy
    – Zero Crossing Rate
    – Spectral Centroid, Bandwidth, Rolloff, Contrast
    – Fundamental Frequency (F0 via YIN)
    – 4 statistics each: mean, std, min, max
    – Total: ~260 features per file
    ↓
[3] Speaker-Independent Split
    – Train: Actor 01–20  |  Test: Actor 21–24
    – No actor appears in both sets
    ↓
[4] StandardScaler (fit on train only)
    ↓
[5] PCA (fit on train only)
    – 4 components (default)
    – Records explained variance
    ↓
  ┌──────────────────────────────────┐
  │  CLASSICAL BRANCH                │
  │  SVM (RBF, C=10, full features)  │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  QUANTUM BRANCH                  │
  │  ZZFeatureMap (4 qubits)         │
  │    ↓                             │
  │  QSVC (quantum kernel)           │
  │  VQC  (variational circuit)      │
  └──────────────────────────────────┘
    ↓
[6] Model Comparison
    Classical SVM | QSVC | VQC
```

### Quantum Concepts

**What is a qubit?**  
A qubit is the quantum analogue of a classical bit. Unlike a bit (0 or 1), a qubit can exist in a *superposition* of 0 and 1 simultaneously: |ψ⟩ = α|0⟩ + β|1⟩.

**Why PCA before quantum?**  
Current quantum simulators require one qubit per input feature dimension. Processing 260 features would need 260 qubits — intractable classically. PCA reduces to 4 dimensions (4 qubits), making simulation feasible.

**Quantum Feature Map (ZZFeatureMap)**  
Classical PCA features x = [x₁, x₂, x₃, x₄] are encoded as rotation angles in a quantum circuit:
- Hadamard gates: put qubits in superposition |+⟩
- Rz(2xᵢ) rotations: encode individual features
- ZZ interactions: Rz(2(π−xᵢ)(π−xⱼ)): encode feature correlations

**Quantum Kernel (QSVC)**  
K(xᵢ, xⱼ) = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²  
The inner product of quantum states serves as a kernel function. The kernel matrix is fed to a classical SVM.

**VQC — Variational Quantum Circuit**  
Adds trainable parameters (θ) via EfficientSU2 ansatz on top of the feature map. Parameters are optimised by SPSA to minimise classification loss.

---

## Installation

### Environment Setup

```bash
# Create virtual environment (Linux/HPC)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Environment

```bash
python -c "import qiskit; print(qiskit.__version__)"
python -c "import qiskit_machine_learning; print(qiskit_machine_learning.__version__)"
python -c "import librosa; print(librosa.__version__)"
```

---

## Usage

### Run Stages Individually

```bash
# Stage 1: Verify dataset exists and parse metadata
python main.py --stage dataset

# Stage 2: Extract features + create speaker-independent split
python main.py --stage features

# Stage 3: Train and evaluate Classical SVM baseline
python main.py --stage classical

# Stage 4: Fit PCA
python main.py --stage pca

# Stage 5: Train and evaluate QSVC
python main.py --stage qsvc

# Stage 6: Train and evaluate VQC
python main.py --stage vqc

# Stage 7: Compare all models
python main.py --stage evaluate

# Optional: Whisper speech-to-text branch
python main.py --stage whisper

# Run everything
python main.py --stage all
```

### Fast Mode (for quick testing)

```bash
# Uses 200 training samples and 30 VQC iterations
python main.py --stage all --mode fast
```

### Advanced Options

```bash
python main.py --stage vqc --n-components 6 --n-qubits 6 --max-iter 200
python main.py --stage features --force-recompute
python main.py --stage all --verbose
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Results

*The following values are populated after actual execution:*

| Model         | Accuracy | Macro F1 | Weighted F1 |
|---------------|----------|----------|-------------|
| Classical SVM | TBD      | TBD      | TBD         |
| QSVC          | TBD      | TBD      | TBD         |
| VQC           | TBD      | TBD      | TBD         |

Results files:
- `results/metrics/classical_svm_results.json`
- `results/metrics/qsvc_results.json`
- `results/metrics/vqc_results.json`
- `results/metrics/model_comparison.csv`

Figures:
- `results/figures/emotion_distribution.png`
- `results/figures/pca_explained_variance.png`
- `results/figures/classical_svm_confusion_matrix.png`
- `results/figures/qsvc_confusion_matrix.png`
- `results/figures/vqc_confusion_matrix.png`
- `results/figures/model_comparison.png`

---

## Limitations

1. **Qubit count**: Only 4 qubits used — severely limits quantum expressivity
2. **PCA compression**: Reducing 260 features to 4 loses significant emotion information
3. **Simulator overhead**: Statevector simulation is exponentially expensive for large N
4. **VQC convergence**: SPSA may not converge to global optimum (barren plateau problem)
5. **Dataset size**: QSVC kernel matrix is O(n²); expensive for large n
6. **No real hardware**: Results are from classical simulation, not actual quantum hardware

---

## Future Work

- Test on actual IBM Quantum hardware (qiskit-ibm-runtime)
- Explore larger qubit counts with sparse encoding
- Investigate quantum convolutional neural networks (QCNN)
- Combine acoustic + linguistic features in the quantum circuit
- Hyperparameter search for quantum models
- Explore shot-noise effects on classification accuracy
- Apply to other emotion datasets (CREMA-D, MSP-IMPROV)

---

## Project Structure

```
Quantum-Audio-Emotion/
│
├── data/raw/RAVDESS/         ← Place dataset here
├── data/processed/           ← Split CSVs, metadata
├── features/                 ← Extracted feature CSVs
├── models/                   ← Saved models
├── results/
│   ├── figures/              ← Plots and visualizations
│   ├── metrics/              ← JSON metrics files
│   └── predictions/          ← CSV prediction outputs
├── src/
│   ├── config.py             ← Central configuration
│   ├── dataset.py            ← RAVDESS parsing
│   ├── preprocessing.py      ← Audio loading/normalisation
│   ├── feature_extraction.py ← Acoustic feature extraction
│   ├── dataset_split.py      ← Speaker-independent split
│   ├── dimensionality_reduction.py  ← PCA
│   ├── classical_model.py    ← SVM baseline
│   ├── quantum_features.py   ← ZZFeatureMap, EfficientSU2
│   ├── qsvc_model.py         ← QSVC
│   ├── vqc_model.py          ← VQC
│   ├── evaluation.py         ← Comparison table
│   ├── visualization.py      ← Plots
│   ├── whisper_transcription.py  ← Optional STT
│   └── multimodal_fusion.py  ← Late fusion
├── tests/
│   └── test_dataset.py
├── main.py                   ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Citation

**RAVDESS Dataset:**
Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE* 13(5): e0196391.

**Quantum kernel methods:**
Havlíček V, et al. (2019) Supervised learning with quantum-enhanced feature spaces. *Nature* 567, 209–212.
