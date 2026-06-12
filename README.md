# Distorted Visual Sequence Pattern Recognition using Deep Learning

## Project Overview

This project focuses on recognizing text sequences from distorted grayscale images containing noise, blur, overlapping characters, irregular spacing, and other visual artifacts. Traditional OCR systems often struggle in such conditions because character boundaries become unclear and visual patterns vary significantly across samples.

To address this challenge, a CRNN (Convolutional Recurrent Neural Network) architecture was implemented. The model combines deep convolutional feature extraction with recurrent sequence modelling and Connectionist Temporal Classification (CTC) Loss, enabling end-to-end text recognition without requiring character-level alignment.

---

## Dataset Exploration

Before training the model, the dataset was analyzed to understand:

* Image dimensions and formats
* Label distributions
* Sequence lengths
* Common distortion patterns
* Training and test set structure

Several visualizations were generated to inspect the quality and diversity of the data. This step helped identify potential challenges and guided preprocessing decisions.

---

## Model Architecture

The solution follows a CRNN + CTC pipeline.

```text
Input Image
      ↓
Modified ResNet18 Feature Extractor
      ↓
Feature Sequence Generation
      ↓
2-Layer Bidirectional LSTM
      ↓
Linear Classification Layer
      ↓
CTC Decoding
      ↓
Predicted Text Sequence
```

### Feature Extraction

A modified ResNet18 backbone is used to extract robust visual representations from grayscale distorted images.

### Sequence Modelling

Extracted feature maps are transformed into sequential representations and processed using Bidirectional LSTM layers, allowing the model to capture contextual information from both directions.

### Sequence Prediction

Since character-level alignments are not available, the model is trained using Connectionist Temporal Classification (CTC) Loss. This enables end-to-end sequence learning without explicitly segmenting characters beforehand.

---

## Training Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Input Size              | 100 × 320         |
| Batch Size              | 32                |
| Train Split             | 90%               |
| Validation Split        | 10%               |
| Optimizer               | Adam              |
| Learning Rate           | 0.0002            |
| Loss Function           | CTC Loss          |
| Scheduler               | ReduceLROnPlateau |
| Maximum Epochs          | 20                |
| Early Stopping Patience | 5                 |

---

## Training Performance

The model converged successfully and demonstrated stable learning behavior throughout training.

<img src="images/Training%20Perfomance.png" width="100%">

### Observations

* Rapid reduction in loss during the initial epochs.
* Stable convergence after approximately 10 epochs.
* Minimal gap between training and validation losses.
* Strong generalization with no significant signs of overfitting.

---

## Training Log

<img src="images/Training%20Log.png" width="100%">

The best-performing checkpoint was automatically saved during training and later used for evaluation and test prediction generation.

---

## Validation Results

<img src="images/Training%20Results.png" width="100%">

### Final Performance

| Metric               | Value      |
| -------------------- | ---------- |
| Best Validation Loss | **0.0264** |
| Best Epoch           | **19**     |
| Validation CER       | **0.0062** |
| Percentage CER       | **0.62%**  |

A Character Error Rate (CER) of **0.62%** indicates that fewer than one character per hundred characters is predicted incorrectly on average, demonstrating the model's ability to accurately recover text despite substantial visual distortions.

---

## Key Takeaways

* Successfully implemented an end-to-end OCR pipeline for distorted text recognition.
* Leveraged a modified ResNet18 backbone for visual feature extraction.
* Used Bidirectional LSTM layers for sequence modelling.
* Trained using CTC Loss without requiring character-level alignment.
* Achieved a validation CER of **0.62%**.
* Generated predictions for the complete test dataset using the best saved checkpoint.

---

## Repository Structure

```text
.
├── images/
│   ├── Training Log.png
│   ├── Training Performance.png
│   └── Training Results.png
│
├── CODE_GUNAKALA_PIYOOSH_PRANAV.ipynb
├── submission_GUNAKALA_PIYOOSH_PRANAV_24114038.csv
└── README.md
```

---

## Requirements

```bash
pip install torch torchvision pandas numpy matplotlib pillow tqdm
```

---

## How to Run

1. Download the dataset.
2. Place the dataset in the appropriate directory.
3. Install the required dependencies.
4. Open the notebook file.
5. Run all cells sequentially.
6. The best model checkpoint will be automatically saved.
7. Generate predictions on the test dataset.

---

## Future Improvements

* Transformer-based sequence models
* Attention-enhanced CRNN architectures
* Beam Search decoding
* Advanced data augmentation strategies
* Ensemble OCR systems

---

## Author

**Gunakala Piyoosh Pranav**

This project demonstrates the effectiveness of CRNN-based sequence recognition for OCR tasks involving noisy and visually distorted text images.
