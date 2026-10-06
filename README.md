# Binary Image Classification with TensorFlow and PyTorch

A comparative deep-learning project for two-class image classification using transfer learning in TensorFlow and PyTorch.

The repository contains two Kaggle-oriented experiments: an InceptionV3 classifier implemented with TensorFlow/Keras and a ResNet18 classifier implemented with PyTorch. It also includes a small OpenCV utility used to extract and crop frames from video when preparing image data.

## Implementations

### TensorFlow / Keras

- InceptionV3 backbone initialized with ImageNet weights
- Frozen convolutional backbone
- Global average pooling, a 256-unit dense layer, dropout, and sigmoid output
- Image resizing to `256 x 256`
- Rescaling, random rotation, and an 80/20 training-validation split
- Early stopping and best-model checkpointing

The saved notebook output reports 7,640 training images and 1,909 validation images. The recorded validation accuracy reached 100% in that run, but the dataset is not included, so this result cannot be independently reproduced from the repository alone.

### PyTorch

- ResNet18 initialized with pretrained weights
- Final classification layer replaced for two output classes
- Image resizing to `224 x 224` and random horizontal flipping
- AdamW optimization with cosine-annealing warm restarts
- Automatic selection of Apple MPS, CUDA, or CPU

The saved notebook output reports 99.79% training accuracy after three epochs. This is a training metric, not a held-out evaluation result, and should not be compared directly with the TensorFlow validation metric.

## Repository structure

```text
.
|-- binary-image-classification-main/
|   |-- binary class tensor.ipynb   # TensorFlow/InceptionV3 experiment
|   |-- binary class pytorch.ipynb  # PyTorch/ResNet18 experiment
|   `-- 15frames.py                 # Video-frame extraction helper
|-- requirements.txt
`-- README.md
```

## Dataset layout

The notebooks expect a Kaggle dataset at `/kaggle/input/bic123` with two class directories:

```text
bic123/
|-- 0/
|   `-- images...
`-- 1/
    `-- images...
```

The dataset and the semantic meaning of labels `0` and `1` are not included. Document the class definitions and data provenance before using this project as evidence for a specific application domain.

## Running the notebooks

The simplest reproducible environment is a Kaggle notebook with the dataset attached at the path above. For a local environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Update the dataset path before running locally. The PyTorch notebook contains historical package-installation cells from its original Kaggle environment; prefer the versions in your current environment rather than running those cells blindly.

## Methodological limitations

- The dataset is not distributed with the repository, preventing independent reproduction.
- The PyTorch notebook does not create a separate validation or test split.
- Consecutive frames extracted from the same video may be highly correlated. Split data by source video or subject, rather than by individual frame, to avoid leakage.
- High accuracy alone is insufficient. A stronger evaluation should include precision, recall, F1-score, a confusion matrix, and class-level error analysis on a held-out test set.
- The two notebooks use different architectures, image sizes, preprocessing, and evaluation strategies, so their stored metrics are not a controlled framework comparison.

## Technologies

Python, TensorFlow/Keras, PyTorch, torchvision, OpenCV, NumPy, pandas, Matplotlib, Seaborn, Plotly, TensorBoard, and Jupyter Notebook.
