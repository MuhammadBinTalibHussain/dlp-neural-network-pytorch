# DLP Assignment 1: Neural Networks from Scratch and in PyTorch

Deep Learning for Perceptron (DLP), Assignment 1, BSCS-7D.

This project builds a two-layer neural network from scratch in NumPy (forward pass, backpropagation, gradient descent), verifies it against PyTorch autograd, and then runs a series of experiments on the Fashion-MNIST dataset. It also includes a tabular regression task on Pakistan cricket batting data.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Assignment Parts](#assignment-parts)
- [Results Summary](#results-summary)
- [Contributors](#contributors)

## Project Overview

The assignment covers the following topics:

1. Backpropagation implemented from scratch with NumPy (784 -> 64 -> 10 network with ReLU and softmax)
2. Gradient verification against a PyTorch model
3. Baseline MLP and activation function study (Sigmoid, Tanh, ReLU)
4. Loss function comparison (Cross-Entropy vs MSE)
5. Tabular regression on Pakistan cricket data
6. Optimiser comparison (SGD, SGD with momentum, RMSProp, Adam)
7. Forcing overfitting with a large MLP on a small training subset
8. Regularisation study (L2, dropout, batch normalisation, early stopping, data augmentation)
9. Hyperparameter tuning with k-fold cross-validation (learning rate, hidden width, dropout rate)

## Repository Structure

```
.
├── DLP_ass#1.ipynb          # Main notebook with all code and experiments
├── DLP Assignment Theory.docx        # Written answers and analysis for each part
├── pakistan_cricket_regression.csv   # Dataset for the tabular regression task
└── README.md
```

Note: the Fashion-MNIST CSV files are not included in this repository because of their size. See the Dataset section below.

## Dataset

### Fashion-MNIST (classification)

Download the CSV version of Fashion-MNIST from Kaggle:
https://www.kaggle.com/datasets/zalando-research/fashionmnist

Place these two files in the same folder as the notebook:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

Each row contains a `label` column (0 to 9) followed by 784 pixel values. Pixel values are scaled to the range 0 to 1 in the notebook, and the training data is split 80/20 into training and validation sets.

### Pakistan Cricket Regression (regression)

`pakistan_cricket_regression.csv` is included in the repository. It has 200 rows with these columns:

| Column | Description |
| --- | --- |
| Player | Player name |
| Balls_Faced | Number of balls faced |
| Strike_Rate | Batting strike rate |
| Batting_Average | Batting average |
| Runs_Scored | Target variable: runs scored |

## Requirements

- Python 3.9 or newer
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- torchvision
- jupyter (or JupyterLab / VS Code with the Jupyter extension)

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn torch torchvision jupyter
   ```

4. Download the Fashion-MNIST CSV files (see the Dataset section) and place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` in the project root.

5. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```

   Then open `DLP_ass#1.ipynb`.

6. Run all cells in order (Kernel -> Restart and Run All). The notebook sets random seeds (`42`) so results are reproducible. Training can take several minutes, especially for the optimiser comparison and k-fold cross-validation sections. A GPU is not required.

### Running on Google Colab

1. Upload the notebook and all CSV files to your Colab session (or mount Google Drive).
2. Make sure the file paths in the notebook point to where the CSV files are stored.
3. Run all cells.

## Assignment Parts

| Part | Topic | Description |
| --- | --- | --- |
| 1 | Backpropagation from scratch | NumPy implementation with gradient comparison against PyTorch |
| 2 | Baseline model and activation study | 784-128-64-10 MLP; Sigmoid vs Tanh vs ReLU, vanishing gradients, dead neurons |
| 3 | Loss functions | Cross-Entropy vs MSE convergence; tabular regression on cricket data |
| 4 | Optimiser comparison | SGD, SGD with momentum, RMSProp and Adam compared on accuracy and training time |
| 5 | Forcing overfitting | Large 5-layer MLP trained on 2000 samples to demonstrate high variance |
| 6 | Regularisation study | L2, dropout, batch normalisation, early stopping and augmentation compared by generalisation gap |
| 7 | Hyperparameter tuning | Grid search with k-fold cross-validation, plus precision, recall, F1 and confusion matrix |

The written analysis for each part is in `DLP Assignment Theory.docx`.

## Results Summary

| Experiment | Outcome |
| --- | --- |
| Baseline (ReLU MLP) validation accuracy | 83.59% |
| Best optimiser | RMSProp, 85.01% accuracy after 30 epochs |
| Overfitting experiment | 100% training accuracy vs 82.82% validation accuracy (gap 17.18%) |
| Best regulariser | L2 with lambda = 0.01, gap reduced from 17.18% to 6.78% |
| Tuned model after k-fold search | 84.65% accuracy (+1.06 points over the baseline) |

Exact values may vary slightly depending on hardware and library versions.

## Contributors

| Name | Roll Number | GitHub |
| --- | --- | --- |
| Muhammad Mursaleen Mustafvi | 23F-0659 | [MMursaleenMustafvi](https://github.com/MMursaleenMustafvi) |

## License

This project was created for academic purposes as part of a university course assignment.
