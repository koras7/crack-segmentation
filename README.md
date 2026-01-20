# Crack Segmentation Project


This project implements a crack-based assessment of reinforced concrete structures using a combination of integral equation solvers and a transformer-based segmentation model.

## Project Overview

- **Goal:** Segment cracks in concrete structures from images.
- **Main Notebook:** `CrackS_Augmentation.ipynb`
- **Supporting Code:** `IE_source/` (contains classes for Galerkin transformer, decoder, solvers, and loss functions)
- **Results:** Saved plots and outputs (optional folder can be included for demo purposes)

## Model Architecture

- **Encoder:** CNN-based feature extractor
- **Transformer:** Galerkin-style attention blocks
- **Decoder:** Neural network for segmentation output
- **Input size:** 256x256
- **Loss:** BCE + Dice loss, alpha=0.5

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/koras7/crack-segmentation.git
   cd crack-segmentation
Create environment:

bash
Copy code
conda env create -f environment.yml
conda activate <env_name>
Usage
Open CrackS_Augmentation.ipynb in Jupyter.

Load your dataset locally. Do not include datasets in the repo.

Run the notebook to train or test the segmentation model.

Dataset
Dataset used: Concrete crack images and masks (local, not included in repo).

Suggested structure:

markdown
Copy code
dataset/
    images/
    masks/
Notes
Only essential code and main notebook are included.

Large datasets, trained models, and intermediate outputs are excluded to keep the repo lightweight.

You can add a single example plot or results folder for demonstration.
