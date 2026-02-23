# JL1-CD-Trees: Woodland Change Detection Test Set

**Purpose:** Benchmark dataset for evaluating generalisation of woodland change detection models pre-trained on other datasets.

**Size:** 408 bi-temporal woodland change instances with binary change masks

**Structure:**
- `A/`: Time 1 images (before)
- `B/`: Time 2 images (after)
- `label/`: Binary change masks (woodland change areas)

**Format:**
- Images: PNG, 512×512 pixels
- Change masks: Binary (0=no change, 255=woodland change)

**Usage:** 
This is a **test-only dataset** for zero-shot evaluation of pre-trained models.
It is NOT intended for training or hyperparameter tuning.

**Evaluation Protocol:**
1. Train your model on a different dataset that incorporates vegetation change objects
2. Test on all 408 samples in this dataset
3. Report: Precision, Recall, F1-Score, IoU

**Citation:**
If you use this dataset, please cite this repository and accompanying paper.

**Note:** 
If you want to use this for training, you must create your own train/val/test splits.
The 408 samples reported in our paper use the entire dataset for zero-shot testing only.
