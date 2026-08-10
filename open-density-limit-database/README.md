# Density Limit Prediction

The aim of this project is to predict density limit using recent plasma measurements. I have drawn data from data from the [**MIT Open Density Limit Database**](https://github.com/MIT-PSFC/open_density_limit_database) which was released to the public for pedagogical purposes. Andrew D. Maris included `demo.ipynb` in the repo which supports the investigation of time-point stability classification throught the use of a linear support vector machine (SVM). I have drawn on this notebook and made a few changes to the code to produce `lvsm.py` and use it as a baseline to compare my three ML models to. I have worked on a 1D convolutional neural network (CNN), a temporal convolutional network (TCN), and a gated recurrent unit network (GRU). 

## (Brief) Background 
- Excessively high density can lead to radiative instabilities and disruptions. 
- The Greenwald limit is widely used as an empirical estimate of the operating boundary. 
- Data-driven boundaries could identify density-limit behaviour accurately and support real-time avoidance.

## Dataset

The file `DL_DataFrame.csv` includes:
* 264,385 timepoints from 2,333 discharges
* A ~0.01s sampling period 
* 6 input signals:
  * `density`
  * `plasma_current`
  * `elongation`
  * `minor_radius`
  * `toroidal_B_field`
  * `triangularity`
* Typical discharge lasts 1.18s
* Binary target: `density_limit_phase` x &isin; {0,1}
* 3, 598 positive (`density_limit_phase ==1`) timepoints (~1.36% of the data)
* 73 discharges containing an observed transition to positive phase
* `discharge_ID` identifies a complete discharge, while `time` gives the sample time within it.

## Attribution
Citation: Maris, A. D., Rea, C., Trevisan, G. L., & the Alcator C-Mod Team. The Open Density Limit Database. MIT Plasma Science and Fusion Center (2025).


## Prediction Task
- Previous 50 samples of all six signals are fed as one model input (equivalent to ~0.5s of history).
- A positive target is defined when `density_limit_phase==1` anywhere in the following 20 samples (~0.2s into the future).
- All windows whose final input sample is already in the density-limit-phase are excluded.
- It was designed to be an early-warning task rather than a detection of an event that had already begun. 
- The window is advanced one sample at a time (to preserve temporal resolution, within the tradeoff that neighbouring examples overlap substantially).

## Experimental design choices

- **Split by discharge**: This prevents identical neighbouring timepoints from leaking between training and evaluation data. 
- **Stratified Splitting**: This allows rare positive discharges to be included in every subset. 
- **70/10/20 train-validation-test split**
- **Class weighting**: Positive windows receive a larger training weight because density-limit events are rare. The weight is calculated from the negative-to-positive ratio in the training set.
- **Recall-first threshold selection**: Among thresholds achieving at least 95% validation recall, the code chooses the one with the highest precision. This is selected on validationd data.
-**PR AUC for early-stopping**

## Repository layout

density-limit-prediction/
├── README.md
└── open-density-limit-database/
    ├── DL_DataFrame.csv
    ├── src/
    │   ├── demo_lvsm.py       # Linear SVM baseline
    │   ├── evaluate.py        # Threshold selection and evaluation
    │   ├── models.py          # CNN, TCN, and GRU definitions
    │   ├── preprocessing.py   # Splitting, scaling, and windowing
    │   ├── train.py           # Training pipeline
    │   └── visualise.py       # Discharge visualization
    └── outputs/               # Saved models, predictions, and metrics

## Running the models
Required downloads: Python, NumPy, pandas, scikit-learn, Matplotlib, TensorFlow/Keras.

From `open-density-limit-database/`, run:

```bash
python -m src.train --model cnn
python -m src.train --model tcn
python -m src.train --model gru
```

Each run saves its model, fitted scaler, training history, configuration, evaluation metrics, and test predictions under `outputs/<model>/`.