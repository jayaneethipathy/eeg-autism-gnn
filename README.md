# EEG Autism Screening — CV + Graph Neural Network

Personal research project. Explores whether framing EEG channels as a graph
(nodes = electrodes, edges = functional connectivity) and applying a Graph
Attention Network improves ASD (autism spectrum disorder) screening over a
plain sequence model on the same signals.

## Approach
1. **Graph construction** — each EEG channel is a node. Edge weights come
   from pairwise **Phase Locking Value (PLV)**, a measure of phase
   synchrony between channel pairs, thresholded to keep the graph sparse.
2. **Two models trained on the same windows:**
   - `LSTMBaseline` — treats each channel as a time series, LSTM over time,
     mean-pool over channels.
   - `EEGGAT` — Graph Attention Network over the PLV graph, with per-window
     node features (band power in delta/theta/alpha/beta/gamma).
3. **Comparison** — accuracy, F1, and AUC on a held-out split, plus a look
   at which channel-pairs the GAT attention weights rank as most
   informative (interpretability angle relevant to clinical screening).

## Data
Built against the Kaggle "EEG Eye State"-style autism EEG datasets, with
ABIDE paediatric EEG identified as a pending-access extension target. This
repo ships a **synthetic data generator** (`synthetic_data.py`) that
produces PLV-graph-shaped fake data so the full pipeline runs end-to-end
without a data access request — swap in `load_kaggle_eeg()` once you have
the real files locally.

## Files
- `plv.py` — Phase Locking Value computation from raw multi-channel EEG.
- `synthetic_data.py` — synthetic EEG + label generator for offline runs.
- `models.py` — `LSTMBaseline` and `EEGGAT` (PyTorch + PyTorch Geometric).
- `train.py` — trains both models on the same split, prints comparison table.

## Running
```bash
pip install -r requirements.txt
python train.py --epochs 20
```

## Status
Research prototype, synthetic data only in this repo. Not a validated
clinical screening tool — PLV-graph GNN methods for ASD EEG are an active,
unsettled research area; treat results here as a methods demo, not evidence.

## Stack
Python · PyTorch · PyTorch Geometric · NumPy · scikit-learn
