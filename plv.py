"""
Phase Locking Value (PLV) — a measure of phase synchrony between two EEG
channels, used here as edge weights for the electrode connectivity graph.

PLV_ij = |mean_t( exp( i * (phase_i(t) - phase_j(t)) ) )|

Values range 0 (no phase relationship) to 1 (perfectly phase-locked).
"""
import numpy as np
from scipy.signal import hilbert


def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase via the Hilbert transform. signal: (samples,)"""
    analytic = hilbert(signal)
    return np.angle(analytic)


def plv(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    phase_a = instantaneous_phase(signal_a)
    phase_b = instantaneous_phase(signal_b)
    diff = phase_a - phase_b
    return float(np.abs(np.mean(np.exp(1j * diff))))


def plv_connectivity_matrix(window: np.ndarray) -> np.ndarray:
    """
    window: (n_channels, n_samples)
    returns: (n_channels, n_channels) symmetric PLV matrix, diagonal = 1
    """
    n_channels = window.shape[0]
    mat = np.eye(n_channels, dtype=np.float32)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            v = plv(window[i], window[j])
            mat[i, j] = mat[j, i] = v
    return mat


def sparsify(matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Zero out edges below threshold to keep the graph sparse for the GNN."""
    out = matrix.copy()
    out[out < threshold] = 0.0
    np.fill_diagonal(out, 1.0)
    return out
