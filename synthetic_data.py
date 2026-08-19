"""
Synthetic multi-channel EEG generator, shaped like a Kaggle-style autism EEG
dataset, so the pipeline runs end-to-end offline. Swap `load_kaggle_eeg()`
in for real access once available (ABIDE paediatric EEG is pending access).

Simulated group difference: the "ASD" class gets slightly reduced
inter-channel phase coupling in the alpha band, loosely modelled on
findings in the connectivity literature — this is a synthetic proxy for
demo purposes only, not a claim about real ASD EEG signatures.
"""
import numpy as np

N_CHANNELS = 8
N_SAMPLES = 512   # per window, e.g. 2s @ 256Hz
FS = 256


def _band_signal(t, freq, coupling_phase_offset, noise_scale, rng):
    base = np.sin(2 * np.pi * freq * t + coupling_phase_offset)
    return base + rng.normal(0, noise_scale, size=t.shape)


def generate_window(label: int, rng: np.random.Generator) -> np.ndarray:
    """label: 0 = control, 1 = ASD (synthetic proxy). Returns (N_CHANNELS, N_SAMPLES)."""
    t = np.arange(N_SAMPLES) / FS
    alpha_freq = 10.0
    # ASD windows get weaker cross-channel phase coupling in alpha band
    coupling_strength = 1.0 if label == 0 else 0.4
    channels = []
    shared_phase = rng.uniform(0, 2 * np.pi)
    for ch in range(N_CHANNELS):
        individual_offset = rng.normal(0, (1 - coupling_strength) * np.pi)
        phase = shared_phase * coupling_strength + individual_offset
        sig = _band_signal(t, alpha_freq, phase, noise_scale=0.5, rng=rng)
        channels.append(sig)
    return np.stack(channels).astype(np.float32)


def band_power_features(window: np.ndarray) -> np.ndarray:
    """Crude band-power proxy per channel via FFT energy in 5 canonical bands."""
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    freqs = np.fft.rfftfreq(window.shape[1], d=1 / FS)
    spectrum = np.abs(np.fft.rfft(window, axis=1)) ** 2
    feats = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(spectrum[:, mask].mean(axis=1))
    return np.stack(feats, axis=1)  # (n_channels, n_bands)


def generate_dataset(n_samples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for _ in range(n_samples):
        label = rng.integers(0, 2)
        windows.append(generate_window(label, rng))
        labels.append(label)
    return np.stack(windows), np.array(labels)


if __name__ == "__main__":
    X, y = generate_dataset(10)
    print("Windows shape:", X.shape, "Labels:", y)
    feats = band_power_features(X[0])
    print("Band power features shape (channels x bands):", feats.shape)
