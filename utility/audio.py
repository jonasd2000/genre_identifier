import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from cupy.fft import rfft

def prepare_waveform(waveform: torch.Tensor, samples: int):
    waveform = waveform_to_mono(waveform)
    clip = clip_samples(waveform, samples)
    rfourier = torch.as_tensor((cp.abs(rfft(cp.asarray(clip), norm="ortho")).astype(np.float32)), device=clip.device)[:, ::3]
    return rfourier



def waveform_to_mono(waveform):
    return torch.mean(waveform, dim=0).reshape(1, -1)

def clip_seconds(waveform: torch.Tensor, seconds: float, samplerate: int, random_start: bool=True, clone: bool=False):
    num_samples = seconds * samplerate
    return clip_samples(waveform, num_samples, random_start=random_start, clone=clone)


def clip_samples(waveform: torch.Tensor, num_samples: int, random_start: bool=True, clone: bool=False):
    start = 0 if not random_start else random.randint(0, max(0, waveform.shape[1]-num_samples-1))

    clip = waveform[:, start:start+num_samples]
    if clone:
        clip = clip.clone()
    return clip


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()