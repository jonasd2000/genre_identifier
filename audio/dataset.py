from typing import Dict
import os
from pathlib import Path
from tqdm import trange, tqdm

import torchaudio
from torch.utils.data import Dataset

import polars as pl
import numpy as np

from utility.audio import waveform_to_mono, prepare_waveform, clip_samples


class TrackGenreDataset(Dataset):
    path: str
    clip_length: float

    track_dataframe: pl.DataFrame
    genre_map: Dict[str, int]
    spectra: list[np.ndarray]


    def __init__(self, path: Path, track_dataframe: pl.DataFrame, genre_map: Dict[str, int], clip_length: float, read_to_memory: bool=False) -> None:
        """
        path: path to data
        clip_length: number of samples used per song for training
        """

        super().__init__()

        self.path = path
        self.clip_length = clip_length
        self.track_dataframe = track_dataframe
        self.genre_map = genre_map

        if read_to_memory:
            self.spectra = self.load_spectra()

    def __len__(self) -> int:
        return len(self.track_dataframe)

    def load_spectrum(self, index: int) -> np.ndarray:
        track_filepath = self.track_dataframe.row(index, named=True)["filepath"]
        waveform, samplerate = torchaudio.load(track_filepath)
        return prepare_waveform(waveform_to_mono(waveform), self.clip_length)

    def load_spectra(self):
        return [self.load_spectrum(index) for index in trange(len(self.track_dataframe))]

    def get_spectrum(self, index: int) -> np.ndarray:
        if self.spectra: # if loaded to memory
            return self.spectra[index]
        else: # if not read from disk
            return self.load_spectrum(index)

    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        spectrum = self.get_spectrum(index)

        return spectrum, self.genre_map[self.track_dataframe.row(index, named=True)["genre"]]
