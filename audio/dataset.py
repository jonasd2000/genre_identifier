from typing import Any
import json
import os

import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from audio.helpers import clip_samples


def prepare_clip(waveform, clip_length):
    cs = clip_samples(waveform, clip_length)
    clip = np.mean(cs.numpy(), axis=0).reshape(1, -1)
    return clip


class TrackGenreDataset(Dataset):
    path: str
    clip_length: float

    genre_map: dict
    track_info: list[dict]
    waveforms: np.ndarray


    def __init__(self, path, track_info: list[dict], clip_length: float, genre_map: dict, load_to_memory: bool=False) -> None:
        """
        path: path to data
        clip_length: number of samples used per song for training
        """

        super().__init__()
        
        self.path = path
        self.clip_length = clip_length
        self.genre_map = genre_map
        self.track_info = track_info

        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self.waveforms = self.load_waveforms()


    def __len__(self) -> int:
        return len(self.track_info)
    
    def load_waveforms(self):
        print("Loading waveforms...")
        waveforms = []
        for i in range(len(self.track_info)):
            waveform, samplerate = torchaudio.load(os.path.join(self.path, self.track_info[i]["filename"] + ".mp3"))
            waveforms.append(waveform)
        return waveforms

    def get_waveform(self, index: int) -> np.ndarray:
        if self.load_to_memory:
            waveform = self.waveforms[index]
        else:
            track_file_name = self.track_info[index]["filename"]
            waveform, samplerate = torchaudio.load(os.path.join(self.path, track_file_name + ".mp3"))
        return waveform

    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        waveform = self.get_waveform(index)

        clip = prepare_clip(waveform, self.clip_length)

        return clip, self.genre_map[self.track_info[index]["genre"]]


def main():
    with open("music/genre.json") as genre_map_file:
        genre_map = json.load(genre_map_file)

    dataset = TrackGenreDataset(
        path="music/training", 
        clip_length=250_000, # 5 seconds at samplerate of 50_000
        genre_map=genre_map
    )

    print(dataset)

    dataloader = DataLoader(dataset, batch_size=16)

    for X, y in dataloader:
        print(X.shape, y)


if __name__ == "__main__":
    main()
