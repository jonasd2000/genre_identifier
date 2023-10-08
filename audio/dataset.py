import json
import os
from tqdm import trange

import torchaudio
from torch.utils.data import Dataset, DataLoader

import numpy as np

from utility.audio import waveform_to_mono, prepare_waveform, clip_samples



class TrackGenreDataset(Dataset):
    path: str
    clip_length: float
    assigned_memory: int

    genre_map: dict
    track_info: list[dict]
    waveforms: list[np.ndarray]


    def __init__(self, path, track_info: list[dict], clip_length: float, genre_map: dict, assigned_memory: int=0) -> None:
        """
        path: path to data
        clip_length: number of samples used per song for training
        """

        super().__init__()
        
        self.path = path
        self.clip_length = clip_length
        self.genre_map = genre_map
        self.track_info = track_info

        self.assigned_memory = assigned_memory * 8 # in bit
        if self.assigned_memory > 0:
            self.waveforms = self.load_waveforms(self.assigned_memory)


    def __len__(self) -> int:
        return len(self.track_info)
    
    def load_waveforms(self, assigned_memory: int):
        # samples per element = memory in bit / (datatype size in bit * number of elements)
        samples_to_load = int(assigned_memory / (32 * len(self))) 

        waveforms = []
        for i in trange(len(self.track_info), desc="Loading waveforms"):
            waveform, samplerate = torchaudio.load(os.path.join(self.path, self.track_info[i]["filename"] + ".mp3"))
            waveforms.append(clip_samples(waveform_to_mono(waveform), samples_to_load, clone=True))
        return waveforms

    def get_waveform(self, index: int) -> np.ndarray:
        if self.assigned_memory: # if loaded to memory
            waveform = self.waveforms[index]
        else: # if not read from disk
            track_file_name = self.track_info[index]["filename"]
            waveform, samplerate = torchaudio.load(os.path.join(self.path, track_file_name + ".mp3"))
            waveform = waveform_to_mono(waveform)
        return waveform

    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        waveform = self.get_waveform(index)

        vector = prepare_waveform(waveform, self.clip_length)

        return vector, self.genre_map.index(self.track_info[index]["genre"])


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
