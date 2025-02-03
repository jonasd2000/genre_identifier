from typing import Iterable, Tuple
import os
import json
from pathlib import Path

import polars as pl

from torch.utils.data import Dataset

from audio.dataset import TrackGenreDataset


data_schema = pl.Schema({"id": int, "source": str, "filepath": str, "genre": str, "is_training_data": bool})

class DataManager:
    path: Path
    data: pl.DataFrame

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data_path = self.path / "data.json"

        # Create data file if it doesn't exist
        if not self.data_path.exists():
            with open(self.data_path, "w") as f:
                json.dump([], f)

        self.data = pl.read_json(
            self.data_path,
            schema=data_schema
        )

    def info(self):
        return f"Data Manager\nPath: {self.path}\nTracks: {len(self.data)}\nGenres: {self.data['genre'].n_unique()}"

    def add_track(self, source: str, file_path: str, genre: str, is_training_data: bool) -> Tuple[pl.DataFrame, bool]:
        # tracks are uniquely identified by source and is_training_data
        track_info = self.data.filter((pl.col("source") == source) & (pl.col("is_training_data") == is_training_data))
        # if track is already in database
        if len(track_info) > 0:
            # return existing track
            return track_info, False
        
        track_id = (self.data["id"].max() or 0) + 1
        track_data = pl.DataFrame([{
            "id": track_id,
            "source": source,
            "filepath": file_path,
            "genre": genre,
            "is_training_data": is_training_data,
        }])

        self.data = self.data.vstack(track_data)
        self.data.write_json(self.data_path)
        
        return track_data, True

    def delete_tracks(self, track_ids: Iterable[int]) -> pl.DataFrame:
        tracks_to_delete = self.data.filter(pl.col("id").is_in(track_ids))

        # Delete files
        for filepath in tracks_to_delete["filepath"]:
            if filepath is not None:
                os.remove(filepath)

        # Remove from data
        self.data = self.data.filter(~pl.col("id").is_in(track_ids))
        self.data.write_json(self.data_path)

        return tracks_to_delete

    def get_training_dataset(self, clip_length: int=131_072, assigned_memory: int=0) -> Dataset:
        training_trackinfo = [entry for entry in self.data["tracks"] if entry["is_training_data"]]
        
        return TrackGenreDataset(
            path=self.path,
            track_info=training_trackinfo,
            clip_length=clip_length,
            genre_map=self.genre_info,
            assigned_memory=assigned_memory
        )

    def get_testing_dataset(self, clip_length: int=131_072, assigned_memory: int=0) -> Dataset:
        testing_trackinfo = [entry for entry in self.data["tracks"] if not entry["is_training_data"]]
        
        return TrackGenreDataset(
            path=self.path,
            track_info=testing_trackinfo,
            clip_length=clip_length,
            genre_map=self.genre_info,
            assigned_memory=assigned_memory
        )
