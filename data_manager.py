import json
from pathlib import Path
import os
import argparse
import subprocess

from pytube import YouTube
from pytube import Playlist

from torch.utils.data import Dataset


from audio.dataset import TrackGenreDataset


GENREINFO_FILENAME = "genre_info.json"
TRACKINFO_FILENAME = "track_info.json"



class YouTubeDownloader:
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path

    def download_video(self, url: str, genre: str, training: bool) -> str:
        video = YouTube(url)
        stream = video.streams.filter(only_audio=True).first()

        filename = f"{hash(video.title)}"
        mp4path = stream.download(self.path, filename=filename+".mp4")
        mp3path = mp4path[:-4] + ".mp3"
        subprocess.run(f"ffmpeg -i {mp4path} -ab 128k -ac 2 -ar 44100 -vn {mp3path}", shell=True)
        os.remove(mp4path)

        print(f"The video {url} has been downloaded to {mp3path}")
        
        with open(self.path / TRACKINFO_FILENAME, 'r') as track_data_file:
            data = json.load(track_data_file)
            data.append({
                "genre": genre,
                "filename": filename,
                "training": training
            })

        with open(self.path / TRACKINFO_FILENAME, 'w') as track_data_file:
            json.dump(data, track_data_file)

        return mp3path
    
    def download_playlist(self, url: str, genre: str, training: bool, max_tracks: int=None) -> None:
        for i, video in enumerate(Playlist(url).videos):
            try:
                filename = self.download_video(video.watch_url, genre, training)

            except Exception as e:
                print(e)

            if max_tracks and i+1 >= max_tracks:
                break




class DataManager:
    path: Path
    genre_info: dict
    track_info: list[dict]

    yt_downloader: YouTubeDownloader

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not os.path.exists(self.path):
            self.create_path()


        self.yt_downloader = YouTubeDownloader(self.path)
        
        self.read_path()

    def __repr__(self) -> str:
        return f"DataManager\n Current path: {self.path}\n Registered genres: {list(self.genre_info.keys())}\n"
    
    # a function that counts the number of tracks per genre
    def count_tracks_per_genre(self) -> dict:
        genre_count = {}
        for track in self.track_info:
            genre = track["genre"]
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1
        return genre_count
    
    # a function that counts the number of tracks that are training
    def count_training_testing_tracks(self) -> tuple[int, int]:
        training_count = 0
        testing_count = 0
        for track in self.track_info:
            if track["training"]:
                training_count += 1
            else:
                testing_count += 1
        return training_count, testing_count
    

    def info(self) -> str:
        training_count, testing_count = self.count_training_testing_tracks()
        return f"{repr(self)} {self.count_tracks_per_genre()}\n Training {training_count} / Testing {testing_count}\n"

    def confirm_path_creation(self):
        while True:
            inp = input(f"The path {self.path} does not yet exist. Do you want to create it? (y/n) ")
            match inp.lower():
                case "y": return True
                case "n": return False
                case _: pass

    def create_path(self):
        if not self.confirm_path_creation():
            return

        os.mkdir(self.path)
        with open(self.path / GENREINFO_FILENAME, "w") as genreinfo_file:
            json.dump({}, genreinfo_file)
        with open(self.path / TRACKINFO_FILENAME, "w") as trackinfo_file:
            json.dump([], trackinfo_file)


    def read_path(self) -> None:
        self.read_genre_info()
        self.read_track_info()

    def read_genre_info(self) -> None:
        with open(self.path / GENREINFO_FILENAME, "r") as genreinfo_file:
            self.genre_info = json.load(genreinfo_file)
    def read_track_info(self) -> None:
        with open(self.path / TRACKINFO_FILENAME, "r") as trackinfo_file:
            self.track_info = json.load(trackinfo_file)

    def write_genre_info(self) -> None:
        with open(self.path / GENREINFO_FILENAME, "w") as genreinfo_file:
            self.genre_info = json.dump(self.genre_info, genreinfo_file)
    def write_track_info(self) -> None:
        with open(self.path / TRACKINFO_FILENAME, "w") as trackinfo_file:
            self.genre_info = json.dump(self.track_info, trackinfo_file)

    def register_genre(self, genre: str) -> None:
        self.genre_info[genre] = max(list(self.genre_info.values()))+1 if self.genre_info.values() else 0
        self.write_genre_info()

    def delete_entry(self, index: int) -> None:
        filename = self.track_info[index]["filename"]
        
        os.remove(self.path / (filename + ".mp3"))
        del self.track_info[index]

        self.write_track_info()

    def delete_entries(self, **properties: dict) -> int:
        indices_to_delete = []

        for i, entry in enumerate(self.track_info):
            all_properties_match = True
            for property, value in properties.items():
                if entry[property] != value:
                    all_properties_match = False

            if all_properties_match:
                indices_to_delete.append(i)

        for index in sorted(indices_to_delete, reverse=True):
            self.delete_entry(index)

        return len(indices_to_delete)

    def get_training_dataset(self, clip_length: int=131_072, genres: list[str]=None, load_to_memory: bool=False) -> Dataset:
        training_trackinfo = [entry for entry in self.track_info if entry["training"]]
        if genres:
            # filter testing_trackinfo based on genres
            training_trackinfo = [entry for entry in training_trackinfo if entry["genre"] in genres]
        
        return TrackGenreDataset(
            path=self.path,
            track_info=training_trackinfo,
            clip_length=clip_length,
            genre_map=self.genre_info,
            load_to_memory=load_to_memory
        )

    def get_testing_dataset(self, clip_length: int=131_072, genres: list[str]=None, load_to_memory: bool=False) -> Dataset:
        testing_trackinfo = [entry for entry in self.track_info if not entry["training"]]
        if genres:
            # filter testing_trackinfo based on genres
            testing_trackinfo = [entry for entry in testing_trackinfo if entry["genre"] in genres]
       
        return TrackGenreDataset(
            path=self.path,
            track_info=testing_trackinfo,
            clip_length=clip_length,
            genre_map=self.genre_info,
            load_to_memory=load_to_memory
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the database to manage.", type=str)
    return parser

def main():
    args = get_parser().parse_args()

    dm = DataManager(args.path)
    print("Succesfully loaded DataManager as 'dm'...\n")
    print(dm)
    while True:
        try:
            exec(input(">>> "))
        except SyntaxError as e:
            print(e)

if __name__ == "__main__":
    main()