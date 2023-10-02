import json
import argparse

import torch
import torchaudio
from torch import nn

from train import get_device
from audio.dataset import prepare_clip

CLIP_LENGTH = 131_072
MODEL_PATH = "4g_faster_model"


def load_track(track_path):
    waveform, sample_rate = torchaudio.load(track_path)
    return waveform, sample_rate

def load_model(path: str):
    return torch.load(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the track to classify.", type=str)
    parser.add_argument("-p", "--passes", help="Number of different samples from the track to analyze.", type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Print more information", action="store_true")
    return parser

def main():
    args = get_parser().parse_args()


    with open("data/music/genre_info.json") as genre_map_file:
        genre_map = json.load(genre_map_file)
        inv_genre_map = {value: key for key, value in genre_map.items()}

    track_path = args.path
    waveform, sample_rate = load_track(track_path)

    device = get_device()
    model = load_model(MODEL_PATH).to(device)


    with torch.no_grad():
        result = torch.Tensor([[0 for _ in range(model.lin1[-1].out_features)]]).to(device)
        for i in range(args.passes):
            clip = torch.Tensor(prepare_clip(waveform, CLIP_LENGTH)).to(device)
            res = model(clip)
            if args.verbose:
                print(f"sample {i+1}: {res}")

            result += res

        confidences = nn.Softmax(1)(result)
        print(f"The genre of this track is {inv_genre_map[confidences.argmax(1).item()]} (confidence: {(100*(confidences.max(1)[0].item())):.2f}%).")

if __name__ == "__main__":
    main()