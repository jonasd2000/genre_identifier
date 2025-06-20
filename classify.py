import json
import argparse

import torch
import torchaudio
from torch import nn

from data_manager import DataManager
from utility.audio import waveform_to_mono, prepare_waveform
from utility.torch import get_device

CLIP_LENGTH = 131_072


def load_track(track_path):
    waveform, sample_rate = torchaudio.load(track_path)
    return waveform_to_mono(waveform), sample_rate

def load_model(path: str):
    return torch.load(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("track_path", help="Path to the track to classify.", type=str)
    parser.add_argument("model_path", help="Path to the model to use.", type=str)
    parser.add_argument("-p", "--passes", help="Number of different samples from the track to analyze.", type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Print more information", action="store_true")
    return parser

def main():
    args = get_parser().parse_args()

    data_manager = DataManager("data/music/")

    track_path = args.track_path
    model_path = args.model_path

    waveform, sample_rate = load_track(track_path)

    device = get_device()
    model = load_model(model_path).to(device)


    with torch.no_grad():
        result = torch.Tensor([[0 for _ in range(model.lin1[-1].out_features)]]).to(device)
        for i in range(args.passes):
            vector = torch.Tensor(prepare_waveform(waveform, CLIP_LENGTH)).to(device)
            res = model(vector)
            if args.verbose:
                print(f"sample {i+1}: {res}")

            result += res

        confidences = nn.Softmax(1)(result)
        print(f"The genre of this track is {data_manager.genre_info[confidences.argmax(1).item()]} (confidence: {(100*(confidences.max(1)[0].item())):.2f}%).")

if __name__ == "__main__":
    main()