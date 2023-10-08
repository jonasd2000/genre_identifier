import argparse
import os

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import psutil

from kbhit.kbhit import NonBlockingConsole

from data_manager import DataManager
from neural_network.nn import NeuralNetwork, train, test
from utility.general import yes_no
from utility.torch import get_device


CLIP_LENGTH = 131_072
LEARINING_RATE = 3e-2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model. If the path does not exist yet a new model will be created.", type=str, default=None)
    parser.add_argument("-d", "--datapath", help="Path to the data used for training. Default is \"data/music/\"", type=str, default="data/music/")
    parser.add_argument("-m", "--memory", help="Fraction of available memory to load samples into.", type=float, default=0)
    parser.add_argument("-p", "--plot", help="If set, plots the results.", action="store_true")
    parser.add_argument("-l", "--learning_rate", help="Learning rate for the optimizer.", type=float, default=LEARINING_RATE)
    parser.add_argument("-b", "--batch_size", help="Batch size for the training.", type=int, default=32)
    parser.add_argument("-o", "--output", help="If set the predictions are printed every n epochs.", type=int, default=None)
    return parser

def main():
    args = get_parser().parse_args()

    device = get_device()

    data_manager = DataManager(args.datapath) # create the data manager

    if os.path.exists(args.model_path):
        print(f"loading model from {args.model_path}...")
        model = torch.load(args.model_path).to(device)
    else:
        if yes_no("Do you really want to create a new model?"):
            print("creating a new model...")
            model = NeuralNetwork(len(data_manager.genre_info)).to(device)
        else:
            quit()

    training_memory = 0
    testing_memory = 0
    if args.memory > 0:
        training_count, testing_count = data_manager.count_training_testing_tracks()
        available_memory = psutil.virtual_memory().available
        training_memory = args.memory * available_memory * training_count / len(data_manager.track_info)
        testing_memory = args.memory * available_memory * testing_count / len(data_manager.track_info)


    training_dataloader = DataLoader(
        data_manager.get_training_dataset(
            CLIP_LENGTH, 
            assigned_memory=training_memory
        ), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    testing_dataloader = DataLoader(
        data_manager.get_testing_dataset(
            CLIP_LENGTH, 
            assigned_memory=testing_memory
        ), 
        batch_size=args.batch_size
    )


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)


    losses = []
    accuracies = []
    if args.plot:
        fig, ax = plt.subplots()
        confidences_line, = ax.plot([], losses, label="confidence")
        accuracies_line, = ax.plot([], accuracies, label="accuracy")
        ax.grid()
        ax.legend()

    with NonBlockingConsole() as nbc:
        t = 1
        while True:
            print(f"Epoch {t}\n-----------------------")
            train(training_dataloader, model, loss_fn, optimizer, device)
            accuracy, avg_loss = test(testing_dataloader, model, loss_fn, device, verbose=args.output and (t%args.output==0))
            print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

            losses.append(avg_loss)
            accuracies.append(accuracy)
            if t > 1:
                slope, intercept = np.polyfit(np.arange(0, min(t, 20)), np.exp(-np.array(losses[-20:])), deg=1)
                print(f"Gaining {(100*slope):.2f}% confidence/epoch.")

            if args.plot:
                confidences = np.exp(-np.array(losses))
                confidences_line.set_data(np.arange(1, t+1), confidences)
                accuracies_line.set_data(np.arange(1, t+1), accuracies)
                ax.set_xbound(0, t+1)
                ax.set_ybound(0, 1)
                ax.set_title(f"confidence: {confidences[-1]:.3f}, accuracy: {accuracies[-1]:.3f}")
                plt.pause(1e-6)
            t+=1
            if nbc.get_data() == "q":
                break
    if yes_no("Do you want to save the model?"):
        print(f"saving model to {args.model_path}")
        torch.save(model, args.model_path)
        
    if args.plot:
        plt.show()



if __name__ == "__main__":
    main()
