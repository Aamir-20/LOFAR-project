# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
    


def print_leaderboard():
    """
    Prints the `runs.csv` file as a leaderboard ordered by the MSE
    column.

    Returns
    -------
    None.

    """
    runs = []
    path = os.path.dirname(__file__) + "/configs/runs.csv" 
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            runs.append(row)

    sorted_runs = sorted(runs, key=lambda x: float(x["mse"]))

    headers = ["#", "MSE", "momentum", "learning_rate", "batch_size", "num_epochs", "filename"]
    rows = []
    for i, run in enumerate(sorted_runs):
        row = [i+1,
               run.get("mse", ""),
               run.get("momentum", "0.9"),
               run.get("learning_rate", ""),
               run.get("batch_size", ""),
               run.get("num_epochs", ""),
               run.get("filename", "")]
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="rst"))


def plot_epoch_loss():
    """
    Plots the `epoch` vs `loss` graph. 
    Both training and validation loss.

    Returns
    -------
    None.

    """
    epoch = list()
    training_loss = list()
    validation_loss = list()
        
    with open("plot.csv", "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        
        for i, row in enumerate(reader):
            epoch.append(float(row[0]))
            training_loss.append(float(row[1]))    
            validation_loss.append(float(row[2]))
    
    plt.plot(epoch, training_loss, label="training")
    plt.plot(epoch, validation_loss, label="validation")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    

print_leaderboard()
plot_epoch_loss()

# 0.001 or 0.01 is one of the best learning rates.

# Load the saved model
#model = torch.load('model.pth')
