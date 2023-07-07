# -*- coding: utf-8 -*-
import csv
from tabulate import tabulate
import os

def print_leaderboard():
    runs = []
    path = os.path.dirname(__file__) + "/configs/runs.csv" 
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            runs.append(row)

    sorted_runs = sorted(runs, key=lambda x: float(x["mse"]))

    headers = ["#", "MSE", "n_inputs", "learning_rate", "batch_size", "num_epochs", "filename"]
    rows = []
    for i, run in enumerate(sorted_runs):
        row = [i+1,
               run.get("mse", ""),
               run.get("n_inputs", ""),
               run.get("learning_rate", ""),
               run.get("batch_size", ""),
               run.get("num_epochs", ""),
               run.get("filename", "")]
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="rst"))

print_leaderboard()
