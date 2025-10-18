import pandas as pd
from main import load_data
import numpy as np

# Read the submission.csv file into a DataFrame object
submission_data = pd.read_csv("submission.csv")
# Load test labels
_, test_labels = load_data("test")

correct = 0
wrong = 0

for row in submission_data.itertuples():
    if test_labels[row[1]][row[2]] == 1:
        correct += 1
    else:
        wrong += 1

print(correct/(correct+wrong))
   