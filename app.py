import pandas
import sys
import tensorflow as tf

if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = input("Enter path to dataset: ")

dataset = pandas.read_csv(path)
for label in dataset.columns:
    if dataset[label].min() == dataset[label].max():
        del dataset[label]
dataset = dataset / dataset.abs().max()
print(dataset[:5])
print(dataset.min())
print(dataset.max())
print(dataset.mean())
