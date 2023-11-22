import pandas
import sys
import tensorflow as tf

if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = input("Enter path to dataset: ")

dataset = pandas.read_csv(path)
print(dataset)
dataset = dataset / dataset.max()
print(dataset[:5])
