import tensorflow as tf
import pandas
import sys

# Read the path to the dataset
if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = input("Enter path to dataset: ")

dataset = pandas.read_csv(path)

# Remove columns where all values are equal
for label in dataset.columns:
    if dataset[label].min() == dataset[label].max():
        dataset.drop(columns=label, inplace=True)

# Normalize the samples
dataset = dataset / dataset.abs().max()

count = dataset.count()[dataset.columns[0]]
print(f"Samples: {count}")
(training, testing) = (dataset.loc[:count / 4 * 3,:], dataset.loc[count / 4 * 3:,:])
print("Training shape:", training.shape)
print("Testing shape:", testing.shape)

# Create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss= tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

# Convert the label column into a numpy array
y = training["fl"].to_numpy()

# Drop the label column from the input
x = training.drop("fl", axis=1).to_numpy()

# Test the model
loss, acc = model.evaluate(x,  y)
print('\nTest accuracy:', acc)

# Train the model
model.fit(x, y)

# Convert the label column into a numpy array
y = testing["fl"].to_numpy()

# Drop the label column from the input
x = testing.drop("fl", axis=1).to_numpy()

# Test the model
loss, acc = model.evaluate(x,  y)
print("Test accuracy:", acc)
print("Test loss:", loss)
