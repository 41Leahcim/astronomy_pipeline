import pandas
import sys
import tensorflow as tf

# Read the path to the dataset
if len(sys.argv) >= 2:
    path = sys.argv[1]
else:
    path = input("Enter path to dataset: ")

# Read the dataset
dataset = pandas.read_csv(path)

# Remove columns where all values are equal
for label in dataset.columns:
    if dataset[label].min() == dataset[label].max():
        dataset.drop(columns=label, inplace=True)

# Normalize the samples
dataset = dataset / dataset.abs().max()

# Print the first 5 samples
print("\nFirst samples:")
print(dataset[:5])

# Print the minimum, maximum, and mean of each column
print("\nMin:")
print(dataset.min())
print("\nMax:")
print(dataset.max())
print("\nMean:")
print(dataset.mean())

# Train one model per column as label
for column in dataset.columns:
    print(f"{column}:")
    # Convert the label column into a numpy array
    y = dataset[column].to_numpy()

    # Drop the label column from the input
    x = dataset.drop(column, axis=1).to_numpy()

    # Create a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Dense(1)
    ])

    # Add the optimizer, loss algorithm, and accuracy metric
    model.compile(optimizer="adam", loss= tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])

    # Train the model
    model.fit(x, y)
