import tensorflow as tf

# Load a model
model = tf.keras.models.load_model("model.keras")

# Print a summary of the model
print(model.summary())
